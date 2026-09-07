import asyncio
from collections.abc import Awaitable
from pathlib import PurePosixPath

from anyio import CapacityLimiter
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.concurrency import run_limited
from app.core.config import (
    COMPANY_NAME,
    EXTRACT_MAX_INPUT_CHARS,
    HEADER_UNIT_COUNT,
)
from app.core.deps import get_extract_client, get_ocr_client
from app.core.logger import get_logger
from app.models.extract import (
    ExtractedUnit,
    ExtractResponse,
    SheetContent,
    UnitExtraction,
)
from app.models.kpi import (
    DocumentMeta,
    HeaderExtraction,
    KpiExtraction,
    KpiPageExtraction,
)
from app.utils.extract_client import ExtractClient
from app.utils.markdown import XLSX_EXTENSION, xlsx_to_markdown
from app.utils.ocr_client import OcrClient
from app.utils.office import CONVERTIBLE_EXTENSIONS, convert_to_pdf
from app.utils.pdf import pdf_to_page_images

logger = get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["extract"])

PDF_EXTENSION = ".pdf"
ALLOWED_EXTENSIONS = {XLSX_EXTENSION, PDF_EXTENSION} | CONVERTIBLE_EXTENSIONS

XLSX_CONCURRENCY = 5
PDF_CONVERSION_CONCURRENCY = 5
PAGE_RENDER_CONCURRENCY = 4

# A single request must finish inside the platform's HTTP window, so cap the
# batch rather than letting one upload set trigger a redirect chain.
MAX_FILES_PER_REQUEST = 20
MAX_FILE_BYTES = 20 * 1024 * 1024
# 10 MB of PDF can be hundreds of pages, and every page is one OCR request.
MAX_PAGES_PER_FILE = 100

# Used when no header pass ran (xlsx) or when it failed. Read-only, so one
# shared instance is safe.
NO_HEADER = HeaderExtraction(holding=None, reporting_period=None)

# Both conversions block, so each gets its own thread limiter rather than
# competing for the default pool FastAPI shares with sync endpoints.
_xlsx_limiter = CapacityLimiter(XLSX_CONCURRENCY)
_pdf_conversion_limiter = CapacityLimiter(PDF_CONVERSION_CONCURRENCY)
# Rasterizing pages blocks too, and holds a full-page bitmap while it runs.
_render_limiter = CapacityLimiter(PAGE_RENDER_CONCURRENCY)


def _extension_of(file: UploadFile) -> str:
    """Return the lowercased extension of the upload's base filename."""
    name = PurePosixPath(file.filename or "").name
    return PurePosixPath(name).suffix.lower()


def _name_of(file: UploadFile) -> str:
    """Return the upload's filename, or a placeholder for an unnamed one."""
    return file.filename or "unnamed"


def _within_size_limit(file: UploadFile) -> bool:
    """True if the upload is small enough to process."""
    if file.size is not None and file.size > MAX_FILE_BYTES:
        logger.warning(
            "Rejecting '%s': %d bytes exceeds the %d byte limit",
            file.filename,
            file.size,
            MAX_FILE_BYTES,
        )
        return False

    return True


def _is_supported(file: UploadFile) -> bool:
    """True if the upload has a handled extension and an acceptable size."""
    return _extension_of(file) in ALLOWED_EXTENSIONS and _within_size_limit(file)


def _split_by_support(
    files: list[UploadFile],
) -> tuple[list[UploadFile], list[str]]:
    """Split uploads into supported files and the names of skipped ones."""
    supported: list[UploadFile] = []
    skipped: list[str] = []

    for file in files:
        if _is_supported(file):
            supported.append(file)
        else:
            skipped.append(_name_of(file))

    return supported, skipped


def _split_by_kind(
    files: list[UploadFile],
) -> tuple[list[UploadFile], list[UploadFile]]:
    """Split supported uploads into xlsx files and files handled as PDFs."""
    xlsx_files = [f for f in files if _extension_of(f) == XLSX_EXTENSION]
    pdf_files = [f for f in files if _extension_of(f) != XLSX_EXTENSION]
    return xlsx_files, pdf_files


async def _or_fallback[T](coro: Awaitable[T], fallback: T, label: str, what: str) -> T:
    """Await `coro`, logging and returning `fallback` if that one item fails.

    Keeps a single bad upload or unit from failing the whole batch. Takes a
    label rather than an UploadFile because extraction is keyed by unit, which
    has no upload of its own.
    """
    try:
        return await coro
    except Exception:
        logger.exception("%s failed for '%s'", what, label)
        return fallback


def _unit(file: UploadFile, extension: str, label: str, content: str) -> ExtractedUnit:
    """Build a unit tagged with the upload it came from."""
    return ExtractedUnit(
        source=_name_of(file),
        extension=extension,
        label=label,
        content=content,
    )


async def _extract_xlsx(file: UploadFile) -> list[SheetContent]:
    """Extract an xlsx spreadsheet as one markdown table per sheet."""
    # Conversion is blocking, so keep it off the event loop.
    sheets = await run_limited(_xlsx_limiter, xlsx_to_markdown, file)
    logger.info("Converted '%s' to markdown (%d sheets)", file.filename, len(sheets))
    return sheets


def _page_content(file: UploadFile, page_num: int, page: str | BaseException) -> str:
    """Return one page's markdown, or empty content if that page failed.

    A placeholder keeps the page numbering of the remaining pages intact.
    """
    if isinstance(page, BaseException):
        logger.warning("OCR failed for '%s' page %d: %s", file.filename, page_num, page)
        return ""

    return page.strip()


def _header_page_failures(pages: list[str | BaseException]) -> list[str]:
    """Return one message per leading page whose OCR raised.

    The header pass reads OCR output, so a failed call and a page that really
    holds no text both arrive here as "nothing to read". Only one of them is
    the document's fault, so they must not be reported the same way.
    """
    return [
        f"page {page_num}: {page}"
        for page_num, page in enumerate(pages, start=1)
        if isinstance(page, BaseException)
    ]


def _log_missing_header(file: UploadFile, page_count: int, failures: list[str]) -> None:
    """Say why the header pass had nothing to read."""
    if failures:
        logger.warning(
            "Header left null for '%s': OCR failed for %d of the first %d page(s) (%s)",
            file.filename,
            len(failures),
            page_count,
            "; ".join(failures),
        )
        return

    logger.warning(
        "No text in the first %d page(s) of '%s'; header left null",
        page_count,
        file.filename,
    )


def _header_text(pages: list[str | BaseException]) -> str:
    """Join the leading pages the header pass should read.

    Truncated rather than rejected when oversized: the holding and the
    reporting period sit at the top of the first page, so a dropped tail costs
    nothing here -- unlike the KPI pass, where it would lose figures.
    """
    usable = [page.strip() for page in pages if isinstance(page, str) and page.strip()]
    return "\n\n".join(usable)[:EXTRACT_MAX_INPUT_CHARS]


async def _header_of(
    page_tasks: list[asyncio.Task[str]], file: UploadFile, client: ExtractClient
) -> HeaderExtraction:
    """Read the document-level header from an upload's first pages.

    Awaits only the leading pages, so this runs while the rest of the document
    is still being OCR'd. Those tasks are awaited again by the caller's gather,
    which is allowed and does not re-run them.

    A failure here is not fatal: the upload's units simply carry a null holding
    and reporting period.
    """
    pages = await asyncio.gather(*page_tasks, return_exceptions=True)
    text = _header_text(pages)

    if not text:
        _log_missing_header(file, len(page_tasks), _header_page_failures(pages))
        return NO_HEADER

    return await _or_fallback(
        client.extract_header(text), NO_HEADER, _name_of(file), "Header extraction"
    )


# Fan-out is bounded by the client's semaphore; batching across pages is
# handled by vLLM + Modal on the server side.
async def _ocr_extract(
    file: UploadFile,
    extension: str,
    pdf: bytes,
    ocr: OcrClient,
    extract: ExtractClient,
) -> tuple[list[ExtractedUnit], HeaderExtraction]:
    """Extract a PDF's pages as markdown via OCR, one unit per page.

    The header pass starts as soon as the leading pages are OCR'd and overlaps
    the OCR of the rest, so for anything longer than a couple of pages it adds
    no wall-clock time.
    """
    # Rendering is blocking, so keep it off the event loop.
    images = await run_limited(
        _render_limiter, pdf_to_page_images, pdf, MAX_PAGES_PER_FILE
    )
    logger.info("Rendered %d page(s) of '%s' for OCR", len(images), file.filename)

    # Tasks rather than bare coroutines, so the header pass can await the
    # leading pages while the gather below awaits all of them.
    page_tasks = [asyncio.create_task(ocr.ocr_page(image)) for image in images]
    header_task = asyncio.create_task(
        _header_of(page_tasks[:HEADER_UNIT_COUNT], file, extract)
    )

    # gather preserves order, so index i holds page i + 1. One bad page must
    # not lose the pages that did extract.
    pages = await asyncio.gather(*page_tasks, return_exceptions=True)
    header = await header_task

    units = [
        _unit(file, extension, str(page_num), _page_content(file, page_num, page))
        for page_num, page in enumerate(pages, start=1)
    ]
    return units, header


async def _read_as_pdf(file: UploadFile, extension: str) -> bytes:
    """Return the upload as PDF bytes, converting pptx/docx if needed."""
    if extension in CONVERTIBLE_EXTENSIONS:
        # convert_to_pdf is blocking, so keep it off the event loop.
        return await run_limited(
            _pdf_conversion_limiter, convert_to_pdf, file, extension
        )

    return await file.read()


async def _extract_pdf(
    file: UploadFile, ocr: OcrClient, extract: ExtractClient
) -> tuple[list[ExtractedUnit], HeaderExtraction]:
    """Convert one upload to PDF and OCR it right away.

    Each stage has its own cap, so a file enters OCR as soon as its own
    conversion finishes instead of waiting for the whole batch to convert.
    """
    extension = _extension_of(file)
    pdf = await _read_as_pdf(file, extension)
    return await _ocr_extract(file, extension, pdf, ocr, extract)


async def _extract_all_xlsx(files: list[UploadFile]) -> list[ExtractedUnit]:
    """Extract every xlsx upload concurrently, one unit per sheet.

    No header pass runs for spreadsheets: their first two sheets are data, not
    a cover page, so holding and reporting period stay null for now.
    """
    sheet_lists = await asyncio.gather(
        *(
            _or_fallback(_extract_xlsx(file), [], _name_of(file), "Markdown conversion")
            for file in files
        )
    )

    return [
        _unit(file, XLSX_EXTENSION, sheet.name, sheet.content)
        for file, sheets in zip(files, sheet_lists, strict=True)
        for sheet in sheets
    ]


type _PdfResult = tuple[list[ExtractedUnit], HeaderExtraction] | None


async def _extract_all_pdfs(
    files: list[UploadFile], ocr: OcrClient, extract: ExtractClient
) -> tuple[list[ExtractedUnit], dict[str, HeaderExtraction]]:
    """Run the convert-then-OCR pipeline for every non-xlsx upload.

    Returns the units plus each upload's header, keyed by filename so it can be
    stamped onto that upload's units later. Files that fail are logged and
    dropped; the response does not report them yet. `None` marks a failure,
    which an empty page list would not.
    """
    results: list[_PdfResult] = await asyncio.gather(
        *(
            _or_fallback(
                _extract_pdf(file, ocr, extract),
                None,
                _name_of(file),
                "PDF extraction",
            )
            for file in files
        )
    )
    pairs = [(file, result) for file, result in zip(files, results, strict=True)]

    _log_failed_pdfs(pairs)

    units = [unit for _, result in pairs if result is not None for unit in result[0]]
    headers = {_name_of(file): r[1] for file, r in pairs if r is not None}
    return units, headers


def _log_failed_pdfs(pairs: list[tuple[UploadFile, _PdfResult]]) -> None:
    """Log the uploads whose convert-then-OCR pipeline failed outright."""
    failed = [_name_of(file) for file, result in pairs if result is None]

    if failed:
        logger.warning(
            "Dropping files that failed PDF extraction: %s", ", ".join(failed)
        )


def _extractable(unit: ExtractedUnit) -> bool:
    """True if the unit is worth spending a GPU call on.

    Empty content means OCR failed for that page. Oversized content would be
    rejected by the server for exceeding its context window, and truncating it
    here would silently drop whatever KPIs were in the tail.
    """
    if not unit.content.strip():
        return False

    if len(unit.content) > EXTRACT_MAX_INPUT_CHARS:
        logger.warning(
            "Skipping extraction for '%s' %s: %d chars exceeds the %d char budget",
            unit.source,
            unit.label,
            len(unit.content),
            EXTRACT_MAX_INPUT_CHARS,
        )
        return False

    return True


def _unit_label(unit: ExtractedUnit) -> str:
    """Identify one unit in logs and in the response's failure list."""
    return f"{unit.source}#{unit.label}"


def _with_document(
    page: KpiPageExtraction, unit: ExtractedUnit, header: HeaderExtraction
) -> KpiExtraction:
    """Combine one unit's figures with the document metadata we already hold.

    None of these four fields comes from the KPI pass. The company is a
    constant, the holding and reporting period were read once from the
    upload's first pages, and the source is the filename -- so the model is
    never asked to repeat information we can supply exactly.
    """
    return KpiExtraction(
        document=DocumentMeta(
            company=COMPANY_NAME,
            holding=header.holding,
            reporting_period=header.reporting_period,
            source=unit.source,
        ),
        kpis=page.kpis,
    )


# Fan-out is bounded by the extraction client's 32-unit semaphore; batching
# across units is handled by vLLM + Modal on the server side.
async def _extract_all_units(
    units: list[ExtractedUnit],
    headers: dict[str, HeaderExtraction],
    client: ExtractClient,
) -> tuple[list[UnitExtraction], list[str]]:
    """Extract KPIs from every eligible unit, reporting the ones that failed.

    One call per unit, so nothing merges figures across pages: a KPI whose
    monthly value and YTD value are printed on different pages comes back as
    two separate extractions.
    """
    eligible = [unit for unit in units if _extractable(unit)]
    logger.info("Extracting KPIs from %d of %d unit(s)", len(eligible), len(units))

    results: list[KpiPageExtraction | None] = await asyncio.gather(
        *(
            _or_fallback(
                client.extract_unit(unit.content),
                None,
                _unit_label(unit),
                "Structured extraction",
            )
            for unit in eligible
        )
    )
    pairs = list(zip(eligible, results, strict=True))

    extractions = [
        UnitExtraction(
            source=unit.source,
            label=unit.label,
            data=_with_document(result, unit, headers.get(unit.source, NO_HEADER)),
        )
        for unit, result in pairs
        if result is not None
    ]
    failed = [_unit_label(unit) for unit, result in pairs if result is None]

    if failed:
        logger.warning("Units that failed extraction: %s", ", ".join(failed))

    return extractions, failed


@router.post("/text")
async def extract_text(
    files: list[UploadFile] = File(...),
    ocr: OcrClient = Depends(get_ocr_client),
    extract: ExtractClient = Depends(get_extract_client),
) -> ExtractResponse:
    """Extract content from the uploaded xlsx/pptx/docx/pdf files, then pull
    structured KPI JSON out of each extracted page or sheet."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files; limit is {MAX_FILES_PER_REQUEST} per request",
        )

    supported, skipped = _split_by_support(files)

    if skipped:
        logger.warning("Skipping unsupported files: %s", ", ".join(skipped))

    xlsx_files, pdf_files = _split_by_kind(supported)

    xlsx_units, (pdf_units, headers) = await asyncio.gather(
        _extract_all_xlsx(xlsx_files),
        _extract_all_pdfs(pdf_files, ocr, extract),
    )
    units = xlsx_units + pdf_units

    # Sequential after OCR, not concurrent with it: a unit cannot be extracted
    # before it has been read. The header pass already ran, during OCR.
    extractions, failed = await _extract_all_units(units, headers, extract)

    return ExtractResponse(
        units=units,
        skipped=skipped,
        extractions=extractions,
        failed_extractions=failed,
    )
