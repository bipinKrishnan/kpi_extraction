import asyncio
from collections.abc import Awaitable
from pathlib import PurePosixPath

from anyio import CapacityLimiter
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.concurrency import run_limited
from app.core.deps import get_ocr_client
from app.core.logger import get_logger
from app.models.extract import ExtractedUnit, ExtractResponse, SheetContent
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
            skipped.append(file.filename or "unnamed")

    return supported, skipped


def _split_by_kind(
    files: list[UploadFile],
) -> tuple[list[UploadFile], list[UploadFile]]:
    """Split supported uploads into xlsx files and files handled as PDFs."""
    xlsx_files = [f for f in files if _extension_of(f) == XLSX_EXTENSION]
    pdf_files = [f for f in files if _extension_of(f) != XLSX_EXTENSION]
    return xlsx_files, pdf_files


async def _or_fallback[T](
    coro: Awaitable[T], fallback: T, file: UploadFile, what: str
) -> T:
    """Await `coro`, logging and returning `fallback` if that one upload fails.

    Keeps a single bad upload from failing the whole batch.
    """
    try:
        return await coro
    except Exception:
        logger.exception("%s failed for '%s'", what, file.filename)
        return fallback


def _unit(file: UploadFile, extension: str, label: str, content: str) -> ExtractedUnit:
    """Build a unit tagged with the upload it came from."""
    return ExtractedUnit(
        source=file.filename or "unnamed",
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


# Fan-out is bounded by the client's semaphore; batching across pages is
# handled by vLLM + Modal on the server side.
async def _ocr_extract(
    file: UploadFile, extension: str, pdf: bytes, ocr: OcrClient
) -> list[ExtractedUnit]:
    """Extract a PDF's pages as markdown via OCR, one unit per page."""
    # Rendering is blocking, so keep it off the event loop.
    images = await run_limited(
        _render_limiter, pdf_to_page_images, pdf, MAX_PAGES_PER_FILE
    )
    logger.info("Rendered %d page(s) of '%s' for OCR", len(images), file.filename)

    # gather preserves order, so index i holds page i + 1. One bad page must
    # not lose the pages that did extract.
    pages = await asyncio.gather(
        *(ocr.ocr_page(image) for image in images), return_exceptions=True
    )

    return [
        _unit(file, extension, str(page_num), _page_content(file, page_num, page))
        for page_num, page in enumerate(pages, start=1)
    ]


async def _read_as_pdf(file: UploadFile, extension: str) -> bytes:
    """Return the upload as PDF bytes, converting pptx/docx if needed."""
    if extension in CONVERTIBLE_EXTENSIONS:
        # convert_to_pdf is blocking, so keep it off the event loop.
        return await run_limited(
            _pdf_conversion_limiter, convert_to_pdf, file, extension
        )

    return await file.read()


async def _extract_pdf(file: UploadFile, ocr: OcrClient) -> list[ExtractedUnit]:
    """Convert one upload to PDF and OCR it right away.

    Each stage has its own cap, so a file enters OCR as soon as its own
    conversion finishes instead of waiting for the whole batch to convert.
    """
    extension = _extension_of(file)
    pdf = await _read_as_pdf(file, extension)
    return await _ocr_extract(file, extension, pdf, ocr)


async def _extract_all_xlsx(files: list[UploadFile]) -> list[ExtractedUnit]:
    """Extract every xlsx upload concurrently, one unit per sheet."""
    sheet_lists = await asyncio.gather(
        *(
            _or_fallback(_extract_xlsx(file), [], file, "Markdown conversion")
            for file in files
        )
    )

    return [
        _unit(file, XLSX_EXTENSION, sheet.name, sheet.content)
        for file, sheets in zip(files, sheet_lists, strict=True)
        for sheet in sheets
    ]


async def _extract_all_pdfs(
    files: list[UploadFile], ocr: OcrClient
) -> list[ExtractedUnit]:
    """Run the convert-then-OCR pipeline for every non-xlsx upload.

    Files that fail are logged and dropped; the response does not report them
    yet. `None` marks a failure, which an empty page list would not.
    """
    batches: list[list[ExtractedUnit] | None] = await asyncio.gather(
        *(
            _or_fallback(_extract_pdf(file, ocr), None, file, "PDF extraction")
            for file in files
        )
    )
    pairs = list(zip(files, batches, strict=True))

    failed = [file.filename or "unnamed" for file, batch in pairs if batch is None]

    if failed:
        logger.warning(
            "Dropping files that failed PDF extraction: %s", ", ".join(failed)
        )

    return [unit for _, batch in pairs if batch is not None for unit in batch]


@router.post("/text")
async def extract_text(
    files: list[UploadFile] = File(...),
    ocr: OcrClient = Depends(get_ocr_client),
) -> ExtractResponse:
    """Extract content from the uploaded xlsx/pptx/docx/pdf files."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files; limit is {MAX_FILES_PER_REQUEST} per request",
        )

    supported, skipped = _split_by_support(files)

    if skipped:
        logger.warning("Skipping unsupported files: %s", ", ".join(skipped))

    xlsx_files, pdf_files = _split_by_kind(supported)

    xlsx_units, pdf_units = await asyncio.gather(
        _extract_all_xlsx(xlsx_files),
        _extract_all_pdfs(pdf_files, ocr),
    )

    return ExtractResponse(units=xlsx_units + pdf_units, skipped=skipped)
