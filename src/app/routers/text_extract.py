import asyncio
from collections.abc import Awaitable
from pathlib import PurePosixPath

from anyio import CapacityLimiter
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.concurrency import run_limited
from app.core.config import (
    COMPANY_NAME,
    HEADER_UNIT_COUNT,
)
from app.core.deps import get_extract_client
from app.core.logger import get_logger
from app.models.extract import (
    ExtractedUnit,
    ExtractResponse,
    UnitExtraction,
)
from app.models.kpi import (
    DocumentMeta,
    HeaderExtraction,
    KpiExtraction,
    KpiPageExtraction,
)
from app.utils.extract_client import ExtractClient
from app.utils.office import CONVERTIBLE_EXTENSIONS, convert_to_pdf
from app.utils.pdf import pdf_to_page_images

logger = get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["extract"])

PDF_EXTENSION = ".pdf"

ALLOWED_EXTENSIONS = {PDF_EXTENSION} | CONVERTIBLE_EXTENSIONS

PDF_CONVERSION_CONCURRENCY = 5
PAGE_RENDER_CONCURRENCY = 4

MAX_FILES_PER_REQUEST = 20
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES_PER_FILE = 100

NO_HEADER = HeaderExtraction(
    holding=None,
    reporting_period=None,
)

_pdf_conversion_limiter = CapacityLimiter(PDF_CONVERSION_CONCURRENCY)
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
    """True if the upload has a handled extension and acceptable size."""
    return (
        _extension_of(file) in ALLOWED_EXTENSIONS
        and _within_size_limit(file)
    )


def _split_by_support(
    files: list[UploadFile],
) -> tuple[list[UploadFile], list[str]]:
    """Split uploads into supported files and skipped filenames."""
    supported: list[UploadFile] = []
    skipped: list[str] = []

    for file in files:
        if _is_supported(file):
            supported.append(file)
        else:
            skipped.append(_name_of(file))

    return supported, skipped


async def _or_fallback[T](
    coro: Awaitable[T],
    fallback: T,
    label: str,
    what: str,
) -> T:
    """Await a coroutine and return fallback if that item fails."""
    try:
        return await coro
    except Exception:
        logger.exception("%s failed for '%s'", what, label)
        return fallback


def _unit(
    file: UploadFile,
    extension: str,
    label: str,
) -> ExtractedUnit:
    """Build a unit containing page metadata only."""
    return ExtractedUnit(
        source=_name_of(file),
        extension=extension,
        label=label,
    )


async def _read_as_pdf(
    file: UploadFile,
    extension: str,
) -> bytes:
    """Return the upload as PDF bytes."""
    if extension in CONVERTIBLE_EXTENSIONS:
        return await run_limited(
            _pdf_conversion_limiter,
            convert_to_pdf,
            file,
            extension,
        )

    return await file.read()


async def _header_of(
    images: list[str],
    file: UploadFile,
    client: ExtractClient,
) -> HeaderExtraction:
    """Extract document metadata from the first page images."""
    header_images = images[:HEADER_UNIT_COUNT]

    if not header_images:
        logger.warning(
            "No page images available for header extraction for '%s'",
            file.filename,
        )
        return NO_HEADER

    return await _or_fallback(
        client.extract_header(header_images),
        NO_HEADER,
        _name_of(file),
        "Header extraction",
    )


async def _extract_pdf(
    file: UploadFile,
    extract: ExtractClient,
) -> tuple[list[ExtractedUnit], HeaderExtraction]:
    """Convert one upload to PDF, render its pages, and extract them."""
    extension = _extension_of(file)

    pdf = await _read_as_pdf(file, extension)

    images = await run_limited(
        _render_limiter,
        pdf_to_page_images,
        pdf,
        MAX_PAGES_PER_FILE,
    )

    logger.info(
        "Rendered %d page(s) of '%s' for structured extraction",
        len(images),
        file.filename,
    )

    header = await _header_of(images, file, extract)

    units = [
        _unit(
            file,
            extension,
            str(page_num),
        )
        for page_num in range(1, len(images) + 1)
    ]

    return units, header, images


type _PdfResult = (
    tuple[list[ExtractedUnit], HeaderExtraction, list[str]]
    | None
)


async def _extract_all_pdfs(
    files: list[UploadFile],
    extract: ExtractClient,
) -> tuple[
    list[ExtractedUnit],
    dict[str, HeaderExtraction],
    dict[str, list[str]],
]:
    """Render every PDF upload and run its header extraction."""
    results: list[_PdfResult] = await asyncio.gather(
        *(
            _or_fallback(
                _extract_pdf(file, extract),
                None,
                _name_of(file),
                "PDF extraction",
            )
            for file in files
        )
    )

    pairs = [
        (file, result)
        for file, result in zip(files, results, strict=True)
    ]

    failed = [
        _name_of(file)
        for file, result in pairs
        if result is None
    ]

    if failed:
        logger.warning(
            "Dropping files that failed PDF extraction: %s",
            ", ".join(failed),
        )

    units = [
        unit
        for _, result in pairs
        if result is not None
        for unit in result[0]
    ]

    headers = {
        _name_of(file): result[1]
        for file, result in pairs
        if result is not None
    }

    images = {
        _name_of(file): result[2]
        for file, result in pairs
        if result is not None
    }

    return units, headers, images


def _unit_label(unit: ExtractedUnit) -> str:
    """Identify one unit in logs and failure responses."""
    return f"{unit.source}#{unit.label}"


def _with_document(
    page: KpiPageExtraction,
    unit: ExtractedUnit,
    header: HeaderExtraction,
) -> KpiExtraction:
    """Attach document metadata to page-level KPI extraction."""
    return KpiExtraction(
        document=DocumentMeta(
            company=COMPANY_NAME,
            holding=header.holding,
            reporting_period=header.reporting_period,
            source=unit.source,
        ),
        kpis=page.kpis,
    )


async def _extract_all_units(
    units: list[ExtractedUnit],
    headers: dict[str, HeaderExtraction],
    images: dict[str, list[str]],
    client: ExtractClient,
) -> tuple[list[UnitExtraction], list[str]]:
    """Extract KPIs directly from the rendered page images."""
    eligible: list[
        tuple[ExtractedUnit, HeaderExtraction, str]
    ] = []

    for unit in units:
        file_images = images.get(unit.source)

        if not file_images:
            logger.warning(
                "No images available for '%s'",
                _unit_label(unit),
            )
            continue

        try:
            page_number = int(unit.label)
        except ValueError:
            logger.warning(
                "Invalid page label '%s' for '%s'",
                unit.label,
                unit.source,
            )
            continue

        image_index = page_number - 1

        if image_index < 0 or image_index >= len(file_images):
            logger.warning(
                "No image for '%s'",
                _unit_label(unit),
            )
            continue

        header = headers.get(unit.source, NO_HEADER)
        image_url = file_images[image_index]

        eligible.append(
            (unit, header, image_url)
        )

    logger.info(
        "Extracting KPIs from %d page(s)",
        len(eligible),
    )

    results: list[KpiPageExtraction | None] = await asyncio.gather(
        *(
            _or_fallback(
                client.extract_unit(
                    image_url,
                    header.reporting_period,
                ),
                None,
                _unit_label(unit),
                "Structured extraction",
            )
            for unit, header, image_url in eligible
        )
    )

    pairs = list(
        zip(
            eligible,
            results,
            strict=True,
        )
    )

    extractions = [
        UnitExtraction(
            source=unit.source,
            label=unit.label,
            data=_with_document(
                result,
                unit,
                header,
            ),
        )
        for (unit, header, _), result in pairs
        if result is not None
    ]

    failed = [
        _unit_label(unit)
        for (unit, _, _), result in pairs
        if result is None
    ]

    if failed:
        logger.warning(
            "Units that failed extraction: %s",
            ", ".join(failed),
        )

    return extractions, failed


@router.post("/text")
async def extract_pdf(
    files: list[UploadFile] = File(...),
    extract: ExtractClient = Depends(get_extract_client),
) -> ExtractResponse:
    """Render uploaded PDF-compatible files and extract KPI JSON directly
    from their page images.
    """
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many files; limit is "
                f"{MAX_FILES_PER_REQUEST} per request"
            ),
        )

    supported, skipped = _split_by_support(files)

    if skipped:
        logger.warning(
            "Skipping unsupported files: %s",
            ", ".join(skipped),
        )

    units, headers, images = await _extract_all_pdfs(
        supported,
        extract,
    )

    extractions, failed = await _extract_all_units(
        units,
        headers,
        images,
        extract,
    )
    # for time being: remove base64 image urls surgically
    for unit in units:
        unit.content = None

    return ExtractResponse(
        units=units,
        skipped=skipped,
        extractions=extractions,
        failed_extractions=failed,
    )
