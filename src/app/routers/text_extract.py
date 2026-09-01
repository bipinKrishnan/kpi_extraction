import asyncio
from collections.abc import Awaitable
from pathlib import PurePosixPath

from anyio import CapacityLimiter
from fastapi import APIRouter, File, UploadFile

from app.core.concurrency import run_limited
from app.core.logger import get_logger
from app.models.extract import ExtractedUnit, ExtractResponse, SheetContent
from app.utils.markdown import XLSX_EXTENSION, xlsx_to_markdown
from app.utils.office import CONVERTIBLE_EXTENSIONS, convert_to_pdf

logger = get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["extract"])

PDF_EXTENSION = ".pdf"
ALLOWED_EXTENSIONS = {XLSX_EXTENSION, PDF_EXTENSION} | CONVERTIBLE_EXTENSIONS

XLSX_CONCURRENCY = 5
PDF_CONVERSION_CONCURRENCY = 5

# Both conversions block, so each gets its own thread limiter rather than
# competing for the default pool FastAPI shares with sync endpoints.
_xlsx_limiter = CapacityLimiter(XLSX_CONCURRENCY)
_pdf_conversion_limiter = CapacityLimiter(PDF_CONVERSION_CONCURRENCY)


def _extension_of(file: UploadFile) -> str:
    """Return the lowercased extension of the upload's base filename."""
    name = PurePosixPath(file.filename or "").name
    return PurePosixPath(name).suffix.lower()


def _split_by_support(
    files: list[UploadFile],
) -> tuple[list[UploadFile], list[str]]:
    """Split uploads into supported files and the names of skipped ones."""
    supported: list[UploadFile] = []
    skipped: list[str] = []

    for file in files:
        if _extension_of(file) in ALLOWED_EXTENSIONS:
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


# concurrency handled by vLLM + modal on server-side
async def _ocr_extract(
    file: UploadFile, extension: str, pdf: bytes
) -> list[ExtractedUnit]:
    """Extract a PDF's pages as custom XML via OCR."""
    # TODO: implement the custom XML extraction, one unit per page.
    logger.info("No OCR extractor yet for '%s' (%d PDF bytes)", file.filename, len(pdf))
    return []


async def _read_as_pdf(file: UploadFile, extension: str) -> bytes:
    """Return the upload as PDF bytes, converting pptx/docx if needed."""
    if extension in CONVERTIBLE_EXTENSIONS:
        # convert_to_pdf is blocking, so keep it off the event loop.
        return await run_limited(
            _pdf_conversion_limiter, convert_to_pdf, file, extension
        )

    return await file.read()


async def _extract_pdf(file: UploadFile) -> list[ExtractedUnit]:
    """Convert one upload to PDF and OCR it right away.

    Each stage has its own cap, so a file enters OCR as soon as its own
    conversion finishes instead of waiting for the whole batch to convert.
    """
    extension = _extension_of(file)
    pdf = await _read_as_pdf(file, extension)
    return await _ocr_extract(file, extension, pdf)


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


async def _extract_all_pdfs(files: list[UploadFile]) -> list[ExtractedUnit]:
    """Run the convert-then-OCR pipeline for every non-xlsx upload.

    Files that fail are logged and dropped; the response does not report them
    yet. `None` marks a failure, which an empty page list would not.
    """
    batches: list[list[ExtractedUnit] | None] = await asyncio.gather(
        *(
            _or_fallback(_extract_pdf(file), None, file, "PDF extraction")
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
async def extract_text(files: list[UploadFile] = File(...)) -> ExtractResponse:
    """Extract content from the uploaded xlsx/pptx/docx/pdf files."""
    supported, skipped = _split_by_support(files)

    if skipped:
        logger.warning("Skipping unsupported files: %s", ", ".join(skipped))

    xlsx_files, pdf_files = _split_by_kind(supported)

    xlsx_units, pdf_units = await asyncio.gather(
        _extract_all_xlsx(xlsx_files),
        _extract_all_pdfs(pdf_files),
    )

    return ExtractResponse(units=xlsx_units + pdf_units, skipped=skipped)
