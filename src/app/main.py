from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logger import get_logger
from app.routers import text_extract
from app.utils.extract_client import ExtractClient
from app.utils.ocr_client import OcrClient
from app.utils.office import CONVERTIBLE_EXTENSIONS, find_soffice

logger = get_logger(__name__)


def _check_libreoffice() -> None:
    """Warn at startup if LibreOffice is missing, since pptx/docx need it."""
    binary = find_soffice()

    if binary is None:
        logger.warning(
            "LibreOffice not found on PATH; uploads of %s cannot be converted "
            "to PDF. Install it (e.g. 'brew install --cask libreoffice').",
            ", ".join(sorted(CONVERTIBLE_EXTENSIONS)),
        )
    else:
        logger.info("Using LibreOffice at '%s'", binary)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_libreoffice()

    # One client each for the whole process, so their connection pools and
    # concurrency semaphores are shared by every request instead of
    # per-request.
    settings = get_settings()
    ocr_client = OcrClient(settings)
    extract_client = ExtractClient(settings)
    app.state.ocr_client = ocr_client
    app.state.extract_client = extract_client

    try:
        yield
    finally:
        # Close both even if the first raises.
        async with AsyncExitStack() as stack:
            stack.push_async_callback(extract_client.aclose)
            stack.push_async_callback(ocr_client.aclose)


app = FastAPI(title="Port", version="0.0.1", lifespan=lifespan)

app.include_router(text_extract.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
