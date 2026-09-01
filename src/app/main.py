from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logger import get_logger
from app.routers import text_extract
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
    yield


app = FastAPI(title="Port", version="0.0.1", lifespan=lifespan)

app.include_router(text_extract.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
