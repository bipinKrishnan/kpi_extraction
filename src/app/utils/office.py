import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import UploadFile

from app.core.logger import get_logger

logger = get_logger(__name__)

CONVERT_TIMEOUT_SECONDS = 120
CONVERTIBLE_EXTENSIONS = {".pptx", ".docx"}


class OfficeConversionError(RuntimeError):
    """Raised when LibreOffice fails to produce a PDF."""


def find_soffice() -> str | None:
    """Return the path to the LibreOffice binary, or None if it is not installed."""
    return shutil.which("soffice") or shutil.which("libreoffice")


def _soffice_binary() -> str:
    """Locate the LibreOffice binary, raising if it is unavailable."""
    binary = find_soffice()

    if binary is None:
        raise OfficeConversionError(
            "LibreOffice not found; install it and ensure 'soffice' is on PATH"
        )

    return binary


def _write_source(file: UploadFile, directory: Path, extension: str) -> Path:
    """Write the upload into `directory` under a fixed, safe name."""
    source = directory / f"source{extension}"
    file.file.seek(0)

    with source.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    return source


def _run_soffice(source: Path, outdir: Path, profile: Path) -> None:
    """Run LibreOffice headless to convert `source` into a PDF in `outdir`."""
    command = [
        _soffice_binary(),
        # A per-call profile keeps concurrent conversions from clashing.
        f"-env:UserInstallation=file://{profile}",
        "--headless",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(outdir),
        str(source),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CONVERT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise OfficeConversionError(
            f"LibreOffice timed out after {CONVERT_TIMEOUT_SECONDS}s"
        ) from error

    if result.returncode != 0:
        raise OfficeConversionError(
            f"LibreOffice exited with {result.returncode}: {result.stderr.strip()}"
        )


def convert_to_pdf(file: UploadFile, extension: str) -> bytes:
    """Convert an uploaded pptx/docx file to PDF bytes via LibreOffice headless.

    `extension` is the validated source extension (for example ".docx"); the
    upload's own filename is never passed to the subprocess.
    """
    if extension not in CONVERTIBLE_EXTENSIONS:
        raise OfficeConversionError(f"Cannot convert '{extension}' to PDF")

    with tempfile.TemporaryDirectory() as workdir:
        directory = Path(workdir)
        source = _write_source(file, directory, extension)
        profile = directory / "profile"

        _run_soffice(source, directory, profile)

        pdf = source.with_suffix(".pdf")

        if not pdf.is_file():
            raise OfficeConversionError(
                f"LibreOffice produced no PDF for '{file.filename}'"
            )

        logger.info("Converted '%s' to PDF via LibreOffice", file.filename)
        return pdf.read_bytes()
