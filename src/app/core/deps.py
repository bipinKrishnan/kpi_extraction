"""FastAPI dependencies for shared, process-wide resources."""

from fastapi import Request

from app.utils.extract_client import ExtractClient
from app.utils.ocr_client import OcrClient


def get_ocr_client(request: Request) -> OcrClient:
    """Return the OCR client created at startup.

    Handing the single instance to endpoints keeps its connection pool and
    concurrency limit shared across requests.
    """
    return request.app.state.ocr_client


def get_extract_client(request: Request) -> ExtractClient:
    """Return the structured-extraction client created at startup.

    Same reasoning as `get_ocr_client`: one instance so the connection pool and
    the 32-unit semaphore are shared across requests rather than per request.
    """
    return request.app.state.extract_client
