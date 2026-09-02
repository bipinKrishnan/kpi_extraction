"""FastAPI dependencies for shared, process-wide resources."""

from fastapi import Request

from app.utils.ocr_client import OcrClient


def get_ocr_client(request: Request) -> OcrClient:
    """Return the OCR client created at startup.

    Handing the single instance to endpoints keeps its connection pool and
    concurrency limit shared across requests.
    """
    return request.app.state.ocr_client
