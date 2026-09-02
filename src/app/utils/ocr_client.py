"""Async client for the Modal-hosted LightOnOCR vLLM server."""

import asyncio
import random
from typing import Any

import httpx

from app.core.config import (
    CHAT_COMPLETIONS_PATH,
    MAX_RETRIES,
    OCR_CONCURRENCY,
    OCR_MAX_TOKENS,
    OCR_MODEL_NAME,
    OCR_TEMPERATURE,
    OCR_TOP_P,
    REQUEST_TIMEOUT_SECONDS,
    Settings,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

# Worth another attempt: rate limits, and the transport/gateway errors a
# scaling-from-zero fleet produces.
TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class OcrError(RuntimeError):
    """Raised when a page could not be OCR'd."""


class _Transient(Exception):
    """An internal failure worth retrying, with the server's hint if any."""

    def __init__(self, reason: str, retry_after: str | None = None) -> None:
        super().__init__(reason)
        self.retry_after = retry_after

    def delay(self, attempt: int) -> float:
        """Honour Retry-After, else exponential backoff with jitter.

        The jitter keeps a burst of failed pages from retrying in lockstep.
        """
        if self.retry_after:
            try:
                return float(self.retry_after)
            except ValueError:
                logger.warning("Ignoring unparseable Retry-After: %s", self.retry_after)

        return (2**attempt) * random.uniform(0.5, 1.5)


def _ocr_payload(data_url: str) -> dict[str, Any]:
    """Build the chat-completions body for one page.

    The message carries the image and nothing else: the model's chat template
    supplies the OCR instruction, so adding our own text prompt would push it
    off-distribution.
    """
    return {
        "model": OCR_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_url}}],
            }
        ],
        "max_tokens": OCR_MAX_TOKENS,
        "temperature": OCR_TEMPERATURE,
        "top_p": OCR_TOP_P,
    }


class OcrClient:
    """One instance per process, owning the HTTP pool and the page semaphore.

    Sharing both is what keeps a large batch of uploads from opening a
    connection per page or fanning out past what the GPU fleet can absorb.
    """

    def __init__(self, settings: Settings) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.ocr_base_url,
            # Modal's proxy authenticates every route ahead of vLLM; the token
            # pair comes from a Modal Secret, never from source.
            headers={
                "Modal-Key": settings.ocr_modal_key.get_secret_value(),
                "Modal-Secret": settings.ocr_modal_secret.get_secret_value(),
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
            limits=httpx.Limits(
                max_connections=OCR_CONCURRENCY,
                max_keepalive_connections=OCR_CONCURRENCY,
            ),
        )
        self._semaphore = asyncio.Semaphore(OCR_CONCURRENCY)

    async def aclose(self) -> None:
        """Release the connection pool. Called once, at shutdown."""
        await self._http.aclose()

    async def ocr_page(self, data_url: str) -> str:
        """Return the model's markdown for one rendered page."""
        async with self._semaphore:
            body = await self._post_with_retry(_ocr_payload(data_url))

        return _content_of(body)

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post one page, retrying transient failures with backoff."""
        reason = "not attempted"

        for attempt in range(MAX_RETRIES):
            try:
                return await self._post_once(payload)
            except _Transient as transient:
                reason = str(transient)
                logger.warning(
                    "OCR attempt %d/%d failed (%s)", attempt + 1, MAX_RETRIES, reason
                )

                if attempt + 1 < MAX_RETRIES:
                    # Held under the semaphore on purpose: backing off should
                    # relieve the server, not free a slot for another page.
                    await asyncio.sleep(transient.delay(attempt))

        raise OcrError(f"OCR failed after {MAX_RETRIES} attempts: {reason}")

    async def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make a single attempt, raising `_Transient` if a retry is warranted."""
        try:
            response = await self._http.post(CHAT_COMPLETIONS_PATH, json=payload)
        except httpx.HTTPError as error:
            raise _Transient(f"transport error: {error}") from error

        if response.is_success:
            return response.json()

        if response.status_code in TRANSIENT_STATUSES:
            raise _Transient(
                f"HTTP {response.status_code}", response.headers.get("Retry-After")
            )

        raise OcrError(
            f"OCR failed with HTTP {response.status_code}: {response.text[:200]}"
        )


def _content_of(body: dict[str, Any]) -> str:
    """Pull the assistant message out of a chat-completions response."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise OcrError(f"Unexpected OCR response shape: {error}") from error

    # A refusal comes back as a null content rather than an error.
    return content or ""
