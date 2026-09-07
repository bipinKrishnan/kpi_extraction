"""Retry policy for the Modal-hosted vLLM services.

Both deployments scale from zero, so transport errors and 5xx from a
still-booting container are routine rather than exceptional. Everything here
concerns *delivery* of a request; whether the answer was any good is the
caller's problem.
"""

import asyncio
import random
from typing import Any

import httpx

from app.core.logger import get_logger

logger = get_logger(__name__)

# Worth another attempt: rate limits, and the transport/gateway errors a
# scaling-from-zero fleet produces.
TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class VllmCallError(RuntimeError):
    """A request to a vLLM service could not be completed."""


class _Transient(Exception):
    """An internal failure worth retrying, with the server's hint if any."""

    def __init__(self, reason: str, retry_after: str | None = None) -> None:
        super().__init__(reason)
        self.retry_after = retry_after

    def delay(self, attempt: int) -> float:
        """Honour Retry-After, else exponential backoff with jitter.

        The jitter keeps a burst of failed units from retrying in lockstep.
        """
        if self.retry_after:
            try:
                return float(self.retry_after)
            except ValueError:
                logger.warning("Ignoring unparseable Retry-After: %s", self.retry_after)

        return (2**attempt) * random.uniform(0.5, 1.5)


async def _post_once(
    http: httpx.AsyncClient, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Make a single attempt, raising `_Transient` if a retry is warranted."""
    try:
        response = await http.post(path, json=payload)
    except httpx.HTTPError as error:
        raise _Transient(f"transport error: {error}") from error

    if response.is_success:
        return response.json()

    if response.status_code in TRANSIENT_STATUSES:
        raise _Transient(
            f"HTTP {response.status_code}", response.headers.get("Retry-After")
        )

    raise VllmCallError(f"HTTP {response.status_code}: {response.text[:200]}")


async def post_with_retry(
    http: httpx.AsyncClient,
    path: str,
    payload: dict[str, Any],
    *,
    max_retries: int,
    what: str,
) -> dict[str, Any]:
    """POST `payload`, retrying transient failures with backoff."""
    reason = "not attempted"

    for attempt in range(max_retries):
        try:
            return await _post_once(http, path, payload)
        except _Transient as transient:
            reason = str(transient)
            logger.warning(
                "%s attempt %d/%d failed (%s)", what, attempt + 1, max_retries, reason
            )

            if attempt + 1 < max_retries:
                # Called with the caller's semaphore still held, on purpose:
                # backing off should relieve the server, not free a slot for
                # another unit.
                await asyncio.sleep(transient.delay(attempt))

    raise VllmCallError(f"{what} failed after {max_retries} attempts: {reason}")
