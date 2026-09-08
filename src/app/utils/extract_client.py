"""Async client for the Modal-hosted Gemma 4 structured-extraction server.

Two kinds of call, one connection pool and one concurrency cap: the KPI pass
(once per OCR unit) and the header pass (once per PDF upload, over its first
pages). They share the semaphore because they share the GPU fleet.
"""

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import (
    CHAT_COMPLETIONS_PATH,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_TOKENS,
    EXTRACT_MODEL_NAME,
    EXTRACT_SCHEMA_NAME,
    EXTRACT_TEMPERATURE,
    EXTRACT_TOP_P,
    HEADER_MAX_TOKENS,
    HEADER_SCHEMA_NAME,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    Settings,
)
from app.core.prompts import (
    HEADER_EXTRACTION_SYSTEM_PROMPT,
    KPI_EXTRACTION_SYSTEM_PROMPT,
)
from app.models.kpi import HeaderExtraction, KpiPageExtraction
from app.utils.vllm_retry import VllmCallError, post_with_retry


class ExtractError(VllmCallError):
    """Raised when a call could not be turned into valid, validated JSON."""


def _response_format(name: str, model: type[BaseModel]) -> dict[str, Any]:
    """Build the guided-decoding block for one schema.

    Built once per schema at import: model_json_schema() walks the whole model
    tree, and the schema is identical for every request.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": model.model_json_schema()},
    }


_KPI_RESPONSE_FORMAT = _response_format(EXTRACT_SCHEMA_NAME, KpiPageExtraction)
_HEADER_RESPONSE_FORMAT = _response_format(HEADER_SCHEMA_NAME, HeaderExtraction)


def _unit_message(text: str, reporting_period: str | None) -> str:
    """Prefix one unit's markdown with the report's reporting period.

    Goes in the user message rather than the system prompt: the system prompt
    has to stay byte-identical for the server's prefix cache to reuse its
    prefill, so anything per-request belongs after it.

    Omitted entirely when the period is unknown -- xlsx uploads, where no
    header pass runs, and uploads whose header pass failed. The prompt covers
    that case, so there is nothing to substitute here.
    """
    if reporting_period is None:
        return text

    return f"Reporting period of this report: {reporting_period}\n\n---\n\n{text}"


def _payload(
    system_prompt: str,
    text: str,
    *,
    max_tokens: int,
    response_format: dict[str, Any],
) -> dict[str, Any]:
    """Build a chat-completions body.

    The system prompt goes first and is byte-identical every time, which is
    what lets the server's prefix cache reuse its prefill. Putting anything
    per-request ahead of it would cost a full re-prefill on every call.

    `response_format` constrains the shape only -- the model never sees the
    schema, so the semantics live in the system prompt.
    """
    return {
        "model": EXTRACT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "max_tokens": max_tokens,
        "temperature": EXTRACT_TEMPERATURE,
        "top_p": EXTRACT_TOP_P,
        "response_format": response_format,
    }


def _content_of(body: dict[str, Any]) -> str:
    """Pull the assistant message out of a chat-completions response.

    Rejects a truncated answer before parsing it: under constrained decoding a
    response cut off at max_tokens is a structurally valid JSON *prefix*, so it
    would otherwise fail as a confusing parse error rather than as the budget
    problem it is.
    """
    try:
        choice = body["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as error:
        raise ExtractError(f"unexpected response shape: {error}") from error

    if finish_reason == "length":
        raise ExtractError("truncated at max_tokens")

    if not content:
        raise ExtractError("empty response content")

    return content


def _validated[T: BaseModel](model: type[T], content: str) -> T:
    """Parse and validate the model's JSON.

    Deliberately outside the retry loop. Transport failures are worth another
    attempt, but this runs at temperature 0: an identical request produces an
    identical invalid answer, so retrying a validation failure only doubles the
    latency and the bill. A failure here means the prompt needs fixing.
    """
    try:
        return model.model_validate_json(content)
    except ValidationError as error:
        raise ExtractError(f"invalid extraction: {error}") from error


class ExtractClient:
    """One instance per process, owning the HTTP pool and the call semaphore.

    Sharing both is what keeps a large batch of pages from opening a connection
    per call or fanning out past what the GPU fleet can absorb.
    """

    def __init__(self, settings: Settings) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.extract_base_url,
            # Modal's proxy authenticates every route ahead of vLLM; the token
            # pair comes from a Modal Secret, never from source.
            headers={
                "Modal-Key": settings.extract_modal_key.get_secret_value(),
                "Modal-Secret": settings.extract_modal_secret.get_secret_value(),
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
            # Modal answers 303 See Other while a container is still
            # cold-starting, pointing at a URL that resumes waiting on the
            # already-queued input. Following it is what turns a cold start
            # back into a slow request instead of a hard failure.
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=EXTRACT_CONCURRENCY,
                max_keepalive_connections=EXTRACT_CONCURRENCY,
            ),
        )
        # Caps structured extraction at EXTRACT_CONCURRENCY calls in flight for
        # the whole worker process, matched to the connection pool above. Both
        # call kinds share it, so header calls cannot push the total past it.
        self._semaphore = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    async def aclose(self) -> None:
        """Release the connection pool. Called once, at shutdown."""
        await self._http.aclose()

    async def extract_unit(
        self, text: str, reporting_period: str | None = None
    ) -> KpiPageExtraction:
        """Return the KPI figures found in one page or sheet.

        `reporting_period` is the document-level period read once by the header
        pass. It is given to the model as context for resolving relative period
        labels, not as a default period for undated figures.
        """
        return await self._call(
            KpiPageExtraction,
            _payload(
                KPI_EXTRACTION_SYSTEM_PROMPT,
                _unit_message(text, reporting_period),
                max_tokens=EXTRACT_MAX_TOKENS,
                response_format=_KPI_RESPONSE_FORMAT,
            ),
            "KPI extraction",
        )

    async def extract_header(self, text: str) -> HeaderExtraction:
        """Return the holding and reporting period from an upload's first pages."""
        return await self._call(
            HeaderExtraction,
            _payload(
                HEADER_EXTRACTION_SYSTEM_PROMPT,
                text,
                max_tokens=HEADER_MAX_TOKENS,
                response_format=_HEADER_RESPONSE_FORMAT,
            ),
            "Header extraction",
        )

    async def _call[T: BaseModel](
        self, model: type[T], payload: dict[str, Any], what: str
    ) -> T:
        """Send one request under the semaphore and validate its answer."""
        async with self._semaphore:
            body = await post_with_retry(
                self._http,
                CHAT_COMPLETIONS_PATH,
                payload,
                max_retries=MAX_RETRIES,
                what=what,
            )

        return _validated(model, _content_of(body))
