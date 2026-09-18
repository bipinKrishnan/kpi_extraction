"""Async client for the Modal-hosted Gemma 4 structured-extraction server.

Two kinds of call, one connection pool and one concurrency cap:
- KPI extraction: once per PDF page
- Header extraction: once per PDF upload, over its first pages

Both calls send PDF pages directly as image_url content. No OCR/text
extraction is performed.
"""

import ast
import asyncio
import json
from typing import Any

import httpx
from json_repair import repair_json
from numind.nuextract_utils import convert_json_schema_to_nuextract_template
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed

from app.core.config import (
    CHAT_COMPLETIONS_PATH,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_TOKENS,
    EXTRACT_MODEL_NAME,
    EXTRACT_SCHEMA_NAME,
    EXTRACT_TEMPERATURE,
    EXTRACT_THINKING_MODE,
    EXTRACT_TOP_P,
    HEADER_MAX_TOKENS,
    HEADER_SCHEMA_NAME,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    Settings,
)
from app.core.logger import get_logger
from app.core.prompts import (
    HEADER_EXTRACTION_SYSTEM_PROMPT,
    KPI_EXTRACTION_SYSTEM_PROMPT,
)
from app.models.kpi import HeaderExtraction, KpiPageExtraction
from app.utils.vllm_retry import VllmCallError, post_with_retry

logger = get_logger(__name__)


class ExtractError(VllmCallError):
    """Raised when a call could not be turned into valid, validated JSON."""


def _response_format(name: str, model: type[BaseModel]) -> dict[str, Any]:
    """Build the guided-decoding block for one schema."""
    conversion = convert_json_schema_to_nuextract_template(
        model.model_json_schema()
    )

    template = conversion["template"]
    return template


_KPI_RESPONSE_FORMAT = _response_format(
    EXTRACT_SCHEMA_NAME,
    KpiPageExtraction,
)

_HEADER_RESPONSE_FORMAT = _response_format(
    HEADER_SCHEMA_NAME,
    HeaderExtraction,
)


def _image_content(image_url: str) -> dict[str, Any]:
    """Build one OpenAI-compatible image_url content item."""
    return {
        "type": "image_url",
        "image_url": {
            "url": image_url,
        },
    }


def _image_message(image_url: str) -> list[dict[str, Any]]:
    """Build user content for a single page image."""
    return [_image_content(image_url)]


def _images_message(image_urls: list[str]) -> list[dict[str, Any]]:
    """Build user content for multiple page images."""
    return [_image_content(image_url) for image_url in image_urls]


def _payload(
    system_prompt: str,
    content: list[dict[str, Any]],
    *,
    max_tokens: int,
    response_format: dict[str, Any],
) -> dict[str, Any]:
    """Build a multimodal chat-completions body.

    The system prompt remains byte-identical between requests so the server's
    prefix cache can reuse its prefill. Per-request page images are placed in
    the user message.
    """
    return {
        "model": EXTRACT_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        "max_tokens": max_tokens,
        "temperature": EXTRACT_TEMPERATURE,
        "top_p": EXTRACT_TOP_P,
        "extract_body": {
            "chat_template_kwargs": {
                "reasoning_effort": "low",
            }
        }
    }


def _content_of(body: dict[str, Any]) -> str:
    """Pull the assistant message out of a chat-completions response."""
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
    """Parse and validate the model's JSON."""
    try:
        # remove single quotes and repair if unparsable
        parsed = repair_json(content)
        return model.model_validate_json(parsed)
    except ValidationError as error:
        raise ExtractError(f"invalid extraction: {error}") from error


def check_for_meaningless_kpis(data):
    """
    Finds KPIs with observation whose values differ but are indistinguishable
    from one another because of same values for other fields.
    """
    results = []
    seen = set()

    for kpi in data.get("kpis", []):
        kpi_name = kpi.get("name")
        observations = kpi.get("observations", [])

        groups = {}

        for obs in observations:
            key = (
                obs.get("period"),
                obs.get("period_type"),
                obs.get("measure"),
                obs.get("unit"),
                obs.get("currency"),
                tuple(obs.get("dimensions", []))
            )

            groups.setdefault(key, []).append(obs)

        for group in groups.values():
            values = {obs.get("value") for obs in group}

            if len(values) > 1:
                for obs in group:
                    dimensions = obs.get("dimensions", [])
                    measure = obs.get("measure")

                    # Use tuple because lists aren't hashable
                    result_key = (kpi_name, tuple(dimensions))

                    if result_key not in seen:
                        results.append({
                            "kpi_name": kpi_name,
                            "dimensions": dimensions,
                            "measure": measure
                        })
                        seen.add(result_key)

    return results


class ExtractClient:
    """One instance per process, owning the HTTP pool and call semaphore."""

    def __init__(self, settings: Settings) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.extract_base_url,
            headers={
                "Modal-Key": settings.extract_modal_key.get_secret_value(),
                "Modal-Secret": settings.extract_modal_secret.get_secret_value(),
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=EXTRACT_CONCURRENCY,
                max_keepalive_connections=EXTRACT_CONCURRENCY,
            ),
        )

        self._semaphore = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    async def aclose(self) -> None:
        """Release the connection pool. Called once, at shutdown."""
        await self._http.aclose()

    async def extract_unit(
        self,
        image_url: str,
        reporting_period: str | None = None,
    ) -> KpiPageExtraction:
        """Return KPI figures found in one PDF page image.

        The page is sent directly to the multimodal model as an image_url.
        """
        content = _image_message(image_url)

        if reporting_period is not None:
            # If you want the reporting period available to the model as
            # textual context, the user content can contain both text and the
            # image. The image remains the primary page input.
            content = [
                {
                    "type": "text",
                    "text": (
                        f"{reporting_period} YTD"
                    ),
                },
                _image_content(image_url),
            ]

        return await self._call(
            KpiPageExtraction,
            _payload(
                KPI_EXTRACTION_SYSTEM_PROMPT,
                content,
                max_tokens=EXTRACT_MAX_TOKENS,
                response_format=_KPI_RESPONSE_FORMAT,
            ),
            "KPI extraction",
        )

    async def extract_header(
        self,
        image_urls: list[str],
    ) -> HeaderExtraction:
        """Return document metadata from the first PDF page images."""
        return await self._call(
            HeaderExtraction,
            _payload(
                HEADER_EXTRACTION_SYSTEM_PROMPT,
                _images_message(image_urls),
                max_tokens=HEADER_MAX_TOKENS,
                response_format=_HEADER_RESPONSE_FORMAT,
            ),
            "Header extraction",
        )

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_fixed(3),
    )
    async def _call[T: BaseModel](
        self,
        model: type[T],
        payload: dict[str, Any],
        what: str,
    ) -> T:
        """Send one request under the semaphore and validate its answer."""
        async with self._semaphore:
            body = await post_with_retry(
                self._http,
                CHAT_COMPLETIONS_PATH,
                payload,
                max_retries=1,
                what=what,
            )

        return _validated(model, _content_of(body))
