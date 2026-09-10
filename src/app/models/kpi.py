"""Schemas for KPI extraction from OCR units.

Two models are sent to the GPU, each with its own prompt in `app.core.prompts`:

  KpiPageExtraction  one call per unit (a page or a sheet), for the figures
  HeaderExtraction   one call per PDF upload, over its first pages, for the
                     document-level holding and reporting period

`KpiExtraction` is the response type and is never sent to the model: every
`document` field is supplied by us, so there is nothing left for the model to
guess at there.

Deliberately free of `pattern`, `ge` and `le`. vLLM's default xgrammar backend
silently ignores those keywords, so putting them in the schema would give false
confidence; value formats are checked by validators here instead, after the
JSON has been parsed.
"""

import re

from pydantic import BaseModel, ConfigDict, field_validator

# Open vocabularies, deliberately typed as plain `str` rather than `Literal`.
# The schema is what vLLM's guided decoding enforces, so a `Literal` here would
# make the listed values the only ones the model can physically emit -- reports
# do use spans and figure variants these four do not cover. The canonical values
# below are what the prompt asks for; anything else is normalised, not rejected.
PeriodType = str
Measure = str

PERIOD_TYPES = ("MTH", "QTR", "YTD", "FY")
MEASURES = ("actual", "actual_cc", "forecast", "forecast_delta")

# Collapse the spellings the model reaches for onto one canonical value, so a
# metric is not split across "budget" and "forecast" downstream.
_PERIOD_TYPE_SYNONYMS = {
    "month": "MTH",
    "monthly": "MTH",
    "quarter": "QTR",
    "quarterly": "QTR",
    "year_to_date": "YTD",
    "full_year": "FY",
    "financial_year": "FY",
    "fiscal_year": "FY",
    "annual": "FY",
}

_MEASURE_SYNONYMS = {
    "budget": "forecast",
    "plan": "forecast",
    "forecast_variance": "forecast_delta",
    "budget_variance": "forecast_delta",
    "variance": "forecast_delta",
    "constant_currency": "actual_cc",
    "actual_constant_currency": "actual_cc",
}

_PERIOD_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _validated_period(value: str | None) -> str | None:
    """Return `value` if it is a "YYYY-MM" month, else raise."""
    if value is not None and not _PERIOD_PATTERN.match(value):
        raise ValueError(f"expected a YYYY-MM period, got {value!r}")

    return value


def _vocabulary_key(field: str, value: str) -> str:
    """Reduce a vocabulary value to snake_case, so lookups ignore spelling.

    Blank is rejected rather than passed through: an unlabelled observation
    cannot be aggregated, and unlike a missing optional field it means the model
    emitted the key without reading a value off the page.
    """
    key = "_".join(value.strip().lower().replace("-", " ").replace("_", " ").split())

    if not key:
        raise ValueError(f"expected a {field}, got {value!r}")

    return key


class _Strict(BaseModel):
    """Base that forbids unknown keys, so the schema sets additionalProperties.

    Without it the model is free to add fields, and a hallucinated key would
    pass validation unnoticed.
    """

    model_config = ConfigDict(extra="forbid")


class Observation(_Strict):
    """One reported figure for one KPI, in one period.

    Currency lives here rather than at the document root: a single page can
    report the same metric in more than one currency.
    """

    period: str
    period_type: PeriodType
    measure: Measure
    value: float
    unit: str | None
    currency: str | None
    dimensions: dict[str, str] | None

    @field_validator("period")
    @classmethod
    def _check_period(cls, value: str) -> str:
        return _validated_period(value)  # type: ignore[return-value]

    @field_validator("period_type")
    @classmethod
    def _normalise_period_type(cls, value: str) -> str:
        """Canonicalise a known span label, upper-casing an unknown one."""
        key = _vocabulary_key("period_type", value)
        return _PERIOD_TYPE_SYNONYMS.get(key, key.upper())

    @field_validator("measure")
    @classmethod
    def _normalise_measure(cls, value: str) -> str:
        """Canonicalise a known figure variant, keeping an unknown one as-is."""
        key = _vocabulary_key("measure", value)
        return _MEASURE_SYNONYMS.get(key, key)


class Kpi(_Strict):
    """One metric named on the page, with every figure reported for it."""

    name: str
    observations: list[Observation]


class KpiPageExtraction(_Strict):
    """What the model returns for one unit: figures only.

    No document metadata, because none of it comes from the model any more.
    `kpis` is empty for most pages, which is the expected answer rather than a
    failure.
    """

    kpis: list[Kpi]


class HeaderExtraction(_Strict):
    """What the model returns for an upload's first pages.

    Both fields are nullable: a cover page may name neither, and for xlsx
    uploads no header pass runs at all.
    """

    holding: str | None
    reporting_period: str | None

    @field_validator("reporting_period")
    @classmethod
    def _check_reporting_period(cls, value: str | None) -> str | None:
        return _validated_period(value)


class DocumentMeta(_Strict):
    """Document-level metadata. Response only -- we fill every field.

    `company` is a constant, `holding` and `reporting_period` come from the
    header pass, and `source` is the upload's filename.
    """

    company: str
    holding: str | None
    reporting_period: str | None
    source: str


class KpiExtraction(_Strict):
    """One unit's full result: the document header plus that unit's figures."""

    document: DocumentMeta
    kpis: list[Kpi]
