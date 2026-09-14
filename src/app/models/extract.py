from pydantic import BaseModel

from app.models.kpi import KpiExtraction


class ExtractedUnit(BaseModel):
    """One page (pdf/pptx/docx) or one sheet (xlsx) of extracted content."""

    source: str
    extension: str
    label: str
    content: str| None = None


class UnitExtraction(BaseModel):
    """One unit's structured extraction, tagged with where it came from.

    Extraction runs per unit, so the source and label are carried here rather
    than inside the model's own answer -- nothing merges pages back together.
    """

    source: str
    label: str
    data: KpiExtraction


class ExtractResponse(BaseModel):
    """Everything extracted for a request, plus what was skipped or failed."""

    units: list[ExtractedUnit]
    skipped: list[str]
    # One entry per unit that extracted cleanly. Pages with no KPI figures
    # appear here with an empty `data.kpis`, which is the expected answer for
    # most pages -- absence from this list means failure, not absence of KPIs.
    extractions: list[UnitExtraction]
    # "<source>#<label>" for units that could not be extracted or validated.
    failed_extractions: list[str]


class SheetContent(BaseModel):
    """One worksheet of a converted xlsx file."""

    name: str
    content: str
