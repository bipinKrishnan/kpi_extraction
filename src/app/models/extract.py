from pydantic import BaseModel


class ExtractedUnit(BaseModel):
    """One page (pdf/pptx/docx) or one sheet (xlsx) of extracted content."""

    source: str
    extension: str
    label: str
    content: str


class ExtractResponse(BaseModel):
    """Everything extracted for a request, plus the uploads that were skipped."""

    units: list[ExtractedUnit]
    skipped: list[str]


class SheetContent(BaseModel):
    """One worksheet of a converted xlsx file."""

    name: str
    content: str
