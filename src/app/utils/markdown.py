import io
from functools import lru_cache

from fastapi import UploadFile
from markitdown import MarkItDown, StreamInfo

from app.models.extract import SheetContent

XLSX_EXTENSION = ".xlsx"


@lru_cache(maxsize=1)
def _get_converter() -> MarkItDown:
    """Shared MarkItDown instance with the OCR plugin loaded (no LLM client).

    Without an `llm_client` the markitdown-ocr plugin skips OCR and falls back
    to the standard converters, so conversion stays local and offline.
    """
    return MarkItDown(enable_plugins=True)


def _is_empty_table(table: str) -> bool:
    """True for the degenerate table markitdown emits for an empty sheet."""
    return not set(table) - set("| -\n")


def _split_sheets(markdown: str) -> list[SheetContent]:
    """Split markitdown's xlsx output into one entry per worksheet.

    Each sheet is emitted as a `## <sheet name>` heading followed by its table.
    Cell text is escaped onto a single row, so a heading can only ever come
    from a real sheet name.
    """
    sheets: list[SheetContent] = []

    for block in markdown.split("\n## "):
        block = block.removeprefix("## ").strip()

        if not block:
            continue

        name, _, table = block.partition("\n")
        table = table.strip()

        if _is_empty_table(table):
            continue

        sheets.append(SheetContent(name=name.strip(), content=table))

    return sheets


def xlsx_to_markdown(file: UploadFile) -> list[SheetContent]:
    """Convert an uploaded xlsx file to one markdown table per worksheet."""
    file.file.seek(0)
    # UploadFile wraps a SpooledTemporaryFile, which markitdown's content
    # sniffer rejects, so hand it a plain BytesIO instead.
    stream = io.BytesIO(file.file.read())

    result = _get_converter().convert_stream(
        stream,
        stream_info=StreamInfo(
            extension=XLSX_EXTENSION,
            filename=file.filename,
        ),
    )
    return _split_sheets(result.text_content)
