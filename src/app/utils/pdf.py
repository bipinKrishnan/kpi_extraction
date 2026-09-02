"""Rasterize PDF pages into the PNG data URLs the OCR model expects.

Quality lives or dies here: the model card asks for 200 DPI renders scaled to
a 1540px longest side, in a lossless format.
"""

import base64
import io

import pypdfium2
from PIL import Image

from app.core.config import MAX_LONGEST_SIDE, PDF_POINTS_PER_INCH, RENDER_DPI
from app.core.logger import get_logger

logger = get_logger(__name__)


def pdf_to_page_images(pdf: bytes, max_pages: int) -> list[str]:
    """Render up to `max_pages` pages of `pdf` as PNG data URLs, in page order.

    Blocking and CPU-bound, so callers run it through `run_limited`.
    """
    document = pypdfium2.PdfDocument(pdf)

    try:
        wanted = _page_count(len(document), max_pages)
        return [_render_page(document[index]) for index in range(wanted)]
    finally:
        document.close()


def _page_count(total: int, max_pages: int) -> int:
    """Clamp the page count, warning when pages are dropped."""
    if total > max_pages:
        logger.warning(
            "PDF has %d pages; only the first %d will be extracted", total, max_pages
        )
        return max_pages

    return total


def _render_page(page: pypdfium2.PdfPage) -> str:
    """Render one page at RENDER_DPI and return it as a PNG data URL."""
    bitmap = page.render(scale=RENDER_DPI / PDF_POINTS_PER_INCH)

    try:
        image = bitmap.to_pil()
    finally:
        bitmap.close()

    return _to_data_url(_fit_longest_side(image, MAX_LONGEST_SIDE))


def _fit_longest_side(image: Image.Image, target: int) -> Image.Image:
    """Scale `image` so its longest side is `target`, preserving aspect ratio.

    Never upscales: that invents detail and slows the vision encoder for
    nothing.
    """
    factor = target / max(image.width, image.height)

    if factor >= 1:
        return image

    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def _to_data_url(image: Image.Image) -> str:
    """Encode the page as a lossless PNG data URL.

    PNG rather than JPEG because JPEG artifacts hurt small glyphs.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"
