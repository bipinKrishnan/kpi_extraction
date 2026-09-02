"""Configuration for the OCR client: fixed tunables plus env-provided secrets.

The constants here are pinned to the deployment in `src/ocr/modal_app.py` and
to the LightOnOCR-2-1B model card, so the client can never drift from the
server it talks to.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- the OCR service ---
OCR_MODEL_NAME = "ocr"  # --served-model-name in src/ocr/modal_app.py
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# --- page rendering (values from the model card) ---
RENDER_DPI = 200
MAX_LONGEST_SIDE = 1540
PDF_POINTS_PER_INCH = 72  # pdfium renders in points, so scale = DPI / 72

# --- sampling ---
OCR_TEMPERATURE = 0.2
OCR_TOP_P = 0.9
# The server runs --max-model-len 8192 and a 1540px page is ~3080 image
# tokens, so a larger budget here would be rejected outright on dense pages.
OCR_MAX_TOKENS = 4096

# --- client concurrency ---
# Pages in flight for the whole worker process. The fleet can absorb
# MAX_CONTAINERS * MAX_NUM_SEQS = 96, but this cap is per app worker and
# several app containers share one fleet, so it stays well below that.
OCR_CONCURRENCY = 32
# A dense page takes tens of seconds, and with min_containers=0 the first
# request after an idle period also waits for a container to boot.
REQUEST_TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 3


class Settings(BaseSettings):
    """Values that differ per deployment and must never be hardcoded.

    Read from the environment: a Modal Secret in deployment, and `.envrc`
    exports (direnv) locally.
    """

    model_config = SettingsConfigDict(extra="ignore")

    # Base URL of the deployed vLLM server, without a trailing path.
    ocr_base_url: str
    # Modal proxy-auth token pair, injected by a Modal Secret at deploy time.
    ocr_modal_key: SecretStr
    ocr_modal_secret: SecretStr


@lru_cache
def get_settings() -> Settings:
    """Load settings once, on first use rather than at import time.

    Deferring the read means a missing variable surfaces as a clear startup
    error instead of breaking every import of this module.
    """
    return Settings()
