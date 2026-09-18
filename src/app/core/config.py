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
RENDER_DPI = 300
PDF_POINTS_PER_INCH = 72  # pdfium renders in points, so scale = DPI / 72
# image resizing when exceeds the longest side
MAX_LONGEST_SIDE = 4001

# --- sampling ---
OCR_TEMPERATURE = 0.2
OCR_TOP_P = 0.9
# The server runs --max-model-len 8192 and a 1540px page is ~3080 image
# tokens, so a larger budget here would be rejected outright on dense pages.
OCR_MAX_TOKENS = 4096

# --- the structured extraction service ---
EXTRACT_MODEL_NAME = "z-ai/glm-5.3-flash"
EXTRACT_SCHEMA_NAME = "kpi-extraction"  # label only; vLLM does not interpret it
# Json extraction must be reproducible, so no sampling at all. Re-running a unit has
# to give the same JSON, otherwise a failed validation is unreproducible.
EXTRACT_TEMPERATURE = 0
EXTRACT_TOP_P = 1.0
EXTRACT_THINKING_MODE = False
EXTRACT_MAX_TOKENS = 131000
# How many units may be in structured extraction at once, for the whole worker
# process. The fleet can absorb MAX_CONTAINERS * MAX_NUM_SEQS = 96, but this is
# per app worker and several app containers share one fleet.
EXTRACT_CONCURRENCY = 16
# we no longer use text -> json extraction
# EXTRACT_MAX_INPUT_CHARS = 12_000

# EXTRACT_PREFIX_RESERVE_CHARS = 128

# --- document-level header pass ---
# Hardcoded for now: every upload is a Kedrion report, so asking the model for
# the company name would only add a way to get it wrong.
COMPANY_NAME = "Kedrion"
HEADER_SCHEMA_NAME = "report-header"  # label only; vLLM does not interpret it
# How many leading units of an upload the header pass reads. The holding and
# reporting period are printed on the cover or the first content page; reading
# further would cost tokens without adding information.
HEADER_UNIT_COUNT = 1
# The answer is two short strings, so it needs a fraction of the KPI budget.
HEADER_MAX_TOKENS = 4096

# --- client concurrency ---
# Pages in flight for the whole worker process. The fleet can absorb
# MAX_CONTAINERS * MAX_NUM_SEQS = 96, but this cap is per app worker and
# several app containers share one fleet, so it stays well below that.
OCR_CONCURRENCY = 32
# A dense page takes tens of seconds, and with min_containers=0 the first
# request after an idle period also waits for a container to boot.
REQUEST_TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 1


class Settings(BaseSettings):
    """Values that differ per deployment and must never be hardcoded.

    Read from the environment: a Modal Secret in deployment, and `.envrc`
    exports (direnv) locally.
    """

    model_config = SettingsConfigDict(extra="ignore")

    # Base URLs of the deployed vLLM servers, without a trailing path.
    ocr_base_url: str
    extract_base_url: str
    # Modal proxy-auth token pairs, injected by Modal Secrets at deploy time.
    # A separate pair per service, so either can be rotated on its own.
    ocr_modal_key: SecretStr
    ocr_modal_secret: SecretStr
    extract_modal_key: SecretStr
    extract_modal_secret: SecretStr


@lru_cache
def get_settings() -> Settings:
    """Load settings once, on first use rather than at import time.

    Deferring the read means a missing variable surfaces as a clear startup
    error instead of breaking every import of this module.
    """
    return Settings()
