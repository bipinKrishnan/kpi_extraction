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

# --- the structured extraction service ---
EXTRACT_MODEL_NAME = "extract"  # --served-model-name in src/extract/modal_app.py
EXTRACT_SCHEMA_NAME = "kpi-extraction"  # label only; vLLM does not interpret it
# Extraction must be reproducible, so no sampling at all. Re-running a unit has
# to give the same JSON, otherwise a failed validation is unreproducible.
EXTRACT_TEMPERATURE = 0.0
EXTRACT_TOP_P = 1.0
# A dense table page can hold many KPIs, each with several observations. Too
# low and the answer is truncated mid-JSON, which is unparseable rather than
# merely short.
#
# This is the largest value the 16384-token window allows: the worst case a
# request can present is EXTRACT_MAX_INPUT_CHARS of dense numeric markdown,
# which tokenizes at roughly 2.5 chars per token rather than the 4 that prose
# does, plus ~800 tokens of system prompt and chat template. Raising it
# further without lowering that cap means vLLM rejects the request outright
# instead of truncating the answer.
EXTRACT_MAX_TOKENS = 11_264
# How many units may be in structured extraction at once, for the whole worker
# process. The fleet can absorb MAX_CONTAINERS * MAX_NUM_SEQS = 96, but this is
# per app worker and several app containers share one fleet.
EXTRACT_CONCURRENCY = 32
# The server runs --max-model-len 16384, shared between the prompt and the
# answer, and EXTRACT_MAX_TOKENS now claims most of it. 12_000 chars is at
# worst ~4.8k tokens, which leaves 16384 - 4800 - ~800 for the system prompt
# comfortably above the output budget.
#
# This no longer binds OCR pages: the OCR pass caps its own answer at
# OCR_MAX_TOKENS, so one page cannot exceed 4096 tokens whatever its character
# count. It binds large xlsx sheets, which have no such ceiling. Units above
# this are reported as failures rather than silently truncated, since a
# dropped tail means silently missing KPIs.
EXTRACT_MAX_INPUT_CHARS = 12_000

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
HEADER_MAX_TOKENS = 256

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
