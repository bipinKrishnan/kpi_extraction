"""Deploy LightOnOCR-2-1B on vLLM via Modal, with GPU memory snapshotting.

Modal does not reimplement vLLM. This module boots a container, runs
`vllm serve` as a subprocess, and reverse-proxies its OpenAI-compatible API to a
public URL. Everything you know about the vLLM server still applies.

Deploy:
    modal deploy ocr/modal_app.py
"""

import json
import subprocess
import time
import urllib.error
import urllib.request

import modal

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# --- model ---
MODEL_NAME = "lightonai/LightOnOCR-2-1B"
# Pin a commit SHA. Tracking `main` can change the weights under you and
# silently invalidate both the Volume cache and the GPU snapshot.
MODEL_REVISION = "c97bd377f04481830395218fa8951df9deaba756"  # main @ 2026-07-08
SERVED_MODEL_NAME = "ocr"  # stable alias clients call; decouples them from the repo id

# --- versions ---
# Pin explicitly: vLLM minor releases change flag names and snapshot behaviour.
VLLM_VERSION = "0.28.0"  # LightOnOCRForConditionalGeneration registered since 0.26.0
CUDA_TAG = "12.9.0-devel-ubuntu22.04"  # -devel: vLLM needs nvcc for some kernels
PYTHON_VERSION = "3.12"

# --- hardware ---
# A 1B BF16 VLM is only ~2.5 GB of weights, so the card is chosen for compute
# and KV capacity, not to fit the model. Note that Ampere is compute capability
# 8.0, below the 8.9 vLLM requires for FP8, so neither FP8 weights nor an FP8
# KV cache are available here -- the KV budget below is BF16 and final.
GPU_TYPE = "A100-40GB"
N_GPU = 1  # MUST stay 1: GPU snapshots are incompatible with multi-GPU
# 0.95 is what Modal's own vLLM snapshot example uses, but that is an 80 GB
# H100 where 5% is 4 GB of slack. On a 40 GB A100 it is 2 GB, which also has to
# cover CUDA graphs and the snapshot machinery. Drop to 0.90 if snapshot
# creation OOMs.
GPU_MEMORY_UTILIZATION = 0.92

# --- vLLM runtime ---
VLLM_PORT = 8000
# Must fit the image tokens for a 1540px page PLUS max_tokens (4096).
# Too low silently truncates pages; too high shrinks the KV cache.
#
# Worst case (square page): 1540px / patch_size 14 / spatial_merge_size 2 = 55
# tokens per side, so 55*55 = 3025 image tokens plus ~55 row-break/end tokens.
# ~3080 + 4096 output = ~7.2k. The model allows 16384 (max_position_embeddings)
# but going there would halve concurrency for pages that never need it.
MAX_MODEL_LEN = 8192
# Per-container batch size. Also your main throughput/latency dial.
#
# KV is ~112 KiB/token (28 layers x 2 x 8 KV heads x 128 head_dim x 2 bytes).
# A 40 GB A100 at 0.92 leaves ~32 GB for KV once weights and CUDA graphs are
# accounted for, which is ~40 sequences at the 7.2k tokens a page really uses.
# 32 stays clear of preemption, which is unusually expensive here: with
# --mm-processor-cache-gb 0 a recomputed sequence re-runs the vision encoder.
MAX_NUM_SEQS = 32
# Prefill budget per scheduler step, deliberately NOT tied to MAX_MODEL_LEN.
# vLLM sets encoder_cache_size from this value, so at 8192 only ~2 pages of
# image embeddings can be in flight and prefill stalls waiting on a slot.
# 16384 buys ~5 pages, at ~6 MB of embeddings each.
MAX_NUM_BATCHED_TOKENS = 16384

# --- Modal fleet ---
MINUTES = 60
MIN_CONTAINERS = 0  # scale to zero when idle: no traffic, no GPU bill
MAX_CONTAINERS = 3  # hard ceiling; a runaway client cannot fan out past 3 GPUs
# How long a container stays warm after its last request. This is the idle-cost
# vs cold-start dial: raise it for bursty traffic, lower it to save money.
SCALEDOWN_WINDOW = 3 * MINUTES
# Spare warm containers kept above current load, so a second concurrent request
# doesn't wait on a boot. Costs one idle GPU per unit whenever anything is
# running. Left at 0 (disabled) — raise it only if burst latency proves painful.
BUFFER_CONTAINERS = 0  # <-- e.g. 1
STARTUP_TIMEOUT = 10 * MINUTES  # the first boot builds the snapshot and is slow
WAKE_TIMEOUT = 2 * MINUTES  # snapshot restores are fast; this is a safety net

# --- auth ---
# Modal's proxy is the ONLY thing authenticating this deployment. Do not turn
# it off, and do not expect vLLM to cover for it.
#
# A Modal web_server is public by default, and its URL is derived from the
# workspace/app/class/method names rather than being random -- obscurity buys
# nothing. vLLM's own --api-key would not have helped much anyway: it only
# guards GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere"), and
# VLLM_SERVER_DEV_MODE=1 (required for GPU snapshotting) registers /sleep and
# /wake_up outside those prefixes. An anonymous POST /sleep would offload the
# weights and stall every request until the container is recovered.
#
# With proxy auth, unauthenticated requests are rejected by Modal and never
# reach the container, so every route is covered -- including the dev-mode
# ones. Callers pass Modal-Key / Modal-Secret header tokens created in the
# Modal dashboard.
REQUIRE_PROXY_AUTH = True

# --- GPU memory snapshotting ---
# Alpha feature. Set to False for a one-line fallback to normal boots if a vLLM
# upgrade breaks snapshot creation.
ENABLE_GPU_SNAPSHOT = True
SLEEP_LEVEL = 1  # 1 = offload weights to host RAM and keep them; 2 discards them

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PYTHON_VERSION)
    .entrypoint([])  # drop the base image's entrypoint
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster weight downloads
            # --- required for GPU snapshotting ---
            "VLLM_SERVER_DEV_MODE": "1",  # exposes /sleep and /wake_up
            # multi-threaded Inductor compilation can break snapshot creation
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

# ---------------------------------------------------------------------------
# Persistent caches. Snapshots do NOT replace these: they don't help with the
# part of startup that is downloading weights.
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("lightonocr-vllm")

# ---------------------------------------------------------------------------
# vLLM command
# ---------------------------------------------------------------------------


def build_vllm_command() -> list[str]:
    """Assemble the `vllm serve` argv. Kept pure so it is easy to read and test."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--uvicorn-log-level=info",
    ]
    cmd += _ocr_flags()
    cmd += _snapshot_flags()
    # Keep compilation ON: with snapshots you pay for it once, at snapshot
    # creation, and every later start restores the already-compiled state.
    cmd += ["--no-enforce-eager"]
    return cmd


def _ocr_flags() -> list[str]:
    """The three flags the LightOnOCR model card requires."""
    return [
        # one page per request; caps the vision budget
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
        # every page is a unique image, so caching processed images only burns RAM
        "--mm-processor-cache-gb",
        "0",
        # pages share no text prefix, so prefix caching is dead weight
        "--no-enable-prefix-caching",
    ]


def _snapshot_flags() -> list[str]:
    """Sleep mode is what makes GPU memory snapshottable."""
    return ["--enable-sleep-mode"] if ENABLE_GPU_SNAPSHOT else []


# ---------------------------------------------------------------------------
# Local HTTP helpers (talk to the vLLM subprocess inside the container)
# ---------------------------------------------------------------------------


def _local_url(path: str) -> str:
    return f"http://127.0.0.1:{VLLM_PORT}{path}"


def _post(path: str) -> None:
    req = urllib.request.Request(_local_url(path), data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=WAKE_TIMEOUT) as resp:
        resp.read()


def _is_healthy() -> bool:
    req = urllib.request.Request(_local_url("/health"))
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _wait_until_healthy(timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_healthy():
            return
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {timeout}s")


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


def _autoscaling_kwargs() -> dict:
    """Only pass buffer_containers when it is actually in use.

    Modal treats 0 and unset the same, but omitting it keeps the deployed config
    clean and avoids failing on Modal versions that predate the argument.
    """
    kwargs = {
        "min_containers": MIN_CONTAINERS,
        "max_containers": MAX_CONTAINERS,
        "scaledown_window": SCALEDOWN_WINDOW,
    }
    if BUFFER_CONTAINERS > 0:
        kwargs["buffer_containers"] = BUFFER_CONTAINERS
    return kwargs


@app.cls(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # ---- GPU memory caching ----
    enable_memory_snapshot=True,  # CPU snapshot; prerequisite for the GPU one
    experimental_options={"enable_gpu_snapshot": ENABLE_GPU_SNAPSHOT},
    # ---- autoscale 0 -> 3 (plus buffer_containers when > 0) ----
    # With min_containers=0 every burst after an idle period pays a cold start,
    # which is exactly what the GPU snapshot above is for.
    **_autoscaling_kwargs(),
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
class VLLMServer:
    """Boots `vllm serve` and lets Modal snapshot it once fully initialized.

    Scales 0 -> MAX_CONTAINERS on demand. The snapshot is what makes scaling
    from zero viable: without it each cold start reloads and recompiles.
    """

    @modal.enter(snap=True)
    def build_snapshot(self):
        """Runs BEFORE the snapshot is taken.

        Everything expensive belongs here: weight loading, torch.compile, CUDA
        graph capture. All of it gets frozen into the snapshot, so the very
        first deploy is slow (minutes) and every start after it is not.

        No warmup inference for now — see "Deferred: pre-snapshot warmup" in
        ocr_pseudocode.md.
        """
        self.process = subprocess.Popen(build_vllm_command())
        _wait_until_healthy(STARTUP_TIMEOUT)

        if ENABLE_GPU_SNAPSHOT:
            # Evict GPU memory to host RAM so Modal can capture it.
            _post(f"/sleep?level={SLEEP_LEVEL}")

    @modal.enter(snap=False)
    def restore(self):
        """Runs on EVERY start, after the snapshot is restored. Must be cheap."""
        if ENABLE_GPU_SNAPSHOT:
            _post("/wake_up")
            _wait_until_healthy(WAKE_TIMEOUT)

    @modal.web_server(
        port=VLLM_PORT,
        startup_timeout=STARTUP_TIMEOUT,
        requires_proxy_auth=REQUIRE_PROXY_AUTH,
    )
    def serve(self):
        """Expose vLLM's OpenAI-compatible API.

        No body needed: the subprocess is already listening on VLLM_PORT and
        Modal just proxies to it. Auth is enforced by Modal's proxy before a
        request ever reaches this container -- see REQUIRE_PROXY_AUTH.
        """

    @modal.exit()
    def stop(self):
        self.process.terminate()
