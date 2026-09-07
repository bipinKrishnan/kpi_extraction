"""Deploy google/gemma-4-E4B-it on vLLM via Modal, with GPU memory snapshotting.

Modal does not reimplement vLLM. This module boots a container, runs
`vllm serve` as a subprocess, and reverse-proxies its OpenAI-compatible API to a
public URL. Everything you know about the vLLM server still applies.

The server is a plain passthrough: it knows nothing about the extraction schema.
Callers send their own system prompt, user text, and `response_format`
(json_schema) per request, so the schema can change without a redeploy. See
"Structured Outputs" in gemma4-vllm-guide.md -- and note that constrained
decoding enforces *shape only*, so the system prompt must still describe what
each field means.

Deploy:
    modal deploy src/extract/modal_app.py
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
MODEL_NAME = "google/gemma-4-E4B-it"  # "effective 4B"; ~16 GB of BF16 weights
# Pin a commit SHA. Tracking `main` can change the weights under you and
# silently invalidate both the Volume cache and the GPU snapshot.
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"  # main @ 2026-07-20
# Stable alias clients call; decouples them from the repo id.
SERVED_MODEL_NAME = "extract"

# --- versions ---
# Gemma 4 support is NOT in a stable vLLM pip release: the recipe ships it as
# nightly wheels or this prebuilt image. The image is the safer of the two --
# it is a known-good build rather than whatever nightly resolves to today.
# It already bundles Python 3.12 and CUDA 12.9, so there is no CUDA_TAG or
# add_python here as there is in the OCR app.
VLLM_IMAGE = "vllm/vllm-openai:gemma4"  # CUDA 12.9 variant

# --- hardware ---
# Unlike the 1B OCR model, the weights are the dominant term here: 16 GB of a
# 40 GB card is gone before any KV cache is allocated. Ampere is compute
# capability 8.0, below the 8.9 vLLM requires for FP8, so neither FP8 weights
# nor an FP8 KV cache are available -- the KV budget below is BF16 and final.
GPU_TYPE = "A100-40GB"
N_GPU = 1  # MUST stay 1: GPU snapshots are incompatible with multi-GPU
# 0.90 rather than the OCR app's 0.92. With 16 GB of weights resident there is
# less absolute slack for CUDA graphs and the snapshot machinery, and this is a
# nightly-lineage vLLM build whose memory profile is less well trodden.
GPU_MEMORY_UTILIZATION = 0.90

# --- vLLM runtime ---
VLLM_PORT = 8000
# Sized for inputs under ~8k tokens plus a JSON answer, with headroom.
# The model allows 131072 (max_position_embeddings) but going anywhere near
# there would spend the whole KV budget on context nobody sends.
MAX_MODEL_LEN = 16384
# Per-container batch size. Also your main throughput/latency dial.
#
# KV cost per token comes from text_config: 2 KV heads, K+V, BF16, over
#   7 x full_attention layers at global_head_dim 512 -> 7 * 2*2*512*2 B = 28 KiB
#   35 x sliding_attention layers at head_dim 256, capped at sliding_window 512
#     -> 35 * 2*2*256*2 B * 512 tok = 35 MiB per sequence, flat
# So a full-length 16k sequence is ~448 MiB + ~35 MiB = ~483 MiB worst case.
# 40 GB at 0.90 leaves roughly 17-18 GB for KV once weights, activations and
# CUDA graphs are paid for, which is ~36 such sequences. 32 keeps a margin.
#
# (config.json also sets num_kv_shared_layers=18. If vLLM implements Gemma 4's
# KV sharing, the real per-token cost is lower and this is conservative --
# which is the direction you want to be wrong in.)
MAX_NUM_SEQS = 32
# Prefill budget per scheduler step. Text-only, so unlike the OCR app this is
# not coupled to any encoder cache -- it is purely a prefill chunking dial.
# 16384 lets two ~8k prompts prefill in one step.
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
# running. Left at 0 (disabled) -- raise it only if burst latency proves painful.
BUFFER_CONTAINERS = 0  # <-- e.g. 1
STARTUP_TIMEOUT = 15 * MINUTES  # first boot downloads 16 GB and builds the snapshot
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
# Alpha feature, and this is a nightly-lineage vLLM build, so treat it as
# unproven until one deploy has succeeded. Set to False for a one-line fallback
# to normal boots if snapshot creation breaks.
ENABLE_GPU_SNAPSHOT = True
# 1 = offload weights to host RAM and keep them; 2 discards them. Level 1 parks
# ~16 GB in host RAM during snapshot creation, so if the container hits a
# memory limit while sleeping, that is the cause.
SLEEP_LEVEL = 1

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

# The vLLM image ships `python3` but no `python`, and Modal's image builder runs
# `python -m pip install ...` as its very first step -- before any command we
# add with .run_commands(). setup_dockerfile_commands is the only hook that
# lands ahead of it, so the symlink has to go here or the build dies with
# "python: not found" (exit 127).
SETUP_COMMANDS = ['RUN ln -sf "$(command -v python3)" /usr/local/bin/python']

# On image builder <= 2024.10 Modal installs its client dependencies from a
# PINNED requirements file straight into the vLLM environment, which downgrades
# typing_extensions below the >=4.13 that vLLM's pydantic_core needs:
#   ImportError: cannot import name 'Sentinel' from 'typing_extensions'
# Modal only declares typing_extensions~=4.6, so putting a current one back
# satisfies both. This runs in a later layer than the builder's own install,
# which is why it repairs rather than races it.
#
# The root-cause fix is `modal config set image_builder_version 2025.06`, which
# mounts client deps at runtime instead of installing them into the image. That
# setting is client-wide though, so it also rebuilds the OCR image.
REPAIR_PACKAGES = ["typing_extensions>=4.15"]

# Nothing else is pip-installed on top: the image's vLLM and huggingface_hub are
# a tested set, and upgrading either from here is how you break the build.
vllm_image = (
    modal.Image.from_registry(VLLM_IMAGE, setup_dockerfile_commands=SETUP_COMMANDS)
    .entrypoint([])
    .run_commands(
        "python -m pip install --no-cache-dir -U "
        + " ".join(f'"{p}"' for p in REPAIR_PACKAGES)
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster weight downloads
            # Keep Xet's cache and its timestamped logs OFF the Volume. They
            # default to $HF_HOME/xet, which is the mounted huggingface-cache,
            # and a snapshot taken with such a log file open fails to restore:
            #   vfs.CompleteRestore() failed ... failed to walk
            #   "vo-.../xet/logs/xet_<timestamp>.log": no such file or directory
            # The container's own disk is part of the snapshot, so an open
            # handle there restores cleanly. Only the chunk cache is sacrificed;
            # the downloaded weights still persist in the Volume's hub/ dir.
            # If this recurs, disable Xet outright with HF_HUB_DISABLE_XET=1
            # (and drop HF_XET_HIGH_PERFORMANCE, which then does nothing).
            "HF_XET_CACHE": "/tmp/xet",
            # --- required for GPU snapshotting ---
            "VLLM_SERVER_DEV_MODE": "1",  # exposes /sleep and /wake_up
            # multi-threaded Inductor compilation can break snapshot creation
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

# ---------------------------------------------------------------------------
# Persistent caches, shared with the OCR deployment. Snapshots do NOT replace
# these: they don't help with the part of startup that is downloading weights.
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("gemma4-extract-vllm")

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
    cmd += _text_only_flags()
    cmd += _snapshot_flags()
    # Keep compilation ON: with snapshots you pay for it once, at snapshot
    # creation, and every later start restores the already-compiled state.
    cmd += ["--no-enforce-eager"]
    return cmd


def _text_only_flags() -> list[str]:
    """Gemma 4 is multimodal; this workload is not.

    Prefix caching is deliberately NOT disabled here (it is on by default),
    which is the opposite of the OCR app. Every request shares the same long
    system prompt, so the cached prefix is reused on essentially every call.
    """
    return [
        # skip multimodal profiling and encoder-cache allocation entirely
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "audio": 0}),
        # overlap scheduling with decoding
        "--async-scheduling",
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
    from zero viable: without it each cold start reloads 16 GB and recompiles.
    """

    @modal.enter(snap=True)
    def build_snapshot(self):
        """Runs BEFORE the snapshot is taken.

        Everything expensive belongs here: weight loading, torch.compile, CUDA
        graph capture. All of it gets frozen into the snapshot, so the very
        first deploy is slow (minutes) and every start after it is not.
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
