"""Deploy numind/NuExtract3 on vLLM via Modal.

The deployment exposes vLLM's OpenAI-compatible /v1/chat/completions API.

NuExtract3 accepts:
  - an optional system message,
  - one image,
  - an optional user text message.

For structured extraction, pass the NuExtract template through
`chat_template_kwargs.template` in the request. The model's chat template then
uses NuExtract's structured extraction mode.

Example request shape:

{
  "model": "extract",
  "temperature": 0,
  "messages": [
    {"role": "system", "content": "Your extraction instructions..."},
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,..."}
        },
        {"type": "text", "text": "Optional user text"}
      ]
    }
  ],
  "max_tokens": 4096,
  "extra_body": {
    "chat_template_kwargs": {
      "template": "{\"name\":\"verbatim-string\"}",
      "enable_thinking": false
    }
  }
}

Deploy:
    modal deploy json_extract_service.py

The model itself is exposed directly; this app does not transform or validate
the extraction schema.
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

MODEL_NAME = "numind/NuExtract3"

# The model is currently based on Qwen3.5-4B and is distributed as a BF16
# checkpoint. Pinning the repo revision makes the deployment reproducible.
MODEL_REVISION = "c99dc8f"  # NuExtract3 main after the published benchmark updates.

# Stable API model name used by clients.
SERVED_MODEL_NAME = "extract"

# NuExtract3's model card documents vLLM deployment. Use an official vLLM image
# rather than installing/upgrading vLLM on top of an unrelated base image.
VLLM_IMAGE = "vllm/vllm-openai:latest"

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

GPU_TYPE = "A100-40GB"
N_GPU = 1

# NuExtract3 is substantially smaller than Gemma 4-E4B, but image processing
# and KV cache still need headroom on a 40 GB card.
GPU_MEMORY_UTILIZATION = 0.90

# The model supports a much larger context, but extraction requests normally
# do not need 131k tokens. Increase this if your documents require it.
MAX_MODEL_LEN = 40480

# One image is the intended workload for this service. Keep concurrency modest
# initially because image tokenization/vision memory is request-dependent.
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 16384

# ---------------------------------------------------------------------------
# NuExtract / vLLM multimodal settings
# ---------------------------------------------------------------------------

# NuExtract3's official vLLM recipe uses OpenAI-format multimodal content and
# allows images. We intentionally allow one image per request for this service.
MAX_IMAGES_PER_PROMPT = 1

# MTP can improve decode throughput. Set False if the installed vLLM build does
# not support qwen3_next_mtp on your chosen GPU/image.
ENABLE_MTP = True
NUM_SPECULATIVE_TOKENS = 2

# ---------------------------------------------------------------------------
# Modal fleet
# ---------------------------------------------------------------------------

MINUTES = 60

# Keep one warm container by default. Change to 0 if cold-start cost matters
# more than latency.
MIN_CONTAINERS = 1
MAX_CONTAINERS = 3
SCALEDOWN_WINDOW = 5 * MINUTES
BUFFER_CONTAINERS = 0

STARTUP_TIMEOUT = 15 * MINUTES
WAKE_TIMEOUT = 2 * MINUTES

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

# Modal proxy authentication protects the public endpoint, including vLLM's
# dev-mode sleep/wake endpoints.
REQUIRE_PROXY_AUTH = True

# ---------------------------------------------------------------------------
# GPU memory snapshotting
# ---------------------------------------------------------------------------

ENABLE_GPU_SNAPSHOT = True
SLEEP_LEVEL = 1

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

# Some Modal image-builder versions expect `python`, while the vLLM image may
# only provide `python3`.
SETUP_COMMANDS = [
    'RUN ln -sf "$(command -v python3)" /usr/local/bin/python'
]

vllm_image = (
    modal.Image.from_registry(
        VLLM_IMAGE,
        setup_dockerfile_commands=SETUP_COMMANDS,
    )
    .entrypoint([])
    .run_commands(
        "python -m pip install --no-cache-dir "
        '"typing_extensions>=4.15" "protobuf==6.33.5"'
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Keep Xet's mutable logs off persistent/snapshotted storage.
            "HF_XET_CACHE": "/tmp/xet",
            # Required for the /sleep and /wake_up endpoints used by snapshots.
            "VLLM_SERVER_DEV_MODE": "1",
            # Avoid snapshot problems caused by concurrent Inductor compilation.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

# Persistent model/cache volumes. The GPU snapshot avoids reinitializing the
# runtime, while these volumes avoid downloading the model repeatedly.
hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache",
    create_if_missing=True,
)
vllm_cache_vol = modal.Volume.from_name(
    "vllm-cache",
    create_if_missing=True,
)

app = modal.App("json-extract-service")


# ---------------------------------------------------------------------------
# vLLM command
# ---------------------------------------------------------------------------


def build_vllm_command() -> list[str]:
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--trust-remote-code",
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
        "--limit-mm-per-prompt",
        json.dumps(
            {
                "image": MAX_IMAGES_PER_PROMPT,
                "video": 0,
            }
        ),
        "--chat-template-content-format",
        "openai",
        "--generation-config",
        "vllm",
        "--uvicorn-log-level=info",
    ]

    if ENABLE_MTP:
        cmd += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "qwen3_next_mtp",
                    "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                }
            ),
        ]

    if ENABLE_GPU_SNAPSHOT:
        cmd += ["--enable-sleep-mode"]

    # Keep compilation enabled; the compiled state is captured by the snapshot.
    cmd += ["--no-enforce-eager"]

    return cmd


# vLLM is always listening on this port inside the Modal container.
VLLM_PORT = 8000


# ---------------------------------------------------------------------------
# Local HTTP helpers
# ---------------------------------------------------------------------------


def _local_url(path: str) -> str:
    return f"http://127.0.0.1:{VLLM_PORT}{path}"


def _post(path: str) -> None:
    req = urllib.request.Request(
        _local_url(path),
        data=b"",
        method="POST",
    )
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
# Modal autoscaling
# ---------------------------------------------------------------------------


def _autoscaling_kwargs() -> dict:
    kwargs = {
        "min_containers": MIN_CONTAINERS,
        "max_containers": MAX_CONTAINERS,
        "scaledown_window": SCALEDOWN_WINDOW,
    }

    if BUFFER_CONTAINERS > 0:
        kwargs["buffer_containers"] = BUFFER_CONTAINERS

    return kwargs


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@app.cls(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={
        "enable_gpu_snapshot": ENABLE_GPU_SNAPSHOT,
    },
    **_autoscaling_kwargs(),
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
class VLLMServer:
    """Run NuExtract3 behind Modal's authenticated vLLM proxy."""

    @modal.enter(snap=True)
    def build_snapshot(self):
        self.process = subprocess.Popen(build_vllm_command())

        _wait_until_healthy(STARTUP_TIMEOUT)

        if ENABLE_GPU_SNAPSHOT:
            # Move model GPU allocations out of the way before Modal captures
            # the container snapshot.
            _post(f"/sleep?level={SLEEP_LEVEL}")

    @modal.enter(snap=False)
    def restore(self):
        if ENABLE_GPU_SNAPSHOT:
            _post("/wake_up")
            _wait_until_healthy(WAKE_TIMEOUT)

    @modal.web_server(
        port=VLLM_PORT,
        startup_timeout=STARTUP_TIMEOUT,
        requires_proxy_auth=REQUIRE_PROXY_AUTH,
    )
    def serve(self):
        """Proxy vLLM's OpenAI-compatible API."""

    @modal.exit()
    def stop(self):
        self.process.terminate()
