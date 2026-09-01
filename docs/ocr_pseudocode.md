# LightOnOCR-2-1B on vLLM + Modal — Design & Pseudocode

> Note on the model name: the card you linked is **`lightonai/LightOnOCR-2-1B`** —
> "2" is the generation, "1B" is the parameter count. There is no `LightOnOCR-2B`.
> Everything below assumes the 1B checkpoint; swapping to a bigger sibling only
> changes `MODEL_NAME` / `MODEL_REVISION` / `GPU`.

**Deployment shape:** autoscaling 0–3 GPU containers, with **GPU memory
snapshotting** enabled so a cold start comes back in seconds instead of
minutes.

---

## 1. Folder structure

```
ocr/
├── README.md                  # how to deploy, how to call, cost notes
├── config.py                  # ALL tunables in one place (no magic numbers elsewhere)
├── modal_app.py               # Modal app: image, volumes, vLLM server class, snapshot hooks
│
├── client/                    # everything that runs OUTSIDE the GPU container
│   ├── __init__.py
│   ├── pdf.py                 # PDF/image bytes -> normalized page images
│   ├── images.py              # resize / DPI / base64 encoding helpers
│   ├── ocr_client.py          # async OpenAI-compatible client (pool + semaphore + retries)
│   └── assemble.py            # per-page markdown -> single document markdown
│
├── scripts/
│   ├── warm_cache.py          # one-off: pull weights into the Modal Volume before deploy
│   └── run_ocr.py             # CLI: run_ocr.py <file.pdf> --out out.md
│
└── tests/
    ├── fixtures/sample.pdf
    ├── test_pdf.py            # rendering + resize math (no GPU needed)
    └── test_smoke.py          # hits a deployed URL, asserts /health + 1 page OCR
```

Why this split:

- `modal_app.py` is the only file Modal ever imports. Keep it thin — it starts a
  subprocess and manages the snapshot lifecycle.
- `client/` is plain Python that talks HTTP. It is testable locally and reusable
  from your existing FastAPI app (`src/app/routers/*`) without importing Modal.
- `config.py` is shared by both, so the client can never drift from the server
  (e.g. asking for 3 images when the server was launched with `image: 1`).

---

## 2. Configs you must decide before deploying

### 2.1 Model / weights

| Config | Options | Recommendation | Why it matters |
|---|---|---|---|
| `MODEL_NAME` | `lightonai/LightOnOCR-2-1B` | as-is | — |
| `MODEL_REVISION` | commit SHA vs `main` | **pin a SHA** | `main` can change under you and silently invalidate both the Volume cache and the snapshot |
| `DTYPE` | `bfloat16` / `auto` | `auto` (card ships BF16) | forcing fp16 on a BF16 checkpoint can degrade quality |
| Gated repo? | Apache-2.0, not gated | no HF token needed | if you later use a gated model you need a `modal.Secret` |

### 2.2 Hardware

| Config | Options | Recommendation |
|---|---|---|
| `GPU` | `T4`, `L4`, `A10G`, `L40S`, `A100`, `H100`, `H200` | **`L4`** to start; **`L40S`** if you need throughput. A 1B BF16 VLM is ~2.5 GB of weights — H100/H200 is pure waste here |
| `N_GPU` (tensor parallel) | 1 / 2 / 4 | **1 — required.** GPU snapshots are documented as incompatible with multi-GPU code, and TP on a 1B model adds latency anyway |
| `GPU_MEM_UTIL` | 0.5 – 0.95 | **0.90** on a dedicated GPU |
| Container count | — | **3** (see §2.5) |

Sizing rule of thumb: weights ≈ 2 × params (BF16) = ~2 GB; the rest of VRAM is
KV cache + vision encoder activations. On a 24 GB L4 you have huge KV headroom,
so `--max-num-seqs` becomes your real batching knob.

### 2.3 vLLM server flags

| Flag | Value | Why |
|---|---|---|
| `--limit-mm-per-prompt` | `{"image": 1}` | one page per request; caps the vision budget. **Required by the model card** |
| `--mm-processor-cache-gb` | `0` | every page is a unique image — caching processed images only burns RAM. **Required by the card** |
| `--no-enable-prefix-caching` | on | no shared text prefix across pages, so prefix caching is dead weight. **Required by the card** |
| `--enable-sleep-mode` | on | **required for GPU snapshotting** — lets vLLM evict GPU memory to host RAM so it can be captured |
| `--max-model-len` | ~8192 | must fit image tokens (~1540px page) + `max_tokens` 4096. **Decide this**: too low → truncated pages, too high → smaller KV batches |
| `--max-num-seqs` | 16 – 64 | your main throughput/latency dial |
| `--max-num-batched-tokens` | ≥ `--max-model-len` | keeps a single large page from being chunked oddly |
| `--served-model-name` | e.g. `ocr` | lets you change checkpoints without changing client code |
| `--no-enforce-eager` | on | **keep compilation ON.** See §2.4 — with snapshots you pay compile once, not per cold start |
| `--api-key` | env-injected | vLLM's OpenAI server is otherwise open |

### 2.4 GPU memory caching / snapshotting  ⬅ the new piece

Modal can checkpoint the container's **GPU + CPU memory** after vLLM has fully
initialized, and restore from that image on every subsequent start. Reported
effect: cold starts drop from ~45s–2min to ~5–10s.

| Config | Value | Notes |
|---|---|---|
| `enable_memory_snapshot` | `True` | the CPU-side snapshot; prerequisite |
| `experimental_options` | `{"enable_gpu_snapshot": True}` | the GPU-side snapshot. **Alpha** — test before trusting it in prod |
| `VLLM_SERVER_DEV_MODE` | `"1"` | env var; exposes the `/sleep` and `/wake_up` endpoints the lifecycle needs |
| `TORCHINDUCTOR_COMPILE_THREADS` | `"1"` | env var; multi-threaded Inductor compilation can make snapshot creation fail |
| Sleep level | `1` | `POST /sleep?level=1` moves weights GPU→host RAM but keeps them; level 2 discards them |
| Pre-snapshot warmup | **skipped for now** | optional; see the note below |
| `N_GPU` | must be `1` | snapshots are incompatible with multi-GPU |

**`FAST_BOOT` becomes the wrong instinct.** Without snapshots, `--enforce-eager`
is a reasonable trade (skip compilation to boot faster). With snapshots, the
compiled kernels and CUDA graphs are captured *inside the snapshot*, so you pay
for compilation exactly once at snapshot-creation time and every later start
restores the compiled state. **Turn compilation on.**

#### Deferred: pre-snapshot warmup

Modal's snapshot examples run a few inference requests before `/sleep` so that
lazily-initialized state (multimodal preprocessor, vision encoder first pass,
cuDNN/cuBLAS autotuning, allocator pool growth) is captured too. **Skipped for
now** — vLLM already captures CUDA graphs at server init, so the remaining
payoff is a few seconds.

Note that scaling to zero strengthens the case for adding it back: cold starts
are now on the user's critical path, so those seconds land in a real request's
latency rather than in a deploy.

Two constraints if you do:

- It must use an **image**, not a text prompt. `--limit-mm-per-prompt {"image": 1}`
  is a ceiling rather than a requirement, so vLLM will accept text-only — but
  LightOnOCR's chat template is built around an image placeholder and will
  likely error or emit garbage.
- You do **not** need a real page file. Only the tensor shape matters, so
  synthesize one and avoid committing a binary:

  ```python
  from PIL import Image
  warmup_img = Image.new("RGB", (1190, 1540), "white")   # ~A4 at 1540px longest side
  ```

Snapshots do **not** help with the part of startup that is weight *downloading* —
that is what the HF Volume is for. The two mechanisms are complementary:

| Mechanism | Solves |
|---|---|
| `huggingface-cache` Volume | don't re-download 2 GB from HF |
| `vllm-cache` Volume | don't re-run torch.compile from scratch |
| GPU memory snapshot | don't re-load, re-compile, re-capture, or re-allocate at all |

### 2.5 Modal deployment knobs (autoscale 0 → 3)

| Config | Value | Trade-off |
|---|---|---|
| `MIN_CONTAINERS` | **0** | no idle GPU bill, but every burst after an idle gap pays a cold start |
| `MAX_CONTAINERS` | **3** | hard ceiling — a runaway client cannot fan out past 3 GPUs |
| `scaledown_window` | **5 min** (tune it) | the real idle-cost vs cold-start dial. Longer = fewer cold starts, more idle spend |
| `buffer_containers` | consider `1` | keeps one spare container above current load, so a *second* concurrent doc doesn't wait on a boot |
| `startup_timeout` | 10 min | the **first** boot creates the snapshot and is slow; later boots are fast |
| `target_concurrency` | ≈ `--max-num-seqs` | Modal's scale-up signal. Peak in-flight capacity ≈ `3 × target_concurrency` |
| `volumes` | `hf-cache` + `vllm-cache` | still needed — snapshots don't replace them |
| Auth | **`requires_proxy_auth=True`** on the web server, plus vLLM `--api-key` | a Modal web server is **public by default**, and vLLM's `--api-key` only guards `/v1/*`. See §2.9 |
| `region` | pin near your callers | cross-region adds latency per page |

Scaling to zero is what makes GPU snapshotting **load-bearing** rather than a
nice-to-have. With `min_containers=0` a cold start sits in the critical path of a
real user request, not just of deploys and preemptions — so the difference
between a ~2 minute boot and a ~10 second boot is user-visible latency on the
first page of an idle-period request.

Two consequences worth planning for:

- **The first request after idle is slow.** Either accept it, warm the fleet on a
  schedule (`modal.Cron`) around known traffic, or raise `scaledown_window`.
- **Client concurrency is now a ceiling, not a target.** Size
  `CLIENT_CONCURRENCY ≈ 3 × MAX_NUM_SEQS` so a large PDF can saturate all 3
  containers once they exist, but note that a fan-out of 100 pages against a
  cold fleet triggers scale-up while the first requests queue.

### 2.6 Image preprocessing (client side — quality lives or dies here)

| Config | Value from the card | Note |
|---|---|---|
| `RENDER_DPI` | `200` | how PDFs are rasterized |
| `MAX_LONGEST_SIDE` | `1540` px | preserve aspect ratio; do **not** letterbox or stretch |
| Format | PNG (lossless) | JPEG artifacts hurt small glyphs |
| Encoding | base64 data URL | `data:image/png;base64,...` |

### 2.7 Sampling / request

| Config | Value |
|---|---|
| `TEMPERATURE` | `0.2` |
| `TOP_P` | `0.9` |
| `MAX_TOKENS` | `4096` |
| Message shape | user message with **only** an `image_url` part — the chat template supplies the OCR instruction |

### 2.8 Client concurrency & resilience

| Config | Recommendation |
|---|---|
| `CLIENT_CONCURRENCY` | `3 × MAX_NUM_SEQS`; one shared semaphore process-wide |
| HTTP client | **one** shared `AsyncClient`/session, connection pool ≥ concurrency |
| Retries | 3 attempts, exponential backoff + jitter, honor `Retry-After` |
| Per-request timeout | generous (a dense page can take tens of seconds) |
| Session affinity | **skip it.** Modal's `Modal-Session-ID` header pins requests to a container for KV-cache reuse — pointless here, prefix caching is disabled and pages share nothing |
| Page failure policy | **decide**: fail the whole doc, or emit a placeholder and continue |

### 2.9 Auth — the endpoint is public unless you say otherwise

A Modal `web_server` publishes an **unauthenticated** URL by default, and that
URL is derived from your workspace / app / class / method names rather than being
random. Obscurity buys nothing.

vLLM's own `--api-key` is not sufficient on its own, because its auth middleware
only guards `/v1/*`. Everything else is served unauthenticated — and since GPU
snapshotting requires `VLLM_SERVER_DEV_MODE=1`, that set includes the admin
endpoints:

| Route | Exposed by | Consequence if public |
|---|---|---|
| `POST /sleep` | dev mode | **one request stalls all inference** until the container recovers |
| `POST /wake_up` | dev mode | forces GPU allocation |
| `POST /reset_prefix_cache` | dev mode | nuisance |
| `GET /health`, `/metrics` | always | information disclosure |

An anonymous `POST /sleep` is a one-request denial of service — a side effect of
a flag we added for snapshotting, not something the deployment would otherwise
have.

**Authenticate at the Modal proxy, above vLLM:**

```python
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT,
                  requires_proxy_auth=True)
```

Modal then rejects unauthenticated requests before they reach the container, so
every route is covered, dev-mode endpoints included. Callers pass `Modal-Key` /
`Modal-Secret` header tokens created in the Modal dashboard. Keep `--api-key` on
as a second layer — it costs nothing.

Note that the container's own calls (`/health`, `/sleep`, `/wake_up` during the
snapshot lifecycle) go to `127.0.0.1` and are unaffected by proxy auth.

---

## 3. Pseudocode

### 3.1 `ocr/config.py`

```python
# Single source of truth. Imported by both the Modal container and the client.

MODEL_NAME     = "lightonai/LightOnOCR-2-1B"
MODEL_REVISION = "<pin-a-commit-sha>"
SERVED_NAME    = "ocr"                 # stable alias clients call

# hardware
GPU            = "L4"
N_GPU          = 1                     # MUST stay 1: GPU snapshots are single-GPU only
GPU_MEM_UTIL   = 0.90

# vLLM
VLLM_PORT      = 8000
MAX_MODEL_LEN  = 8192
MAX_NUM_SEQS   = 32

# modal: autoscale 0 -> 3
MINUTES            = 60
MIN_CONTAINERS     = 0                 # scale to zero when idle
MAX_CONTAINERS     = 3                 # hard ceiling
SCALEDOWN_WINDOW   = 5  * MINUTES      # idle-cost vs cold-start dial
STARTUP_TIMEOUT    = 10 * MINUTES      # first boot builds the snapshot
TARGET_CONCURRENCY = MAX_NUM_SEQS      # per container

# gpu memory snapshotting
SLEEP_LEVEL = 1                        # 1 = offload weights to host RAM, keep them

# preprocessing (from the model card)
RENDER_DPI        = 200
MAX_LONGEST_SIDE  = 1540

# sampling
TEMPERATURE = 0.2
TOP_P       = 0.9
MAX_TOKENS  = 4096

# client: ceiling sized against a FULLY scaled-up fleet, not one container
CLIENT_CONCURRENCY = MAX_CONTAINERS * MAX_NUM_SEQS
REQUEST_TIMEOUT_S  = 300
MAX_RETRIES        = 3
```

### 3.2 `ocr/modal_app.py` — the GPU side

```python
import modal
from config import *

# --- 1. the container image -------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])                       # drop the base image's entrypoint
        .uv_pip_install("vllm==<pinned>", "huggingface_hub[hf_transfer]", "httpx")
        .env({
            "HF_XET_HIGH_PERFORMANCE": "1",
            # --- required for GPU snapshotting ---
            "VLLM_SERVER_DEV_MODE": "1",           # exposes /sleep and /wake_up
            "TORCHINDUCTOR_COMPILE_THREADS": "1",  # multi-threaded compile breaks snapshots
        })
)

# --- 2. persistent caches (still needed — snapshots don't replace them) -----
hf_cache   = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache",        create_if_missing=True)

app = modal.App("lightonocr-vllm")


# --- 3. the server ----------------------------------------------------------
# Modal does NOT reimplement vLLM. It boots a container, runs `vllm serve` as a
# subprocess, and reverse-proxies :8000 to a public URL.
@app.cls(
    image=image,
    gpu=f"{GPU}:{N_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache,
             "/root/.cache/vllm":        vllm_cache},

    # ---- GPU memory caching ----
    enable_memory_snapshot=True,                          # CPU snapshot (prerequisite)
    experimental_options={"enable_gpu_snapshot": True},   # GPU snapshot (alpha)

    # ---- autoscale 0 -> 3 ----
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,

    secrets=[modal.Secret.from_name("vllm-api-key")],
)
@modal.concurrent(max_inputs=TARGET_CONCURRENCY)   # per-container in-flight requests
class VLLMServer:

    # ======================================================================
    # PHASE 1 — runs BEFORE the snapshot is taken.
    # Everything expensive belongs here: load weights, compile, capture CUDA
    # graphs. All of it gets frozen into the snapshot. On the very first deploy
    # this is slow (minutes). Every start after that skips it entirely.
    #
    # No warmup inference for now -- see "Deferred: pre-snapshot warmup".
    # ======================================================================
    @modal.enter(snap=True)
    def build_snapshot(self):
        self.process = subprocess.Popen(build_vllm_command())
        wait_until_healthy(local_url(), timeout=STARTUP_TIMEOUT)

        # Tell vLLM to evict GPU memory to host RAM so it can be captured.
        # Without --enable-sleep-mode + this call, there is nothing snapshottable.
        post_local(f"/sleep?level={SLEEP_LEVEL}")
        # <-- Modal takes the GPU + CPU memory snapshot here.

    # ======================================================================
    # PHASE 2 — runs on EVERY start, after restoring from the snapshot.
    # Must be cheap. Just pull the weights back onto the GPU.
    # ======================================================================
    @modal.enter(snap=False)
    def restore(self):
        post_local("/wake_up")
        wait_until_healthy(local_url(), timeout=2 * MINUTES)

    # ======================================================================
    # PHASE 3 — expose vLLM's OpenAI-compatible API publicly.
    # ======================================================================
    @modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT,
                      requires_proxy_auth=True)   # see 2.9 -- public by default
    def serve(self):
        pass          # the subprocess is already listening; Modal just proxies

    @modal.exit()
    def stop(self):
        self.process.terminate()


def build_vllm_command() -> list[str]:
    """Kept separate so it is unit-testable and readable."""
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(N_GPU),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_MODEL_LEN),
    ]
    # --- the three flags the model card requires for OCR workloads ---
    cmd += ["--limit-mm-per-prompt", json.dumps({"image": 1})]
    cmd += ["--mm-processor-cache-gb", "0"]     # pages are never repeated
    cmd += ["--no-enable-prefix-caching"]       # no shared prefix across pages

    # --- required for GPU snapshotting ---
    cmd += ["--enable-sleep-mode"]

    # --- keep compilation ON: with snapshots you pay for it once, at snapshot
    #     creation, and every later start restores the already-compiled state.
    cmd += ["--no-enforce-eager"]

    # --- auth: read from the Modal Secret, never hardcoded ---
    cmd += ["--api-key", os.environ["VLLM_API_KEY"]]
    return cmd


def ocr_payload(image, max_tokens=MAX_TOKENS) -> dict:
    """One definition of an OCR request, imported by the client."""
    return {
        "model": SERVED_NAME,
        "messages": [{"role": "user",
                      "content": [{"type": "image_url",
                                   "image_url": {"url": to_data_url(image)}}]}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
```

### 3.3 `ocr/scripts/warm_cache.py` — pre-pull weights

```python
# Run once after changing MODEL_REVISION. Downloads into the Volume on a cheap
# CPU container so the first GPU boot (the one that builds the snapshot) doesn't
# also pay for the download.
@app.function(image=image, volumes={"/root/.cache/huggingface": hf_cache},
              timeout=30 * MINUTES)
def warm():
    huggingface_hub.snapshot_download(MODEL_NAME, revision=MODEL_REVISION)
    hf_cache.commit()          # persist so other containers see it
```

### 3.4 `ocr/client/pdf.py` — document → page images

```python
def pdf_to_page_images(pdf_bytes) -> list[Image]:
    """Rasterize each page at RENDER_DPI, then downscale to MAX_LONGEST_SIDE."""
    doc = pypdfium2.PdfDocument(pdf_bytes)
    scale = RENDER_DPI / 72                      # pdfium works in points
    for page in doc:
        img = page.render(scale=scale).to_pil()
        yield fit_longest_side(img, MAX_LONGEST_SIDE)


def fit_longest_side(img, target):
    """Preserve aspect ratio. Never upscale — that invents detail and slows the
    vision encoder for nothing."""
    factor = target / max(img.width, img.height)
    if factor >= 1:
        return img
    return img.resize((round(img.width * factor), round(img.height * factor)),
                      resample=LANCZOS)


def to_data_url(img) -> str:
    return "data:image/png;base64," + b64encode(png_bytes(img)).decode()
```

### 3.5 `ocr/client/ocr_client.py` — the async caller

```python
class OcrClient:
    """One instance per process. Owns the shared HTTP pool and the semaphore."""

    def __init__(self, base_url, api_key):
        # ONE client, reused for every request. Pool sized to our concurrency.
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT_S,
            limits=httpx.Limits(max_connections=CLIENT_CONCURRENCY,
                                max_keepalive_connections=CLIENT_CONCURRENCY),
        )
        # Bounds in-flight requests. Sized for the WHOLE 3-container fleet:
        # too low and 2 GPUs sit idle; too high and requests just queue.
        self._sem = asyncio.Semaphore(CLIENT_CONCURRENCY)

    async def ocr_page(self, img) -> str:
        # NOTE: image only. The model's chat template injects the OCR
        # instruction — adding your own text prompt is off-distribution.
        payload = ocr_payload(img)
        async with self._sem:
            resp = await self._post_with_retry("/v1/chat/completions", payload)
        return resp["choices"][0]["message"]["content"]

    async def _post_with_retry(self, path, payload):
        for attempt in range(MAX_RETRIES):
            resp = await self._http.post(path, json=payload)
            if resp.status_code < 400:
                return resp.json()
            if not is_transient(resp.status_code):     # 4xx other than 429
                raise OcrError(resp)
            await asyncio.sleep(backoff_delay(attempt, resp.headers))
        raise OcrError("retries exhausted")

    def backoff_delay(attempt, headers):
        # Respect the server first, then exponential backoff WITH jitter so a
        # burst of failed pages doesn't retry in lockstep.
        if "Retry-After" in headers:
            return float(headers["Retry-After"])
        return (2 ** attempt) * random.uniform(0.5, 1.5)

    async def aclose(self):
        await self._http.aclose()
```

### 3.6 `ocr/client/assemble.py` + top-level flow

```python
async def ocr_document(pdf_bytes, client) -> str:
    pages = list(pdf_to_page_images(pdf_bytes))

    # Fan out; the semaphore inside the client does the throttling and Modal
    # load-balances across the 3 containers.
    results = await asyncio.gather(
        *(client.ocr_page(p) for p in pages),
        return_exceptions=True,          # one bad page shouldn't kill the doc
    )

    return join_pages(results)


def join_pages(results) -> str:
    """Order is preserved by gather(). Decide your failure policy here."""
    out = []
    for i, r in enumerate(results, start=1):
        if isinstance(r, Exception):
            out.append(f"<!-- page {i}: OCR failed: {r} -->")   # or: raise
        else:
            out.append(r.strip())
    return "\n\n---\n\n".join(out)       # page separator of your choosing
```

### 3.7 `ocr/scripts/run_ocr.py` — local entrypoint

```python
@app.local_entrypoint()
async def main(path: str, out: str = "out.md"):
    url = VLLMServer().serve.web_url            # deployed server URL
    await wait_for_health(url)                  # poll /health, 200 == ready
    client = OcrClient(url, os.environ["VLLM_API_KEY"])
    try:
        markdown = await ocr_document(read_bytes(path), client)
    finally:
        await client.aclose()
    write(out, markdown)
```

---

## 4. Execution flow, end to end

```
deploy time
  scripts/warm_cache.py         -> weights land in hf-cache Volume (cheap CPU container)
  modal deploy ocr/modal_app.py -> image built once, cached

FIRST boot of a container (slow, once per code/model change)
  @modal.enter(snap=True)
      `vllm serve --enable-sleep-mode ...`
      weights read from hf-cache Volume
      torch.compile + CUDA graph capture      <- expensive, but only here
      POST /sleep?level=1                     <- GPU mem -> host RAM
  ===== Modal captures the GPU + CPU memory snapshot =====

EVERY later boot (deploy, preempt, crash, replacement)
  restore snapshot  (~seconds, not minutes)
  @modal.enter(snap=False) -> POST /wake_up   <- host RAM -> GPU mem
  @modal.web_server        -> port 8000 exposed behind Modal proxy auth

steady state
  0 containers when idle; scales up to 3 under load, each batching up to
  --max-num-seqs pages. Peak in-flight capacity = 3 x TARGET_CONCURRENCY.
  After SCALEDOWN_WINDOW with no traffic: back to 0, next burst restores
  from the snapshot.

per document (client side, no GPU)
  pdf bytes
    -> pypdfium2 render @200 DPI
    -> resize longest side to 1540, keep aspect
    -> PNG -> base64 data URL
    -> N concurrent POST /v1/chat/completions   [semaphore-bounded, shared pool]
    -> Modal load-balances across the 3 containers
    -> vLLM continuous-batches pages on each GPU
    -> per-page markdown
    -> join in page order
  markdown document
```

---

## 5. Open decisions to lock in before you deploy

1. **GPU class** — L4 (cheap) vs L40S (throughput). Benchmark pages/sec/$ on a
   representative 50-page PDF before committing.
2. **`scaledown_window`** — with `min_containers=0` this is the knob that decides
   how often your users eat a cold start. Needs a look at your real traffic
   pattern, not a default.
3. **`--max-model-len`** — measure actual image-token count for a 1540px page,
   add 4096, round up. Guessing here causes silent truncation.
4. **`--max-num-seqs`** — sets per-container batch size *and*, times 3, your
   total concurrency budget.
5. **Accept the alpha risk?** GPU snapshotting is experimental. Have a one-line
   kill switch (`experimental_options={}`) so you can fall back to normal boots
   if a vLLM upgrade breaks snapshot creation.
6. **Who gets a proxy-auth token?** The auth mechanism is settled (§2.9); what
   remains is issuing and rotating a token per caller.
7. **Sync vs async API surface** — does your FastAPI app await the OCR inline,
   or enqueue a job and return an id? Long PDFs make inline requests fragile.
8. **Per-page failure policy** — placeholder-and-continue vs fail-the-document.
9. **Where this plugs into `src/app/`** — a new router that calls `client/`, or
   swap the backend behind the existing `markitdown-ocr` path.
