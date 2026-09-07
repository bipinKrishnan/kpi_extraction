# Gemma 4 extraction service — post-deploy notes

Everything you have to get right *after* `modal deploy src/extract/modal_app.py`,
split into the Modal side and the client side. Companion to
`src/extract/modal_app.py`; the tunables themselves are documented inline there.

---

## 1. Deploy and first boot

```
modal deploy src/extract/modal_app.py
```

The **first** deploy is slow — minutes. It downloads ~16 GB of weights into the
`huggingface-cache` Volume, runs `torch.compile`, captures CUDA graphs, then
`/sleep`s so Modal can snapshot GPU memory. `STARTUP_TIMEOUT` is 15 minutes for
exactly this. Every boot after that restores the snapshot and is fast.

Get the real URL from the deploy output — do not hand-assemble it:

```
modal app list
```

It will look like
`https://<workspace>--gemma4-extract-vllm-vllmserver-serve.modal.run`.

Watch the boot:

```
modal app logs gemma4-extract-vllm
```

### What invalidates the snapshot

The snapshot is keyed to the image and the class configuration. Changing any of
these means the next boot rebuilds it from scratch (slow, once):

- `VLLM_IMAGE` tag
- `MODEL_NAME` or `MODEL_REVISION`
- anything feeding `build_vllm_command()` — `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
  `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION`, the flag helpers
- `ENABLE_GPU_SNAPSHOT` / `SLEEP_LEVEL`

Editing only comments or the docstring does not.

---

## 2. Modal server side

### Proxy auth tokens

`REQUIRE_PROXY_AUTH = True` is the **only** thing authenticating this
deployment. Modal rejects unauthenticated requests before they reach the
container, which matters more than usual here: `VLLM_SERVER_DEV_MODE=1` (needed
for snapshotting) exposes `POST /sleep` and `POST /wake_up` outside vLLM's own
guarded prefixes, so an anonymous `/sleep` would offload the weights and stall
every request.

Create the token pair in the Modal dashboard → **Settings → Proxy Auth Tokens**.
You get a `wk-…` key and a `ws-…` secret.

A token pair is workspace-wide, so the existing OCR pair would work — **use a
separate pair anyway**, so you can rotate the extraction service without
breaking OCR.

### The Modal Secret the client reads

The FastAPI app gets its config from a Modal Secret, never from source. Mirror
the existing `ocr-service` secret:

```
modal secret create extract-service \
  EXTRACT_BASE_URL=https://<workspace>--gemma4-extract-vllm-vllmserver-serve.modal.run \
  EXTRACT_MODAL_KEY=wk-xxxxxxxxxxxxxxxx \
  EXTRACT_MODAL_SECRET=ws-xxxxxxxxxxxxxxxx
```

Then add it to the FastAPI deployment in `fastapi_deploy.py`:

```
secrets=[
    modal.Secret.from_name("ocr-service"),
    modal.Secret.from_name("extract-service"),
],
```

**`EXTRACT_BASE_URL` has no trailing path and no trailing slash** — the client
appends `/v1/chat/completions` itself.

Secret changes are picked up on container start, so after editing a secret:

```
modal deploy fastapi_deploy.py
```

Locally, the same three variables go in `.env` (already gitignored) or an
`.envrc`. Never commit them.

### Shared Volumes

This app reuses the OCR app's `huggingface-cache` and `vllm-cache` Volumes on
purpose — one cache, two tenants. Consequences:

- Do not prune either Volume assuming it belongs to one service.
- `huggingface-cache` now carries an extra ~16 GB; Volume storage is billed.
- Deleting the compile cache in `vllm-cache` costs a recompile on both apps.

### Fleet limits worth remembering

| Setting | Value | What it means downstream |
|---|---|---|
| `MIN_CONTAINERS` | 0 | Scales to zero. The first request after an idle gap waits for a snapshot restore — the client timeout must absorb it. |
| `MAX_CONTAINERS` | 3 | Hard ceiling. Fleet capacity is `3 × 32 = 96` concurrent sequences. |
| `MAX_NUM_SEQS` | 32 | Also the `@modal.concurrent(max_inputs=…)` value, so Modal will not over-admit a container. |
| `SCALEDOWN_WINDOW` | 3 min | Idle-cost vs cold-start dial. |
| `BUFFER_CONTAINERS` | 0 | No spare warm GPU. Raise to 1 only if burst latency proves painful — it costs an idle GPU whenever anything runs. |

### Verify it is alive

```
curl -H "Modal-Key: $EXTRACT_MODAL_KEY" \
     -H "Modal-Secret: $EXTRACT_MODAL_SECRET" \
     "$EXTRACT_BASE_URL/v1/models"
```

The `id` in the response is **`extract`** (the `--served-model-name` alias), not
`google/gemma-4-E4B-it`. That alias is what clients must send, and it exists so
they never have to change when the repo id or revision does.

---

## 3. Client side

### The server is a passthrough — the schema lives here

`src/extract/modal_app.py` knows nothing about the extraction schema. The client
sends the system prompt, the user text, **and** `response_format` on every
request. Upside: schema changes need no redeploy. Downside: forget
`response_format` and you get prose back, not JSON.

### The one gotcha that bites everyone

Constrained decoding enforces **shape only**. The model never sees the JSON
schema, its field names' meaning, or any `Field(description=...)`. So:

> Put the field semantics in the **system prompt**, and use `response_format`
> purely for structural enforcement.

For `sample_output.json` that means spelling out, in the prompt: `period` is
`YYYY-MM`; the preferred `period_type` values (`MTH`, `QTR`, `YTD`, …); the
preferred `measure` values (`actual`, `actual_cc`, `forecast`,
`forecast_delta`) and that both vocabularies are open rather than closed;
the sign convention for `forecast_delta`; that `value` is a plain number with no
thousands separators; which currency to report; and what to emit when a figure
is absent.

Also: vLLM's default `xgrammar` backend does **not** enforce `pattern`,
`minimum`, or `maximum`. Never rely on the schema to police value *formats* —
parse, then validate with a Pydantic model, and treat a validation failure as a
failed request.

### Request contract

| Field | Value | Why |
|---|---|---|
| `model` | `"extract"` | The `--served-model-name` alias. |
| `temperature` | `0.0` | Extraction wants determinism, not OCR's `0.2`. |
| `top_p` | `1.0` | Leave sampling alone at `temperature=0`. |
| `max_tokens` | 4096–8192 | A report with 30 KPIs × 5 observations is a lot of JSON. |
| `response_format` | `{"type": "json_schema", "json_schema": {"name": …, "schema": Model.model_json_schema()}}` | Structural enforcement. |
| `stream` | omit | No benefit for a machine-consumed JSON blob. |

**Always check `finish_reason` before `json.loads`.** With constrained decoding,
hitting `max_tokens` yields a *structurally valid prefix* that is still
truncated and unparseable. `finish_reason == "length"` means the answer was cut
off — retrying with the same budget will cut it off again.

**Total token budget is 16384** (`MAX_MODEL_LEN`), covering system prompt +
document text + `max_tokens`. Exceed it and vLLM returns a 400 rather than
truncating. Count tokens client-side, or chunk the document, before you send.

### Do not send

- **Images or audio.** The server runs `--limit-mm-per-prompt image=0,audio=0`,
  so any `image_url` / `audio_url` content part is rejected. Text extraction
  runs on the *output of the OCR service*, not on page images.
- **`chat_template_kwargs: {"enable_thinking": true}`.** Thinking mode is off
  and no `--reasoning-parser gemma4` is configured, so the raw
  `<|channel>thought` delimiters would leak into `content` and break JSON
  parsing. Turning it on is a server-side change (parser flag + custom chat
  template + a bigger `MAX_MODEL_LEN`), not a client flag.

### Keep the prefix cache working

Prefix caching is **on** for this deployment (unlike OCR, where every page was
unique). It only pays off if requests share a byte-identical leading prefix:

- Put the system prompt **first** and keep it byte-stable across requests —
  a timestamp, a request id, or a reordered instruction list in the system
  prompt destroys every cache hit.
- Put the per-document text in the **user** message, after it.

### Reuse one client, cap concurrency

Follow `src/app/utils/ocr_client.py` — the pattern is already right:

- **One `httpx.AsyncClient` per process**, created at startup, closed at
  shutdown, handed to endpoints via a dependency. Never one client per request.
- Proxy-auth headers set once on the client:
  `{"Modal-Key": …, "Modal-Secret": …}` from the Secret-provided settings.
- `httpx.Limits(max_connections=N, max_keepalive_connections=N)` matched to a
  process-wide `asyncio.Semaphore(N)`.
- `N` is **per app worker**, and several app containers share one GPU fleet, so
  keep it well under the fleet's 96. The OCR client uses 32 for the same reason.
- Retries: exponential backoff **with jitter**, honouring `Retry-After`, on
  `{408, 409, 425, 429, 500, 502, 503, 504}`. A fleet scaling from zero
  produces these legitimately.
- Request timeout generous — `min_containers=0` means the first call after an
  idle period also pays a container wake. OCR uses 300 s.

### Config constants to add

`src/app/core/config.py` already pins its OCR constants to the deployment.
Do the same here so client and server cannot drift: `EXTRACT_MODEL_NAME =
"extract"`, the sampling values, `EXTRACT_MAX_TOKENS`, `EXTRACT_CONCURRENCY`,
and the three `extract_*` `Settings` fields (`SecretStr` for the token pair).

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM during snapshot creation | 16 GB of weights plus snapshot machinery on a 40 GB card | Drop `GPU_MEMORY_UTILIZATION` to 0.85, or set `ENABLE_GPU_SNAPSHOT = False` |
| Build fails at step 2: `python: not found` (exit 127) | The vLLM image has `python3` but no `python`, and Modal's builder runs `python -m pip install` before any `.run_commands()` | Already handled by `SETUP_COMMANDS` / `setup_dockerfile_commands` in the script — it is the only hook that runs ahead of the builder's own steps |
| `ImportError: cannot import name 'Sentinel' from 'typing_extensions'` when `vllm serve` starts | On image builder ≤ 2024.10 Modal installs its client deps from a *pinned* requirements file into the vLLM environment, downgrading `typing_extensions` below the ≥4.13 `pydantic_core` needs | Handled by `REPAIR_PACKAGES` in the script (a `.run_commands()` layer after the builder's install). Same class of bug can hit `protobuf` / `grpclib` — add the offending pin there |
| Repeated dependency conflicts between Modal's client deps and vLLM | Same root cause: builder ≤ 2024.10 installs into the image env | `modal config set image_builder_version 2025.06` mounts client deps at runtime instead. Client-wide, so the OCR image rebuilds too |
| Container killed while sleeping | `SLEEP_LEVEL = 1` parks ~16 GB in host RAM | Raise container memory, or use `SLEEP_LEVEL = 2` (discards weights; slower wake) |
| `starting container ... vfs.CompleteRestore() failed ... failed to walk "vo-.../xet/logs/xet_<ts>.log"` | The snapshot was taken with a HuggingFace Xet log file open **inside the mounted Volume** (`HF_XET_CACHE` defaults to `$HF_HOME/xet`); the file is gone by restore time | Handled by `HF_XET_CACHE=/tmp/xet` in the script, which keeps Xet's cache and logs on the container's own disk. Escalate to `HF_HUB_DISABLE_XET=1` if it recurs. **General rule: never hold an open handle to a Volume file across a snapshot** |
| `Model architectures ['Gemma4ForConditionalGeneration'] are not supported` | The `vllm/vllm-openai:gemma4` tag moved, or a fallback to a stable wheel happened | Re-pin the image; Gemma 4 is not in stable vLLM releases |
| 400, prompt longer than max context | Input + `max_tokens` over 16384 | Chunk the document, lower `max_tokens`, or raise `MAX_MODEL_LEN` (costs KV cache and re-snapshots) |
| Truncated / unparseable JSON | `finish_reason == "length"` | Raise `max_tokens`, or split the extraction into smaller documents |
| Valid JSON, wrong values | Semantics only in `Field(description=…)`, which the model never sees | Move them into the system prompt |
| 401/403 on every call | Missing or stale `Modal-Key` / `Modal-Secret` | Check the `extract-service` secret, then redeploy the FastAPI app |
| Every request slow, no cache hits | System prompt not byte-stable, or per-request text placed before it | Stabilise the prefix |
| First request each morning takes minutes | Scale-from-zero cold start | Expected; raise `MIN_CONTAINERS`/`BUFFER_CONTAINERS` to trade money for latency |

---

## 5. Re-check on every version bump

Bumping `VLLM_IMAGE` is the riskiest change here, because Gemma 4 support comes
from a nightly lineage rather than a stable release. After any bump:

1. Flag names — vLLM minor releases rename them (`--limit-mm-per-prompt`,
   `--async-scheduling`, the structured-output backend flag).
2. Snapshot creation still succeeds; if not, fall back to
   `ENABLE_GPU_SNAPSHOT = False` and redeploy.
3. One real extraction request end to end, with `response_format`, and validate
   the result against the Pydantic model.
4. `MODEL_REVISION` still resolves. Re-pin with:

```
python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('google/gemma-4-E4B-it').sha)"
```
