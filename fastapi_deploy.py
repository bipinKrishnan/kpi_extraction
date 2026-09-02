import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    # LibreOffice backs the pptx/docx -> PDF path in app.utils.office; only the
    # writer/impress components are needed, and fonts avoid glyph substitution.
    .apt_install(
        "libreoffice-writer",
        "libreoffice-impress",
        "fonts-liberation",
        "fonts-dejavu-core",
    )
    # Installs the locked default dependency group. uv_sync deliberately skips
    # the project itself, so the source is added separately below and code
    # changes do not rebuild the dependency layer.
    .uv_sync(frozen=True)
    # LOG_TO_FILE is left unset on purpose: container disks are ephemeral and
    # Modal already captures stdout.
    .env({"PYTHONPATH": "/root/src"})
    # Only the FastAPI package; src/ocr is a separate Modal app with its own
    # GPU image and is deployed on its own. add_local_* must come last, so the
    # source is mounted at container startup instead of triggering a rebuild on
    # every code change.
    .add_local_dir("src/app", remote_path="/root/src/app", ignore=["**/__pycache__"])
)

# The Modal CLI looks for an App named `app`, so keep this name.
app = modal.App("port-extract", image=image)


@app.function(
    # OCR_BASE_URL plus the Modal-Key / Modal-Secret proxy-auth pair the OCR
    # deployment requires, injected as OCR_MODAL_KEY / OCR_MODAL_SECRET. Never
    # baked into the image.
    secrets=[modal.Secret.from_name("ocr-service")],
    cpu=2.0,
    # LibreOffice conversions are memory-hungry, and several run concurrently.
    memory=4096,
    # The default 300s is short for a batch of conversions.
    timeout=900,
    # Keep containers warm between bursts so LibreOffice boots are not repaid.
    scaledown_window=300,
)
# Without this a container serves one request at a time, which would idle the
# router's 5 xlsx + 5 PDF conversion thread limiters. target_inputs leaves
# headroom so Modal scales out before a container is saturated.
@modal.concurrent(max_inputs=10, target_inputs=8)
@modal.asgi_app(label="port-extract", requires_proxy_auth=True)
def fastapi_app():
    """Serve the FastAPI app, including POST /extract/text."""
    # Aliased so it does not read as the modal.App above.
    from app.main import app as web_app

    return web_app
