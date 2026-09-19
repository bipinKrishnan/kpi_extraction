import subprocess

import modal


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(frozen=True)
    .add_local_dir(
            "src/frontend/xlsx",
            "/root/xlsx",
            copy=True,
    )
    .add_local_file(
        "src/frontend/web_app.py",
        "/root/web_app.py",
    )
)

app = modal.App("vereport", image=image)


@app.function(
    secrets=[
        modal.Secret.from_name("kpi-streamlit-secrets"),
    ],
    cpu=2,
    memory=4096,
    min_containers=1,
    max_containers=1,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(
    8501,
    startup_timeout=120,
)
def vereport():

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "/root/web_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--server.enableWebsocketCompression=false",
            "--browser.gatherUsageStats=false",
        ]
    )