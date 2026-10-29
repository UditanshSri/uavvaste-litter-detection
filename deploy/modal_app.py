"""Serves api.py on Modal. From the repo root:

    modal deploy deploy/modal_app.py

The container scales to zero when idle; the first request after that takes
about a minute while the models load.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.15.0", "torchvision", "ultralytics==8.4.192", "fastapi", "python-multipart",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .workdir("/root")
    # bake SAM 2 weights into the image instead of downloading on every cold start
    .run_commands("python -c \"from ultralytics import SAM; SAM('sam2_s.pt')\"")
    .add_local_file(ROOT / "weights" / "best.pt", "/root/weights/best.pt")
    .add_local_file(ROOT / "pipeline.py", "/root/pipeline.py")
    .add_local_file(ROOT / "api.py", "/root/api.py")
)

app = modal.App("uavvaste-litter-api")


@app.function(image=image, cpu=2, memory=4096, timeout=300, scaledown_window=300)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/root")
    from api import app as fastapi_app
    return fastapi_app
