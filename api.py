"""REST API for the pipeline.

    uvicorn api:app --port 8000
    curl -F file=@image.jpg "http://127.0.0.1:8000/predict?conf=0.12"

Interactive docs at http://127.0.0.1:8000/docs
"""
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import coverage, detect, load_models, mask_polygons

app = FastAPI(title="UAVVaste litter detection")
# public API, called from the web page on another domain
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
detector, sam = load_models()


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "docs": "/docs"}


@app.post("/predict")
def predict(file: UploadFile = File(...), conf: float = 0.12):
    bgr = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    boxes, scores, masks = detect(detector, sam, bgr, conf)
    return {
        "width": bgr.shape[1],
        "height": bgr.shape[0],
        "count": len(boxes),
        "coverage_pct": round(coverage(masks), 3),
        "detections": [
            {"box": [round(float(v), 1) for v in box],
             "score": round(float(score), 4),
             "mask_area_px": int(mask.sum()),
             "polygons": mask_polygons(mask)}
            for box, score, mask in zip(boxes, scores, masks)
        ],
    }
