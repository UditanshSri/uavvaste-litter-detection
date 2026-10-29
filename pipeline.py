"""NWD-YOLOv8m boxes -> SAM 2 masks. Shared by the Streamlit app and the API."""
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO

WEIGHTS = Path(__file__).parent / "weights" / "best.pt"

# ultralytics predictors keep per-image state, so serialise calls across sessions
_lock = threading.Lock()


def load_models(weights=WEIGHTS, sam_weights="sam2_s.pt"):
    """sam2_s.pt is downloaded by ultralytics on first use."""
    return YOLO(str(weights)), SAM(sam_weights)


def detect(detector, sam, bgr, conf=0.12, imgsz=1024):
    """BGR image -> boxes (N, 4) xyxy, scores (N,), masks (N, H, W) bool."""
    with _lock:
        res = detector(bgr, conf=conf, imgsz=imgsz, verbose=False)[0].boxes
        boxes, scores = res.xyxy.cpu().numpy(), res.conf.cpu().numpy()
        if len(boxes) == 0:
            return boxes, scores, np.zeros((0, *bgr.shape[:2]), dtype=bool)
        masks = sam(bgr, bboxes=boxes, verbose=False)[0].masks.data.cpu().numpy().astype(bool)
    return boxes, scores, masks


def coverage(masks):
    """Percent of the image covered by at least one mask."""
    return float(masks.any(axis=0).mean() * 100) if len(masks) else 0.0


def mask_polygons(mask, eps=1.0):
    """Outline(s) of a mask as lists of [x, y] points, for drawing client-side."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, eps, True).reshape(-1, 2).tolist() for c in contours if len(c) >= 3]


def draw_boxes(bgr, boxes):
    out = bgr.copy()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out


def draw_masks(bgr, masks):
    out = bgr.copy()
    for m in masks:
        out[m] = (out[m] * 0.5 + np.array([255, 120, 0]) * 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out
