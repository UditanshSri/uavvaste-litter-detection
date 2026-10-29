# UAVVaste litter detection

Detects and segments litter in drone images from the [UAVVaste](https://github.com/PUTvision/UAVVaste) dataset.

1. **Detection:** YOLOv8m, fine-tuned at 1024 px. Half of the box loss is replaced with a Normalized Wasserstein Distance (NWD) term, which copes better with the very small objects in this dataset.
2. **Segmentation:** each detected box is passed to SAM 2 (Hiera Small) as a prompt, giving a pixel mask per object and the share of the frame covered by litter.

## Results

Random 80/20 split (617 train / 155 val images, seed 42), COCO bbox metrics on the val split:

| | mAP@50 | mAP@50:95 | AP small | AR small |
|---|---|---|---|---|
| Reference numbers used for comparison (UAVVaste paper) | 51.2 | 28.4 | 14.8 | 20.1 |
| NWD-YOLOv8m | 59.3 | 35.2 | 21.9 | 30.7 |
| NWD-YOLOv8m + SAM 2 (box refitted to mask) | 59.2 | 33.8 | 21.9 | 32.5 |

The paper used its own split, so this is not a strict like-for-like comparison. See also the known issues below.

## Repo layout

```
streamlit_app.py   web app (upload an image, see boxes, masks, coverage)
api.py             REST API (FastAPI), same pipeline
pipeline.py        model loading, inference and drawing, shared by both
weights/best.pt    trained NWD-YOLOv8m weights
notebooks/         Kaggle notebook: data prep, training, evaluation
```

SAM 2 weights (`sam2_s.pt`) are downloaded automatically by Ultralytics the first time the app runs.

## Running locally

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run streamlit_app.py          # web app on http://localhost:8501
uvicorn api:app --port 8000             # API, docs on http://localhost:8000/docs
```

Calling the API:

```
curl -F file=@image.jpg "http://localhost:8000/predict?conf=0.12"
```

```json
{"width": 3840, "height": 2160, "count": 12, "coverage_pct": 0.41,
 "detections": [{"box": [1022.5, 640.1, 1061.0, 671.3], "score": 0.81, "mask_area_px": 702}]}
```

`box` is `[x1, y1, x2, y2]` in pixels. `conf` is the confidence threshold: lower finds more litter but adds false alarms. The default of 0.12 favours recall, because SAM can't recover an object the detector missed.

## Training

Open `notebooks/uavvaste_nwd_yolo_sam2.ipynb` on Kaggle, attach the UAVVaste dataset, enable a GPU (T4) and internet, and use Save & Run All. Training takes about 1.2 h for 100 epochs on a T4.

## Known issues

- The notebook's evaluation converts images to RGB before passing them to YOLO, but Ultralytics expects BGR for in-memory arrays, so the val metrics above were computed on channel-swapped images. Training reads files from disk and is unaffected, and the app and API pass BGR correctly.
- The notebook collects val images with `glob("*.jpg")`, which misses files with an upper-case `.JPG` extension. Those images still count in the ground truth, which lowers the reported scores.
- The NWD constant is fixed at 12, but Ultralytics computes the box loss in stride units, so the effective scale is larger than intended.
