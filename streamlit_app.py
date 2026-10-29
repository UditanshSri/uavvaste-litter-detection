import cv2
import numpy as np
import streamlit as st

from pipeline import coverage, detect, draw_boxes, draw_masks, load_models

st.set_page_config(page_title="UAVVaste litter detection", layout="wide")
st.title("UAVVaste litter detection")
st.caption("Upload a drone photo. YOLOv8m (trained with an NWD box loss) finds litter, "
           "SAM 2 turns each box into a mask.")


@st.cache_resource(show_spinner="Loading models (first run downloads SAM 2)...")
def models():
    return load_models()


file = st.file_uploader("Aerial image", type=["jpg", "jpeg", "png", "bmp"])
conf = st.slider("Confidence threshold", 0.05, 0.9, 0.12, 0.01,
                 help="Boxes scoring below this are dropped. Lower finds more litter "
                      "but adds false alarms; higher is stricter but misses more.")

if file is None:
    st.stop()

bgr = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Couldn't read that file as an image.")
    st.stop()

with st.spinner("Detecting and segmenting..."):
    boxes, scores, masks = detect(*models(), bgr, conf)

a, b, c = st.columns(3)
a.metric("Detections", len(boxes))
b.metric("Litter coverage", f"{coverage(masks):.2f}%")
c.metric("Mean confidence", f"{scores.mean():.2f}" if len(scores) else "-")

left, right = st.columns(2)
left.image(draw_boxes(bgr, boxes), channels="BGR", caption="YOLO boxes", width="stretch")
right.image(draw_masks(bgr, masks), channels="BGR", caption="SAM 2 masks", width="stretch")

if len(boxes):
    st.dataframe(
        [{"x1": round(float(x1)), "y1": round(float(y1)), "x2": round(float(x2)), "y2": round(float(y2)),
          "score": round(float(s), 3), "mask area (px)": int(m.sum())}
         for (x1, y1, x2, y2), s, m in zip(boxes, scores, masks)],
        width="stretch",
    )
