"""Streamlit app: Brain Tumor MRI classifier.

Run:  streamlit run app.py
Needs the `models/` folder created by the notebook:
    models/*.h5, models/class_names.json, models/model_info.json
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

MODEL_DIR = Path(__file__).parent / "models"
IMG_SIZE = (224, 224)
LOW_CONFIDENCE = 0.70  # below this we warn the user

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")


# ---------- helpers ----------
@st.cache_data
def load_json(path: str):
    """Read a small JSON file; return None if it does not exist."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def load_model(h5_path: str):
    """Load a trained Keras model once and reuse it between reruns."""
    import tensorflow as tf
    return tf.keras.models.load_model(h5_path, compile=False)


def preprocess(arr: np.ndarray, kind: str) -> np.ndarray:
    """Apply the SAME preprocessing that was used when the model was trained."""
    import tensorflow as tf
    x = arr.astype("float32")
    if kind == "rescale":
        return x / 255.0
    if kind == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)
    if kind == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(x)
    return x  # EfficientNet expects raw 0-255 pixels


def read_image(file) -> Image.Image:
    """Open the upload as an RGB image or raise a clear error."""
    try:
        return Image.open(file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ValueError("This file is not a valid image. Upload a JPG or PNG.")


# ---------- load config ----------
class_names = load_json(str(MODEL_DIR / "class_names.json"))
info = load_json(str(MODEL_DIR / "model_info.json"))

st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Classifies a brain MRI into Glioma, Meningioma, Pituitary tumor or No Tumor.")

if not class_names or not info:
    st.error("`models/class_names.json` or `models/model_info.json` not found. "
             "Run the notebook first and copy the `models/` folder next to app.py.")
    st.stop()

models_cfg = info["models"]
names = list(models_cfg.keys())
default_idx = names.index(info["best"]) if info.get("best") in names else 0

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", names, index=default_idx)
    st.write(f"Best model on validation data: **{info.get('best', 'n/a')}**")
    if info.get("metrics"):
        st.subheader("Test-set results")
        st.dataframe(info["metrics"], width="stretch")
    st.warning("Educational project. Not a medical device. "
               "Never use it for real diagnosis.")

# ---------- main ----------
upload = st.file_uploader("Upload a brain MRI image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if upload is None:
    st.info("Upload an MRI image to get a prediction.")
    st.stop()

try:
    img = read_image(upload)
except ValueError as e:
    st.error(str(e))
    st.stop()

cfg = models_cfg[model_name]
model_path = MODEL_DIR / cfg["file"]
if not model_path.exists():
    st.error(f"Model file missing: {model_path.name}")
    st.stop()

col_img, col_res = st.columns([1, 1.2])
col_img.image(img, caption="Uploaded image", width="stretch")

with st.spinner("Predicting..."):
    model = load_model(str(model_path))
    batch = np.expand_dims(preprocess(np.array(img.resize(IMG_SIZE, Image.BILINEAR)), cfg["preprocess"]), 0)
    probs = model.predict(batch, verbose=0)[0]

top = int(np.argmax(probs))
conf = float(probs[top])

with col_res:
    st.subheader(f"Prediction: {class_names[top]}")
    st.metric("Confidence", f"{conf * 100:.1f}%")
    if conf < LOW_CONFIDENCE:
        st.warning("Low confidence. The model is unsure about this image. Treat the result as unreliable.")
    st.write("**Probability for each class**")
    for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1]):
        st.progress(float(p), text=f"{name}: {p * 100:.1f}%")
    st.bar_chart({n: float(p) for n, p in zip(class_names, probs)})
