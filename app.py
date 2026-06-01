import streamlit as st
from PIL import Image

from yolo_detector import (
    BASE_DIR,
    CLASS_NAMES,
    CUSTOM_YOLO_PATH,
    PRETRAINED_YOLO_PATH,
    detect_objects,
    draw_detections,
    get_model_path,
    load_yolo_model,
)


st.set_page_config(page_title="SmartVision AI", layout="wide")


@st.cache_resource
def get_cached_model():
    return load_yolo_model()


def read_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def show_results(image, confidence):
    model = get_cached_model()

    if model is None:
        st.image(image, caption="Input image", use_container_width=True)
        st.error(f"YOLO model not found: {PRETRAINED_YOLO_PATH}")
        return

    detections = detect_objects(image, model, confidence)
    output_image = draw_detections(image, detections)

    image_col, result_col = st.columns([2, 1])

    with image_col:
        st.image(output_image, caption="Detected objects", use_container_width=True)

    with result_col:
        st.metric("Objects detected", len(detections))

        if not detections:
            st.warning("No selected class detected.")
            return

        best_detection = detections[0]
        st.metric("Top object", best_detection["class"])
        st.metric("Confidence", f"{best_detection['confidence']:.2f}%")

        st.write("All detections")
        for detection in detections:
            st.progress(
                detection["confidence"] / 100,
                text=f"{detection['class']} - {detection['confidence']:.2f}%",
            )


st.title("SmartVision AI - YOLO Object Detection")
st.write("Upload or capture an image to detect selected COCO objects with YOLO.")

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05,
)

upload_tab, camera_tab, info_tab = st.tabs(["Upload Image", "Camera", "Model Info"])

with upload_tab:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        show_results(read_image(uploaded_file), confidence)

with camera_tab:
    camera_file = st.camera_input("Capture image")
    if camera_file is not None:
        show_results(read_image(camera_file), confidence)

with info_tab:
    model_path = get_model_path()

    col1, col2 = st.columns(2)
    col1.metric("Selected classes", len(CLASS_NAMES))
    col2.metric("Active model", model_path.name if model_path else "Missing")

    st.write("Classes")
    st.write(", ".join(CLASS_NAMES))

    st.write("Model paths")
    st.code(
        f"Custom YOLO: {CUSTOM_YOLO_PATH}\n"
        f"Pretrained YOLO: {PRETRAINED_YOLO_PATH}",
        language="text",
    )

st.sidebar.markdown("### SmartVision AI")
st.sidebar.write(f"Project folder: `{BASE_DIR}`")
st.sidebar.write(f"Active model: `{get_model_path()}`")
