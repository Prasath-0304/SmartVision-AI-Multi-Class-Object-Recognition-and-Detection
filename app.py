from pathlib import Path

import json
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_best.pth"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
YOLO_PATH = BASE_DIR / "detection" / "yolov8n_training" / "weights" / "best.pt"
PRETRAINED_YOLO_PATH = BASE_DIR / "yolov8n.pt"

DEFAULT_CLASSES = [
    "airplane",
    "bed",
    "bench",
    "bicycle",
    "bird",
    "bottle",
    "bowl",
    "bus",
    "cake",
    "car",
    "cat",
    "chair",
    "couch",
    "cow",
    "cup",
    "dog",
    "elephant",
    "horse",
    "motorcycle",
    "person",
    "pizza",
    "potted plant",
    "stop sign",
    "traffic light",
    "train",
    "truck",
]


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return DEFAULT_CLASSES


CLASSES = load_class_names()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_WEIGHTS = models.MobileNet_V2_Weights.DEFAULT

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
IMAGENET_TRANSFORM = IMAGENET_WEIGHTS.transforms()


st.set_page_config(
    page_title="SmartVision AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_classification_model():
    if not MODEL_PATH.exists():
        return None

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_imagenet_model():
    model = models.mobilenet_v2(weights=IMAGENET_WEIGHTS)
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_yolo_model():
    if not YOLO_PATH.exists():
        return None

    from ultralytics import YOLO

    return YOLO(str(YOLO_PATH))


@st.cache_resource
def load_pretrained_yolo_model():
    if not PRETRAINED_YOLO_PATH.exists():
        return None

    from ultralytics import YOLO

    return YOLO(str(PRETRAINED_YOLO_PATH))


def imagenet_to_project_class(label):
    label = label.lower()

    if "dog" in label or any(
        breed in label
        for breed in (
            "retriever",
            "terrier",
            "spaniel",
            "hound",
            "poodle",
            "shepherd",
            "mastiff",
            "husky",
            "malamute",
            "chihuahua",
            "boxer",
            "beagle",
            "dalmatian",
            "rottweiler",
            "pug",
            "corgi",
            "collie",
            "doberman",
            "schnauzer",
            "pinscher",
            "dachshund",
            "newfoundland",
            "samoyed",
            "basenji",
            "leonberg",
        )
    ):
        return "dog"
    if "cat" in label or label in {"tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat"}:
        return "cat"
    if "bird" in label or any(name in label for name in ("hen", "cock", "ostrich", "brambling", "jay", "magpie", "robin")):
        return "bird"
    if "horse" in label or "zebra" in label:
        return "horse"
    if "cow" in label or "ox" in label or "bull" in label:
        return "cow"
    if "elephant" in label:
        return "elephant"
    if "bicycle" in label or "bike" in label:
        return "bicycle"
    if "motorcycle" in label or "moped" in label:
        return "motorcycle"
    if "car" in label or "cab" in label or "limousine" in label:
        return "car"
    if "truck" in label or "lorry" in label or "trailer truck" in label:
        return "truck"
    if "bus" in label:
        return "bus"
    if "airplane" in label or "airliner" in label or "warplane" in label:
        return "airplane"
    if "train" in label or "locomotive" in label:
        return "train"
    if "traffic light" in label:
        return "traffic light"
    if "stop sign" in label:
        return "stop sign"
    if "bench" in label or "park bench" in label:
        return "bench"
    if "bottle" in label:
        return "bottle"
    if "cup" in label or "mug" in label:
        return "cup"
    if "bowl" in label:
        return "bowl"
    if "pizza" in label:
        return "pizza"
    if "cake" in label:
        return "cake"
    if "chair" in label:
        return "chair"
    if "couch" in label or "sofa" in label:
        return "couch"
    if "potted plant" in label or "pot" in label and "plant" in label:
        return "potted plant"
    if "bed" in label:
        return "bed"
    if "person" in label or "man" in label or "woman" in label:
        return "person"

    return None


def predict_classification(image, model):
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        top_confidences, top_indices = torch.topk(probs, k=min(5, len(CLASSES)), dim=1)

    top_predictions = [
        {
            "class": CLASSES[class_index],
            "confidence": float(class_confidence) * 100,
        }
        for class_index, class_confidence in zip(
            top_indices[0].tolist(),
            top_confidences[0].tolist(),
        )
    ]

    return CLASSES[predicted.item()], confidence.item() * 100, top_predictions


def predict_imagenet_project_class(image, model):
    img_tensor = IMAGENET_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    aggregated = {class_name: 0.0 for class_name in CLASSES}
    for imagenet_index, probability in enumerate(probs.tolist()):
        project_class = imagenet_to_project_class(IMAGENET_WEIGHTS.meta["categories"][imagenet_index])
        if project_class in aggregated:
            aggregated[project_class] += probability * 100

    top_predictions = sorted(
        (
            {"class": class_name, "confidence": confidence}
            for class_name, confidence in aggregated.items()
        ),
        key=lambda item: item["confidence"],
        reverse=True,
    )[:5]

    best = top_predictions[0]
    return best["class"], best["confidence"], top_predictions


def predict_detected_project_class(image, model):
    if model is None:
        return None, 0.0, []

    results = model.predict(image, conf=0.25, verbose=False)
    if not results or results[0].boxes is None:
        return None, 0.0, []

    width, height = image.size
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if class_name not in CLASSES:
            continue

        confidence = float(box.conf[0].item()) * 100
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height)
        score = confidence * (0.75 + min(area_ratio, 0.25))
        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "score": score,
            }
        )

    if not detections:
        return None, 0.0, []

    top_predictions = sorted(detections, key=lambda item: item["score"], reverse=True)[:5]
    best = top_predictions[0]
    return best["class"], best["confidence"], top_predictions


def detect_objects(image, model, confidence=0.3):
    if model is None:
        return []

    results = model.predict(image, conf=confidence, verbose=False)
    if not results or results[0].boxes is None:
        return []

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if class_name not in CLASSES:
            continue

        detections.append(
            {
                "class": class_name,
                "confidence": float(box.conf[0].item()) * 100,
                "box": box.xyxy[0].tolist(),
            }
        )

    return detections


def draw_green_boxes(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['class']} {detection['confidence']:.1f}%"
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=4)

        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 8)
        draw.rectangle((x1, label_y, x1 + text_width + 8, label_y + text_height + 8), fill="lime")
        draw.text((x1 + 4, label_y + 4), label, fill="black", font=font)

    return annotated


st.title("SmartVision AI - Multi-Class Object Recognition")
st.write("Classify and detect selected COCO object classes using lightweight models.")

classification_tab, live_detection_tab, detection_tab, info_tab = st.tabs(
    ["Classification", "Live Detection", "Detection", "Dataset Info"]
)

with classification_tab:
    st.subheader("Image Classification")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="classification_upload",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns(2)

        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with right:
            model = load_classification_model()
            if model is None:
                st.error(f"Classification model not found: {MODEL_PATH}")
            else:
                class_name, confidence, custom_predictions = predict_classification(image, model)
                imagenet_model = load_imagenet_model()
                fallback_class, fallback_confidence, fallback_predictions = predict_imagenet_project_class(
                    image,
                    imagenet_model,
                )
                detected_class, detected_confidence, detected_predictions = predict_detected_project_class(
                    image,
                    load_pretrained_yolo_model(),
                )
                detections = detect_objects(image, load_pretrained_yolo_model())

                top_predictions = custom_predictions
                source = "custom model"

                if detected_class and detected_confidence >= 35:
                    class_name = detected_class
                    confidence = detected_confidence
                    top_predictions = detected_predictions
                    source = "object detector"
                elif confidence < 70 and fallback_confidence >= 10:
                    class_name = fallback_class
                    confidence = fallback_confidence
                    top_predictions = fallback_predictions
                    source = "pretrained fallback"

                st.metric("Predicted class", class_name)
                st.metric("Confidence", f"{confidence:.2f}%")
                st.caption(f"Prediction source: {source}")
                if confidence < 70:
                    st.warning("Low confidence prediction. Check the top predictions below.")
                st.write("Top predictions")
                for prediction in top_predictions:
                    st.progress(
                        prediction["confidence"] / 100,
                        text=f"{prediction['class']} - {prediction['confidence']:.2f}%",
                    )

                if source == "object detector" and detections:
                    st.image(
                        draw_green_boxes(image, detections),
                        caption="Output with green bounding boxes",
                        use_container_width=True,
                    )

with live_detection_tab:
    st.subheader("Live Detection")
    camera_file = st.camera_input("Capture image")

    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")
        yolo_model = load_yolo_model() or load_pretrained_yolo_model()

        if yolo_model is None:
            st.info(f"YOLO model not found: {YOLO_PATH} or {PRETRAINED_YOLO_PATH}")
        else:
            detections = detect_objects(image, yolo_model)
            st.image(
                draw_green_boxes(image, detections),
                caption="Live detection output",
                use_container_width=True,
            )
            st.success(f"Detected {len(detections)} objects")

with detection_tab:
    st.subheader("Object Detection")
    uploaded_file = st.file_uploader(
        "Choose an image for detection",
        type=["jpg", "jpeg", "png"],
        key="detection_upload",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        yolo_model = load_yolo_model() or load_pretrained_yolo_model()

        if yolo_model is None:
            st.image(image, caption="Uploaded image", use_container_width=True)
            st.info(f"YOLO model not found: {YOLO_PATH} or {PRETRAINED_YOLO_PATH}")
        else:
            detections = detect_objects(image, yolo_model)
            st.image(
                draw_green_boxes(image, detections),
                caption="Detected objects",
                use_container_width=True,
            )
            st.success(f"Detected {len(detections)} objects")

with info_tab:
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total classes", len(CLASSES))
    col2.metric("Images per class", "100")
    col3.metric("Total images", "2,600")

    st.write("Classes included")
    st.write(", ".join(CLASSES))

st.sidebar.markdown("### SmartVision AI")
st.sidebar.write(f"App folder: `{BASE_DIR}`")
st.sidebar.write(f"Device: `{DEVICE}`")
st.sidebar.write("Put trained model files in these locations:")
st.sidebar.code(
    "models/mobilenetv2_best.pth\n"
    "detection/yolov8n_training/weights/best.pt",
    language="text",
)
