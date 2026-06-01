from pathlib import Path

from PIL import ImageDraw, ImageFont
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
CUSTOM_YOLO_PATH = BASE_DIR / "detection" / "yolov8n_training" / "weights" / "best.pt"
PRETRAINED_YOLO_PATH = BASE_DIR / "yolov8n.pt"
BOX_COLOR = "lime"
BOX_WIDTH = 4

SMARTVISION_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "traffic light": 9,
    "stop sign": 11,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "cow": 19,
    "elephant": 20,
    "bottle": 39,
    "cup": 41,
    "bowl": 45,
    "pizza": 53,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
}

CLASS_NAMES = list(SMARTVISION_CLASSES.keys())
CLASS_IDS = list(SMARTVISION_CLASSES.values())


def get_model_path():
    if CUSTOM_YOLO_PATH.exists():
        return CUSTOM_YOLO_PATH
    if PRETRAINED_YOLO_PATH.exists():
        return PRETRAINED_YOLO_PATH
    return None


def load_yolo_model():
    model_path = get_model_path()
    if model_path is None:
        return None
    return YOLO(str(model_path))


def detect_objects(image, model, confidence=0.3):
    if model is None:
        return []

    results = model.predict(image, conf=confidence, classes=CLASS_IDS, verbose=False)
    if not results or results[0].boxes is None:
        return []

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        detections.append(
            {
                "class_id": class_id,
                "class": model.names[class_id],
                "confidence": float(box.conf[0].item()) * 100,
                "box": box.xyxy[0].tolist(),
            }
        )

    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def draw_detections(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['class']} {detection['confidence']:.1f}%"
        draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR, width=BOX_WIDTH)

        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 8)
        draw.rectangle((x1, label_y, x1 + text_width + 8, label_y + text_height + 8), fill=BOX_COLOR)
        draw.text((x1 + 4, label_y + 4), label, fill="black", font=font)

    return annotated
