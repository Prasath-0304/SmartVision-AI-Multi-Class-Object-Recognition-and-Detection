from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
YOLO_WEIGHT_PATH = BASE_DIR / "yolov8n.pt"


def main():
    print("Preparing official COCO-pretrained YOLOv8n weights...")
    model = YOLO("yolov8n.pt")
    source_path = Path(model.ckpt_path).resolve()

    if source_path != YOLO_WEIGHT_PATH.resolve():
        YOLO_WEIGHT_PATH.write_bytes(source_path.read_bytes())

    print(f"Ready: {YOLO_WEIGHT_PATH}")


if __name__ == "__main__":
    main()
