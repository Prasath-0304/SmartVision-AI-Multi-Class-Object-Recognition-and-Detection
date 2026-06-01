# SmartVision AI

SmartVision AI is a Streamlit-based object detection app that uses YOLOv8 to detect selected COCO object classes from uploaded images and live camera captures.

## Features

- Detects objects using YOLOv8
- Supports 26 selected COCO object classes
- Shows prediction confidence scores
- Displays green bounding boxes for detected objects
- Supports image upload detection
- Supports live camera detection using Streamlit
- Uses pretrained YOLOv8n weights by default
- Optimized for laptop and low-power systems

## Object Classes

The project supports the following classes:

airplane, bed, bench, bicycle, bird, bottle, bowl, bus, cake, car, cat, chair, couch, cow, cup, dog, elephant, horse, motorcycle, person, pizza, potted plant, stop sign, traffic light, train, truck

## Tech Stack

- Python
- Streamlit
- YOLOv8
- Ultralytics
- PyTorch
- Pillow
- NumPy

## Project Structure

```text
smartvision_dataset/
├── app.py
├── yolo_detector.py
├── download_pretrained_weights.py
├── requirements.txt
└── yolov8n.pt
