# SmartVision AI

SmartVision AI is a Streamlit-based object detection app that uses pretrained YOLOv8n weights to detect selected COCO objects from uploaded images or live camera captures.

## Features

- Classifies images into 26 selected COCO object classes
- Detects objects using YOLOv8
- Shows prediction confidence scores
- Displays green bounding boxes for detected objects
- Supports image upload prediction
- Supports live camera detection using Streamlit
- Optimized for laptop and low-power systems

## Object Classes

The project supports the following classes:

airplane, bed, bench, bicycle, bird, bottle, bowl, bus, cake, car, cat, chair, couch, cow, cup, dog, elephant, horse, motorcycle, person, pizza, potted plant, stop sign, traffic light, train, truck

## Tech Stack

- Python
- Streamlit
- PyTorch
- Torchvision
- MobileNetV2
- YOLOv8
- Pillow
- OpenCV
- NumPy

## Project Structure

```text
smartvision_dataset/
├── app.py
├── train_local.py
├── requirements.txt
├── models/
│   ├── mobilenetv2_best.pth
│   └── class_names.json
├── detection/
│   └── yolov8n_training/
│       └── weights/
│           └── best.pt
└── yolov8n.pt
```

## Model Details

- MobileNetV2 is used for image classification.
- YOLOv8 is used for object detection and bounding box output.
- The Streamlit app uses YOLOv8 first for real-world uploaded images, then fallback models if needed.

## Installation

Clone the repository and install the required packages:

```bash
pip install -r smartvision_dataset/requirements.txt
```

## Run The App

```bash
cd smartvision_dataset
streamlit run app.py
```

Open the local URL shown by Streamlit:

```text
http://localhost:8501
```

## Retrain The Classification Model

```bash
cd smartvision_dataset
python train_local.py --images-per-class 100 --epochs 50 --force-download
```

The trained model will be saved at:

```text
smartvision_dataset/models/mobilenetv2_best.pth
```

## Output

The app displays:

- Predicted class
- Confidence score
- Prediction source
- Top predictions
- Green bounding boxes for detected objects

## Dataset

The dataset is prepared from selected COCO classes using the Hugging Face COCO dataset stream. Images are organized into train, validation, and test folders for classification training.

## Author

SmartVision AI - Intelligent Multi-Class Object Recognition System
