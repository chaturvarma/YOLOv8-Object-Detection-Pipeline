# YOLOv8 Object Detection Pipeline

This project implements an object detection solution using the Ultralytics YOLOv8 model. The solution supports three inference modes:

- Image: Object detection on a single image.
- Video: Object detection and tracking in video files.
- Webcam: Live object detection from webcam feed.

Model weights and key detection parameters like confidence and IOU thresholds are read from a configuration file (config.yaml), while the inference mode and input/output file paths are specified dynamically via command-line arguments.

## Model Implementation

- Loads the YOLOv8 model and detection thresholds (conf_threshold and iou_threshold) from config.yaml.
- Takes the inference mode (image, video, or webcam) and input/output paths as command-line parameters.
- Detects objects and draws bounding boxes and labels on detected objects.
- Saves the results based on the mode.

## Configuration (config.yaml)

This file allows you to change model weights and fine-tune detection settings without modifying your code.

- model: Specifies the YOLOv8 model file to use.
- conf_threshold: The confidence threshold to filter out low-confidence predictions.
- iou_threshold: The IOU threshold for non-maximum suppression, used to eliminate overlapping boxes

All other runtime inputs (such as input/output paths and mode selection) are handled via command-line arguments

## Setup and Dependencies

Ensure that you have Python 3.8 or higher installed on your system

Use the following command to install the necessary dependencies:

```bash
pip install ultralytics opencv-python pyyaml
```

## How to Run

Before running any of the detection modes, ensure that the `config.yaml` file is properly configured with your desired YOLOv8 model and inference thresholds

To perform object detection on a single image, run:

```bash
python inference.py --mode image --input ./inputs/street.jpg --output ./outputs/predicted_street.jpg
```

To perform detection on a video file, run:

```bash
python inference.py --mode video --input ./inputs/highway.mp4 --output ./outputs/predicted_highway.mp4
```

To use your system's webcam for real-time object detection, run:

```bash
python inference.py --mode webcam
```

The `inputs/` folder contains sample image and video files that can be used for testing directly. The `outputs/` folder contains sample output results corresponding to the input files.