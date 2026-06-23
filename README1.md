# Object Detection Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-TorchVision-red)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-purple)
![Object Detection](https://img.shields.io/badge/Task-Object%20Detection-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Benchmarking pretrained object detection models on real-world style images.**

</div>

---

## 📌 Overview

This repository contains an Object Detection benchmark project developed for the course:

**Advanced Topics in Computer Vision**

The project evaluates and compares three pretrained object detection models:

- **YOLOv5m**
- **Faster R-CNN ResNet50 FPN**
- **SSD300 VGG16**

The goal is to detect and label objects in images, visualize bounding boxes, estimate confidence scores, and compare the models based on:

- detection quality,
- confidence scores,
- inference speed,
- number of detected objects,
- robustness in complex scenes.

---

## 🎯 Project Goals

The main objectives of this assignment are:

- apply pretrained neural networks for object detection,
- detect objects in real-world style images,
- visualize bounding boxes and predicted labels,
- evaluate confidence scores,
- compare different detection architectures,
- analyze speed versus detection quality.

---

## 🖼️ Dataset

The benchmark uses **11 AI-generated images** with different indoor and outdoor scenarios.

The image set includes scenes such as:

- coastal landscape with person, bench, and bicycle,
- home office with books, plants, chair, and monitor,
- family dinner table,
- motorbike riders,
- luxury vehicles,
- city bus in motion,
- modern workspace,
- city skyline,
- supercar and driver,
- air show with aircraft,
- western cowboy scene with horses and dog.

All input images should be placed inside the `images/` directory.

```text
images/
├── coastal-reflection.jpg
├── cozy-home-office-with-a-thoughtful-professional.jpg
├── family-dinner-spread-with-a-roasted-chicken.jpg
├── group-of-riders-showcasing-powerful-motorbikes.jpg
├── luxury-vehicles-on-display.jpg
├── maroon-city-bus-in-motion.jpg
├── modern-workspace-with-natural-light.jpg
├── skyline-view-of-a-modern-city.jpg
├── sleek-mclaren-supercar-with-dapper-driver.jpg
├── spectacular-air-show-with-precision-aircraft.jpg
└── western-adventure-cowboy-canine-companion.jpg
```

---

## 🧠 Models

### YOLOv5m

YOLOv5m is used through the `ultralytics` library.

```python
from ultralytics import YOLO

model = YOLO("yolov5m.pt")
```

YOLOv5 performs preprocessing internally and returns detections with labels, confidence scores, and bounding boxes.

### Faster R-CNN ResNet50 FPN

Faster R-CNN is loaded from `torchvision.models.detection`.

```python
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn
)

model = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
```

This model uses COCO class labels and generally provides detailed object localization.

### SSD300 VGG16

SSD300 is also loaded from `torchvision.models.detection`.

```python
from torchvision.models.detection import (
    SSD300_VGG16_Weights,
    ssd300_vgg16
)

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
```

SSD300 is faster than Faster R-CNN but showed lower detection quality in this benchmark.

---

## ⚙️ Preprocessing

The preprocessing depends on the selected model.

| Model | Preprocessing |
|---|---|
| YOLOv5m | Handled internally by Ultralytics |
| Faster R-CNN | `torchvision.transforms.ToTensor()` |
| SSD300 | `torchvision.transforms.ToTensor()` |

The Faster R-CNN and SSD300 script also filters detections using:

```python
threshold = 0.5
```

Only detections with confidence score greater than or equal to `0.5` are displayed.

---

## 📊 Result Summary

For each image, bounding boxes were drawn along with predicted class labels and confidence scores.

Indicative results from the first three images:

| Image | YOLOv5m | Faster R-CNN | SSD300 |
|---|---:|---:|---:|
| Image0 | 3 objects / 133.2 ms | 3 objects / 1.053 sec | 3 objects / 0.146 sec |
| Image1 | 13 objects / 128.1 ms | Many detections / 1.039 sec | 4 objects / 0.147 sec |
| Image2 | 15 objects / 121.3 ms | 23 objects / 1.046 sec | 4 objects / 0.145 sec |

---

## 🔍 Qualitative Evaluation

### YOLOv5m

YOLOv5m was the most balanced model overall.

It provided:

- high inference speed,
- strong detection quality,
- correct labels for many object categories,
- high confidence scores,
- reliable performance across simple and complex scenes.

### Faster R-CNN

Faster R-CNN detected many objects, especially in complex scenes.

However:

- it required more processing time,
- it sometimes produced too many bounding boxes,
- some detections had wrong labels despite high confidence scores.

### SSD300

SSD300 was fast, but less accurate in this benchmark.

It showed:

- fewer detections,
- more missed objects,
- weaker labeling consistency,
- occasional `unknown` labels due to dataset/class limitations.

---

## ✅ Main Conclusions

Based on the benchmark:

- **YOLOv5m** achieved the best balance between speed and accuracy.
- **Faster R-CNN** was more exhaustive but slower and less consistent in labels.
- **SSD300** was fast but produced fewer and less reliable detections.
- Complex backgrounds, small objects, lighting, and object scale strongly affected performance.

YOLOv5m is the recommended model for this project because it combines strong object detection quality with efficient inference time.

---

## 📁 Repository Structure

```text
Object-Detection-Benchmark/
│
├── README.md
├── requirements.txt
│
├── YOLOv5.py
├── FasterR-CNN_&_SSD_Object_Detection.py
│
├── images/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
│
├── results/
│   ├── yolo/
│   └── faster_rcnn_vs_ssd/
│
└── docs/
    └── report.docx
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Place all input images inside the `images/` folder.

```text
images/
├── image1.jpg
├── image2.jpg
└── image3.jpg
```

Run YOLOv5 detection:

```bash
python YOLOv5.py
```

Run Faster R-CNN and SSD300 comparison:

```bash
python "FasterR-CNN_&_SSD_Object_Detection.py"
```

The scripts will:

1. load all images from the `images/` directory,
2. run object detection,
3. draw bounding boxes,
4. display class labels and confidence scores,
5. show inference time per model.

---

## 📦 Requirements

Core dependencies:

```text
torch
torchvision
ultralytics
opencv-python
pillow
matplotlib
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🧪 Expected Output

YOLOv5 displays each image with detections plotted directly by Ultralytics.

Faster R-CNN and SSD300 are displayed side by side for each image:

```text
Faster R-CNN | SSD300 (VOC)
```

Each visualization includes:

- bounding boxes,
- predicted class labels,
- confidence scores,
- inference time.

---

## 🔧 Possible Improvements

The accuracy and robustness of the benchmark could be improved with:

- fine-tuning on a custom dataset,
- using newer architectures such as YOLOv8, EfficientDet, or DETR,
- applying data augmentation,
- increasing input image quality,
- using ensemble methods,
- adjusting confidence thresholds per model.

---

## 👤 Author

**Κωνσταντίνος Κιούσης**  
Student ID: **2121129**

Course: **Advanced Topics in Computer Vision**

---

## 📄 License

This project is intended for academic use.
