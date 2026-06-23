# Computer Vision Benchmarks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-TorchVision-red)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv5-purple)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Two coursework projects for image classification and object detection in Advanced Topics in Computer Vision.**

</div>

## Overview

This repository contains two independent computer vision assignments:

| Project | Folder | Description |
| --- | --- | --- |
| Fashion-MNIST Classification | [`classification/`](classification/) | Compares shallow and deep dense neural networks on Fashion-MNIST. |
| Object Detection Benchmark | [`object_detection/`](object_detection/) | Compares YOLOv5m, Faster R-CNN, and SSD300 on 11 test images. |

Each project includes its own source code, requirements file, README, and original report.

## Preview

Classification training curves:

![Deep network training curves](classification/docs/plots/deep_network_metrics.png)

Faster R-CNN vs SSD300 comparison:

![Faster R-CNN and SSD300 street comparison](object_detection/docs/model_comparisons/faster_rcnn_ssd_street.png)

YOLOv5m detection output:

![YOLO detection example](object_detection/docs/yolo_detections/plot_2025-06-25%2017-28-01_0.png)

More saved detections are available in [`object_detection/docs/yolo_detections/`](object_detection/docs/yolo_detections/).

## Repository Structure

```text
ComputerVisionBenchmarks/
|-- README.md
|-- .gitignore
|-- classification/
|   |-- README.md
|   |-- requirements.txt
|   |-- shallow_model.py
|   |-- deep_model.py
|   `-- docs/
|       |-- plots/
|       `-- classification_report.pdf
`-- object_detection/
    |-- README.md
    |-- requirements.txt
    |-- yolo_detect.py
    |-- faster_rcnn.py
    |-- ssd300.py
    |-- faster_rcnn_ssd_compare.py
    |-- images/
    |-- models/
    `-- docs/
        |-- model_comparisons/
        |-- yolo_detections/
        `-- object_detection_report.docx
```

## Quick Start

Use a separate virtual environment for each project because the TensorFlow and PyTorch dependency stacks are large.

### Classification

```bash
cd classification
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python shallow_model.py --no-plots
```

On macOS or Linux:

```bash
cd classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python shallow_model.py --no-plots
```

### Object Detection

```bash
cd object_detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python yolo_detect.py --no-show
```

On macOS or Linux:

```bash
cd object_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python yolo_detect.py --no-show
```

## Results Summary

### Classification

| Metric | Shallow Model | Deep Model |
| --- | ---: | ---: |
| Test accuracy | 0.8846 | 0.8961 |
| Validation accuracy | 0.8923 | 0.8973 |
| Training time | 30.66 sec | 64.47 sec |

The deep model produced the best test accuracy, while the shallow model trained faster and stayed competitive.

### Object Detection

| Model | Main observation |
| --- | --- |
| YOLOv5m | Best balance of speed and detection quality. |
| Faster R-CNN | More exhaustive detections, but slower. |
| SSD300 | Fast, but missed more objects and produced weaker labels. |

## Reports

- [`classification/docs/classification_report.pdf`](classification/docs/classification_report.pdf)
- [`object_detection/docs/object_detection_report.docx`](object_detection/docs/object_detection_report.docx)

## GitHub Notes

- Sample images and saved PNG result plots are included.
- Large model/checkpoint files such as `*.pt`, `*.pth`, and `*.onnx` are ignored.
- Generated outputs such as `results/`, `runs/`, and `output/` are ignored.
- The local YOLO model file can be kept under `object_detection/models/`, but it should not be committed.

## Author

Konstantinos Kiousis  
Student ID: 2121129
