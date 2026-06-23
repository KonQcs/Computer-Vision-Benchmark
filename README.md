# Computer Vision Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Benchmark-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Benchmarking image classification models and object detection outputs for a Computer Vision assignment.**

</div>

---

## 📌 Overview

This repository contains a Computer Vision benchmark project developed for the course:

**Advanced Topics in Computer Vision**

The project focuses on two main tasks:

1. **Image Classification** using the Fashion-MNIST dataset.
2. **Object Detection Result Visualization** using bounding boxes, predicted classes, and confidence scores on different real-world image scenarios.

The main goal is to compare model depth and evaluate how neural network complexity affects:

- accuracy,
- validation performance,
- test performance,
- training time,
- generalization,
- possible overfitting or underfitting.

---

## 🧠 Image Classification Task

The classification experiment uses the **Fashion-MNIST** dataset.

Fashion-MNIST contains grayscale images of clothing items and is commonly used as a benchmark dataset for image classification models.

### Dataset Details

| Property | Value |
|---|---:|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28 × 28 pixels |
| Number of classes | 10 |
| Input format | Flattened vector of 784 values |
| Normalization | Pixel values scaled to `[0, 1]` |

---

## 🏗️ Model Architectures

Two fully connected neural networks were implemented and compared.

### Shallow Neural Network

The shallow model contains three hidden Dense layers.

```python
Dense(100, activation="relu", input_shape=(784,))
Dense(50, activation="relu")
Dense(100, activation="relu")
Dense(10, activation="softmax")
```

### Deep Neural Network

The deep model contains five hidden Dense layers with gradually decreasing width.

```python
Dense(512, activation="relu", input_shape=(784,))
Dense(256, activation="relu")
Dense(128, activation="relu")
Dense(64, activation="relu")
Dense(32, activation="relu")
Dense(10, activation="softmax")
```

The deeper architecture is designed to learn richer feature representations through multiple nonlinear transformations.

---

## ⚙️ Training Configuration

Both models were trained using the same configuration.

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Categorical Cross-Entropy |
| Validation split | 0.2 |
| Batch size | 128 |
| Epochs | 50 |
| Hidden activation | ReLU |
| Output activation | Softmax |

---

## 📊 Results

| Metric | Shallow Model | Deep Model |
|---|---:|---:|
| Train Accuracy | 0.9625 | 0.9716 |
| Validation Accuracy | 0.8923 | 0.8973 |
| Test Accuracy | 0.8846 | 0.8961 |
| Validation Loss | 0.4713 | 0.4887 |
| Training Time | 30.66 sec | 64.47 sec |

### Result Analysis

The deep model achieved better test accuracy:

```text
Deep Model Test Accuracy:    0.8961
Shallow Model Test Accuracy: 0.8846
```

However, this improvement came with a significantly higher computational cost.

The deep model required approximately twice the training time compared to the shallow model. Also, although it achieved higher accuracy, its validation loss was slightly worse, which may indicate mild overfitting.

### Key Observations

- The **deep model** achieved the best final test accuracy.
- The **shallow model** trained faster and remained competitive.
- The accuracy improvement of the deep model was relatively small compared to the extra training time.
- For simpler datasets such as Fashion-MNIST, a shallow network can be a strong and efficient baseline.
- Deeper networks are more useful when the dataset or visual patterns are more complex.

---

## 🖼️ Object Detection Examples

The repository also includes example visualizations of object detection results on real-world style images.

The detections include:

- bounding boxes,
- predicted class labels,
- confidence scores.

Example scenes include:

- coastal reflection scene,
- home office scene,
- family dinner table,
- motorbike riders,
- luxury vehicles,
- city bus in motion,
- modern workspace,
- city skyline,
- supercar and driver,
- air show aircraft,
- western cowboy scene.

Suggested folder structure for detection result images:

```text
results/
└── detection_examples/
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

## 📁 Repository Structure

```text
Computer-Vision-Benchmark/
│
├── README.md
├── requirements.txt
│
├── shallow_model.py
├── deep_model.py
│
├── results/
│   └── detection_examples/
│
└── docs/
    └── report.pdf
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/KonQcs/Computer-Vision-Benchmark.git
cd Computer-Vision-Benchmark
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

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the required packages manually:

```bash
pip install tensorflow matplotlib
```

---

## ▶️ Usage

Run the shallow model:

```bash
python shallow_model.py
```

Run the deep model:

```bash
python deep_model.py
```

Each script performs the following steps:

1. Loads the Fashion-MNIST dataset.
2. Normalizes the image data.
3. Converts labels to categorical format.
4. Builds the neural network model.
5. Trains the model for 50 epochs.
6. Evaluates the model on the test set.
7. Displays accuracy and loss plots.

---

## 📦 Requirements

The project requires:

```text
tensorflow
matplotlib
```

Recommended Python version:

```text
Python 3.9+
```

---

## 📈 Expected Output

After training, each script prints the final test accuracy and the total training time.

Example output:

```text
Deep Model Test Accuracy: 0.8961
Training Time: 64.47 seconds
```

The scripts also generate training curves for:

- training accuracy,
- validation accuracy,
- training loss,
- validation loss.

---

## ✅ Conclusion

This benchmark shows that increasing neural network depth can improve classification accuracy, but the improvement is not always proportional to the additional computational cost.

For Fashion-MNIST, the deep model achieved slightly better accuracy, while the shallow model provided faster training and competitive performance.

Therefore:

- use a **shallow model** when speed, simplicity, and low computational cost are important;
- use a **deep model** when maximum accuracy is more important and the available computational resources are sufficient.

---

## 👤 Author

**Κωνσταντίνος Κιούσης**  
Student ID: **2121129**

Course: **Advanced Topics in Computer Vision**

---

## 📄 License

This project is intended for academic use.
