# 🧠 Brain Tumor Detection using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning project for binary classification of brain MRI images to detect the presence of tumors. Three different architectures (Custom CNN, VGG-inspired, and Lightweight ResNet) are implemented and compared.

## 📁 Dataset Structure

```
./Tumor/
├── Brain Tumor/     # MRI images with tumors (class 1)
└── Healthy/         # Healthy MRI images (class 0)
```

**Data Processing:**
- Original image size: 512×512 pixels
- Preprocessing: Converted to grayscale and resized to 128×128
- Dataset loaded using `os.walk()` and processed with OpenCV
- Labels: 1 for tumor, 0 for healthy (tumor images loaded first)

## 🏗️ Model Architectures

### 1️⃣ Custom CNN (Baseline Model)
A lightweight convolutional network designed for small datasets:

```
Input (128,128,1)
↓
Conv2D(8) → BatchNorm → MaxPooling
↓
Conv2D(16) → BatchNorm → MaxPooling
↓
Conv2D(24) → BatchNorm → MaxPooling
↓
Flatten → Dense(64) → Dropout
↓
Output (1, Sigmoid)
```

**Configuration:**
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy
- Epochs: 15
- Batch Size: 32

**Performance:** ✅ **98.97% Accuracy**

---

### 2️⃣ VGG-inspired Architecture
Deeper architecture with multiple consecutive convolutions per block:

```
Input (128,128,1)
↓
[Conv2D(32) → Conv2D(32)] → BatchNorm → MaxPooling
↓
[Conv2D(64) → Conv2D(64)] → BatchNorm → MaxPooling
↓
[Conv2D(128) → Conv2D(128)] → BatchNorm → MaxPooling
↓
Flatten → Dense(64) → Dropout
↓
Output (1, Sigmoid)
```

**Features:**
- 3×3 kernels with 'same' padding
- Batch Normalization after each convolution
- Progressive filter increase
- 15 epochs, batch size 32

**Performance:** ✅ **97.37% Accuracy**

---

### 3️⃣ Lightweight ResNet
Residual architecture with skip connections to prevent gradient vanishing:

```
Input (128,128,1)
↓
Conv2D(64) → BatchNorm → MaxPooling
    ↓
    ┌── Residual Block (64 filters) ──┐
    ↓                                 ↓
Conv2D(32) → BatchNorm → MaxPooling ← Skip Connection
    ↓
    ┌── Residual Block (32 filters) ──┐
    ↓                                 ↓
Conv2D(16) → BatchNorm → MaxPooling ← Skip Connection
    ↓
    ┌── Residual Block (16 filters) ──┐
    ↓                                 ↓
Global Average Pooling → Dense(128) → Dropout
↓
Output (1, Sigmoid)
```

**Features:**
- Skip connections for better gradient flow
- Global Average Pooling instead of Flatten
- Progressive filter reduction (64 → 32 → 16)
- 15 epochs, batch size 32

**Performance:** 🏆 **99.29% Accuracy** (Best Model)

## 📊 Results Summary

| Architecture | Accuracy | Key Feature |
|-------------|----------|-------------|
| Custom CNN | 98.97% | Simple, efficient, low overfitting |
| VGG-inspired | 97.37% | Deeper, multiple conv layers |
| Lightweight ResNet | **99.29%** | Skip connections, best performance |

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --model resnet  # or cnn/vgg
```

## 📦 Requirements

- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- scikit-learn
- Matplotlib

## 📝 License

This project is licensed under the MIT License.

---

**⭐ If you find this project useful, please consider giving it a star!**
