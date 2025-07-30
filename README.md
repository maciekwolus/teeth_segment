# 🦷 Tooth Segmentation on Dental X-Rays Using Deep Learning

This project compares two semantic segmentation architectures — the classical **U-Net** and the transformer-based **SegFormer (MiT-B4)** — for the task of detecting and segmenting teeth in panoramic dental X-ray images. It was developed as part of my master’s thesis in applied computer science and explores deep learning's potential in assisting digital dentistry diagnostics.

---

## 🧠 Project Overview

Accurate segmentation of dental X-rays is a crucial step in building intelligent dental support systems. This project evaluates and contrasts two neural network models:

- **U-Net** — A convolutional encoder-decoder architecture widely used in medical imaging.
- **SegFormer (MiT-B4)** — A modern transformer-based model capable of capturing global context and delivering sharp boundary segmentations.

---

## 📊 Research Goals

- Compare segmentation quality between U-Net and SegFormer on the same dataset.
- Use identical preprocessing, augmentation, and evaluation metrics for a fair comparison.
- Analyze inference quality and potential clinical applicability.

---

## 📁 Dataset

- **Source**: [Kaggle - Teeth Segmentation on Dental X-ray Images](https://www.kaggle.com/datasets)
- **Content**: Grayscale panoramic dental X-ray images with manually annotated masks.

---

## 🔬 Models & Techniques

### ✅ U-Net Model
- Custom PyTorch implementation
- Trained with **Tversky Loss** (α=0.3, β=0.7) to handle class imbalance
- Dice coefficient for evaluation
- Image size reduced to `256x256` due to GPU memory constraints

### 🔷 SegFormer Model
- Pretrained **nvidia/mit-b4** encoder from Hugging Face Transformers
- Fine-tuned for binary segmentation (`teeth` vs `background`)
- Uses **Cross-Entropy Loss** with Dice score metric
- Benefits from global attention and transformer-based feature extraction

---

## 📦 Tech Stack

- Python 3.10
- PyTorch
- Hugging Face Transformers
- Albumentations
- OpenCV, NumPy
- TQDM, Matplotlib (for training visuals)

---

## ⚙️ Training Details

| Parameter               | U-Net                   | SegFormer               |
|------------------------|-------------------------|-------------------------|
| Input Size             | 256×256                 | 256×256                 |
| Loss Function          | Tversky Loss            | Cross-Entropy Loss      |
| Optimizer              | Adam                    | Adam                    |
| Metrics                | Dice Score              | Dice Score              |
| Epochs                 | 50                      | 10                      |
| Batch Size             | 8                       | 4                       |
| Scheduler              | ReduceLROnPlateau       | -                       |

---

## 📈 Results Summary

- Both models performed comparably on standard metrics.
- **SegFormer** delivered sharper contours and better preserved tooth shapes, making it a promising candidate for clinical use.
- **U-Net** was faster to train but less effective at capturing fine details.

---

## 🚀 How to Run

### U-Net Training
```bash
python unet_segmentation_TverskyLoss.py
