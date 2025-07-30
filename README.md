# 🦷 Tooth Segmentation on Dental X-Rays Using Deep Learning

This project compares two semantic segmentation architectures — the classical **U-Net** and the transformer-based **SegFormer (MiT-B4)** — for the task of detecting and segmenting teeth in panoramic dental X-ray images. It was developed as part of my master’s thesis in applied computer science and explores deep learning's potential in assisting digital dentistry diagnostics.

Important part - images were resized because my hardware could not handle training at the original resolution.

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

## 📦 Tech Stack

- Python 3.10
- PyTorch
- Hugging Face Transformers
- Albumentations
- OpenCV, NumPy
- TQDM, Matplotlib (for training visuals)

---

## 📈 Results Summary

- Both models performed comparably on standard metrics.
- **SegFormer** delivered sharper contours and better preserved tooth shapes, making it a promising candidate for clinical use.
- **U-Net** was faster to train but less effective at capturing fine details.

### 📊 Quantitative Results

The table below summarizes the performance of both models — **U-Net** and **SegFormer (MiT-B4)** — evaluated on the validation dataset. All models were trained using the same preprocessing and augmentation pipeline to ensure a fair comparison.

| Model      | Accuracy | Precision | Recall  | Dice Score | F2 Score | Jaccard Index |
|------------|----------|-----------|---------|------------|----------|----------------|
| **U-Net**     | 0.9491   | 0.7513    | 0.9215  | 0.8261     | 0.8803   | 0.7069         |
| **SegFormer** | 0.9681   | 0.8909    | 0.8440  | 0.8657     | 0.8523   | 0.7711         |

### 🖼️ Sample Segmentation Results

Below is an example illustrating the performance of the trained model on a validation image.
<img width="990" height="1805" alt="example" src="https://github.com/user-attachments/assets/e722efe6-ad01-478d-ba2c-a055f26e5ad5" />
