import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from datetime import datetime  # For timestamp

# Generate filename with current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"metrics_SEGFORMER_{timestamp}.xlsx"

# Paths
valid_images_path = r"C:/mgr/data/VALID_IMAGES"
masks_path = r"C:/mgr/data/MASKS"
output_excel_path = os.path.join(r"C:/mgr/data/METRICS_OUTPUT", file_name)
save_path = r"C:/mgr/data/segformer-teeth_segment_10ep_b4_GPU"  # replace with your SegFormer checkpoint path

# Ensure output directory exists
os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

# Load model and feature extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegformerForSemanticSegmentation.from_pretrained(save_path).to(device)
feature_extractor = SegformerImageProcessor.from_pretrained(save_path)
model.eval()

# Metric functions
def dice_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def precision_score(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    pp = pred.sum()
    if pp == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return tp / pp


def recall_score(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    ap = gt.sum()
    if ap == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return tp / ap


def f1_score(pred, gt):
    p = precision_score(pred, gt)
    r = recall_score(pred, gt)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def f2_score(pred, gt):
    p = precision_score(pred, gt)
    r = recall_score(pred, gt)
    denom = 4 * p + r
    if denom == 0:
        return 0.0
    return 5 * (p * r) / denom


def jaccard_score(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def accuracy_score(pred, gt):
    matches = (pred == gt).sum()
    total = pred.size
    return matches / total

# Process images and collect metrics
results = []
image_files = [f for f in os.listdir(valid_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for image_file in image_files:
    img_path = os.path.join(valid_images_path, image_file)
    mask_path = os.path.join(masks_path, image_file)
    if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
        print(f"Skipping {image_file}: corresponding mask not found.")
        continue
    # Load and preprocess input
    image = Image.open(img_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    # Inference with timing
    start = time.time()
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        pred_labels = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    inference_time = (time.time() - start) * 1000  # ms

    # Binarize and resize masks
    pred_bin = (pred_labels > 0).astype(np.uint8)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_resized = cv2.resize(pred_bin, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Compute metrics
    acc = accuracy_score(pred_resized, gt_bin)
    prec = precision_score(pred_resized, gt_bin)
    rec = recall_score(pred_resized, gt_bin)
    f1 = f1_score(pred_resized, gt_bin)
    f2 = f2_score(pred_resized, gt_bin)
    dice = dice_score(pred_resized, gt_bin)
    jac = jaccard_score(pred_resized, gt_bin)

    results.append({
        "Image": image_file,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "F2 Score": f2,
        "Dice Score": dice,
        "Jaccard Index": jac,
        "Inference Time (ms)": inference_time
    })

# Save to Excel
df = pd.DataFrame(results, columns=[
    "Image", "Accuracy", "Precision", "Recall",
    "F1 Score", "F2 Score", "Dice Score",
    "Jaccard Index", "Inference Time (ms)"
])
# Compute averages
avg_vals = {col: df[col].mean() for col in df.columns if col != "Image"}
avg_row = {"Image": "Average", **avg_vals}
df.loc[-1] = avg_row
df.index = df.index + 1
df.sort_index(inplace=True)
df.to_excel(output_excel_path, index=False)
print(f"Results saved to {output_excel_path}")
