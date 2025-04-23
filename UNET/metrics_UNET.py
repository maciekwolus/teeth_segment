import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from datetime import datetime  # For timestamp

# Variables to change
from unet_segmentation_final import UNet  # Ensure this import is from correct file
save_path = "C:/mgr/data/UNET/unet_best.pth" # Name of saved model must be correct

# Generate filename with current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"metrics_UNET_{timestamp}.xlsx"

# Paths
masks_path = "C:/mgr/data/MASKS"
valid_images_path = "C:/mgr/data/VALID_IMAGES"
output_excel_path = os.path.join("C:/mgr/data/METRICS_OUTPUT", file_name)

# Load the model
model = UNet()
model.load_state_dict(torch.load(save_path))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("L")
    return transform(image).unsqueeze(0)

# Metrics functions
def dice_score(predicted_mask, ground_truth_mask):
    pred = (predicted_mask > 0).astype(np.uint8)
    gt = (ground_truth_mask > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if inter == 0 else 0.0
    return 2.0 * inter / total

def precision_score(predicted_mask, ground_truth_mask):
    pred = (predicted_mask > 0).astype(np.uint8)
    gt = (ground_truth_mask > 0).astype(np.uint8)
    tp = np.logical_and(pred, gt).sum()
    pp = pred.sum()
    if pp == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return tp / pp

def recall_score(predicted_mask, ground_truth_mask):
    pred = (predicted_mask > 0).astype(np.uint8)
    gt = (ground_truth_mask > 0).astype(np.uint8)
    tp = np.logical_and(pred, gt).sum()
    ap = gt.sum()
    if ap == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return tp / ap

def f1_score(predicted_mask, ground_truth_mask):
    p = precision_score(predicted_mask, ground_truth_mask)
    r = recall_score(predicted_mask, ground_truth_mask)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def f2_score(predicted_mask, ground_truth_mask):
    p = precision_score(predicted_mask, ground_truth_mask)
    r = recall_score(predicted_mask, ground_truth_mask)
    denom = 4 * p + r
    if denom == 0:
        return 0.0
    return 5 * (p * r) / denom

def jaccard_score(predicted_mask, ground_truth_mask):
    pred = (predicted_mask > 0).astype(np.uint8)
    gt = (ground_truth_mask > 0).astype(np.uint8)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, 1 - gt).sum()
    fn = np.logical_and(1 - pred, gt).sum()
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / denom

def accuracy_score(predicted_mask, ground_truth_mask):
    pred = (predicted_mask > 0).astype(np.uint8)
    gt = (ground_truth_mask > 0).astype(np.uint8)
    match = (pred == gt).sum()
    total = pred.size
    return match / total

# Directory checks
for path in [valid_images_path, masks_path]:
    if not os.path.exists(path):
        print(f"Error: Directory {path} does not exist.")
        exit(1)

# Ensure output directory exists
out_dir = os.path.dirname(output_excel_path)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Process images and collect metrics
results = []
for img_file in os.listdir(valid_images_path):
    img_path = os.path.join(valid_images_path, img_file)
    mask_path = os.path.join(masks_path, img_file)
    if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
        print(f"Skipping {img_file}: mask not found.")
        continue
    try:
        tensor = preprocess_image(img_path).to(device)
        start = time.time()
        with torch.no_grad():
            output = model(tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        infer_time = (time.time() - start) * 1000  # ms
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred_resized = cv2.resize((pred_mask > 0.5).astype(np.uint8),
                                  (gt_mask.shape[1], gt_mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        # Compute metrics
        acc = accuracy_score(pred_resized, gt_mask)
        prec = precision_score(pred_resized, gt_mask)
        rec = recall_score(pred_resized, gt_mask)
        f1 = f1_score(pred_resized, gt_mask)
        f2 = f2_score(pred_resized, gt_mask)
        dice = dice_score(pred_resized, gt_mask)
        jac = jaccard_score(pred_resized, gt_mask)
        results.append({
            "Image": img_file,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "F2 Score": f2,
            "Dice Score": dice,
            "Jaccard Index": jac,
            "Inference Time (ms)": infer_time
        })
    except Exception as e:
        print(f"Error processing {img_file}: {e}")

# Save to Excel
if results:
    cols = ["Image", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score",
            "Dice Score", "Jaccard Index", "Inference Time (ms)"]
    df = pd.DataFrame(results, columns=cols)
    # Compute averages
    avg_vals = {col: df[col].mean() for col in cols if col != "Image"}
    avg_row = {**{"Image": "Average"}, **avg_vals}
    df.loc[-1] = avg_row
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")
else:
    print("No results to save.")
