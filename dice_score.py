import os
import cv2
import numpy as np
import torch
import pandas as pd
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image

# Define paths
valid_images_path = r"C:/mgr/data/VALID_IMAGES"
masks_path = r"C:/mgr/data/MASKS"
save_path = r"C:/mgr/data/segformer-teeth_segment_10ep_b4"
output_excel_path = r"C:/mgr/data/DICE_SCORE/dice_scores.xlsx"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

# Load model and feature extractor
model = SegformerForSemanticSegmentation.from_pretrained(save_path)
feature_extractor = SegformerImageProcessor.from_pretrained(save_path)
model.eval()

def dice_score(predicted_mask, ground_truth_mask):
    """
    Compute Dice Score between predicted and ground-truth masks.
    """
    predicted_mask = (predicted_mask > 0).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    total_pixels = predicted_mask.sum() + ground_truth_mask.sum()
    
    if total_pixels == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / total_pixels

def preprocess_image(image_path):
    """Load and preprocess image for model inference."""
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def get_predicted_mask(image_path):
    """Run inference and get predicted segmentation mask."""
    inputs = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=inputs["pixel_values"].shape[-2:], mode="bilinear", align_corners=False
        )
        predicted_labels = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    return predicted_labels

def evaluate_model():
    """Evaluate Dice Score for all images in the validation dataset and save results to an Excel file."""
    image_files = [f for f in os.listdir(valid_images_path) if f.endswith('.jpg')]
    total_dice = 0
    count = 0
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(valid_images_path, image_file)
        mask_path = os.path.join(masks_path, image_file)
        
        if not os.path.exists(mask_path):
            continue
        
        predicted_mask = get_predicted_mask(image_path)
        ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if ground_truth_mask is None:
            continue
        
        ground_truth_mask = cv2.resize(ground_truth_mask, (predicted_mask.shape[1], predicted_mask.shape[0]))
        dice = dice_score(predicted_mask, ground_truth_mask)
        total_dice += dice
        count += 1
        results.append((image_file, dice))
    
    avg_dice = total_dice / count if count > 0 else 0.0
    
    # Save to Excel
    df = pd.DataFrame(results, columns=["Image Name", "Dice Score"])
    df.loc[-1] = ["Average Dice Score", avg_dice]  # Add average dice score at the top
    df.index = df.index + 1  # Shift index
    df.sort_index(inplace=True)
    df.to_excel(output_excel_path, index=False)

# Run the evaluation
evaluate_model()
