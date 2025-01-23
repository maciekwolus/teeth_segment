import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from unet_segmentation_5_COMBO_2_3 import UNet # has to be proper

# Paths
save_path = "C:/mgr/data/unet_teeth_segmentation.pth"
masks_path = "C:/mgr/data/MASKS"
valid_images_path = "C:/mgr/data/VALID_IMAGES"
output_excel_path = "C:/mgr/data/DICE_SCORE/dice_score_UNET.xlsx"

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
    image = Image.open(image_path).convert("L")  # Ensure grayscale
    return transform(image).unsqueeze(0)

# Function to compute Dice Score
def dice_score(predicted_mask, ground_truth_mask):
    predicted_mask = predicted_mask.astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    total_pixels = predicted_mask.sum() + ground_truth_mask.sum()
    
    if total_pixels == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / total_pixels

# Directory check
if not os.path.exists(valid_images_path):
    print(f"Error: Directory {valid_images_path} does not exist.")
    exit(1)

if not os.path.exists(masks_path):
    print(f"Error: Directory {masks_path} does not exist.")
    exit(1)

if not os.path.exists(os.path.dirname(output_excel_path)):
    os.makedirs(os.path.dirname(output_excel_path))

# Process each image
results = []
for image_file in os.listdir(valid_images_path):
    image_path = os.path.join(valid_images_path, image_file)
    mask_path = os.path.join(masks_path, image_file)

    if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
        print(f"Skipping {image_file}: corresponding mask not found.")
        continue

    try:
        image_tensor = preprocess_image(image_path).to(device)

        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy() > 0.5  # Threshold at 0.5

        # Load ground-truth mask
        ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8),
                                            (ground_truth_mask.shape[1], ground_truth_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)

        # Compute Dice score
        dice = dice_score(predicted_mask_resized, ground_truth_mask)
        results.append({"Image": image_file, "Dice Score": dice})
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# Save to Excel
if results:
    df = pd.DataFrame(results)
    avg_dice = df["Dice Score"].mean()
    df.loc[-1] = {"Image": "Average", "Dice Score": avg_dice}  # Add average row at the top
    df.index = df.index + 1  # Shift index
    df.sort_index(inplace=True)
    df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")
else:
    print("No results to save.")
