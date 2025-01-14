import sys
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Example run
# python check_model.py 563

# Define the paths
save_path = r'C:/mgr/data/segformer-teeth_segment_10ep_b4'
masks_path = r"C:/mgr/data/MASKS"
valid_images_path = r"C:/mgr/data/VALID_IMAGES"

# Load the model and feature extractor
model = SegformerForSemanticSegmentation.from_pretrained(save_path)
feature_extractor = SegformerImageProcessor.from_pretrained(save_path)

# Ensure the model is in evaluation mode
model.eval()

# Get the image number from command-line argument
if len(sys.argv) < 2:
    print("Usage: python check_model.py <image_number>")
    sys.exit(1)

image_number = sys.argv[1]  # Get the input image number
image_path = f"{valid_images_path}/{image_number}.jpg"  # Construct full path

# Prepare an Image for Inference
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Load image and convert to RGB
    inputs = feature_extractor(images=image, return_tensors="pt")  # Process the image
    return inputs

# Check if the file exists
try:
    inputs = preprocess_image(image_path)
except FileNotFoundError:
    print(f"Error: Image {image_number}.jpg not found in {valid_images_path}")
    sys.exit(1)

# Run Inference
with torch.no_grad():  # Disable gradient calculations for inference
    outputs = model(**inputs)
    logits = outputs.logits  # Get raw model outputs
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=inputs["pixel_values"].shape[-2:],  # Resize to match input image size
        mode="bilinear",
        align_corners=False
    )

    # Ensure predicted labels are a valid numpy array
    predicted_labels = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    # Convert to uint8 for OpenCV compatibility
    predicted_labels = predicted_labels.astype(np.uint8)

    # Ensure the mask has a valid shape
    if predicted_labels.size == 0:
        raise ValueError("Segmentation mask is empty. Check model outputs.")

# Function to compute Dice Score
def dice_score(predicted_mask, ground_truth_mask):
    predicted_mask = (predicted_mask > 0).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    total_pixels = predicted_mask.sum() + ground_truth_mask.sum()
    
    if total_pixels == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / total_pixels

# Visualize the Results
def display_segmentation(image_path, mask, masks_path):
    """
    Display:
    - Original Image
    - Ground-Truth Mask
    - Predicted Segmentation Mask Overlay
    - Compute Dice Score
    """
    
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB

    # Get the corresponding mask filename
    mask_filename = image_path.split("/")[-1]  # Extract filename
    ground_truth_mask_path = f"{masks_path}/{mask_filename}"  # Path to ground-truth mask

    # Load ground-truth mask
    ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure ground-truth mask is valid
    if ground_truth_mask is None:
        print(f"Warning: Ground-truth mask not found at {ground_truth_mask_path}")
        return

    # Ensure predicted mask is in correct format
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_colored = np.zeros_like(image)  # Create an empty RGB image
    mask_colored[mask == 1] = [255, 0, 0]  # Color teeth regions in Red

    # Compute Dice Score
    dice = dice_score(mask, ground_truth_mask)

    # Overlay predicted mask on the original image
    overlay = cv2.addWeighted(image, 0.6, mask_colored, 0.4, 0)

    # Display the images
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Ground-Truth Mask
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask, cmap="gray")
    plt.title("Ground-Truth Mask")
    plt.axis("off")

    # Predicted Segmentation Mask
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Predicted Mask Overlay\nDice Score: {dice:.4f}")
    plt.axis("off")

    plt.show()

# Visualize segmentation
display_segmentation(image_path, predicted_labels, masks_path)
