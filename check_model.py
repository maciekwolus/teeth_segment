from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the path where the model was saved
save_path = r'C:/mgr/data/segformer-teeth_segment_10ep_b4'
# Define the mask storage path
masks_path = r"C:/mgr/data/MASKS"

# Load the model and feature extractor
model = SegformerForSemanticSegmentation.from_pretrained(save_path)
feature_extractor = SegformerImageProcessor.from_pretrained(save_path)

# Ensure the model is in evaluation mode
model.eval()

# Prepare an Image for Inference
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Load image and convert to RGB
    inputs = feature_extractor(images=image, return_tensors="pt")  # Process the image
    return inputs

# Example image
image_path = r"C:/mgr/data/VALID_IMAGES/6.jpg"  # Change this to an actual test image path
inputs = preprocess_image(image_path)

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

#  Dice Score = 2×∣A∩B∣​/(∣A∣+∣B∣)
"""
1.0 = Perfect segmentation
0.9+ - Excellent segmentation
0.8 - 0.9 = Good segmentation
0.6 - 0.8 = Acceptable, but could improve
< 0.6 = Poor segmentation, model needs improvement
"""
def dice_score(predicted_mask, ground_truth_mask):
    """
    Compute Dice Score between predicted and ground-truth masks.
    
    Args:
        predicted_mask (numpy.ndarray): Model's predicted segmentation mask.
        ground_truth_mask (numpy.ndarray): Ground-truth mask from dataset.
        
    Returns:
        float: Dice Score (between 0 and 1)
    """
    # Ensure masks are binary (0 or 1)
    predicted_mask = (predicted_mask > 0).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    # Calculate intersection and sum
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    total_pixels = predicted_mask.sum() + ground_truth_mask.sum()

    # Avoid division by zero
    if total_pixels == 0:
        return 1.0 if intersection == 0 else 0.0

    # Compute Dice Score
    dice = (2.0 * intersection) / total_pixels
    return dice

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
    mask_filename = image_path.split("/")[-1]  # Extract filename (assumes same name structure)
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
