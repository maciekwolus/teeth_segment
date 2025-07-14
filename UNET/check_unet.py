import sys
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
from unet_segmentation_TverskyLoss import UNet # has to be proper

# Example run:
# python check_unet.py 563


# Define paths
save_path = "C:/mgr/data/UNET/unet_best_1.pth"
masks_path = "C:/mgr/data/MASKS"
valid_images_path = "C:/mgr/data/VALID_IMAGES"

# Load the model
model = UNet()
model.load_state_dict(torch.load(save_path, weights_only=True))

#model.load_state_dict(torch.load(save_path, weights_only=True))
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

# Get the image number from command-line argument
if len(sys.argv) < 2:
    print("Usage: python check_unet_segmentation.py <image_number>")
    sys.exit(1)

image_number = sys.argv[1]  # Get the input image number
image_path = f"{valid_images_path}/{image_number}.jpg"  # Construct full path

# Check if the file exists
try:
    image_tensor = preprocess_image(image_path).to(device)
except FileNotFoundError:
    print(f"Error: Image {image_number}.jpg not found in {valid_images_path}")
    sys.exit(1)

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy() > 0.5  # Threshold at 0.5

# Function to compute Dice Score
def dice_score(predicted_mask, ground_truth_mask):
    predicted_mask = predicted_mask.astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    total_pixels = predicted_mask.sum() + ground_truth_mask.sum()
    
    if total_pixels == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / total_pixels

# Visualize the Results
def display_segmentation(image_path, predicted_mask, masks_path):
    """
    Display:
    - Original Image
    - Ground-Truth Mask
    - Predicted Segmentation Mask Overlay
    - Compute Dice Score
    """

    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB

    # Get the corresponding mask filename
    mask_filename = image_path.split("/")[-1]  # Extract filename
    ground_truth_mask_path = f"{masks_path}/{mask_filename}"  # Path to ground-truth mask

    # Load ground-truth mask
    ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
    if ground_truth_mask is None:
        print(f"Warning: Ground-truth mask not found or could not be read at {ground_truth_mask_path}")
        return

    # Resize predicted mask to match original image size
    predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8),
                                        (image.shape[1], image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

    # Resize ground-truth mask to match predicted mask size
    ground_truth_mask_resized = cv2.resize(ground_truth_mask,
                                           (predicted_mask_resized.shape[1], predicted_mask_resized.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

    # Compute Dice Score
    dice = dice_score(predicted_mask_resized, ground_truth_mask_resized)

    # Overlay predicted mask on the original image
    mask_colored = np.zeros_like(image)
    mask_colored[predicted_mask_resized == 1] = [255, 0, 0]  # Color in red
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
    plt.title(f"UNet predicted Mask Overlay\nDice Score: {dice:.4f}")
    plt.axis("off")

    plt.show()


# Visualize segmentation
display_segmentation(image_path, predicted_mask, masks_path)
