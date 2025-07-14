import sys
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torchvision import transforms
from unet_segmentation_TverskyLoss import UNet

# Paths
SEGFORMER_MODEL_PATH = r"C:/mgr/data/SEGFORMER/3"
UNET_WEIGHTS_PATH    = r"C:/mgr/data/UNET/unet_best_1.pth"
MASKS_PATH           = r"C:/mgr/data/MASKS"
VALID_IMAGES_PATH    = r"C:/mgr/data/VALID_IMAGES"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Segformer model
seg_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_PATH).to(device)
feature_extractor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_PATH)
seg_model.eval()

# Load UNet model
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load(UNET_WEIGHTS_PATH, map_location=device))
unet_model.eval()

# UNet preprocessing pipeline
unet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def preprocess_segformer(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    return inputs

def predict_segformer(inputs):
    with torch.no_grad():
        outputs = seg_model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=inputs["pixel_values"].shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        mask = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return mask

def predict_unet(image_path):
    img = Image.open(image_path).convert("L")
    tensor = unet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = unet_model(tensor)
        mask = (torch.sigmoid(out).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return mask

def overlay_mask(image_rgb, mask):
    mask_col = np.zeros_like(image_rgb)
    mask_col[mask == 1] = [255, 0, 0]
    return cv2.addWeighted(image_rgb, 0.6, mask_col, 0.4, 0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merged_segmentation.py <image_number>")
        sys.exit(1)

    img_num = sys.argv[1]
    img_path = f"{VALID_IMAGES_PATH}/{img_num}.jpg"
    gt_path  = f"{MASKS_PATH}/{img_num}.jpg"

    # Load original and ground-truth
    orig = cv2.imread(img_path)
    if orig is None:
        print(f"Error: could not load image {img_path}")
        sys.exit(1)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        print(f"Error: could not load ground-truth mask {gt_path}")
        sys.exit(1)
    gt_bin = (gt > 0).astype(np.uint8)

    # Segformer prediction
    seg_inputs = preprocess_segformer(img_path)
    seg_mask  = predict_segformer(seg_inputs)

    # UNet prediction
    unet_mask = predict_unet(img_path)

    # Resize masks to original image size
    h, w = orig_rgb.shape[:2]
    seg_resized  = cv2.resize(seg_mask,  (w, h), interpolation=cv2.INTER_NEAREST)
    unet_resized = cv2.resize(unet_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    gt_resized   = cv2.resize(gt_bin,    (w, h), interpolation=cv2.INTER_NEAREST)

    # Create overlays
    overlay_gt    = overlay_mask(orig_rgb, gt_resized)
    overlay_unet  = overlay_mask(orig_rgb, unet_resized)
    overlay_segf  = overlay_mask(orig_rgb, seg_resized)

    # Display vertically: top=GT, middle=UNet, bottom=Segformer
    plt.figure(figsize=(8, 24))

    plt.subplot(3, 1, 1)
    plt.imshow(overlay_gt)
    plt.title("Zdjęcie oryginalne z nałożoną maską referenycjną")
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.imshow(overlay_unet)
    plt.title("Zdjęcie oryginalne z maską przewidzianą przez model U-Net")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.imshow(overlay_segf)
    plt.title("Zdjęcie oryginalne z maską przewidzianą przez model SegFormer")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
