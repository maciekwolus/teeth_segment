import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
import numpy as np

# Define U-Net Architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(1, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = up_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)

        self.up3 = up_block(512, 256)
        self.decoder3 = conv_block(512, 256)

        self.up2 = up_block(256, 128)
        self.decoder2 = conv_block(256, 128)

        self.up1 = up_block(128, 64)
        self.decoder1 = conv_block(128, 64)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output_layer(d1)


# Dice Loss Implementation
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to normalize predictions
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


# Dataset Class
class XRayDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_names = sorted(os.listdir(images_dir))
        self.mask_names = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_names[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])

        image = read_image(image_path).float() / 255.0  # Normalize image to [0, 1]
        mask = read_image(mask_path).float() / 255.0   # Normalize mask to [0, 1]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Training Parameters
train_images_dir = "C:/mgr/data/TRAIN_IMAGES"
train_masks_dir = "C:/mgr/data/MASKS"
valid_images_dir = "C:/mgr/data/VALID_IMAGES"
valid_masks_dir = "C:/mgr/data/MASKS"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
])

# Datasets and Dataloaders
train_dataset = XRayDataset(train_images_dir, train_masks_dir, transform=transform)
valid_dataset = XRayDataset(valid_images_dir, valid_masks_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

if __name__ == "__main__":
    # Model, Loss, and Optimizer
    model = UNet().to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    # Combined Loss Function
    def combined_loss(outputs, targets):
        bce = bce_loss(outputs, targets)
        dice = dice_loss(outputs, targets)
        return bce + dice

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss/len(train_loader):.4f}")

    # Save Model
    model_save_path = "C:/mgr/data/unet_teeth_segmentation.pth"
    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)
