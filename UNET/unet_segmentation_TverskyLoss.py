import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# U-Net with BatchNorm, Dropout, and flexible upsampling
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024 // factor))
        if bilinear:
            self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv4 = DoubleConv(1024, 512 // factor)
        else:
            self.up4 = nn.ConvTranspose2d(1024, 512 // factor, kernel_size=2, stride=2)
            self.conv4 = DoubleConv(1024, 512 // factor)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv4(x)
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)
        return self.outc(x)


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'}, is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'}, is_check_shapes=False)


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = read_image(img_path).float().numpy().transpose(1, 2, 0)
        mask = read_image(mask_path).float().numpy().transpose(1, 2, 0)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(-1)
        mask = mask / 255.0
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image, mask


# Tversky Loss implementation
class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation.
    Uogólnienie Dice'a z wagami dla falszywych pozytywów i negatywów.
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        prob_flat = prob.view(prob.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        TP = (prob_flat * target_flat).sum(dim=1)
        FP = (prob_flat * (1 - target_flat)).sum(dim=1)
        FN = ((1 - prob_flat) * target_flat).sum(dim=1)
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return 1.0 - tversky.mean()


def dice_coeff(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    return (2 * intersection + eps) / (union + eps)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_dice = 0.0, 0.0
    for imgs, masks in tqdm(loader, desc='Training'):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_dice += dice_coeff(outputs, masks).item()
    return running_loss / len(loader), running_dice / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, running_dice = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Validation'):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            running_dice += dice_coeff(outputs, masks).item()
    return running_loss / len(loader), running_dice / len(loader)


if __name__ == "__main__":
    # Directories
    train_images_dir = r"C:/mgr/data/TRAIN_IMAGES"
    train_masks_dir  = r"C:/mgr/data/MASKS"
    val_images_dir   = r"C:/mgr/data/VALID_IMAGES"
    val_masks_dir    = r"C:/mgr/data/MASKS"

    # Output directory for saving best model
    save_dir = r"C:/mgr/data/UNET"
    os.makedirs(save_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Criterion: Tversky Loss
    criterion = TverskyLoss(alpha=0.3, beta=0.7)

    # Datasets & Loaders
    train_ds = SegmentationDataset(train_images_dir, train_masks_dir, transform=get_transforms(train=True))
    val_ds   = SegmentationDataset(val_images_dir, val_masks_dir, transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Model, Optimizer, Scheduler
    model     = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    n_epochs = 75
    best_val_loss = float('inf')
    start_time = time.time()
    for epoch in range(1, n_epochs+1):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_dice   = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Dice {train_dice:.4f}; Val Loss {val_loss:.4f}, Val Dice {val_dice:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "unet_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
