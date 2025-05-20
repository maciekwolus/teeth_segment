import cv2
import os
import shutil
import numpy as np
import json
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        targets = targets.float()
        bce_loss = self.bce(inputs, targets)

        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1.) / (probs.sum() + targets.sum() + 1.)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def show_image(image_path):
    """
    Wyświetla obraz w oknie o stałej wysokości 300px.
    """
    image = cv2.imread(image_path)
    tgt_height = 300
    height, width = image.shape[:2]
    scale_factor = tgt_height / height
    tgt_width = int(width * scale_factor)
    resized_image = cv2.resize(image, (tgt_width, tgt_height))
    cv2.imshow("Image Viewer", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def copy_files(source_path, file_list, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for file_name in file_list:
        src = os.path.join(source_path, file_name)
        dst = os.path.join(destination_dir, file_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for f in os.listdir(directory_path):
            fp = os.path.join(directory_path, f)
            try:
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            except Exception as e:
                print(f"Nie udało się usunąć {fp}: {e}")

def split_data(train_ratio, source_path, train_path, valid_path):
    clear_directory(train_path)
    clear_directory(valid_path)
    imgs = [f for f in os.listdir(source_path) if f.endswith('.jpg')]
    total = len(imgs)
    train_cnt = int(total * train_ratio)
    train_files = imgs[:train_cnt]
    val_files = imgs[train_cnt:]
    copy_files(source_path, train_files, train_path)
    copy_files(source_path, val_files, valid_path)
    print(f"Total images: {total}, train: {len(train_files)}, val: {len(val_files)}")

def json_to_mask(json_folder, output_folder, image_shape):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    clear_directory(output_folder)
    for jf in os.listdir(json_folder):
        if not jf.endswith('.json'):
            continue
        data = json.load(open(os.path.join(json_folder, jf)))
        coords = [np.array(obj['points']['exterior'], np.int32)
                  for obj in data.get('objects', []) if 'points' in obj]
        mask = np.ones(image_shape, np.uint8) * 255
        for poly in coords:
            cv2.fillPoly(mask, [poly], 0)
            cv2.polylines(mask, [poly], True, 255, 3)
        mask = cv2.bitwise_not(mask)
        out_name = os.path.splitext(os.path.splitext(jf)[0])[0] + ".jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), mask)

def get_file_list(data_root, subfolder):
    return sorted([os.path.join(data_root, subfolder, f)
                   for f in os.listdir(os.path.join(data_root, subfolder))])

def get_image_mask_paths(img_dir, mask_dir):
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(img_dir)])
    return imgs, masks

def create_dataset(img_paths, mask_paths):
    ds = Dataset.from_dict({"pixel_values": img_paths, "label": mask_paths})
    ds = ds.cast_column("pixel_values", Image())
    ds = ds.cast_column("label", Image())
    return ds

# Konwersja obrazu na 3-kanałowy RGB, jeśli jest w skali szarości
# Funkcja konwertująca obraz do 3 kanałów, jeśli jest czarno-biały
def to_three_channel(img_array):
    if img_array.ndim == 2:
        return np.stack([img_array] * 3, axis=-1)
    return img_array

# Pipeline augmentacji dla treningu (wyłączamy sprawdzanie zgodności kształtów)
train_augmentations = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={"mask": "mask"}, is_check_shapes=False)

# Pipeline preprocessingu dla walidacji (wyłączamy sprawdzanie zgodności kształtów)
val_augmentations = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={"mask": "mask"}, is_check_shapes=False)

# Transformacje dla HF Dataset: train
def train_transforms(example_batch):
    imgs = example_batch["pixel_values"]
    msks = example_batch["label"]
    batch_imgs = []
    batch_masks = []
    for img, msk in zip(imgs, msks):
        arr_img = np.array(img)
        arr_img = to_three_channel(arr_img)
        arr_msk = np.array(msk)
        if arr_msk.ndim == 3:
            arr_msk = arr_msk[:, :, 0]
        aug = train_augmentations(image=arr_img, mask=arr_msk)
        img_tensor = aug["image"]
        mask_tensor = aug["mask"].squeeze(0).long()
        # remap mask: 0->0, 255->1
        mask_tensor = (mask_tensor > 0).long()
        batch_imgs.append(img_tensor)
        batch_masks.append(mask_tensor)
    return {"pixel_values": batch_imgs, "labels": batch_masks}

# Transformacje dla HF Dataset: validation
def val_transforms(example_batch):
    imgs = example_batch["pixel_values"]
    msks = example_batch["label"]
    batch_imgs = []
    batch_masks = []
    for img, msk in zip(imgs, msks):
        arr_img = np.array(img)
        arr_img = to_three_channel(arr_img)
        arr_msk = np.array(msk)
        if arr_msk.ndim == 3:
            arr_msk = arr_msk[:, :, 0]
        aug = val_augmentations(image=arr_img, mask=arr_msk)
        img_tensor = aug["image"]
        mask_tensor = aug["mask"].squeeze(0).long()
        mask_tensor = (mask_tensor > 0).long()
        batch_imgs.append(img_tensor)
        batch_masks.append(mask_tensor)
    return {"pixel_values": batch_imgs, "labels": batch_masks}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        labels = labels.to(logits.device)
        logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],  # (H, W)
            mode="bilinear",
            align_corners=False
        )

        logits = logits[:, 1, :, :]  # tylko kanał dla klasy „teeth”
        labels = labels.float()

        loss_fct = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root   = r'C:/mgr/data'
    src_img     = data_root + r'/Teeth Segmentation PNG/d2/img'
    train_img   = data_root + r'/TRAIN_IMAGES'
    valid_img   = data_root + r'/VALID_IMAGES'
    mask_folder = data_root + r'/MASKS'
    ann_folder  = data_root + r'/Teeth Segmentation PNG/d2/ann'
    img_shape   = (1024, 2041)

    # split_data(0.8, src_img, train_img, valid_img)
    # json_to_mask(ann_folder, mask_folder, img_shape)

    train_imgs, train_masks = get_image_mask_paths(train_img, mask_folder)
    val_imgs,   val_masks   = get_image_mask_paths(valid_img, mask_folder)

    train_ds = create_dataset(train_imgs, train_masks)
    val_ds   = create_dataset(val_imgs,   val_masks)
    dataset  = DatasetDict({"train": train_ds, "validation": val_ds})

    dataset["train"].set_transform(train_transforms)
    dataset["validation"].set_transform(val_transforms)

    model_name = "nvidia/mit-b4"
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        id2label={0: 'background', 1: 'teeth'},
        label2id={'background': 0, 'teeth': 1}
    )
    model.to(device)

    save_dir = r'C:\mgr\data\SEGFORMER'
    os.makedirs(save_dir, exist_ok=True)
    os.environ["WANDB_DISABLED"] = "true"
    args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=1e-4,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=10,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        fp16=False
    )

    def dice_score(preds, labels, ignore_index=255):
        scores = []
        for lab in range(labels.max() + 1):
            if lab == ignore_index:
                continue
            pred_mask = preds == lab
            true_mask = labels == lab
            if true_mask.sum()==0 and pred_mask.sum()==0:
                scores.append(0.0)
            else:
                scores.append((2.0 * (pred_mask & true_mask).sum()) / (pred_mask.sum()+true_mask.sum()))
        return float(torch.tensor(scores).mean())

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.from_numpy(logits)
        upsampled = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).argmax(dim=1)
        return {"dice_score": dice_score(upsampled, labels)}



    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Save model
    model.save_pretrained(save_dir)
    

if __name__ == "__main__":
    main()
