# Supervised U-Net baseline, PUMA nuclei segmentation

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

class PUMANucleiDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, file_ids, img_size=512):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.file_ids = file_ids
        self.img_size = img_size

    def __len__(self):
        return len(self.file_ids)

    def _mask_from_geojson(self, geojson_path, original_size, target_size):
        # Draw the mask at the image's ORIGINAL resolution (GeoJSON polygon
        mask = Image.new("L", original_size, 0)
        draw = ImageDraw.Draw(mask)
        with open(geojson_path) as f:
            data = json.load(f)
        for feature in data.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "Polygon":
                for ring in geom["coordinates"]:
                    pts = [(x, y) for x, y in ring]
                    draw.polygon(pts, fill=1)
            elif geom.get("type") == "MultiPolygon":
                for polygon in geom["coordinates"]:
                    for ring in polygon:
                        pts = [(x, y) for x, y in ring]
                        draw.polygon(pts, fill=1)
        mask = mask.resize(target_size, resample=Image.NEAREST)
        return np.array(mask, dtype=np.float32)

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        img_path = os.path.join(self.images_dir, f"{fid}.png")
        ann_path = os.path.join(self.annotations_dir, f"{fid}.geojson")

        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        img = img.resize((self.img_size, self.img_size))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # C,H,W

        mask_arr = self._mask_from_geojson(ann_path, original_size, (self.img_size, self.img_size))
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # 1,H,W

        return img_tensor, mask_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))

def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)

def dice_loss(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)

def train_baseline(images_dir, annotations_dir, file_ids, epochs=20, lr=1e-4, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"

    split = int(0.8 * len(file_ids))
    train_ids, val_ids = file_ids[:split], file_ids[split:]

    train_ds = PUMANucleiDataset(images_dir, annotations_dir, train_ids)
    val_ds = PUMANucleiDataset(images_dir, annotations_dir, val_ids)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = dice_loss(preds, masks) + F.binary_cross_entropy(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_dice += dice_score(preds, masks).item()

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss/len(train_loader):.4f} "
              f"| val_dice={val_dice/len(val_loader):.4f}")

    torch.save(model.state_dict(), "baseline_unet.pt")
    return model

def generate_pseudo_labels(model, unlabeled_loader, confidence_thresh=0.9, device="cpu"):
    """
    For each unlabeled image, run the trained baseline model, keep only
    pixels where the model is confident (prob > thresh or < 1-thresh),
    and treat the rest as 'ignore' in the loss (standard pseudo-labeling).
    Returns a list of (image, pseudo_mask, valid_mask) to be folded back
    into training as additional (noisier) supervision.
    """
    model.eval()
    pseudo_data = []
    with torch.no_grad():
        for imgs in unlabeled_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            valid_mask = (preds > confidence_thresh) | (preds < 1 - confidence_thresh)
            pseudo_mask = (preds > 0.5).float()
            pseudo_data.append((imgs.cpu(), pseudo_mask.cpu(), valid_mask.cpu()))
    return pseudo_data

if __name__ == "__main__":
    # Example usage once PUMA is downloaded and file_ids populated:
    print("Populate data/images and data/annotations from the PUMA dataset, "
          "then set file_ids and call train_baseline().")
