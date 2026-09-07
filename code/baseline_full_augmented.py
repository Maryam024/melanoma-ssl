# Full baseline (205 labeled images) WITH augmentation

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

file_ids = sorted(set(
    f[:-4] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".tif")
))
split = int(0.8 * len(file_ids))
train_ids, val_ids = file_ids[:split], file_ids[split:]

train_ds = AugmentedPUMADataset(IMAGES_DIR, ANNOTATIONS_DIR, train_ids, img_size=1024, augment=True)
val_ds = AugmentedPUMADataset(IMAGES_DIR, ANNOTATIONS_DIR, val_ids, img_size=1024, augment=False)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

EPOCHS = 40
best_dice = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            preds = model(imgs)
        preds = preds.float()
        loss = dice_loss(preds, masks) + F.binary_cross_entropy(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                preds = model(imgs)
            preds = preds.float()
            val_dice += dice_score(preds, masks).item()
    val_dice /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} | val_dice={val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "baseline_augmented_best.pt"))

print(f"\nBest val_dice with augmentation: {best_dice:.4f}")
print(f"Previous (no augmentation): 0.9038")
