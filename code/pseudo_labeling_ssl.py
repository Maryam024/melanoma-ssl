# Pseudo-labeling SSL: simplest standard SSL baseline

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

all_file_ids = sorted(set(
    f[:-4] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".tif")
))
np.random.seed(42)
np.random.shuffle(all_file_ids)

N_LABELED = 15
labeled_ids = all_file_ids[:N_LABELED]
unlabeled_ids = all_file_ids[N_LABELED:180]
val_ids = all_file_ids[180:]

IMG_SIZE = 1024
BATCH_SIZE = 2
CONFIDENCE_THRESH = 0.9  # only trust pseudo-labels the model is this confident about

labeled_ds = PUMANucleiDataset(IMAGES_DIR, ANNOTATIONS_DIR, labeled_ids, img_size=IMG_SIZE)
val_ds = PUMANucleiDataset(IMAGES_DIR, ANNOTATIONS_DIR, val_ids, img_size=IMG_SIZE)
labeled_loader = DataLoader(labeled_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


class UnlabeledPUMADataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, file_ids, img_size=1024):
        self.images_dir = images_dir
        self.file_ids = file_ids
        self.img_size = img_size

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        img = Image.open(os.path.join(self.images_dir, f"{fid}.tif")).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


unlabeled_ds = UnlabeledPUMADataset(IMAGES_DIR, unlabeled_ids, img_size=IMG_SIZE)
unlabeled_loader = DataLoader(unlabeled_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

# --- Stage 1: train on labeled data only ---
STAGE1_EPOCHS = 40
for epoch in range(STAGE1_EPOCHS):
    model.train()
    for imgs, masks in labeled_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            preds = model(imgs)
        preds = preds.float()
        loss = dice_loss(preds, masks) + F.binary_cross_entropy(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
print("Stage 1 (labeled-only) done")

# --- Generate pseudo-labels on unlabeled data ---
model.eval()
pseudo_imgs, pseudo_masks = [], []
with torch.no_grad():
    for imgs in unlabeled_loader:
        imgs = imgs.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            preds = model(imgs)
        preds = preds.float()
        confident = (preds > CONFIDENCE_THRESH) | (preds < 1 - CONFIDENCE_THRESH)
        frac_confident = confident.float().mean(dim=[1, 2, 3])
        for i in range(imgs.shape[0]):
            if frac_confident[i] > 0.8:  # only keep images where most pixels are confident
                pseudo_imgs.append(imgs[i].cpu())
                pseudo_masks.append((preds[i] > 0.5).float().cpu())

print(f"Kept {len(pseudo_imgs)} / {len(unlabeled_ds)} unlabeled images as confident pseudo-labels")

# --- Stage 2: retrain on labeled + pseudo-labeled combined ---
if len(pseudo_imgs) > 0:
    pseudo_imgs_t = torch.stack(pseudo_imgs)
    pseudo_masks_t = torch.stack(pseudo_masks)
    pseudo_ds = torch.utils.data.TensorDataset(pseudo_imgs_t, pseudo_masks_t)
    pseudo_loader = DataLoader(pseudo_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    STAGE2_EPOCHS = 40
    best_dice = 0.0
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        for loader in [labeled_loader, pseudo_loader]:
            for imgs, masks in loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                    preds = model(imgs)
                preds = preds.float()
                loss = dice_loss(preds, masks) + F.binary_cross_entropy(preds, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
        print(f"Stage 2 Epoch {epoch+1}/{STAGE2_EPOCHS} | val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pseudo_label_best.pt"))

    print(f"\nBest pseudo-labeling val_dice: {best_dice:.4f}")
else:
    print("No confident pseudo-labels found — model too uncertain on unlabeled data at this stage.")
