# Test-time augmentation (TTA): average predictions over flips/rotations of each image

import os
import torch
from torch.utils.data import DataLoader

CHECKPOINT_TO_EVAL = "baseline_augmented_best.pt"  # change to whichever model you want to evaluate

file_ids = sorted(set(
    f[:-4] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".tif")
))
split = int(0.8 * len(file_ids))
val_ids = file_ids[split:]

eval_ds = PUMANucleiDataset(IMAGES_DIR, ANNOTATIONS_DIR, val_ids, img_size=1024)
eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, CHECKPOINT_TO_EVAL)))
model.eval()


def tta_predict(model, img):
    preds = []
    variants = [
        (img, lambda x: x),
        (torch.flip(img, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (torch.flip(img, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (torch.rot90(img, 1, dims=[2, 3]), lambda x: torch.rot90(x, -1, dims=[2, 3])),
    ]
    for variant_img, undo in variants:
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            pred = model(variant_img)
        pred = pred.float()
        preds.append(undo(pred))
    return torch.stack(preds).mean(dim=0)


total_dice = 0.0
n = 0
with torch.no_grad():
    for imgs, masks in eval_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = tta_predict(model, imgs)
        total_dice += dice_score(preds, masks).item()
        n += 1

print(f"Dice with TTA: {total_dice/n:.4f}")
