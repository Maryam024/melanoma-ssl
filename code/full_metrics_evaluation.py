# Full metrics on baseline model: Dice/F1, IoU, Precision, Recall, Pixel Accuracy

import os
import torch
from torch.utils.data import DataLoader

file_ids = sorted(set(
    f[:-4] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".tif")
))
split = int(0.8 * len(file_ids))
val_ids_eval = file_ids[split:]

eval_ds = PUMANucleiDataset(IMAGES_DIR, ANNOTATIONS_DIR, val_ids_eval, img_size=1024)
eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, num_workers=2)

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "baseline_unet_best.pt")))
model.eval()


def compute_metrics(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_acc = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    return dice.item(), iou.item(), precision.item(), recall.item(), pixel_acc.item()


totals = [0.0] * 5
n = 0
with torch.no_grad():
    for imgs, masks in eval_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            preds = model(imgs)
        preds = preds.float()
        m = compute_metrics(preds, masks)
        totals = [t + v for t, v in zip(totals, m)]
        n += 1

names = ["Dice/F1", "IoU/Jaccard", "Precision", "Recall", "Pixel Accuracy"]
for name, total in zip(names, totals):
    print(f"{name}: {total/n:.4f}")
