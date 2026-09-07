# Mean Teacher SSL with real out-of-domain unlabeled data (TCGA), not in-domain PUMA split

import sys
import types

try:
    import triton.backends.compiler
except Exception:
    import triton.backends
    dummy = types.ModuleType("triton.backends.compiler")
    dummy.AttrsDescriptor = object
    sys.modules["triton.backends.compiler"] = dummy
    triton.backends.compiler = dummy
# fixes: AttributeError: module 'triton.backends' has no attribute 'compiler'
# torch.optim tries to lazy-import torch._dynamo, which needs this triton submodule
# to exist even though we never call torch.compile — this patches it in directly

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TCGA_PATCH_DIR = "/kaggle/working/tcga_patches"


class TCGAUnlabeledDataset(Dataset):
    def __init__(self, patch_dir, img_size=1024):
        self.patch_dir = patch_dir
        self.files = [f for f in os.listdir(patch_dir) if f.endswith(".tif")]
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.patch_dir, self.files[idx])).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


all_file_ids = sorted(set(
    f[:-4] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".tif")
))
np.random.seed(42)
np.random.shuffle(all_file_ids)

N_LABELED = 15
labeled_ids = all_file_ids[:N_LABELED]
val_ids = all_file_ids[180:]

print(f"Labeled: {len(labeled_ids)} | Val: {len(val_ids)}")

IMG_SIZE = 1024
BATCH_SIZE = 2

labeled_ds = PUMANucleiDataset(IMAGES_DIR, ANNOTATIONS_DIR, labeled_ids, img_size=IMG_SIZE)
unlabeled_ds = TCGAUnlabeledDataset(TCGA_PATCH_DIR, img_size=IMG_SIZE)
val_ds = PUMANucleiDataset(IMAGES_DIR, ANNOTATIONS_DIR, val_ids, img_size=IMG_SIZE)

print(f"Real unlabeled TCGA patches: {len(unlabeled_ds)}")

labeled_loader = DataLoader(labeled_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
unlabeled_loader = DataLoader(unlabeled_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

student = UNet().to(DEVICE)
teacher = copy.deepcopy(student).to(DEVICE)
for p in teacher.parameters():
    p.requires_grad = False  # teacher: EMA only, no backprop

optimizer = torch.optim.Adam(student.parameters(), lr=5e-5)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

EMA_DECAY = 0.99
EPOCHS = 60
MAX_CONSISTENCY_WEIGHT = 1.0
RAMPUP_EPOCHS = 10


def update_teacher_ema(student, teacher, decay):
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1 - decay)


def consistency_weight(epoch):
    if epoch >= RAMPUP_EPOCHS:
        return MAX_CONSISTENCY_WEIGHT
    return MAX_CONSISTENCY_WEIGHT * (epoch / RAMPUP_EPOCHS)


best_dice = 0.0
unlabeled_iter = iter(unlabeled_loader)

for epoch in range(EPOCHS):
    student.train()
    total_sup_loss, total_cons_loss = 0.0, 0.0
    cw = consistency_weight(epoch)

    for imgs, masks in labeled_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        try:
            unl_imgs = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unl_imgs = next(unlabeled_iter)
        unl_imgs = unl_imgs.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            sup_preds = student(imgs)
        sup_preds = sup_preds.float()
        sup_loss = dice_loss(sup_preds, masks) + F.binary_cross_entropy(sup_preds, masks)

        brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.8
        contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.8
        noise = torch.randn_like(unl_imgs) * 0.08

        angle = (torch.rand(1).item() - 0.5) * 30
        translate_x = (torch.rand(1).item() - 0.5) * 0.1 * unl_imgs.shape[-1]
        translate_y = (torch.rand(1).item() - 0.5) * 0.1 * unl_imgs.shape[-2]
        theta = torch.tensor([[
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), translate_x / (unl_imgs.shape[-1] / 2)],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), translate_y / (unl_imgs.shape[-2] / 2)],
        ]], dtype=torch.float32, device=unl_imgs.device).repeat(unl_imgs.shape[0], 1, 1)
        grid = F.affine_grid(theta, unl_imgs.shape, align_corners=False)
        warped = F.grid_sample(unl_imgs, grid, align_corners=False, padding_mode="reflection")

        augmented = torch.clamp(warped * contrast_factor + (brightness_factor - 1.0) + noise, 0.0, 1.0)

        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            student_unl_preds = student(augmented)
            with torch.no_grad():
                teacher_unl_preds = teacher(unl_imgs)
        student_unl_preds = student_unl_preds.float()
        teacher_unl_preds = teacher_unl_preds.float()
        cons_loss = F.mse_loss(student_unl_preds, teacher_unl_preds)

        loss = sup_loss + cw * cons_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_teacher_ema(student, teacher, EMA_DECAY)

        total_sup_loss += sup_loss.item()
        total_cons_loss += cons_loss.item()

    student.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                preds = student(imgs)
            preds = preds.float()
            val_dice += dice_score(preds, masks).item()
    val_dice /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | sup_loss={total_sup_loss/len(labeled_loader):.4f} "
          f"| cons_loss={total_cons_loss/len(labeled_loader):.4f} | val_dice={val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, "ssl_tcga_student_best.pt"))

print(f"\nBest SSL (real TCGA unlabeled) val_dice: {best_dice:.4f}")
print("Compare against supervised_only_comparison.py's 0.8702 for 15 labeled.")
