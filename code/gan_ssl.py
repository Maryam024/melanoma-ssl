# Semi-supervised GAN for nuclei segmentation (Hung et al. 2018 style adversarial approach)
# Generator = UNet producing masks. Discriminator judges image+mask pairs as real/fake.
# Unlabeled data contributes via the adversarial loss, without needing ground truth.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base * 8, 1, 1),
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        return self.net(x).view(-1)


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
unlabeled_loader = DataLoader(unlabeled_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

generator = UNet().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

opt_g = torch.optim.Adam(generator.parameters(), lr=5e-5, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))

ADV_WEIGHT = 0.01  # keep small — adversarial loss is a regularizer, not the main signal
EPOCHS = 60
best_dice = 0.0
unlabeled_iter = iter(unlabeled_loader)

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()
    total_sup_loss, total_d_loss, total_g_adv_loss = 0.0, 0.0, 0.0

    for imgs, masks in labeled_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        try:
            unl_imgs = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unl_imgs = next(unlabeled_iter)
        unl_imgs = unl_imgs.to(DEVICE)

        # --- train discriminator: real (labeled img, gt mask) vs fake (any img, generator's mask) ---
        opt_d.zero_grad()
        with torch.no_grad():
            fake_mask_labeled = generator(imgs)
            fake_mask_unlabeled = generator(unl_imgs)
        d_real = discriminator(imgs, masks)
        d_fake_l = discriminator(imgs, fake_mask_labeled)
        d_fake_u = discriminator(unl_imgs, fake_mask_unlabeled)
        d_loss = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                  + F.binary_cross_entropy_with_logits(d_fake_l, torch.zeros_like(d_fake_l))
                  + F.binary_cross_entropy_with_logits(d_fake_u, torch.zeros_like(d_fake_u))) / 3
        d_loss.backward()
        opt_d.step()

        # --- train generator: supervised loss on labeled + adversarial loss on both ---
        opt_g.zero_grad()
        preds_labeled = generator(imgs)
        preds_unlabeled = generator(unl_imgs)
        sup_loss = dice_loss(preds_labeled, masks) + F.binary_cross_entropy(preds_labeled, masks)

        d_on_fake_l = discriminator(imgs, preds_labeled)
        d_on_fake_u = discriminator(unl_imgs, preds_unlabeled)
        g_adv_loss = (F.binary_cross_entropy_with_logits(d_on_fake_l, torch.ones_like(d_on_fake_l))
                      + F.binary_cross_entropy_with_logits(d_on_fake_u, torch.ones_like(d_on_fake_u))) / 2

        g_loss = sup_loss + ADV_WEIGHT * g_adv_loss
        g_loss.backward()
        opt_g.step()

        total_sup_loss += sup_loss.item()
        total_d_loss += d_loss.item()
        total_g_adv_loss += g_adv_loss.item()

    generator.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = generator(imgs)
            val_dice += dice_score(preds, masks).item()
    val_dice /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | sup_loss={total_sup_loss/len(labeled_loader):.4f} "
          f"| d_loss={total_d_loss/len(labeled_loader):.4f} "
          f"| g_adv_loss={total_g_adv_loss/len(labeled_loader):.4f} | val_dice={val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, "gan_ssl_best.pt"))

print(f"\nBest GAN-SSL val_dice: {best_dice:.4f}")
print("Compare against: supervised-only 0.8702, Mean Teacher 0.8661, autoencoder-diff-lr 0.8521 (all 15 labeled)")
