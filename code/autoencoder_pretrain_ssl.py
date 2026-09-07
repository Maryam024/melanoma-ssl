# Autoencoder pretraining + fine-tune, single lr

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

all_images_ids = labeled_ids + unlabeled_ids  # every image with a photo,
all_images_ds = UnlabeledPUMADataset(IMAGES_DIR, all_images_ids, img_size=IMG_SIZE)
all_images_loader = DataLoader(all_images_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print(f"Autoencoder pretraining pool: {len(all_images_ids)} images (no masks used)")

ae_model = UNet(in_ch=3, out_ch=3).to(DEVICE)  # out_ch=3 to reconstruct RGB, not
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-4)
ae_scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

AE_EPOCHS = 30

for epoch in range(AE_EPOCHS):
    ae_model.train()
    total_recon_loss = 0.0
    for imgs in all_images_loader:
        imgs = imgs.to(DEVICE)
        ae_optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            recon = ae_model(imgs)
        recon = recon.float()
        loss = F.mse_loss(recon, imgs)  # pure reconstruction loss, no labels
        ae_scaler.scale(loss).backward()
        ae_scaler.step(ae_optimizer)
        ae_scaler.update()
        total_recon_loss += loss.item()

    print(f"[Autoencoder pretrain] Epoch {epoch+1}/{AE_EPOCHS} "
          f"| recon_loss={total_recon_loss/len(all_images_loader):.5f}")

torch.save(ae_model.state_dict(), os.path.join(OUTPUT_DIR, "autoencoder_pretrained.pt"))
print("Saved pretrained autoencoder.\n")

seg_model = UNet(in_ch=3, out_ch=1).to(DEVICE)

ae_state = ae_model.state_dict()
seg_state = seg_model.state_dict()
encoder_prefixes = ("enc1.", "enc2.", "enc3.", "bottleneck.")
transplanted = 0
for key in seg_state.keys():
    if key.startswith(encoder_prefixes) and key in ae_state:
        seg_state[key] = ae_state[key]
        transplanted += 1
seg_model.load_state_dict(seg_state)
print(f"Transplanted {transplanted} encoder tensors from the pretrained autoencoder.\n")

seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=5e-5)
seg_scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

FT_EPOCHS = 60
best_ae_dice = 0.0

for epoch in range(FT_EPOCHS):
    seg_model.train()
    total_loss = 0.0
    for imgs, masks in labeled_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        seg_optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            preds = seg_model(imgs)
        preds = preds.float()
        loss = dice_loss(preds, masks) + F.binary_cross_entropy(preds, masks)
        seg_scaler.scale(loss).backward()
        seg_scaler.step(seg_optimizer)
        seg_scaler.update()
        total_loss += loss.item()

    seg_model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                preds = seg_model(imgs)
            preds = preds.float()
            val_dice += dice_score(preds, masks).item()
    val_dice /= len(val_loader)

    print(f"[AE-pretrained, {N_LABELED} labeled] Epoch {epoch+1}/{FT_EPOCHS} "
          f"| loss={total_loss/len(labeled_loader):.4f} | val_dice={val_dice:.4f}")

    if val_dice > best_ae_dice:
        best_ae_dice = val_dice
        torch.save(seg_model.state_dict(), os.path.join(OUTPUT_DIR, "ae_pretrained_seg_best.pt"))

print(f"\n=== THREE-WAY COMPARISON ({N_LABELED} labeled examples) ===")
print(f"Supervised-only, random init:        val_dice = 0.8702  (from your earlier run)")
print(f"Mean Teacher SSL (consistency):      val_dice = 0.8661  (from your earlier run)")
print(f"Autoencoder-pretrained + fine-tune:   val_dice = {best_ae_dice:.4f}")
