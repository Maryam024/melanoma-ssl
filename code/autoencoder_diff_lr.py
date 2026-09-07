# Autoencoder pretraining + fine-tune, differential encoder/decoder lr

import torch
import torch.nn.functional as F

seg_model_v2 = UNet(in_ch=3, out_ch=1).to(DEVICE)

ae_state = ae_model.state_dict()  # reuse the already-pretrained autoencoder from
seg_state = seg_model_v2.state_dict()
encoder_prefixes = ("enc1.", "enc2.", "enc3.", "bottleneck.")
for key in seg_state.keys():
    if key.startswith(encoder_prefixes) and key in ae_state:
        seg_state[key] = ae_state[key]
seg_model_v2.load_state_dict(seg_state)

encoder_params = []
decoder_params = []
for name, param in seg_model_v2.named_parameters():
    if name.startswith(encoder_prefixes):
        encoder_params.append(param)
    else:
        decoder_params.append(param)

print(f"Encoder params: {len(encoder_params)} tensors (lr=1e-5, gentle) | "
      f"Decoder params: {len(decoder_params)} tensors (lr=3e-4, fast)")

optimizer_v2 = torch.optim.Adam([
    {"params": encoder_params, "lr": 1e-5},   # pretrained — small nudges only
    {"params": decoder_params, "lr": 3e-4},   # random init — needs to learn fast
])
scaler_v2 = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

FT_EPOCHS_V2 = 60
best_ae_v2_dice = 0.0

for epoch in range(FT_EPOCHS_V2):
    seg_model_v2.train()
    total_loss = 0.0
    for imgs, masks in labeled_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer_v2.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            preds = seg_model_v2(imgs)
        preds = preds.float()
        loss = dice_loss(preds, masks) + F.binary_cross_entropy(preds, masks)
        scaler_v2.scale(loss).backward()
        scaler_v2.step(optimizer_v2)
        scaler_v2.update()
        total_loss += loss.item()

    seg_model_v2.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                preds = seg_model_v2(imgs)
            preds = preds.float()
            val_dice += dice_score(preds, masks).item()
    val_dice /= len(val_loader)

    print(f"[AE-pretrained v2, diff-lr] Epoch {epoch+1}/{FT_EPOCHS_V2} "
          f"| loss={total_loss/len(labeled_loader):.4f} | val_dice={val_dice:.4f}")

    if val_dice > best_ae_v2_dice:
        best_ae_v2_dice = val_dice
        torch.save(seg_model_v2.state_dict(), os.path.join(OUTPUT_DIR, "ae_pretrained_v2_best.pt"))

print(f"\n=== FOUR-WAY COMPARISON ({N_LABELED} labeled examples) ===")
print(f"Supervised-only, random init:               val_dice = 0.8702")
print(f"Mean Teacher SSL (consistency):             val_dice = 0.8661")
print(f"Autoencoder-pretrained, single lr:           val_dice = 0.7692")
print(f"Autoencoder-pretrained, differential lr:     val_dice = {best_ae_v2_dice:.4f}")
