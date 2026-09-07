# Supervised-only baseline on same labeled subset, for SSL comparison

torch.manual_seed(0)  # same seed as a fair-comparison

sup_only_model = UNet().to(DEVICE)
sup_optimizer = torch.optim.Adam(sup_only_model.parameters(), lr=5e-5)
sup_scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

SUP_EPOCHS = 60  # match the SSL run's epoch

best_sup_dice = 0.0
for epoch in range(SUP_EPOCHS):
    sup_only_model.train()
    total_loss = 0.0
    for imgs, masks in labeled_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        sup_optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            preds = sup_only_model(imgs)
        preds = preds.float()
        loss = dice_loss(preds, masks) + F.binary_cross_entropy(preds, masks)
        sup_scaler.scale(loss).backward()
        sup_scaler.step(sup_optimizer)
        sup_scaler.update()
        total_loss += loss.item()

    sup_only_model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                preds = sup_only_model(imgs)
            preds = preds.float()
            val_dice += dice_score(preds, masks).item()
    val_dice /= len(val_loader)

    print(f"[Supervised-only, 40 labeled] Epoch {epoch+1}/{SUP_EPOCHS} "
          f"| loss={total_loss/len(labeled_loader):.4f} | val_dice={val_dice:.4f}")

    if val_dice > best_sup_dice:
        best_sup_dice = val_dice
        torch.save(sup_only_model.state_dict(), os.path.join(OUTPUT_DIR, "sup_only_best.pt"))

print(f"\n=== RESULT ===")
print(f"Supervised-only (15 labeled):        best val_dice = {best_sup_dice:.4f}")
print(f"Mean Teacher SSL (15 labeled + unlabeled): best val_dice = <paste your SSL number here>")
print("If SSL's number is meaningfully higher, that's your real, defensible evidence "
      "that the unlabeled data helped — which is the actual research question this "
      "project asks.")
