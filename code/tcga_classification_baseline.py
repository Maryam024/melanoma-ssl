import json
import os
import re
import numpy as np
import openslide
import requests
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SVS_DIR = "/kaggle/working/tcga_svs"
PATCH_DIR = "/kaggle/working/tcga_patches"
os.makedirs(SVS_DIR, exist_ok=True)
os.makedirs(PATCH_DIR, exist_ok=True)

PATCH_SIZE = 1024
PATCHES_PER_SLIDE = 40
TISSUE_STD_THRESHOLD = 15
SLIDES_PER_CLASS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_tissue(patch_arr):
    return patch_arr.std() > TISSUE_STD_THRESHOLD


def extract_patches(svs_path, n_patches):
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.dimensions
    saved, attempts = 0, 0
    while saved < n_patches and attempts < n_patches * 10:
        attempts += 1
        x = np.random.randint(0, max(1, w - PATCH_SIZE))
        y = np.random.randint(0, max(1, h - PATCH_SIZE))
        patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        if not is_tissue(np.array(patch)):
            continue
        stem = os.path.splitext(os.path.basename(svs_path))[0]
        patch.save(os.path.join(PATCH_DIR, f"{stem}_patch{saved:03d}.tif"))
        saved += 1
    slide.close()
    return saved

def download_and_process(sample_type, n_slides):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-SKCM"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": [sample_type]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size",
        "format": "JSON",
        "size": str(n_slides),
    }
    hits = requests.get("https://api.gdc.cancer.gov/files", params=params).json()["data"]["hits"]
    print(f"{sample_type}: found {len(hits)} slides on GDC")

    for h in hits:
        file_id, name = h["file_id"], h["file_name"]
        stem = os.path.splitext(name)[0]
        if any(f.startswith(stem) for f in os.listdir(PATCH_DIR)):
            continue
        out_path = os.path.join(SVS_DIR, name)
        r = requests.get(f"https://api.gdc.cancer.gov/data/{file_id}", stream=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        n_saved = extract_patches(out_path, PATCHES_PER_SLIDE)
        print(f"  {name}: extracted {n_saved} patches")
        os.remove(out_path)  # keep disk usage low; only patches are kept

def get_sample_type(filename):
    match = re.search(r"TCGA-\w+-\w+-(\d{2})", filename)
    if not match:
        return None
    code = match.group(1)
    if code == "01":
        return 0  # primary
    if code == "06":
        return 1  # metastatic
    return None


class TCGAClassificationDataset(Dataset):
    def __init__(self, file_label_list, patch_dir, transform):
        self.data = file_label_list
        self.patch_dir = patch_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, label = self.data[idx]
        img = Image.open(os.path.join(self.patch_dir, fname)).convert("RGB")
        return self.transform(img), label

def main():
    download_and_process("Primary Tumor", SLIDES_PER_CLASS)
    download_and_process("Metastatic", SLIDES_PER_CLASS)

    all_files = [f for f in os.listdir(PATCH_DIR) if f.endswith(".tif")]
    labeled_files = []
    for f in all_files:
        label = get_sample_type(f)
        if label is not None:
            slide_id = f.split("_patch")[0]
            labeled_files.append((f, label, slide_id))
    print(f"Total patches: {len(all_files)} | Usable (labeled): {len(labeled_files)}")

    primary_slides = sorted(set(s for _, l, s in labeled_files if l == 0))
    meta_slides = sorted(set(s for _, l, s in labeled_files if l == 1))
    print(f"Primary slides: {len(primary_slides)} | Metastatic slides: {len(meta_slides)}")

    np.random.seed(42)
    np.random.shuffle(primary_slides)
    np.random.shuffle(meta_slides)

    n_val_primary = max(1, len(primary_slides) // 4)
    n_val_meta = max(1, len(meta_slides) // 4)
    val_slides = set(primary_slides[:n_val_primary] + meta_slides[:n_val_meta])

    train_files = [(f, l) for f, l, s in labeled_files if s not in val_slides]
    val_files = [(f, l) for f, l, s in labeled_files if s in val_slides]
    print(f"Train patches: {len(train_files)} | Val patches: {len(val_files)}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        TCGAClassificationDataset(train_files, PATCH_DIR, train_transform),
        batch_size=16, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        TCGAClassificationDataset(val_files, PATCH_DIR, val_transform),
        batch_size=16, shuffle=False, num_workers=2,
    )

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 20
    best_acc = 0.0
    best_preds, best_labels = None, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/len(train_loader):.4f} | val_acc={acc:.4f} | val_f1={f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_preds, best_labels = all_preds.copy(), all_labels.copy()
            torch.save(model.state_dict(), "/kaggle/working/tcga_baseline_classifier_best.pt")

    print(f"\nBest TCGA baseline (primary vs. metastatic) val_acc: {best_acc:.4f}")
    print("Confusion matrix at best epoch (rows=true, cols=predicted):")
    print(confusion_matrix(best_labels, best_preds))

if __name__ == "__main__":
    main()
