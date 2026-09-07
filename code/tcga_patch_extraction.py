# Extract tissue patches from TCGA SVS files, matching PUMA's 1024x1024 format

import os
import numpy as np
import openslide
from PIL import Image

SVS_DIR = "/kaggle/working/tcga_svs"
OUT_DIR = "/kaggle/working/tcga_patches"
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = 1024
PATCHES_PER_SLIDE = 40
TISSUE_STD_THRESHOLD = 15  # low std = blank/background patch, skip it


def is_tissue(patch_arr):
    return patch_arr.std() > TISSUE_STD_THRESHOLD


def extract_patches(svs_path, n_patches):
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.dimensions
    saved = 0
    attempts = 0
    while saved < n_patches and attempts < n_patches * 10:
        attempts += 1
        x = np.random.randint(0, max(1, w - PATCH_SIZE))
        y = np.random.randint(0, max(1, h - PATCH_SIZE))
        patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        patch_arr = np.array(patch)
        if not is_tissue(patch_arr):
            continue
        stem = os.path.splitext(os.path.basename(svs_path))[0]
        patch.save(os.path.join(OUT_DIR, f"{stem}_patch{saved:03d}.tif"))
        saved += 1
    slide.close()
    return saved


total = 0
for fname in os.listdir(SVS_DIR):
    if not fname.lower().endswith(".svs"):
        continue
    n = extract_patches(os.path.join(SVS_DIR, fname), PATCHES_PER_SLIDE)
    print(f"{fname}: saved {n} patches")
    total += n

print(f"Total patches extracted: {total}")
