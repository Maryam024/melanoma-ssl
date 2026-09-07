# Augmented PUMA dataset: random flip, rotation, color jitter for labeled training

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageEnhance


class AugmentedPUMADataset(Dataset):
    def __init__(self, images_dir, annotations_dir, file_ids, img_size=1024, augment=True):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.file_ids = file_ids
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.file_ids)

    def _mask_from_geojson(self, geojson_path, original_size):
        mask = Image.new("L", original_size, 0)
        draw = ImageDraw.Draw(mask)
        with open(geojson_path) as f:
            data = json.load(f)
        for feature in data.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "Polygon":
                for ring in geom["coordinates"]:
                    draw.polygon([(x, y) for x, y in ring], fill=1)
            elif geom.get("type") == "MultiPolygon":
                for polygon in geom["coordinates"]:
                    for ring in polygon:
                        draw.polygon([(x, y) for x, y in ring], fill=1)
        return mask

    def _augment(self, img, mask):
        # flips
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        # rotation (90/180/270, keeps mask alignment exact)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        # color jitter (image only, mask untouched)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        return img, mask

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        img_path = os.path.join(self.images_dir, f"{fid}.tif")
        ann_path = os.path.join(self.annotations_dir, f"{fid}_nuclei.geojson")

        img = Image.open(img_path).convert("RGB")
        mask = self._mask_from_geojson(ann_path, img.size)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = img.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        img_arr = np.array(img, dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.float32)

        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return img_tensor, mask_tensor
