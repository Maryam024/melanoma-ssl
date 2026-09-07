# Download TCGA-SKCM diagnostic slides via GDC API

import json
import os
import requests

OUT_DIR = "/kaggle/working/tcga_svs"
os.makedirs(OUT_DIR, exist_ok=True)

N_SLIDES = 8  # keep small — each slide is 1-4GB

filters = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-SKCM"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
    ],
}

params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,file_size",
    "format": "JSON",
    "size": str(N_SLIDES),
}

resp = requests.get("https://api.gdc.cancer.gov/files", params=params)
hits = resp.json()["data"]["hits"]
print(f"Found {len(hits)} slides")

for h in hits:
    file_id, name = h["file_id"], h["file_name"]
    out_path = os.path.join(OUT_DIR, name)
    if os.path.exists(out_path):
        print(f"skip {name}, already downloaded")
        continue
    print(f"downloading {name} ({h['file_size'] / 1e9:.1f} GB)")
    r = requests.get(f"https://api.gdc.cancer.gov/data/{file_id}", stream=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

print("done:", os.listdir(OUT_DIR))
