# Semi-Supervised Melanoma Nuclei Segmentation

Exploring semi-supervised learning for melanoma detection in H&E-stained histopathological images — preparatory work ahead of a MITACS Globalink application, aligned with Dr. Mrinal Mandal's (University of Alberta) project on melanoma CAD via nuclei segmentation and semi-supervised learning.

## Motivation

Medical image labeling is expensive, and most real-world histopathology data is unlabeled or coarsely labeled. This project asks a concrete question: **given only a small labeled set, can semi-supervised learning (SSL) meaningfully improve nuclei segmentation by also using unlabeled images?** Four distinct SSL mechanisms — Mean Teacher (consistency regularization), a semi-supervised GAN (adversarial), autoencoder pretraining (reconstruction), and confidence-thresholded pseudo-labeling (self-training) — are tested and compared honestly against a supervised-only baseline, at two label-scarcity levels (15 and 40 labeled images), including negative results and their diagnosis.

## Related work this builds on

- Akbarpour et al. (2025), *Deep Learning-Based Nuclei Segmentation and Melanoma Detection...* — the direct reference paper for this project (Dr. Mandal's group), reporting ~91.6% Dice on their private dataset.
- Alheejawi et al. (2021) — earlier foundation of the same nuclei-segmentation → melanoma-region pipeline.
- Yu et al. (2021), *Accurate recognition of colorectal cancer with semi-supervised deep learning...* — source of the Mean Teacher method adapted here.
- Hung et al. (2018), *Adversarial Learning for Semi-Supervised Semantic Segmentation* — source of the GAN-based SSL method adapted here.
- Requa et al. (2023) — multi-stage supervised/semi-supervised skin-neoplasm detection, motivating the labeled/coarsely-labeled/unlabeled framing.

## Dataset

[PUMA — Panoptic segmentation of nUclei and tissue in MelanomA](https://puma.grand-challenge.org/dataset/) (GigaScience, 2025). 103 primary + 103 metastatic melanoma H&E ROIs (1024×1024, 40× magnification) with expert nuclei annotations. **Not included in this repo** — download directly from the [Zenodo record](https://zenodo.org/records/14869398) (`01_training_dataset_tif_ROIs.zip` + `01_training_dataset_geojson_nuclei.zip`).

This is a public stand-in for the lab's actual clinical dataset — useful for validating methodology, but not identical in scanner, staining protocol, or patient population. Results here should be read as evidence about the *methods*, not final numbers for the real target dataset.

The out-of-domain extension additionally uses real TCGA-SKCM whole-slide images, downloaded via the GDC API (see `code/tcga_download.py`).

## Method

1. **Baseline**: fully supervised U-Net, trained on all 205 labeled PUMA ROIs at full 1024×1024 resolution.
2. **Scarce-label experiments**: subsample to 15 or 40 "labeled" images, treat the rest as "unlabeled" (masks withheld), and compare supervised-only training against four SSL mechanisms:
   - **Mean Teacher SSL** (Yu et al., 2021): student network trained with a supervised loss on labeled data + a consistency loss (student vs. EMA teacher) on unlabeled data. Tested with both mild and strong consistency-branch augmentation, and with both in-domain (PUMA) and out-of-domain (TCGA-SKCM) unlabeled data.
   - **GAN-based SSL** (Hung et al., 2018): a discriminator judges (image, mask) pairs as real/fake; the generator (segmentation UNet) gets a supervised loss on labeled data plus a small adversarial loss on both labeled and unlabeled data.
   - **Autoencoder-pretrained SSL**: encoder pretrained via image reconstruction on all available images (no masks needed), then transplanted into the segmentation model and fine-tuned on the labeled subset — first with a single learning rate, then with differential encoder/decoder learning rates.
   - **Pseudo-labeling (self-training)**: train on labeled data, generate predictions on unlabeled data, keep only images where the model is highly confident (≥90% of pixels above/below a 0.9 threshold, on ≥80% of the image) as pseudo-labels, retrain on labeled + pseudo-labeled data combined.

## Results

**Full-data supervised baseline** (205 labeled images): **val_dice = 0.9038**

**With data augmentation** (flips, rotations, brightness/contrast jitter): **val_dice = 0.9093** — a genuine improvement from making better use of the same 205 images.

**With test-time augmentation on top**: val_dice = 0.9072 — no further gain, since the model had already learned flip/rotation invariance from training augmentation.

**Scarce-label comparison, 15 labeled examples (165 unlabeled, 25 held-out val):**

| Method | val_dice | vs. supervised |
|---|---|---|
| Supervised-only, random init | **0.8702** | target |
| **GAN-SSL (adversarial)** | **0.8693** | **essentially tied (-0.0009)** |
| Mean Teacher, in-domain PUMA, mild aug | 0.8661 | -0.0041 |
| Mean Teacher, out-of-domain TCGA, strong aug | 0.8454 | -0.0248 |
| Autoencoder-pretrained, differential lr | 0.8521 | -0.0181 |
| Mean Teacher, in-domain PUMA, strong aug | 0.8323 | -0.0379 |
| Autoencoder-pretrained, single lr | 0.7692 | -0.1010 |
| Pseudo-labeling (self-training) | n/a | 0 confident pseudo-labels generated — no result |

**Scarce-label comparison, 40 labeled examples (140 unlabeled, 25 held-out val):**

| Method | val_dice | vs. supervised |
|---|---|---|
| Supervised-only, random init | **0.8903** | target |
| Mean Teacher SSL (consistency) | 0.8883 | -0.0020 |
| GAN-SSL (adversarial) | 0.8872 | -0.0031 |

Full epoch-by-epoch logs for every run are in `results/`.

## Honest conclusion

**No SSL method clearly beat supervised-only training in this setup**, but the picture is more nuanced than a flat negative result, and one method came within noise-level distance of the target.

**GAN-SSL is the standout result.** At 15 labeled images, adversarial training reached 0.8693 — a 0.0009 gap from supervised-only, effectively a tie given the ~0.01-0.03 epoch-to-epoch noise visible throughout these runs. This is the one method tested here that is genuinely competitive with fully supervised training under extreme label scarcity.

**That advantage narrows at 40 labels.** Re-run with 40 labeled images, GAN-SSL (0.8872) fell slightly behind both Mean Teacher (0.8883) and supervised-only (0.8903) — all three within 0.003 of each other, i.e. close to indistinguishable given run-to-run noise. This is consistent with the general expectation for SSL: extra unlabeled data helps most exactly where labeled data is scarcest, and that advantage shrinks as more labels become available.

**Mean Teacher's story required isolating two variables.** An initial run (mild consistency-branch augmentation, in-domain PUMA unlabeled data) scored 0.8661, just under supervised. Two follow-up changes were tested independently for a fair comparison: (1) strengthening the consistency-branch augmentation (affine warp + wider jitter) on the *same* in-domain unlabeled data dropped performance to 0.8323 — the stronger perturbation was too aggressive for the model to learn reliable invariance from at only 15 labeled anchors; (2) holding that same stronger augmentation fixed and swapping in real out-of-domain unlabeled data (320 TCGA-SKCM patches from 8 slides, downloaded via the GDC API) raised performance to 0.8454 — better than the same-augmentation in-domain run, supporting the hypothesis that genuine domain diversity gives consistency regularization more real signal to exploit, though still short of the supervised target.

**Autoencoder pretraining underperformed most.** The larger initial gap (single learning rate: 0.7692) was substantially — but not fully — explained by a learning-rate mismatch between the pretrained encoder and randomly-initialized decoder; using differential learning rates (gentle for the encoder, fast for the decoder) recovered most but not all of the shortfall (0.8521).

**Pseudo-labeling produced no usable result at 15 labels.** Confidence-thresholded self-training found 0 of 165 unlabeled images confident enough to pass the filter. The Stage-1 model simply isn't reliable enough yet at 15 labels to bootstrap its own training signal — a genuine negative finding about this method's label-count floor, not a bug.

**Overall**: semi-supervised learning's value here is real but narrow — visible only in the best-performing method (GAN-based adversarial training) and only at the most extreme label scarcity tested. Different SSL mechanisms respond differently to the same conditions (stronger augmentation helped nothing tested; domain diversity helped Mean Teacher but wasn't enough to close its gap; adversarial training came closest to parity, and only at the lowest label count). This is evidence that mechanism choice matters more than any single "SSL vs. supervised" verdict would suggest.

## TCGA extension (real out-of-domain unlabeled data)

8 TCGA-SKCM slides were downloaded via the GDC API and 320 real tissue patches (1024×1024, tissue-content filtered) were extracted successfully at 100% yield. The training run initially hit a PyTorch/Triton version incompatibility specific to the Kaggle notebook environment (`torch.optim.Adam` triggers an internal `torch._dynamo` import that fails on this image's Triton version); `mean_teacher_tcga.py` includes a monkeypatch at the top of the file that stubs the missing `triton.backends.compiler` module before torch's lazy import reaches it. The run completed successfully after this fix — result above (0.8454), logged in full in `results/06_mean_teacher_tcga_out_of_domain.txt`.

**Known limitation, not yet corrected**: TCGA patches were extracted without confirming their microns-per-pixel matches PUMA's 40x scans. If base magnification differs between the two sources, nuclei could appear at different apparent scales in TCGA patches, which would be an additional confound alongside genuine staining/scanner domain diversity.

## Repository structure

```
code/
  baseline_nuclei_segmentation.py       # local baseline U-Net training
  kaggle_baseline_nuclei_segmentation.py # Kaggle-adapted version (used for all results here)
  mean_teacher_ssl.py                    # Mean Teacher SSL implementation (in-domain unlabeled data)
  mean_teacher_tcga.py                   # Mean Teacher SSL using real TCGA out-of-domain unlabeled data
  gan_ssl.py                             # semi-supervised GAN (Hung et al. 2018 style)
  pseudo_labeling_ssl.py                 # confidence-thresholded pseudo-labeling / self-training
  supervised_only_comparison.py          # fair supervised-only baseline for comparison
  autoencoder_pretrain_ssl.py            # autoencoder pretraining + fine-tune (single lr)
  autoencoder_diff_lr.py                 # autoencoder pretraining + fine-tune (differential lr)
  tcga_download.py                       # download TCGA-SKCM slides via GDC API
  tcga_patch_extraction.py               # extract tissue patches from those slides
  augmented_dataset.py                   # data augmentation (flips, rotation, color jitter)
  baseline_full_augmented.py             # full baseline retrained with augmentation
  tta_evaluation.py                      # test-time augmentation evaluation
  full_metrics_evaluation.py             # Dice/IoU/Precision/Recall/Pixel Accuracy
results/
  01_baseline_full_supervised.txt
  02_scarce_label_comparison_15.txt
  03_scarce_label_comparison_40.txt
  04_full_metrics.txt
  05_augmentation_and_tta.txt
  06_mean_teacher_tcga_out_of_domain.txt
  07_mean_teacher_indomain_stronger_aug.txt
  08_gan_ssl.txt
  09_pseudo_labeling.txt
```

## Reproducing

```bash
pip install -r requirements.txt
```

Download PUMA from the Zenodo link above, arrange as `images/` (.tif) and `annotations/` (.geojson) with matching filename stems (annotations have a `_nuclei` suffix). For the TCGA extension, run `tcga_download.py` then `tcga_patch_extraction.py` (requires `openslide-tools` / `openslide-python`). Then run the remaining scripts in `code/` — each is self-contained given the shared dataset/model/loss definitions established in the baseline script.
