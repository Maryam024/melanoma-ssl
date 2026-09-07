# Semi-Supervised Melanoma Nuclei Segmentation

Research project investigating **semi-supervised learning (SSL) for nuclei segmentation in H&E-stained melanoma histopathology images under limited annotation**.

The project evaluates four SSL approaches—**Mean Teacher, GAN-based SSL, autoencoder pretraining, and pseudo-labeling**—against supervised-only baselines at different levels of label scarcity.

## Motivation

Expert annotation of histopathology images is expensive. This project investigates whether unlabeled images can provide useful training signal when only a small labeled set is available.

The study is motivated by recent work on **melanoma detection through nuclei segmentation**, particularly Akbarpour et al. (2025).

## Dataset

Experiments primarily use the **PUMA (Panoptic segmentation of nUclei and tissue in MelanomA)** dataset:

* 205 melanoma H&E ROIs
* 1024×1024 resolution
* 40× magnification
* Expert nuclei annotations

The dataset is not included in this repository. It can be obtained from the [PUMA dataset](https://puma.grand-challenge.org/dataset/) and [Zenodo](https://zenodo.org/records/14869398).

An additional out-of-domain experiment uses **TCGA-SKCM** whole-slide images. Eight slides were used to extract **320 tissue patches** for unlabeled training.

## Methods

* **Supervised U-Net:** full-data and scarce-label baselines
* **Mean Teacher:** consistency regularization with in-domain PUMA and out-of-domain TCGA unlabeled data
* **GAN-SSL:** adversarial semi-supervised segmentation
* **Autoencoder pretraining:** reconstruction-based encoder pretraining with single and differential learning rates
* **Pseudo-labeling:** confidence-thresholded self-training
* **Data augmentation and TTA:** evaluated for the full-data baseline

## Results

### Full-data supervised baseline

| Experiment                    | Validation Dice |
| ----------------------------- | --------------: |
| Supervised U-Net (205 images) |      **0.9038** |
| + Data augmentation           |      **0.9093** |
| + Test-time augmentation      |      **0.9072** |

### 15 labeled images

| Method                              |   Val Dice |            vs. supervised |
| ----------------------------------- | ---------: | ------------------------: |
| **Supervised-only**                 | **0.8702** |                         — |
| **GAN-SSL**                         | **0.8693** |               **-0.0009** |
| Mean Teacher, in-domain, mild aug   |     0.8661 |                   -0.0041 |
| Mean Teacher, TCGA, strong aug      |     0.8454 |                   -0.0248 |
| Autoencoder, differential LR        |     0.8521 |                   -0.0181 |
| Mean Teacher, in-domain, strong aug |     0.8323 |                   -0.0379 |
| Autoencoder, single LR              |     0.7692 |                   -0.1010 |
| Pseudo-labeling                     |        N/A | 0 confident pseudo-labels |

### 40 labeled images

| Method              |   Val Dice | vs. supervised |
| ------------------- | ---------: | -------------: |
| **Supervised-only** | **0.8903** |              — |
| Mean Teacher        |     0.8883 |        -0.0020 |
| GAN-SSL             |     0.8872 |        -0.0031 |

## Key Findings

* No SSL method clearly outperformed the corresponding supervised-only baseline.
* **GAN-SSL was the closest to supervised performance at 15 labeled images**, with only a 0.0009 Dice difference.
* Stronger Mean Teacher augmentation reduced performance at 15 labels.
* Replacing in-domain PUMA unlabeled data with TCGA-SKCM data improved Mean Teacher from **0.8323 to 0.8454**, although it remained below the supervised baseline.
* Differential learning rates substantially improved autoencoder-pretrained segmentation compared with a single learning rate.
* Pseudo-labeling produced **0 confident pseudo-labels from 165 unlabeled images** at the 15-label setting.

These results suggest that the effectiveness of SSL depends strongly on the **learning mechanism, label availability, augmentation strategy, and domain of the unlabeled data**.

## Repository Structure

```text
code/
├── baseline_nuclei_segmentation.py
├── kaggle_baseline_nuclei_segmentation.py
├── baseline_full_augmented.py
├── supervised_only_comparison.py
├── mean_teacher_ssl.py
├── mean_teacher_tcga.py
├── gan_ssl.py
├── pseudo_labeling_ssl.py
├── autoencoder_pretrain_ssl.py
├── autoencoder_diff_lr.py
├── augmented_dataset.py
├── tta_evaluation.py
├── full_metrics_evaluation.py
├── tcga_download.py
└── tcga_patch_extraction.py

results/
└── experiment logs (.txt)
```

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the PUMA dataset from the official sources and arrange the images and GeoJSON annotations according to the project structure.

For the TCGA extension, run:

```bash
python code/tcga_download.py
python code/tcga_patch_extraction.py
```

Individual training and evaluation scripts are provided in `code/`, with corresponding experiment logs in `results/`.

## Limitations

* PUMA is a public dataset and is not identical to a private clinical dataset.
* TCGA and PUMA may differ in staining, scanner characteristics, patient population, and image scale.
* TCGA patch magnification/microns-per-pixel was not explicitly normalized to PUMA.
* Current comparisons are primarily based on individual training runs; multiple seeds would provide stronger statistical evidence.

## Related Work

* Akbarpour et al. (2025) — nuclei segmentation and melanoma detection
* Alheejawi et al. (2021) — nuclei segmentation for melanoma analysis
* Yu et al. (2021) — Mean Teacher / semi-supervised learning
* Hung et al. (2018) — adversarial semi-supervised segmentation
* Requa et al. (2023) — semi-supervised skin-neoplasm detection
