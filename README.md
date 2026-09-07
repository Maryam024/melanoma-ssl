# Semi-Supervised Melanoma Nuclei Segmentation

A research study investigating whether **semi-supervised learning (SSL) can improve nuclei segmentation in melanoma histopathology when only a small number of expert-labeled images are available**.

The project evaluates multiple SSL strategies on H&E-stained melanoma tissue and compares them against carefully matched supervised baselines. The study focuses not only on performance improvements, but also on understanding **when SSL helps, when it fails, and why**.

The work is motivated by research on computer-aided melanoma analysis based on nuclei segmentation, including work by **Akbarpour et al. (2025)** and related research from the University of Alberta.

---

## Research Question

Expert annotation of histopathological images is expensive and time-consuming, while large collections of medical images may remain unlabeled.

This project investigates:

> **Can unlabeled histopathology images provide useful training signal for nuclei segmentation when only a small labeled set is available?**

Four different SSL mechanisms were implemented and evaluated:

* **Mean Teacher** — consistency regularization
* **GAN-based SSL** — adversarial learning
* **Autoencoder pretraining** — unsupervised representation learning
* **Pseudo-labeling** — confidence-based self-training

Each method was compared against a **supervised-only baseline using the same labeled subset**.

---

## Dataset

The primary experiments use the public **PUMA (Panoptic segmentation of nUclei and tissue in MelanomA)** dataset.

* 205 melanoma H&E regions of interest
* 103 primary melanoma ROIs
* 103 metastatic melanoma ROIs
* 1024 × 1024 resolution
* 40× magnification
* Expert nuclei annotations

The dataset is **not included in this repository**. It can be obtained from the official [PUMA dataset](https://puma.grand-challenge.org/dataset/) and its [Zenodo record](https://zenodo.org/records/14869398).

PUMA serves as a public experimental dataset for validating the methodology. It is not identical to the private clinical datasets used in the target research setting, so the reported results should be interpreted as **methodological evidence rather than expected performance on a clinical dataset**.

### Out-of-domain extension

To investigate whether domain diversity affects SSL, additional unlabeled tissue patches were extracted from **TCGA-SKCM** whole-slide images.

* 8 TCGA-SKCM whole-slide images
* 320 tissue patches
* 1024 × 1024 patches
* Tissue-content filtering
* Downloaded through the GDC API

This extension was designed to test whether **genuinely different unlabeled data** provides a more useful consistency signal than additional images from the same distribution.

---

## Experimental Design

### 1. Full-data supervised baseline

A U-Net was trained using all 205 annotated PUMA ROIs.

**Best validation Dice: 0.9038**

Training augmentation improved this to:

**Dice: 0.9093**

Test-time augmentation did not provide an additional improvement:

**Dice: 0.9072**

This establishes a strong supervised reference before evaluating SSL.

### 2. Limited-label experiments

Two label-scarcity settings were evaluated:

* **15 labeled images + 165 unlabeled images**
* **40 labeled images + 140 unlabeled images**

For each setting, a supervised-only model was trained using only the labeled subset. SSL methods then received the same labeled data plus the corresponding unlabeled images.

This design allows the benefit of unlabeled data to be evaluated more directly.

---

## Methods

### Mean Teacher

A student U-Net is trained using:

* supervised segmentation loss on labeled images
* consistency loss between the student and an EMA teacher on unlabeled images

Two additional factors were investigated:

* strength of consistency-branch augmentation
* source of unlabeled data (in-domain PUMA vs. out-of-domain TCGA-SKCM)

### GAN-based SSL

Following the adversarial semi-supervised segmentation framework of Hung et al. (2018), a discriminator distinguishes plausible image-mask pairs from generated predictions.

The segmentation network combines supervised segmentation loss with a small adversarial objective.

### Autoencoder Pretraining

The encoder was first pretrained using image reconstruction without requiring segmentation masks.

The pretrained encoder was then transferred to the segmentation network and fine-tuned using the limited labeled set.

Two optimization strategies were tested:

* single learning rate
* differential learning rates for pretrained encoder and randomly initialized decoder

### Pseudo-labeling

A segmentation model was first trained using the labeled subset.

Predictions on unlabeled images were retained only when they satisfied a high-confidence criterion. The selected pseudo-labels were then used for a second training stage.

At 15 labeled images, the confidence criterion selected **0 of 165 unlabeled images**, preventing a meaningful second-stage experiment.

---

# Results

## 15 labeled images

| Method                                        | Validation Dice |          Difference |
| --------------------------------------------- | --------------: | ------------------: |
| **Supervised-only**                           |      **0.8702** |                   — |
| **GAN-SSL**                                   |      **0.8693** |             -0.0009 |
| Mean Teacher — in-domain, mild augmentation   |          0.8661 |             -0.0041 |
| Mean Teacher — TCGA, strong augmentation      |          0.8454 |             -0.0248 |
| Autoencoder — differential LR                 |          0.8521 |             -0.0181 |
| Mean Teacher — in-domain, strong augmentation |          0.8323 |             -0.0379 |
| Autoencoder — single LR                       |          0.7692 |             -0.1010 |
| Pseudo-labeling                               |             N/A | 0 confident samples |

## 40 labeled images

| Method              | Validation Dice | Difference |
| ------------------- | --------------: | ---------: |
| **Supervised-only** |      **0.8903** |          — |
| Mean Teacher        |          0.8883 |    -0.0020 |
| GAN-SSL             |          0.8872 |    -0.0031 |

---

# Key Findings

### 1. SSL did not automatically outperform supervised learning

None of the evaluated SSL approaches produced a clear improvement over the matched supervised-only baseline.

This is an important result rather than simply a failure: **the usefulness of unlabeled histopathology data depends strongly on the SSL mechanism, label availability, augmentation strategy, and data distribution.**

### 2. GAN-based SSL was the most competitive under extreme label scarcity

With only **15 labeled images**, GAN-SSL achieved a Dice score of **0.8693**, compared with **0.8702** for supervised-only training.

The difference was only **0.0009**, making adversarial SSL the closest approach to the supervised baseline in this experiment.

At 40 labeled images, however, GAN-SSL no longer provided an advantage:

* Supervised-only: 0.8903
* Mean Teacher: 0.8883
* GAN-SSL: 0.8872

This suggests that any potential advantage of SSL may become less pronounced as the number of labeled examples increases.

### 3. Stronger consistency augmentation was not necessarily beneficial

For Mean Teacher with 15 labeled images, increasing the consistency-branch augmentation substantially reduced performance:

* Mild augmentation: **0.8661**
* Strong augmentation: **0.8323**

With very limited labeled supervision, the stronger perturbation may have made the consistency objective too difficult for the model to satisfy reliably.

### 4. Domain diversity improved Mean Teacher relative to the stronger in-domain setup

Replacing PUMA unlabeled images with **320 TCGA-SKCM tissue patches** increased the Mean Teacher result from:

**0.8323 → 0.8454**

The result remained below the supervised baseline, but the improvement provides evidence that **unlabeled data from a different domain can provide a different and potentially useful consistency signal**.

### 5. Autoencoder pretraining was sensitive to optimization

Autoencoder pretraining initially produced a large performance gap:

**0.7692 Dice**

Using differential learning rates for the pretrained encoder and randomly initialized decoder improved performance to:

**0.8521 Dice**

This suggests that optimization and the mismatch between pretrained and randomly initialized components can substantially influence the effectiveness of unsupervised pretraining.

### 6. Pseudo-labeling reached a label-count limitation

With only 15 labeled images, the initial segmentation model did not produce sufficiently confident predictions under the selected filtering criterion.

As a result, **0/165 unlabeled images** were accepted as pseudo-labels.

Rather than artificially relaxing the threshold to obtain a result, the experiment was recorded as a negative finding. This highlights a practical limitation of self-training when the initial model is too weak to generate reliable pseudo-labels.

---

# Interpretation

The experiments suggest that **semi-supervised learning should not be treated as a universally beneficial replacement for supervised learning** in histopathological segmentation.

The strongest observation was not a large performance gain, but the different behavior of the SSL mechanisms under the same label constraints.

* Adversarial learning remained close to supervised performance at extreme label scarcity.
* Mean Teacher was sensitive to augmentation strength and data distribution.
* Out-of-domain unlabeled data improved Mean Teacher relative to a stronger in-domain perturbation setup, although it did not surpass the supervised baseline.
* Autoencoder pretraining benefited substantially from appropriate optimization.
* Pseudo-labeling failed to bootstrap effectively at the lowest label count.

These findings motivate further investigation using **larger and genuinely diverse unlabeled clinical datasets**, controlled magnification/scale normalization, stronger experimental replication, and more systematic ablation studies.

---

# Limitations

Several limitations should be considered when interpreting these results.

### Dataset scale

The primary experiments use 205 PUMA ROIs. This is sufficient for controlled experimentation but limited for drawing broad conclusions about clinical deployment.

### Domain mismatch

PUMA and TCGA-SKCM may differ in scanner characteristics, staining, tissue preparation, patient population, and image scale.

### TCGA magnification

The TCGA patches were extracted without explicitly confirming that their microns-per-pixel resolution matched the 40× PUMA images. Differences in apparent nuclear scale therefore remain a potential confounding factor.

### Statistical robustness

The current experiments primarily report validation Dice from individual training runs. Multiple random seeds and statistical testing would strengthen the comparison.

### Public vs. clinical data

PUMA is used as a public methodological proxy. Performance on this dataset should not be interpreted as equivalent to performance on a private clinical melanoma dataset.

---

# Reproducibility

The repository contains the training and evaluation scripts used for the experiments together with the recorded results.

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
├── 01_baseline_full_supervised.txt
├── 02_scarce_label_comparison_15.txt
├── 03_scarce_label_comparison_40.txt
├── 04_full_metrics.txt
├── 05_augmentation_and_tta.txt
├── 06_mean_teacher_tcga_out_of_domain.txt
├── 07_mean_teacher_indomain_stronger_aug.txt
├── 08_gan_ssl.txt
└── 09_pseudo_labeling.txt
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

Download the PUMA dataset from the official sources and arrange the images and annotations according to the expected directory structure.

For the TCGA extension:

```bash
python code/tcga_download.py
python code/tcga_patch_extraction.py
```

The remaining scripts contain the individual training and evaluation procedures used in the experiments.

---

# Related Work

* **Akbarpour et al. (2025)** — *Deep Learning-Based Nuclei Segmentation and Melanoma Detection...*; primary reference motivating the nuclei-segmentation approach.
* **Alheejawi et al. (2021)** — foundational work connecting nuclei segmentation with melanoma-region analysis.
* **Yu et al. (2021)** — semi-supervised learning using Mean Teacher-style consistency regularization.
* **Hung et al. (2018)** — adversarial learning for semi-supervised semantic segmentation.
* **Requa et al. (2023)** — supervised/semi-supervised approaches for skin neoplasm detection.

---

## Research Direction

This project serves as a methodological investigation into **label-efficient learning for melanoma histopathology**.

The results motivate future work toward models that can better exploit large quantities of heterogeneous, unlabeled histopathological data while remaining robust to differences in staining, acquisition, magnification, and clinical domain.

The central objective is not simply to obtain a higher benchmark score, but to understand **how unlabeled medical images can be converted into reliable learning signal under realistic annotation constraints**.
