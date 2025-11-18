# Skin Cancer Detection using CNNs (HAM10000 Dataset)

This repository contains a complete end-to-end implementation of a **skin lesion classification pipeline** trained and evaluated on the **HAM10000** dataset.
It includes **full EDA**, **preprocessing**, **multiple model trials**, and two major high-quality training phases:

* **Phase 4:** EfficientNet-based retraining (baseline advanced model)
* **Phase 5:** Improved, stable, balanced retraining with preprocessing, targeted augmentation, and focal loss

Going forward, only the **notebook** and the **.pth model files** will be committed, as requested.

---

# Repository Contents

* `skin-caner-mnist-ham.ipynb` — Complete notebook (EDA, preprocessing, models, Phase 4 and Phase 5 training)
* `phase5_best_model.pth` — Best Phase 5 model checkpoint
* `requirements.txt` — Dependencies
* `README.md` — This file

(Additional scripts, checkpoints, and artifacts are intentionally **not included**.)

---

# PHASE 4 — Baseline Advanced Model (EfficientNet-B0)

Phase 4 establishes the first strong baseline using EfficientNet-B0 with advanced augmentations.

## Stage 1 — Initial EDA

* Class distribution and imbalance visualization
* Basic statistical summaries
* Sample visualization per class

## Stage 2 — Preprocessing

* Image resolution consistency checks
* Custom preprocessing: hair removal (blackhat inpainting) + CLAHE

## Stage 3 — Metadata Analysis

* Exploration of age, sex, and localization fields
* Verification of no data leakage

## Stage 4 — Data Splitting

* Stratified **train/val/test** split
* Patient-aware splitting recommended

## Stage 5 — Augmentation

* Horizontal/vertical flips
* Rotations
* ColorJitter
* CutMix and MixUp (integrated into training loop)

## Stage 6 — Model Experiments

* Custom CNNs
* ResNet18
* EfficientNet (final choice for Phase 4)

## Stage 7 — Final Phase 4 Training

* EfficientNet-B0 pretrained
* WeightedRandomSampler
* CutMix + MixUp
* AdamW + Cosine Annealing
* Mixed precision
* EMA
* Grad-CAM extraction

### Key Dataset Stats

| Label | Count |
| ----: | ----: |
|    nv |  4693 |
|   mel |   779 |
|   bkl |   769 |
|   bcc |   360 |
| akiec |   229 |
|  vasc |    99 |
|    df |    81 |

Severe imbalance affects macro-F1 significantly.

### Final Phase 4 Results

* Best macro-F1: **0.1617**
* EMA macro-F1: 0.0891
* Accuracy: ~11%

Phase 4 therefore provides a baseline, but suffers from high imbalance sensitivity.

---

# PHASE 5 — Balanced Retrain (Improved Pipeline)

Phase 5 includes major quality and stability improvements over Phase 4.

## Objectives of Phase 5

* Improve generalization
* Directly address class imbalance
* Use a stable backbone (after EfficientNet & ConvNeXt GPU failures)
* Strengthen augmentation for rare classes
* Introduce preprocessing before augmentation

## Improvements Introduced

### 1. Preprocessing Pipeline

* Hair removal using blackhat + inpainting
* CLAHE normalization

### 2. Targeted Augmentation Strategy

Rare classes receive strong augmentation:

* Rotations
* Color jitter
* RandomAffine
* RandomResizedCrop
* Translations

Common class `nv` uses light augmentation.

### 3. Class Rebalancing

* WeightedRandomSampler
* Focal Loss with per-class alpha (inverse frequency)

### 4. Stable Backbone

EfficientNet and ConvNeXt repeatedly crashed with CUDA misaligned memory errors.

Final backbone: **MobileNetV3-Large**, chosen because it is:

* Lightweight
* GPU-stable
* High-speed
* Performs well with balanced training

### 5. Improved Training Setup

* Optimizer: AdamW
* Scheduler: CosineAnnealingWarmRestarts
* Mixed Precision Training (AMP)
* Batch size: 48
* 40 epochs

---

# Phase 5 Results

## Best Checkpoint (Epoch 32)

* Train F1: **0.708**
* Validation F1: **0.410** (best achieved so far)
* Loss: **0.0002**

This is a **2.5x improvement** over Phase 4.

## Observations

* Validation F1 stabilizes and improves steadily
* Preprocessing + focal loss + rebalanced augmentation significantly help rare classes
* MobileNetV3-Large provides reliable performance and convergence
* Phase 5 establishes a strong foundation for future work

---

# What to Keep in the Repository

Since only the **notebook** and **pth model files** will be committed, keep:

* `skin-caner-mnist-ham.ipynb` (full source of truth)
* `phase5_best_model.pth`
* `requirements.txt`
* `README.md`

Do **not** include:

* Checkpoints folder
* Very large Grad-CAM exports
* TensorBoard logs

---

# Recommended Minimal Repo Structure

```
ham10000/
│
├── skin-caner-mnist-ham.ipynb
├── phase5_best_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Future Directions

Phase 6 (next phase) will add:

* Offline augmented dataset generation for extreme imbalance
* Metadata fusion via MLP
* Multi-backbone ensemble (ConvNeXt, EfficientNet-B3, MobileNetV3)
* Class-balanced focal loss
* TTA + ensemble soft voting
* Larger, deeper backbones where GPU allows

---

# License & Attribution

If publishing the repository, include:

* MIT License
* Citation to the HAM10000 dataset creators

---

This README now fully describes both **Phase 4** and **Phase 5**, and matches the repository format you will push to Git. If you want a short version or a GitHub-optimized version, I can generate it as well.
