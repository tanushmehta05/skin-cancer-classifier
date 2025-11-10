# 🩺 Skin Cancer Detection using CNNs (HAM10000 Dataset)

This repository contains a complete end-to-end implementation of a **skin lesion classification pipeline** using the **HAM10000** dataset.
It includes **extensive Exploratory Data Analysis (EDA)**, multiple model experiments, and a final robust retraining phase (Phase 4) using EfficientNet-B0 with advanced augmentations and training tricks.

## Repository contents (summary)
- `skin-caner-mnist-ham.ipynb` — Full notebook containing EDA, preprocessing, modeling experiments, Phase 4 retraining and evaluation (source of truth).
- `phase4_ham10000_retrain.py` — (optional) Extracted training script version of the final Phase 4 pipeline (not included automatically; notebook contains the runnable script cells).
- `requirements.txt` — Python dependencies used across the notebook.
- `README.md` — This file.
- `phase4_checkpoints/` — Checkpoints, predictions CSV and Grad-CAM exports created by training (large files; add to `.gitignore` by default).

## Quick summary of the workflow (what the notebook covers)
1. **STAGE 1 — INITIAL EDA**
   - Class distribution, sample counts and basic statistics.
   - Visualizations of class imbalance and per-class sample examples.

2. **STAGE 2 — IMAGE QUALITY & PREPROCESSING**
   - Image size/shape checks.
   - Custom preprocessing pipeline applied before transforms:
     - Morphological blackhat inpainting for occlusions
     - CLAHE on L channel in LAB color space for contrast normalization
   - Functions: `preprocess_image_cv`, demo pipeline on samples.

3. **STAGE 3 — METADATA ANALYSIS & LEAKAGE CHECKS**
   - Checked `age`, `sex`, `localization` correlations with labels.
   - Ensured patient-level splitting to avoid leakage (images from same lesion/patient across splits).

4. **STAGE 4 — DATA SPLITTING & SAMPLING**
   - Stratified train/val/test split (patient-aware recommended).
   - WeightedRandomSampler used in training to help imbalance.

5. **STAGE 5 — AUGMENTATION & TRANSFORMS**
   - Basic augmentations: horizontal/vertical flips, rotations, ColorJitter.
   - Advanced mixing: CutMix and MixUp implemented in training loop (Phase 4).

6. **STAGE 6 — MODEL TRIALS**
   - Baseline custom CNNs, ResNet18 transfer learning experiments, then EfficientNet (timm/torchvision).
   - Training strategies: freeze-head → fine-tune, optimizer sweeps, scheduler (ReduceLROnPlateau).

7. **STAGE 7 — PHASE 4: FINAL ROBUST RETRAIN (the final script)**
   - EfficientNet-B0 (ImageNet pretrained)
   - WeightedRandomSampler + class weights
   - CutMix + MixUp augmentations with configurable probabilities
   - Mixed precision (torch.amp), EMA weight tracking
   - Checkpointing (best, last, best_ema)
   - TensorBoard logging and simple Grad-CAM export (one sample per class)
   - Early stopping based on validation macro-F1

## Key dataset statistics (from the notebook)
| Label | Count |
|-------:|------:|
| nv    | 4693 |
| mel   | 779  |
| bkl   | 769  |
| bcc   | 360  |
| akiec | 229  |
| vasc  | 99   |
| df    | 81   |

> **Severe class imbalance** — `nv` dominates (~46% of samples). This impacts macro-F1 heavily; minority classes (e.g., `df`, `vasc`) have very few samples.

## Final Phase 4 results (as reported in the notebook)
- Device: 2× GPUs used (DataParallel)
- Best model test macro-F1: **0.1617**
- EMA model test macro-F1: **0.0891**
- Test accuracy (best model): ~11% (dataset-level, weighted metrics show skew to majority class)
- Classification reports are saved to the notebook outputs (also exported to `phase4_checkpoints/test_predictions_best.csv`).

## Observations & Findings
- The preprocessing pipeline (inpainting + CLAHE) helps with illumination and artifact removal.
- WeightedRandomSampler and class-weighted loss help but do not fully solve minority-class poor recall.
- CutMix and MixUp provided modest gains in validation F1 during the run but require careful hyperparameter tuning per class.
- EMA weights did not outperform the best checkpoint in this run (can vary per run).

## Recommendations — what to keep in the repo
Keep the following files and code cells (required for reproducibility and further work):
- The full notebook `skin-caner-mnist-ham.ipynb` (source of truth; keep all EDA and final training cells).
- `requirements.txt` capturing the environment used.
- `phase4_checkpoints/` **(do not commit large model files to GitHub).** Keep locally or store on cloud storage; include a small example or pointer in README.
- Utility functions and modules:
  - `preprocess_image_cv` and preprocessing demo cells.
  - Data split & patient-leakage check cells.
  - Training loop cells that include CutMix/MixUp/EMA logic (Phase 4).
  - Evaluation and Grad-CAM export cells.
- Example inference cell demonstrating how to load `best_macroF1.pth` and run a sample prediction (keep, but remove heavy checkpoint references or make them optional).

## Recommendations — what to remove or keep out of Git
- **Remove large binary artifacts** from the repo (or add to `.gitignore`):
  - `*.pth`, `*.pt`, `/phase4_checkpoints/*` (unless you want to upload small sample weights)
  - TensorBoard logs (`/tb_logs/*`)
  - Large export images (>5MB) — keep low-res sample or link externally
- **Sensitive or local-only paths**: Replace absolute Kaggle paths in the notebook with relative or configurable `DATA_DIR` variables.
- **System-specific commands**: Any Kaggle-only magic (e.g., `%tensorboard`) should be made conditional or converted to shell-safe calls.
- **Verbose prints** and intermediate large DataFrame displays: keep a few key summary tables/figures; collapse redundant outputs to keep notebook compact.

## Suggested repo structure (cleaned for GitHub)
```
skin-cancer-mnist-ham/
│
├── notebooks/
│   └── skin-caner-mnist-ham.ipynb         # full EDA + experiments + Phase 4
├── src/
│   ├── data.py                            # data loaders, preprocessors
│   ├── train.py                           # Phase 4 training script (extracted)
│   ├── models.py                          # model creation helpers (EfficientNet wrapper)
│   ├── utils.py                           # CutMix, MixUp, EMA, metrics, Grad-CAM functions
│   └── inference.py                       # inference + simple Gradio demo
├── requirements.txt
├── README.md
├── .gitignore
└── examples/
    └── small_sample_images/               # tiny sample images for demo (optional)
```

## Next steps (future work — summarize in README)
- **Address class imbalance more aggressively**:
  - Focal loss, class-balanced loss, or re-sampling strategies targeted per-class
  - Oversampling minority classes using strong augmentation (AutoAugment / RandAugment)
  - Semi-supervised learning (pseudo-labeling) using unlabeled clinical images
  - Few-shot learning tricks or metric-learning approaches for very small classes
- **Model upgrades**:
  - Try EfficientNet-B3 / B4 or ConvNeXt / Swin-Tiny for improved capacity
  - Use per-class learning rate or loss scaling for rare classes
- **Evaluation & deployment**:
  - Use patient-level cross-validation to better estimate generalization
  - Deploy a lightweight Gradio demo or HuggingFace Space for quick manual checks

---
## License & Attribution
If you plan to release, add a LICENSE (MIT recommended for research prototypes) and cite the HAM10000 dataset source.

---
---
---