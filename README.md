# LEMed - Medical Image Fusion

LEMed is a deep learning model for fusing paired medical images (e.g. PET+MRI, SPECT+MRI) using a learned fusion network trained with a multi-component loss.

> This work is currently under review. A thesis submission is also in progress. Please do not redistribute or build upon this code without permission.

---

## Repository Structure

```
LEMed/
├── net.py              # Model architecture (LEMed + Unet_fuser)
├── vgg.py              # PerceptualLossVgg16ExDark (required by net.py)
├── img_utils.py        # Colour-space conversions, filters, Sobel operator
├── dataset.py          # Patch extraction + HavardDataset loader
├── losses.py           # L_en, L_tex, L_per, DWA weighting, total loss
├── train.py            # K-Fold training loop
├── test.py             # Inference (auto-aligns mismatched image sizes)
├── setup.py            # Dependency + file check, device/seed config
└── run_pipeline.py     # End-to-end pipeline runner
```

---

## Directory Convention

The pipeline reads from and writes to fixed directories - no arguments needed:

```
LEMed/
├── data/
│   ├── train/
│   │   ├── pet/            ← raw PET training images (.png)
│   │   └── mri/            ← raw MRI training images (.png)
│   ├── patches/
│   │   ├── pet/            ← extracted patches (auto-created)
│   │   └── mri/            ← (auto-created)
│   └── test/
│       ├── pet/            ← test PET images (.png)
│       └── mri/            ← test MRI images (.png)
├── model/
│   ├── checkpoints/        ← per-fold .pth files + training_log.txt (auto-created)
│   └── LEMed_final.pth     ← final model (auto-created)
└── results/                ← fused output images (auto-created by test.py)
```

Filenames in `pet/` and `mri/` must match exactly (`001.png` in both) as pairs are matched by filename.

> **Note:** `data/`, `model/checkpoints/`, and `results/` are excluded from version control (see `.gitignore`). Datasets must be sourced separately. The final model weights (`model/*.pth`) are included in the repo.

---

## Environment

**Python:** 3.10

| Package | Version | Notes |
|---|---|---|
| torch | 2.11.0+cu130 | CUDA 13.0 build |
| torchvision | 0.26.0+cu130 | |
| numpy | 1.26.4 | |
| opencv-python | 4.13.0.92 | imported as `cv2` |
| pillow | 12.1.1 | imported as `PIL` |
| scikit-image | 0.25.2 | imported as `skimage` |
| scikit-learn | 1.7.2 | imported as `sklearn` |
| matplotlib | 3.10.8 | |
| einops | 0.8.2 | |
| tqdm | 4.67.3 | |

Install all at once:
```bash
pip install torch torchvision numpy opencv-python pillow scikit-image scikit-learn matplotlib einops tqdm
```

> If you're on a different CUDA version, get the right torch build from https://pytorch.org/get-started/locally

---

## Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

Patch extraction is skipped automatically if `./data/patches/` already contains enough patches from a previous run.

### Override defaults

All paths and hyperparameters can be overridden at runtime even though defaults are set:

```bash
python run_pipeline.py --lr 1e-4 --num_epochs 30 --k_folds 3
python test.py --ckpt_path ./model/checkpoints/LEMed_best_fold02.pth
```

### Run setup check only

```bash
python setup.py
```

### Extract patches only

```bash
python dataset.py \
    --pet_dir     ./data/train/pet \
    --mri_dir     ./data/train/mri \
    --out_pet_dir ./data/patches/pet \
    --out_mri_dir ./data/patches/mri
```

### Train only

```bash
python train.py
```

### Test / Inference

```bash
python test.py
```

By default, test.py loads ./model/LEMed_final.pth. To use a specific checkpoint instead (such as a per-fold best or a model trained on a different modality), pass it via --ckpt_path

```bash
python test.py --ckpt_path ./model/checkpoints/LEMed_fold02.pth
```

If a PET/MRI pair has mismatched sizes, the smaller image is automatically upscaled to match the larger one before fusion.

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `patch_size` | 128 | Side length of extracted training patches (px) |
| `patch_count` | 1642 | Number of patch pairs to extract |
| `lr` | 2e-4 | Adam learning rate |
| `k_folds` | 5 | Number of K-Fold cross-validation splits |
| `num_epochs` | 50 | Max epochs per fold |
| `early_stop_patience` | 15 | Epochs without val improvement before stopping |

---

## Model Output

The fused output images are RGB `.png` files. Fusion is performed in YCbCr colour space (only the Y (luminance) channel is passed through the network; the Cb and Cr chrominance channels are carried over from the PET image and recombined with the fused Y before saving). This preserves the colour information of the PET while fusing the structural content of both modalities in luminance.

---

## Loss Functions

| Loss | Description |
|---|---|
| L_en | Enhancement loss: promotes detail via histogram equalization of max/min channels |
| L_tex | Texture loss: per-channel Sobel gradient matching against source images |
| L_per | Dual perceptual loss: VGG feature matching against both PET and MRI |
| L_total | Weighted sum via Dynamic Weight Averaging (DWA); L_en boosted ×10 |

---

## Pretrained Weights

Pretrained models are available in the [Releases](https://github.com/odeiii/LEMed-Medical-Image-Fusion/releases/tag/ptw) section:

| File | Modality |
|---|---|
| `LEMed_pet.pth` | PET-MRI fusion |
| `LEMed_ct.pth` | CT-MRI fusion |

**To use:**
1. Download the `.pth` file(s) from the release page
2. Create a `model/` folder in the project root if it doesn't exist
3. Place the downloaded `.pth` file(s) inside it
4. Run inference pointing to the relevant file:

```bash
# PET-MRI
python test.py --ckpt_path ./model/LEMed_pet.pth

# CT-MRI
python test.py --ckpt_path ./model/LEMed_ct.pth
```

---

## Pipeline Smoke Tests

Before training starts, the pipeline runs quick sanity checks on Stages 2–4 using dummy tensors not real images. This is just to confirm everything loads and runs without errors before committing to a full training run.

| Stage | What it checks | What to expect |
|---|---|---|
| Stage 2: img_utils | YCbCr round-trip and Sobelxy gradient on a random tensor | Output shape matches input, values stay in [0, 1] |
| Stage 4: L_en | Enhancement loss on two identical random RGB tensors | Small non-zero value |
| Stage 4: L_tex | Texture loss on two identical random RGB tensors | Non-zero value |
| Stage 4: L_per | Perceptual loss on two identical random tensors | **Always 0.0000** .. VGG features are identical when both inputs are the same tensor, so L1 difference is zero. This is expected and not a bug. |
| Stage 4: L_total | Combined weighted loss | Non-zero value driven by L_en and L_tex |

These values have no bearing on actual training performance. Real loss values during Stage 5 will look very different since PET and MRI images are genuinely different.

---

## Training Log

Each training run writes `./model/checkpoints/training_log.txt` with a full record of every epoch's train and validation loss, which epoch was saved for each fold, and early stopping events. Example:

```
LEMed Training Log
============================================================
Run started : 2025-05-23 14:02:11
Device      : cuda
Dataset     : 1642 pairs
Epochs      : 50
LR          : 0.0002
K-Folds     : 5
Early stop  : 15
============================================================

----- Fold 1/5 -----
  Epoch 001/50  |  train: 0.8421  |  val: 0.7913
  ✓ Checkpoint saved  (epoch 1, val loss 0.7913)
  Epoch 002/50  |  train: 0.7102  |  val: 0.6887
  ✓ Checkpoint saved  (epoch 2, val loss 0.6887)
  ...
  Best checkpoint: epoch 18  |  val loss 0.3214
```

---


## Citation

This code accompanies a paper currently under review and a thesis in progress. If you use this work, please check back for the full citation once published.

```bibtex
@article{lemedFusion,
  title   = {LEMed: A Low-Light Enhancement Inspired Fusion Network for Multi-Modal Medical Imaging},
  author  = {Boadu, Kwasi Odei and Jin, Qi. and Odeh, Victor Adeyi, and Audu, David Ocho and Akwaboah, Christopher},
  note = {Under Review at Biomedical Signal Processing and Control},
  year    = {2026*}
}
```

---

## .gitignore

Add the following to your `.gitignore` to avoid pushing datasets, weights, and results to GitHub:

```
data/
model/
results/
__pycache__/
*.pyc
*.pth
```

---
