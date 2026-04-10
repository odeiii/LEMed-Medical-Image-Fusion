"""
run_pipeline.py  —  LEMed end-to-end training pipeline

Runs in order:
  1. setup.py         (dependency check, device, seeds, model smoke test)
  2. img_utils        (smoke-test core functions)
  3. dataset          (patch extraction — skipped if patches already exist)
  4. losses           (smoke-test loss functions)
  5. train            (full K-Fold training loop)

Expected directory structure:
    ./data/train/pet/       raw PET training images
    ./data/train/mri/       raw MRI training images
    ./data/patches/pet/     extracted patches (auto-created)
    ./data/patches/mri/     extracted patches (auto-created)
    ./model/checkpoints/    per-fold checkpoints (auto-created)
    ./model/                final model saved here

Usage:
    python run_pipeline.py
"""

import os
import sys
import argparse
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "pet_dir":     "./data/train/pet",
    "mri_dir":     "./data/train/mri",
    "out_pet_dir": "./data/patches/pet",
    "out_mri_dir": "./data/patches/mri",
    "save_dir":    "./model/checkpoints",
    "model_dir":   "./model",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def patches_exist(out_pet_dir, out_mri_dir, patch_count):
    """Return True if both patch dirs already contain enough images."""
    if not os.path.isdir(out_pet_dir) or not os.path.isdir(out_mri_dir):
        return False
    pet_count = len([f for f in os.listdir(out_pet_dir) if f.endswith('.png')])
    mri_count = len([f for f in os.listdir(out_mri_dir) if f.endswith('.png')])
    return pet_count >= patch_count and mri_count >= patch_count


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — img_utils
# ─────────────────────────────────────────────────────────────────────────────

def stage_img_utils(device):
    section("Stage 2 — img_utils")

    from img_utils import RGB2YCrCb, YCbCr2RGB, Sobelxy, clamp

    dummy = torch.rand(1, 3, 64, 64).to(device)
    Y, Cr, Cb = RGB2YCrCb(dummy)
    rgb_back  = YCbCr2RGB(Y, Cb, Cr)

    assert rgb_back.shape == dummy.shape, "YCbCr round-trip shape mismatch"
    assert rgb_back.min() >= 0 and rgb_back.max() <= 1, "YCbCr round-trip out of [0,1]"

    sobel = Sobelxy().to(device)
    grad  = sobel(Y)
    assert grad.shape == Y.shape, "Sobelxy output shape mismatch"

    print("  ✓  RGB2YCrCb / YCbCr2RGB round-trip")
    print("  ✓  Sobelxy gradient")
    print("\nimg_utils OK.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Dataset / Patch Extraction
# ─────────────────────────────────────────────────────────────────────────────

def stage_dataset(args, device):
    section("Stage 3 — Dataset & Patch Extraction")

    from dataset import extract_patches, HavardDataset
    from torch.utils.data import DataLoader

    if patches_exist(args.out_pet_dir, args.out_mri_dir, args.patch_count):
        pet_count = len([f for f in os.listdir(args.out_pet_dir) if f.endswith('.png')])
        print(f"  Patches already exist ({pet_count} pairs found) — skipping extraction.")
    else:
        print(f"  Extracting {args.patch_count} patches (size {args.patch_size}x{args.patch_size})...")
        extract_patches(
            args.pet_dir, args.mri_dir,
            args.out_pet_dir, args.out_mri_dir,
            patch_size=args.patch_size,
            patch_count=args.patch_count,
        )
        print(f"  Patches saved to:\n    {args.out_pet_dir}\n    {args.out_mri_dir}")

    dataset = HavardDataset(args.out_pet_dir, args.out_mri_dir, device=device)
    loader  = DataLoader(dataset, batch_size=4, shuffle=True)
    pet_y, mri_y, cb, cr = next(iter(loader))
    assert pet_y.shape[1] == 1, "Expected single-channel Y"

    print(f"  ✓  Dataset loaded  ({len(dataset)} pairs)")
    print(f"  ✓  Sample batch shape: {pet_y.shape}")
    print("\nDataset OK.")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Losses
# ─────────────────────────────────────────────────────────────────────────────

def stage_losses(device):
    section("Stage 4 — Losses")

    from losses import (
        fusion_enhancement_loss,
        texture_loss_rgb_weighted,
        dual_perceptual_loss,
        compute_dwa_weights,
        total_loss_with_dwa,
    )
    from net import LEMed

    model = LEMed().to(device)
    model.eval()

    dummy_rgb    = torch.rand(1, 3, 64, 64).to(device)
    dummy_y      = torch.rand(1, 1, 64, 64).to(device)
    loss_history = {"en": [1.0, 1.0], "per": [1.0, 1.0], "tex": [1.0, 1.0]}

    with torch.no_grad():
        l_en  = fusion_enhancement_loss(dummy_rgb, dummy_rgb, dummy_y)
        l_tex = texture_loss_rgb_weighted(dummy_rgb, dummy_rgb, dummy_y)
        l_per = dual_perceptual_loss(dummy_rgb, dummy_rgb, dummy_rgb, model.content)
        w     = compute_dwa_weights(loss_history)
        total, *_ = total_loss_with_dwa(
            dummy_rgb, dummy_rgb, dummy_rgb, dummy_y, dummy_y, model.content, loss_history
        )

    print(f"  ✓  L_en  = {l_en.item():.4f}")
    print(f"  ✓  L_tex = {l_tex.item():.4f}")
    print(f"  ✓  L_per = {l_per.item():.4f}")
    print(f"  ✓  DWA weights: { {k: round(v,3) for k,v in w.items()} }")
    print(f"  ✓  L_total = {total.item():.4f}")
    print("\nLosses OK.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Train
# ─────────────────────────────────────────────────────────────────────────────

def stage_train(args):
    section("Stage 5 — Training")
    from train import train
    train(args)
    print("\nTraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEMed end-to-end training pipeline")

    parser.add_argument("--pet_dir",     default=DEFAULTS["pet_dir"],     help="Source PET images")
    parser.add_argument("--mri_dir",     default=DEFAULTS["mri_dir"],     help="Source MRI images")
    parser.add_argument("--out_pet_dir", default=DEFAULTS["out_pet_dir"], help="Output PET patch directory")
    parser.add_argument("--out_mri_dir", default=DEFAULTS["out_mri_dir"], help="Output MRI patch directory")
    parser.add_argument("--save_dir",    default=DEFAULTS["save_dir"],    help="Checkpoint output directory")
    parser.add_argument("--model_dir",   default=DEFAULTS["model_dir"],   help="Final model output directory")

    parser.add_argument("--patch_size",  type=int, default=128)
    parser.add_argument("--patch_count", type=int, default=1642)

    parser.add_argument("--num_epochs",          type=int,   default=50)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--k_folds",             type=int,   default=5)
    parser.add_argument("--early_stop_patience", type=int,   default=15)

    args = parser.parse_args()

    # Stage 1 — runs first, exits immediately if anything is missing
    from setup import setup
    device = setup()

    stage_img_utils(device)
    stage_dataset(args, device)
    stage_losses(device)
    stage_train(args)

    print("\n" + "="*60)
    print("  Pipeline finished successfully.")
    print("="*60)
