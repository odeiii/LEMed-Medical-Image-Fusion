import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

from net import LEMed
from dataset import HavardDataset
from losses import total_loss_with_dwa
from img_utils import RGB2YCrCb, YCbCr2RGB


DEFAULTS = {
    "pet_dir":  "./data/patches/pet",
    "mri_dir":  "./data/patches/mri",
    "save_dir": "./model/checkpoints",
    "model_dir":"./model",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_log(log_path, message):
    with open(log_path, 'a') as f:
        f.write(message + '\n')
    print(message)


def train(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = HavardDataset(args.pet_dir, args.mri_dir, device=device)
    print(f"Loaded {len(dataset)} image pairs")

    model = LEMed().to(device)
    model.train()

    loss_history = {
        "en":  [1.0, 1.0],
        "per": [1.0, 1.0],
        "tex": [1.0, 1.0],
    }

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── Training log ─────────────────────────────────────────────────────────
    log_path = os.path.join(args.save_dir, "training_log.txt")
    run_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'w') as f:
        f.write(f"LEMed Training Log\n")
        f.write(f"{'='*60}\n")
        f.write(f"Run started : {run_start}\n")
        f.write(f"Device      : {device}\n")
        f.write(f"Dataset     : {len(dataset)} pairs\n")
        f.write(f"Epochs      : {args.num_epochs}\n")
        f.write(f"LR          : {args.lr}\n")
        f.write(f"K-Folds     : {args.k_folds}\n")
        f.write(f"Early stop  : {args.early_stop_patience}\n")
        f.write(f"{'='*60}\n\n")

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        fold_num = fold + 1
        write_log(log_path, f"\n----- Fold {fold_num}/{args.k_folds} -----")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=1, shuffle=False)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_val_loss      = float('inf')
        early_stop_counter = 0
        best_epoch         = -1

        for epoch in range(args.num_epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Fold {fold_num} | Epoch {epoch + 1}/{args.num_epochs}")

            for pet_y, mri_y, cb, cr in pbar:
                pet_y, mri_y, cb, cr = (
                    pet_y.to(device), mri_y.to(device),
                    cb.to(device),    cr.to(device),
                )

                output_y  = model.fuser(pet_y, mri_y)
                fused_rgb = YCbCr2RGB(output_y, cb, cr)
                pet_rgb   = YCbCr2RGB(pet_y, cb, cr)
                mri_rgb   = YCbCr2RGB(mri_y, cb, cr)

                total_loss, loss_en, loss_per, loss_tex, weights = total_loss_with_dwa(
                    fused_rgb, pet_rgb, mri_rgb, cb, cr, model.content, loss_history
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                pbar.set_postfix({
                    "Total":  f"{total_loss.item():.4f}",
                    "L_en":   f"{loss_en.item():.4f}",
                    "L_per":  f"{loss_per.item():.4f}",
                    "L_tex":  f"{loss_tex.item():.4f}",
                })

            scheduler.step()

            loss_history['en'].append(loss_en.item())
            loss_history['per'].append(loss_per.item())
            loss_history['tex'].append(loss_tex.item())
            for key in loss_history:
                loss_history[key] = loss_history[key][-2:]

            # Validation
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for pet_y, mri_y, cb, cr in val_loader:
                    pet_y, mri_y, cb, cr = (
                        pet_y.to(device), mri_y.to(device),
                        cb.to(device),    cr.to(device),
                    )
                    output_y  = model.fuser(pet_y, mri_y)
                    fused_rgb = YCbCr2RGB(output_y, cb, cr)
                    pet_rgb   = YCbCr2RGB(pet_y,    cb, cr)
                    mri_rgb   = YCbCr2RGB(mri_y,    cb, cr)
                    loss, *_  = total_loss_with_dwa(
                        fused_rgb, pet_rgb, mri_rgb, cb, cr, model.content, loss_history
                    )
                    val_loss_total += loss.item()

            val_loss_avg = val_loss_total / len(val_loader)

            # Log every epoch
            epoch_log = (
                f"  Epoch {epoch + 1:03d}/{args.num_epochs}"
                f"  |  train: {running_loss / len(train_loader):.4f}"
                f"  |  val: {val_loss_avg:.4f}"
            )
            write_log(log_path, epoch_log)

            if val_loss_avg < best_val_loss:
                best_val_loss      = val_loss_avg
                best_epoch         = epoch + 1
                early_stop_counter = 0
                best_path = os.path.join(args.save_dir, f"LEMed_fold{fold_num:02d}.pth")
                torch.save({'model': model.state_dict()}, best_path)
                write_log(log_path, f"  Checkpoint saved  (epoch {best_epoch}, val loss {best_val_loss:.4f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_patience:
                    write_log(log_path, f"  Early stopping at epoch {epoch + 1}")
                    break

            model.train()

        write_log(log_path, f"  Best checkpoint: epoch {best_epoch}  |  val loss {best_val_loss:.4f}")

    final_path = os.path.join(args.model_dir, "LEMed.pth")
    torch.save({'model': model.state_dict()}, final_path)

    run_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_log(log_path, f"\n{'='*60}")
    write_log(log_path, f"Final model saved to: {final_path}")
    write_log(log_path, f"Run ended : {run_end}")
    write_log(log_path, f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LEMed")
    parser.add_argument("--pet_dir",             default=DEFAULTS["pet_dir"],   help="PET patch directory")
    parser.add_argument("--mri_dir",             default=DEFAULTS["mri_dir"],   help="MRI patch directory")
    parser.add_argument("--save_dir",            default=DEFAULTS["save_dir"],  help="Checkpoint output directory")
    parser.add_argument("--model_dir",           default=DEFAULTS["model_dir"], help="Final model output directory")
    parser.add_argument("--num_epochs",          type=int,   default=50)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--k_folds",             type=int,   default=5)
    parser.add_argument("--early_stop_patience", type=int,   default=15)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    train(args)