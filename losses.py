import math
import torch
import torch.nn.functional as F

from img_utils import Sobelxy, RGB2YCrCb, histogram_equalization


# ---------------------------------------------------------------------------
# ℒ_en  —  Enhancement Loss
# ---------------------------------------------------------------------------

def fusion_enhancement_loss(fused_rgb, pet_rgb, mri_img):
    """
    ℒ_en = L1( max_channel(fused), max(hist_eq(max_channel(PET)), MRI) )
         + 0.8 * L1( min_channel(fused), min(hist_eq(min_channel(PET)), MRI) )
    """
    # MAX branch
    pet_max, _ = torch.max(pet_rgb, dim=1, keepdim=True)
    pet_max_eq  = histogram_equalization(pet_max)
    fused_max, _ = torch.max(fused_rgb, dim=1, keepdim=True)
    loss_max = F.l1_loss(fused_max, torch.max(pet_max_eq, mri_img))

    # MIN branch
    pet_min, _ = torch.min(pet_rgb, dim=1, keepdim=True)
    pet_min_eq  = histogram_equalization(pet_min)
    fused_min, _ = torch.min(fused_rgb, dim=1, keepdim=True)
    loss_min = F.l1_loss(fused_min, torch.min(pet_min_eq, mri_img))

    return loss_max + 0.8 * loss_min


# ---------------------------------------------------------------------------
# ℒ_tex  —  Texture Loss
# ---------------------------------------------------------------------------

def texture_loss_rgb_weighted(fused_rgb, pet_rgb, mri_img, lambda_g=1, lambda_b=1):
    """
    ℒ_tex = ℒ_tex^R + λ₁·ℒ_tex^G + λ₂·ℒ_tex^B
    Each channel's gradient is computed separately and weighted accordingly.
    """
    sobel = Sobelxy().to(fused_rgb.device)

    fused_r = fused_rgb[:, 0:1]; fused_g = fused_rgb[:, 1:2]; fused_b = fused_rgb[:, 2:3]
    pet_r   = pet_rgb[:,   0:1]; pet_g   = pet_rgb[:,   1:2]; pet_b   = pet_rgb[:,   2:3]

    grad_mri = sobel(mri_img.expand_as(pet_r))

    loss_r = F.l1_loss(sobel(fused_r), torch.max(sobel(pet_r), grad_mri))
    loss_g = F.l1_loss(sobel(fused_g), torch.max(sobel(pet_g), grad_mri))
    loss_b = F.l1_loss(sobel(fused_b), torch.max(sobel(pet_b), grad_mri))

    return loss_r + lambda_g * loss_g + lambda_b * loss_b


# ---------------------------------------------------------------------------
# ℒ_per  —  Dual Perceptual Loss
# ---------------------------------------------------------------------------

def dual_perceptual_loss(fused_rgb, pet_rgb, mri_rgb, vgg_loss_fn, lambda_mri=1):
    """
    ℒ_per = VGG(fused, PET) + λ·VGG(fused, MRI)
    """
    return vgg_loss_fn(fused_rgb, pet_rgb) + lambda_mri * vgg_loss_fn(fused_rgb, mri_rgb)


# ---------------------------------------------------------------------------
# Dynamic Weight Averaging (DWA)
# ---------------------------------------------------------------------------

def compute_dwa_weights(loss_hist, T=2, K=3):
    """
    Compute per-task weights using Dynamic Weight Averaging.

    Args:
        loss_hist: dict with keys 'en', 'per', 'tex', each a list of at least
                   2 recent loss values (oldest first).
        T:         DWA temperature (default 2).
        K:         Number of tasks — scales the softmax output (default 3).

    Returns:
        dict of weights with same keys as loss_hist.
    """
    wk = {
        key: loss_hist[key][-1] / (loss_hist[key][-2] + 1e-8)
        for key in loss_hist
    }
    exp_terms = {k: math.exp(v / T) for k, v in wk.items()}
    total = sum(exp_terms.values())
    return {k: K * exp_terms[k] / total for k in exp_terms}


# ---------------------------------------------------------------------------
# ℒ_total  —  Combined Loss with DWA
# ---------------------------------------------------------------------------

def total_loss_with_dwa(fused_rgb, pet_rgb, mri_rgb, cb, cr, vgg_loss_fn, loss_history):
    """
    Final combined loss:
      ℒ_total = w_en·ℒ_en + w_per·ℒ_per + w_tex·ℒ_tex
    Weights are computed via DWA; ℒ_en weight is boosted ×10.
    """
    mri_y_only, _, _ = RGB2YCrCb(mri_rgb)

    loss_en = fusion_enhancement_loss(fused_rgb, pet_rgb, mri_y_only)

    # Ensure 3-channel inputs for VGG
    if mri_rgb.shape[1] == 1:
        mri_rgb = mri_rgb.repeat(1, 3, 1, 1)
    if pet_rgb.shape[1] == 1:
        pet_rgb = pet_rgb.repeat(1, 3, 1, 1)

    loss_per = dual_perceptual_loss(fused_rgb, pet_rgb, mri_rgb, vgg_loss_fn, lambda_mri=1)
    loss_tex = texture_loss_rgb_weighted(fused_rgb, pet_rgb, mri_y_only, lambda_g=1, lambda_b=1)

    weights = compute_dwa_weights(loss_history)
    weights['en'] *= 10  # Boost ℒ_en

    total = (
        weights['en']  * loss_en  +
        weights['per'] * loss_per +
        weights['tex'] * loss_tex
    )

    return total, loss_en, loss_per, loss_tex, weights
