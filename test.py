import os
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from net import LEMed
from img_utils import image_read_cv2, RGB2YCrCb, YCbCr2RGB


DEFAULTS = {
    "ckpt_path": "./model/LEMed_final.pth",
    "pet_path":  "./data/test/pet",
    "mri_path":  "./data/test/mri",
    "out_path":  "./results",
}


def align_to_larger(pet, mri):
    """
    If PET and MRI are different sizes, resize the smaller one up to match
    the larger. If they're already the same size, no-op.
    """
    ph, pw = pet.shape[:2]
    mh, mw = mri.shape[:2]

    if (ph, pw) == (mh, mw):
        return pet, mri, None

    th = max(ph, mh)
    tw = max(pw, mw)
    target = (tw, th)

    note = f"PET {pw}x{ph} / MRI {mw}x{mh} → {tw}x{th}"

    if (ph, pw) != (th, tw):
        pet = cv2.resize(pet, target, interpolation=cv2.INTER_LINEAR)
    if (mh, mw) != (th, tw):
        mri = cv2.resize(mri, target, interpolation=cv2.INTER_LINEAR)

    return pet, mri, note


def test(ckpt_path, pet_path, mri_path, out_path):
    """
    Run inference on aligned PET/MRI pairs and save fused RGB outputs.
    If a pair has mismatched sizes the smaller image is upscaled to the
    larger one's dimensions before fusion.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    fuser = LEMed().to(device)
    fuser.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    fuser.eval()
    print(f"Model loaded from {ckpt_path}")

    os.makedirs(out_path, exist_ok=True)

    pet_images = sorted([f for f in os.listdir(pet_path) if f.lower().endswith('.png')])
    mri_images = sorted([f for f in os.listdir(mri_path) if f.lower().endswith('.png')])

    if set(pet_images) != set(mri_images):
        print("Warning: PET and MRI folders don't have matching filenames")
        common = set(pet_images) & set(mri_images)
        pet_images = sorted(common)
        print(f"Using {len(common)} common files")

    print(f"\nProcessing {len(pet_images)} image pairs...")
    print("-" * 60)

    with torch.no_grad():
        for idx, img_name in enumerate(pet_images):
            pet = image_read_cv2(os.path.join(pet_path, img_name), mode='RGB')
            mri = image_read_cv2(os.path.join(mri_path, img_name), mode='RGB')

            pet, mri, resize_note = align_to_larger(pet, mri)

            pet_t = torch.FloatTensor(np.transpose(pet[np.newaxis] / 255.0, (0, 3, 1, 2))).to(device)
            mri_t = torch.FloatTensor(np.transpose(mri[np.newaxis] / 255.0, (0, 3, 1, 2))).to(device)

            pet_y, cr, cb = RGB2YCrCb(pet_t)
            mri_y, _,  _  = RGB2YCrCb(mri_t)

            out = fuser(pet_y, mri_y)
            out_min, out_max = out.min(), out.max()
            out = (out - out_min) / (out_max - out_min) if out_max > out_min else out - out_min

            out = YCbCr2RGB(out, cb, cr)
            out = (out * 255.0).cpu().numpy().squeeze(0).astype('uint8')
            out = np.transpose(out, (1, 2, 0))
            Image.fromarray(out).save(os.path.join(out_path, img_name))

            if (idx + 1) % 10 == 0 or idx == 0:
                note = f"  ↑ resized: {resize_note}" if resize_note else ""
                print(f"  [{idx + 1}/{len(pet_images)}] {img_name}{note}")

    print(f"\nDone! Fused images saved to: {out_path}")
    print(f"Total processed: {len(pet_images)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LEMed fusion model")
    parser.add_argument("--ckpt_path", default=DEFAULTS["ckpt_path"], help="Path to checkpoint .pth file")
    parser.add_argument("--pet_path",  default=DEFAULTS["pet_path"],  help="Directory of PET images")
    parser.add_argument("--mri_path",  default=DEFAULTS["mri_path"],  help="Directory of MRI images")
    parser.add_argument("--out_path",  default=DEFAULTS["out_path"],  help="Output directory for fused images")
    args = parser.parse_args()

    test(args.ckpt_path, args.pet_path, args.mri_path, args.out_path)
