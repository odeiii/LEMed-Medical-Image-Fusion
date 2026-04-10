import os
import random
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from img_utils import image_read_cv2, RGB2YCrCb


def extract_patches(pet_dir, mri_dir, out_pet_dir, out_mri_dir, patch_size=128, patch_count=1642):
    """
    Randomly extract aligned patches from PET/MRI image pairs and save them to disk.

    Args:
        pet_dir:      Path to source PET images.
        mri_dir:      Path to source MRI images.
        out_pet_dir:  Output directory for PET patches.
        out_mri_dir:  Output directory for MRI patches.
        patch_size:   Side length of square patches (default 128).
        patch_count:  Total number of patch pairs to extract (default 1642).
    """
    os.makedirs(out_pet_dir, exist_ok=True)
    os.makedirs(out_mri_dir, exist_ok=True)

    img_names = sorted(os.listdir(pet_dir))
    i = 0
    pbar = tqdm(total=patch_count)

    while i < patch_count:
        img_name = random.choice(img_names)
        pet_path = os.path.join(pet_dir, img_name)
        mri_path = os.path.join(mri_dir, img_name)

        pet_img = cv2.imread(pet_path)
        mri_img = cv2.imread(mri_path)

        if pet_img is None or mri_img is None:
            print(f"Skipping {img_name} (couldn't read)")
            continue

        h, w, _ = pet_img.shape
        if h < patch_size or w < patch_size:
            print(f"Skipping {img_name} (image too small)")
            continue

        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)

        pet_patch = pet_img[y:y + patch_size, x:x + patch_size]
        mri_patch = mri_img[y:y + patch_size, x:x + patch_size]

        cv2.imwrite(os.path.join(out_pet_dir, f"{i:04d}.png"), pet_patch)
        cv2.imwrite(os.path.join(out_mri_dir, f"{i:04d}.png"), mri_patch)

        i += 1
        pbar.update(1)

    pbar.close()


class HavardDataset(Dataset):
    """
    Dataset for paired PET/MRI patches.
    Returns (pet_y, mri_y, cb, cr) tensors in YCbCr colour space.
    """

    def __init__(self, pet_dir, mri_dir, device='cpu'):
        self.pet_dir = pet_dir
        self.mri_dir = mri_dir
        self.filenames = sorted(os.listdir(pet_dir))
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        pet_path = os.path.join(self.pet_dir, self.filenames[idx])
        mri_path = os.path.join(self.mri_dir, self.filenames[idx])

        pet_img = image_read_cv2(pet_path, mode='RGB') / 255.0
        mri_img = image_read_cv2(mri_path, mode='RGB') / 255.0

        pet_tensor = torch.FloatTensor(pet_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mri_tensor = torch.FloatTensor(mri_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        pet_y, cr, cb = RGB2YCrCb(pet_tensor)
        mri_y, _, _  = RGB2YCrCb(mri_tensor)

        return pet_y.squeeze(0), mri_y.squeeze(0), cb.squeeze(0), cr.squeeze(0)


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Extract patches or verify dataset")
    parser.add_argument("--pet_dir",     required=True, help="Source PET images")
    parser.add_argument("--mri_dir",     required=True, help="Source MRI images")
    parser.add_argument("--out_pet_dir", required=True, help="Output PET patches")
    parser.add_argument("--out_mri_dir", required=True, help="Output MRI patches")
    parser.add_argument("--patch_size",  type=int, default=128)
    parser.add_argument("--patch_count", type=int, default=1642)
    args = parser.parse_args()

    extract_patches(
        args.pet_dir, args.mri_dir,
        args.out_pet_dir, args.out_mri_dir,
        args.patch_size, args.patch_count,
    )

    dataset = HavardDataset(args.out_pet_dir, args.out_mri_dir)
    loader  = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Loaded {len(dataset)} image pairs")
    for pet_y, mri_y, cb, cr in loader:
        print("Sample batch shape:", pet_y.shape, mri_y.shape)
        break
