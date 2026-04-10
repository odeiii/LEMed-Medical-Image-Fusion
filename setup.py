"""
setup.py  —  LEMed environment setup & dependency check

Order of checks:
  1. Required .py files are present in the same directory
  2. All packages are importable (with exact pip install names)
  3. torch.distributed patched for single-machine use
  4. Device + seeds configured
  5. LEMed forward-pass smoke test

Import and call setup() from run_pipeline.py as the very first step.
"""

import sys
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1. File presence check
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FILES = [
    "vgg.py",
    "net.py",
    "img_utils.py",
    "dataset.py",
    "losses.py",
    "train.py",
    "test.py",
]

def check_files():
    here = os.path.dirname(os.path.abspath(__file__))
    missing = []

    for fname in REQUIRED_FILES:
        fpath = os.path.join(here, fname)
        if os.path.isfile(fpath):
            print(f"  ✓  {fname}")
        else:
            print(f"  ✗  {fname}  ← not found")
            missing.append(fname)

    if missing:
        print(f"\n{'─'*60}")
        print(f"  {len(missing)} missing file(s):")
        for f in missing:
            print(f"    - {f}")
        print("\n  Make sure all pipeline files are in the same directory as run_pipeline.py.")
        sys.exit(1)

    print("\n  All required files present.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dependency check
# ─────────────────────────────────────────────────────────────────────────────

# (import_name, pip_install_name)  — kept separate since they often differ
REQUIRED_PACKAGES = [
    ("torch",       "torch"),
    ("torchvision", "torchvision"),
    ("cv2",         "opencv-python"),
    ("numpy",       "numpy"),
    ("matplotlib",  "matplotlib"),
    ("PIL",         "pillow"),
    ("skimage",     "scikit-image"),
    ("sklearn",     "scikit-learn"),
    ("tqdm",        "tqdm"),
    ("einops",      "einops"),
    ("numbers",     None),              # stdlib, no pip install needed
]

def check_dependencies():
    missing = []

    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            print(f"  ✓  {import_name}")
        except ImportError:
            install_name = pip_name or import_name
            print(f"  ✗  {import_name}  ← not installed  (pip install {install_name})")
            missing.append(install_name)

    if missing:
        print(f"\n{'─'*60}")
        print(f"  {len(missing)} missing package(s). Install with:")
        print(f"\n    pip install {' '.join(missing)}\n")
        sys.exit(1)

    print("\n  All dependencies satisfied.")


# ─────────────────────────────────────────────────────────────────────────────
# Main setup()
# ─────────────────────────────────────────────────────────────────────────────

def setup():
    """
    Run full environment setup. Returns the active torch.device.
    Exits with a clear message at the first sign of trouble.
    """

    # ── file check ───────────────────────────────────────────────────────────
    print("="*60)
    print("  Checking required files")
    print("="*60)
    check_files()

    # ── dependency check ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Checking dependencies")
    print("="*60)
    check_dependencies()

    # ── safe to import now ───────────────────────────────────────────────────
    import random
    import numpy as np
    import torch
    import torch.distributed as dist

    # Patch distributed — not needed for single-machine training
    # and can cause hangs/errors on some environments
    dist.is_available = lambda: False

    # ── device ───────────────────────────────────────────────────────────────
    print()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── seeds ────────────────────────────────────────────────────────────────
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Seed   : {seed}")

    # ── model smoke test ─────────────────────────────────────────────────────
    print("\n  Loading LEMed model...")
    try:
        from net import LEMed
        model = LEMed().to(device)
        model.eval()
        dummy = torch.rand(1, 1, 64, 64).to(device)
        with torch.no_grad():
            out = model(dummy, dummy)
        assert out.shape == dummy.shape, f"Unexpected output shape: {out.shape}"
        print(f"  ✓  LEMed loaded  (forward pass: {list(dummy.shape)} → {list(out.shape)})")
        del model, dummy, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗  LEMed failed to load: {e}")
        sys.exit(1)

    print("\n  Setup complete.")
    print("="*60)

    return device


if __name__ == "__main__":
    setup()
