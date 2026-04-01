# ============================================================
# Quality per Joule — Training Utilities
# utils/training.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================
# Shared utilities for all training scripts:
#   - print_device_info()    : prints CPU/GPU hardware info
#   - print_training_config(): prints config, params, device
#   - save_sample_grid()     : saves 16 images in 2 rows of 8
# ============================================================

import matplotlib
matplotlib.use('Agg')

import os
import platform
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Prints CPU and GPU hardware info at the start of training.
def print_device_info(device: torch.device):
    """Prints CPU and GPU hardware info at the start of training."""
    print("\n" + "=" * 55)
    print("  Hardware Configuration")
    print("=" * 55)

    # CPU info
    cpu_name = platform.processor() or "Unknown"
    cpu_count = os.cpu_count() or 0
    print(f"  CPU             : {cpu_name}")
    print(f"  CPU cores       : {cpu_count}")

    # GPU info
    if device.type == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
        print(f"  GPU             : {name}")
        print(f"  GPU memory      : {mem:.1f} GB")
        print(f"  CUDA version    : {torch.version.cuda}")
    else:
        print(f"  GPU             : None (using CPU)")

    print(f"  PyTorch version : {torch.__version__}")
    print(f"  Device          : {device}")
    print("=" * 55)

# Prints training configuration, total params, and device info at the start of training.
def print_training_config(config: dict, total_params: int, device: torch.device):
    """Prints device info and training configuration at the start of training."""
    print_device_info(device)

    print("\n" + "=" * 55)
    print(f"  Training Configuration — {config.get('model_name', 'Unknown').upper()}")
    print("=" * 55)

    # Print config entries (skip nested dicts)
    for key, val in config.items():
        if isinstance(val, (str, int, float, bool)):
            print(f"  {key:<18}: {val}")

    print(f"  {'total_params':<18}: {total_params:,}")
    print("=" * 55 + "\n")

# Saves a grid of generated images (2 rows × 8 columns by default).
def save_sample_grid(
    images: torch.Tensor,
    path: str,
    title: str = "Generated Samples",
    n: int = 16,
    nrow: int = 8,
    denorm_fn=None,
):
    imgs = images[:n].cpu()
    if denorm_fn is not None:
        imgs = denorm_fn(imgs)
    imgs = imgs.clamp(0, 1)

    grid = make_grid(imgs, nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(nrow * 2, 2 * 2))
    ax.imshow(np_img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
