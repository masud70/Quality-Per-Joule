# ============================================================
# Quality per Joule — CelebA Data Pipeline
# data/celeba_loader.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================

import torch
import os as _os
import numpy as np
import shutil as _shutil
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config.config import DATA_CONFIG as CONFIG

# Copying data to /tmp can significantly speed up training on some systems
def _setup_fast_data_dir(src_dir: str) -> str:
    """
    Copies CelebA data to /tmp for faster I/O if not already there.
    Falls back to the original directory if /tmp copy fails.
    """
    fast_dir = "/tmp/celeba_cache"
    if _os.path.isdir(fast_dir) and _os.listdir(fast_dir):
        print(f"  Data cache      : {fast_dir} (already cached)")
        return fast_dir
    if not _os.path.isdir(src_dir):
        return src_dir
    try:
        print(f"  Copying data to /tmp for faster I/O ... ", end="", flush=True)
        _shutil.copytree(src_dir, fast_dir)
        print(f"done ({fast_dir})")
        return fast_dir
    except Exception as e:
        print(f"failed ({e}), using original path")
        return src_dir

DATA_DIR = _setup_fast_data_dir(CONFIG["data_dir"])


# Random seed setup for reproducibility
def set_seed(seed: int = CONFIG["seed"]):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# Transforms: CenterCrop → Resize → (Optional) RandomHorizontalFlip → ToTensor → Normalize
def get_transforms(augment: bool = False) -> transforms.Compose:
    transform_list = [
        transforms.CenterCrop(178),
        transforms.Resize(CONFIG["image_size"]),
    ]

    if augment:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ]
    return transforms.Compose(transform_list)


# Data Loaders
def get_train_loader(
    batch_size  : int  = CONFIG["batch_size"],
    augment     : bool = False,
    num_workers : int  = CONFIG["num_workers"],
    seed        : int  = CONFIG["seed"],
) -> DataLoader:
    set_seed(seed)
    dataset = datasets.CelebA(
        root=DATA_DIR,
        split="train",
        target_type="attr",
        download=False,
        transform=get_transforms(augment=augment)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=lambda _: np.random.seed(seed),
    )

# Test data loader
def get_test_loader(
    batch_size  : int = CONFIG["batch_size"],
    num_workers : int = CONFIG["num_workers"],
    seed        : int = CONFIG["seed"],
) -> DataLoader:
    """
    CelebA official test split (~20K images).
    Used for AE/VAE reconstruction error evaluation.
    """
    set_seed(seed)
    dataset = datasets.CelebA(
        root=DATA_DIR,
        split="test",
        target_type="attr",
        download=False,
        transform=get_transforms(augment=False)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

# FID reference data loader
def get_fid_reference_loader(
    n_samples   : int = CONFIG["n_samples"],
    batch_size  : int = CONFIG["batch_size"],
    num_workers : int = CONFIG["num_workers"],
    seed        : int = CONFIG["seed"],
) -> DataLoader:
    set_seed(seed)
    dataset = datasets.CelebA(
        root=DATA_DIR,
        split="valid",
        target_type="attr",
        download=False,
        transform=get_transforms(augment=False)
    )
    n_samples = min(n_samples, len(dataset))
    indices   = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(seed)
    )[:n_samples]
    subset = Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# Utility: Denormalize [-1,1] → [0,1] for visualization
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverses [-1,1] normalization to [0,1] for visualization."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


# Utility: Visualize a batch of images in a grid
def visualize_batch(
    loader    : DataLoader,
    n         : int  = 16,
    title     : str  = "CelebA Sample Batch (32x32)",
    save_path : str  = None,
):
    """Displays a grid of n images from any loader."""
    images, _ = next(iter(loader))
    images = denormalize(images[:n])
    grid   = make_grid(images, nrow=int(n ** 0.5), padding=2)
    np_img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(np_img)
    plt.title(title, fontsize=13)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
####################### Sanity Check #########################
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  CelebA Data Pipeline — Sanity Check")
    print("=" * 55)

    train_loader = get_train_loader(batch_size=64,  augment=False)
    test_loader  = get_test_loader(batch_size=64)
    fid_loader   = get_fid_reference_loader(n_samples=10000)

    print(f"\n  Train images  : {len(train_loader.dataset):,}")
    print(f"  Test images   : {len(test_loader.dataset):,}")
    print(f"  FID ref images: {len(fid_loader.dataset):,}")
    print(f"  Train batches : {len(train_loader)}")

    # Inspect a batch
    images, attrs = next(iter(train_loader))
    print(f"\n  Batch shape   : {images.shape}  (expected: [64, 3, 32, 32])")
    print(f"  Attrs shape   : {attrs.shape}   (40 binary attributes per image)")
    print(f"  Pixel min/max : {images.min():.3f} / {images.max():.3f}"
          f"  (expected: ~-1.0 / ~1.0)")
    print(f"  Dtype         : {images.dtype}")

    assert images.shape == (64, 3, 32, 32), "Unexpected batch shape!"
    assert images.min() >= -1.0 and images.max() <= 1.0, "Pixel range error!"
    print(f"\n  Shape check   : PASS")
    print(f"  Range check   : PASS")

    # Determinism check on FID loader
    imgs_a, _ = next(iter(get_fid_reference_loader(seed=42)))
    imgs_b, _ = next(iter(get_fid_reference_loader(seed=42)))
    assert torch.equal(imgs_a, imgs_b), "FID loader is not deterministic!"
    print(f"  FID determinism: PASS")

    print(f"\n  All checks passed. CelebA pipeline is ready.")
    print("=" * 55)

    # Optional: visualize
    visualize_batch(train_loader, n=16, save_path="results/celeba_sample.png")