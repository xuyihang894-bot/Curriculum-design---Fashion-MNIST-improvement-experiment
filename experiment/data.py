"""Data loading and OOD construction for Fashion-MNIST OOD experiment.

Training distribution: black background + white foreground (binarized).
Test distribution: random-color background + random-color foreground,
ensuring the two colors are perceptually different.
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass

import numpy as np

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fashion")


def _load_idx(path: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte.gz")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte.gz")
    with gzip.open(labels_path, "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    return images, labels


def load_raw(data_dir: str = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw Fashion-MNIST. Returns (x_train, y_train, x_test, y_test) as uint8."""
    x_train, y_train = _load_idx(data_dir, "train")
    x_test, y_test = _load_idx(data_dir, "t10k")
    return x_train, y_train, x_test, y_test


def binarize(images: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Convert grayscale images to strict binary (0 or 255)."""
    return (images > threshold).astype(np.uint8) * 255


def make_train_bw(images: np.ndarray) -> np.ndarray:
    """Build the in-distribution training set: black background + white foreground.

    Returns float32 array of shape (N, 3, 28, 28) in [0, 1] for CNN, and the same
    info is also available as a flat 784-dim grayscale for traditional ML.
    """
    binary = binarize(images)              # (N, 28, 28) in {0,255}
    rgb = np.stack([binary, binary, binary], axis=1)  # (N, 3, 28, 28)
    return rgb.astype(np.float32) / 255.0


def _sample_color_pairs(n: int, min_distance: int = 80, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample n (bg_color, fg_color) RGB pairs with L1 distance >= min_distance."""
    rng = rng or np.random.default_rng()
    bg = rng.integers(0, 256, size=(n, 3), dtype=np.int32)
    fg = rng.integers(0, 256, size=(n, 3), dtype=np.int32)
    # Resample any pair that's too close in color
    for _ in range(8):
        dist = np.abs(bg - fg).sum(axis=1)
        too_close = dist < min_distance
        if not too_close.any():
            break
        k = int(too_close.sum())
        fg[too_close] = rng.integers(0, 256, size=(k, 3), dtype=np.int32)
    return bg.astype(np.uint8), fg.astype(np.uint8)


def make_ood_test(images: np.ndarray, seed: int = 42, min_distance: int = 80) -> np.ndarray:
    """Build OOD test set: each image gets random bg color + random fg color
    (different colors). Returns float32 (N, 3, 28, 28) in [0, 1]."""
    rng = np.random.default_rng(seed)
    binary = binarize(images)                                  # (N, 28, 28) in {0,255}
    mask = (binary > 0).astype(np.float32)[:, None, :, :]      # (N, 1, 28, 28)
    bg, fg = _sample_color_pairs(images.shape[0], min_distance, rng)
    bg = bg.astype(np.float32)[:, :, None, None] / 255.0       # (N, 3, 1, 1)
    fg = fg.astype(np.float32)[:, :, None, None] / 255.0
    out = mask * fg + (1.0 - mask) * bg                        # (N, 3, 28, 28)
    return out.astype(np.float32)


def to_grayscale_flat(images_rgb: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 28, 28) float images to (N, 784) grayscale features."""
    if images_rgb.ndim == 4 and images_rgb.shape[1] == 3:
        gray = 0.299 * images_rgb[:, 0] + 0.587 * images_rgb[:, 1] + 0.114 * images_rgb[:, 2]
    elif images_rgb.ndim == 3:
        gray = images_rgb.astype(np.float32) / 255.0 if images_rgb.dtype == np.uint8 else images_rgb
    else:
        raise ValueError(f"unexpected shape {images_rgb.shape}")
    return gray.reshape(gray.shape[0], -1)


@dataclass
class Datasets:
    x_train_rgb: np.ndarray     # (Ntr, 3, 28, 28) float32 in [0,1]
    y_train: np.ndarray
    x_test_id_rgb: np.ndarray   # in-distribution test (still B/W)
    y_test_id: np.ndarray
    x_test_ood_rgb: np.ndarray  # OOD test (random colors)
    y_test_ood: np.ndarray


def build_datasets(seed: int = 42) -> Datasets:
    x_train, y_train, x_test, y_test = load_raw()
    x_tr = make_train_bw(x_train)
    x_te_id = make_train_bw(x_test)          # in-distribution test = same domain as train
    x_te_ood = make_ood_test(x_test, seed=seed)
    return Datasets(
        x_train_rgb=x_tr,
        y_train=y_train.astype(np.int64),
        x_test_id_rgb=x_te_id,
        y_test_id=y_test.astype(np.int64),
        x_test_ood_rgb=x_te_ood,
        y_test_ood=y_test.astype(np.int64),
    )


if __name__ == "__main__":
    ds = build_datasets()
    print("train:", ds.x_train_rgb.shape, ds.x_train_rgb.dtype, "min/max:", ds.x_train_rgb.min(), ds.x_train_rgb.max())
    print("test ID:", ds.x_test_id_rgb.shape)
    print("test OOD:", ds.x_test_ood_rgb.shape, "mean:", ds.x_test_ood_rgb.mean())
