"""Visualization helpers: sample grids, confusion matrices, t-SNE."""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from data import CLASS_NAMES

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _to_hwc(img: np.ndarray) -> np.ndarray:
    """Take a (3, H, W) or (H, W) float image and return HxWxC in [0,1]."""
    if img.ndim == 3 and img.shape[0] == 3:
        return np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    return img


def save_sample_grid(images: np.ndarray, labels: np.ndarray, n_per_class: int = 4,
                     title: str = "samples", filename: str = "samples.png") -> str:
    n_classes = 10
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 1.4, n_classes * 1.4))
    for c in range(n_classes):
        idx = np.where(labels == c)[0][:n_per_class]
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(np.clip(_to_hwc(images[i]), 0, 1))
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(CLASS_NAMES[c], fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def save_tsne(features: np.ndarray, labels: np.ndarray, title: str, filename: str,
              n_samples: int = 2000, seed: int = 0) -> str:
    if features.shape[0] > n_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(features.shape[0], n_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed, max_iter=500)
    emb = tsne.fit_transform(features.astype(np.float32))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    palette = sns.color_palette("tab10", 10)
    for c in range(10):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6, color=palette[c], label=CLASS_NAMES[c])
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_curves(history: dict, title: str, filename: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train", marker="o")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val", marker="s")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title(f"{title} – loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train", marker="o")
    if "val_acc_id" in history:
        axes[1].plot(epochs, history["val_acc_id"], label="val(ID)", marker="s")
    if "val_acc_ood" in history:
        axes[1].plot(epochs, history["val_acc_ood"], label="val(OOD)", marker="^")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].set_title(f"{title} – accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def save_comparison_bar(results: dict, filename: str = "summary.png") -> str:
    """results: {method_name: {"train": acc, "test_id": acc, "test_ood": acc}}"""
    methods = list(results.keys())
    metrics = ["train", "test_id", "test_ood"]
    metric_labels = ["Train", "Test (ID, B/W)", "Test (OOD, color)"]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        vals = [results[k][m] for k in methods]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=metric_labels[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Method comparison: train vs in-distribution vs OOD")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def save_misclassified(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                        title: str, filename: str, n: int = 24) -> str:
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        return ""
    rng = np.random.default_rng(0)
    pick = rng.choice(wrong, size=min(n, wrong.size), replace=False)
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    for ax, i in zip(np.atleast_1d(axes).ravel(), pick):
        ax.imshow(np.clip(_to_hwc(images[i]), 0, 1))
        ax.set_title(f"T:{CLASS_NAMES[y_true[i]][:6]}\nP:{CLASS_NAMES[y_pred[i]][:6]}", fontsize=7)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
