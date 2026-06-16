"""Train a small CNN baseline and a color-augmented version.

Both share the same model architecture and hyperparameters; the only
difference is whether `random_color_recolor` is applied to each mini-batch.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from augment import random_color_recolor
from models import SmallCNN

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def make_loaders(x_train, y_train, x_test_id, y_test_id, x_test_ood, y_test_ood,
                  batch_size: int = 256):
    def loader(x, y, shuffle):
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return (
        loader(x_train, y_train, True),
        loader(x_test_id, y_test_id, False),
        loader(x_test_ood, y_test_ood, False),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    preds_all = []
    labels_all = []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total_n += y.size(0)
        preds_all.append(preds.cpu().numpy())
        labels_all.append(y.cpu().numpy())
    return total_loss / total_n, correct / total_n, np.concatenate(preds_all), np.concatenate(labels_all)


@torch.no_grad()
def extract_features(model: SmallCNN, x: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    feats = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        z = model.embed(xb)
        feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


def train_cnn(x_train, y_train, x_test_id, y_test_id, x_test_ood, y_test_ood, *,
              epochs: int = 8, lr: float = 1e-3, batch_size: int = 256,
              use_color_aug: bool = False, seed: int = 0,
              device: Optional[str] = None) -> tuple[SmallCNN, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, id_loader, ood_loader = make_loaders(
        x_train, y_train, x_test_id, y_test_id, x_test_ood, y_test_ood, batch_size)
    model = SmallCNN(num_classes=10).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_acc_id": [], "val_acc_ood": []}

    tag = "CNN+ColorAug" if use_color_aug else "CNN"
    print(f"[{tag}] device={device} epochs={epochs} lr={lr} batch={batch_size}")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        running_correct = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            if use_color_aug:
                x = random_color_recolor(x, min_distance=0.30, p=1.0)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += float(loss.item()) * y.size(0)
            running_n += y.size(0)
            running_correct += int((logits.argmax(1) == y).sum().item())
        scheduler.step()
        tr_loss = running_loss / running_n
        tr_acc = running_correct / running_n

        _, id_acc, _, _ = evaluate(model, id_loader, device)
        _, ood_acc, _, _ = evaluate(model, ood_loader, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc_id"].append(id_acc)
        history["val_acc_ood"].append(ood_acc)
        print(f"  ep{ep:02d} loss={tr_loss:.4f} train_acc={tr_acc:.4f}  ID={id_acc:.4f}  OOD={ood_acc:.4f}")

    print(f"[{tag}] total {time.time() - t0:.1f}s")
    return model, history
