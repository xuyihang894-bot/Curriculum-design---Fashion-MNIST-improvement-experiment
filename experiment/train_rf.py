"""Random Forest baseline.

For a fair OOD comparison, the RF sees grayscale features (784-dim flatten):
- train: original B/W training set, grayscale (foreground bright, bg dark)
- test ID: same domain, grayscale
- test OOD: colored test images converted to grayscale via standard luminance.

This means the model literally cannot see the color of the image — but the
luminance of fg/bg can still collide for some color pairs, which is the
OOD failure mode we will analyze.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data import to_grayscale_flat


def train_rf(x_train_rgb: np.ndarray, y_train: np.ndarray, *,
              n_estimators: int = 200, max_depth: Optional[int] = None,
              n_jobs: int = -1, seed: int = 0) -> RandomForestClassifier:
    x_tr = to_grayscale_flat(x_train_rgb)
    print(f"[RF] training with {x_tr.shape[0]} samples, n_estimators={n_estimators}")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs,
        random_state=seed, verbose=0,
    )
    clf.fit(x_tr, y_train)
    print(f"[RF] trained in {time.time() - t0:.1f}s")
    return clf


def predict_rf(clf: RandomForestClassifier, x_rgb: np.ndarray) -> np.ndarray:
    x_flat = to_grayscale_flat(x_rgb)
    return clf.predict(x_flat)
