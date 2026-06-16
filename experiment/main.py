"""End-to-end experiment driver for the OOD Fashion-MNIST task.

Run from the experiment/ directory:
    python main.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from data import CLASS_NAMES, build_datasets
from train_cnn import evaluate, extract_features, train_cnn
from train_rf import predict_rf, train_rf
from visualize import (
    save_comparison_bar,
    save_confusion,
    save_misclassified,
    save_sample_grid,
    save_training_curves,
    save_tsne,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def main(epochs: int = 8, rf_estimators: int = 200, seed: int = 0):
    t_start = time.time()

    print("== Building datasets ==")
    ds = build_datasets(seed=42)

    # ---------- sample previews ----------
    save_sample_grid(ds.x_train_rgb, ds.y_train, n_per_class=4,
                     title="Training set: black bg / white fg", filename="samples_train.png")
    save_sample_grid(ds.x_test_id_rgb, ds.y_test_id, n_per_class=4,
                     title="Test (ID): same domain as train", filename="samples_test_id.png")
    save_sample_grid(ds.x_test_ood_rgb, ds.y_test_ood, n_per_class=4,
                     title="Test (OOD): random colors", filename="samples_test_ood.png")

    summary = {}

    # ---------- Random Forest ----------
    print("\n== Random Forest ==")
    rf = train_rf(ds.x_train_rgb, ds.y_train, n_estimators=rf_estimators, seed=seed)
    rf_pred_train = predict_rf(rf, ds.x_train_rgb)
    rf_pred_id    = predict_rf(rf, ds.x_test_id_rgb)
    rf_pred_ood   = predict_rf(rf, ds.x_test_ood_rgb)
    rf_acc_train = accuracy_score(ds.y_train, rf_pred_train)
    rf_acc_id    = accuracy_score(ds.y_test_id, rf_pred_id)
    rf_acc_ood   = accuracy_score(ds.y_test_ood, rf_pred_ood)
    print(f"RF train acc = {rf_acc_train:.4f}  ID acc = {rf_acc_id:.4f}  OOD acc = {rf_acc_ood:.4f}")
    save_confusion(ds.y_test_ood, rf_pred_ood, "Random Forest – OOD confusion", "cm_rf_ood.png")
    save_confusion(ds.y_test_id, rf_pred_id, "Random Forest – ID confusion", "cm_rf_id.png")
    save_misclassified(ds.x_test_ood_rgb, ds.y_test_ood, rf_pred_ood,
                       "RF misclassified on OOD", "errors_rf_ood.png")
    summary["RandomForest"] = {"train": rf_acc_train, "test_id": rf_acc_id, "test_ood": rf_acc_ood}

    # ---------- CNN baseline ----------
    print("\n== CNN baseline ==")
    cnn, hist_baseline = train_cnn(
        ds.x_train_rgb, ds.y_train,
        ds.x_test_id_rgb, ds.y_test_id,
        ds.x_test_ood_rgb, ds.y_test_ood,
        epochs=epochs, use_color_aug=False, seed=seed,
    )
    save_training_curves(hist_baseline, "CNN baseline", "curves_cnn_baseline.png")
    device = next(cnn.parameters()).device

    from torch.utils.data import DataLoader, TensorDataset
    def _loader(x, y):
        return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                          batch_size=512, shuffle=False)
    _, cnn_acc_train, _, _ = evaluate(cnn, _loader(ds.x_train_rgb, ds.y_train), str(device))
    _, cnn_acc_id, cnn_pred_id, _ = evaluate(cnn, _loader(ds.x_test_id_rgb, ds.y_test_id), str(device))
    _, cnn_acc_ood, cnn_pred_ood, _ = evaluate(cnn, _loader(ds.x_test_ood_rgb, ds.y_test_ood), str(device))
    print(f"CNN train acc = {cnn_acc_train:.4f}  ID acc = {cnn_acc_id:.4f}  OOD acc = {cnn_acc_ood:.4f}")
    save_confusion(ds.y_test_ood, cnn_pred_ood, "CNN baseline – OOD confusion", "cm_cnn_ood.png")
    save_confusion(ds.y_test_id, cnn_pred_id, "CNN baseline – ID confusion", "cm_cnn_id.png")
    save_misclassified(ds.x_test_ood_rgb, ds.y_test_ood, cnn_pred_ood,
                       "CNN baseline misclassified on OOD", "errors_cnn_ood.png")
    summary["CNN (baseline)"] = {"train": cnn_acc_train, "test_id": cnn_acc_id, "test_ood": cnn_acc_ood}

    # ---------- CNN + color augmentation ----------
    print("\n== CNN + color augmentation ==")
    cnn_aug, hist_aug = train_cnn(
        ds.x_train_rgb, ds.y_train,
        ds.x_test_id_rgb, ds.y_test_id,
        ds.x_test_ood_rgb, ds.y_test_ood,
        epochs=epochs, use_color_aug=True, seed=seed,
    )
    save_training_curves(hist_aug, "CNN + ColorAug", "curves_cnn_aug.png")
    _, aug_acc_train, _, _ = evaluate(cnn_aug, _loader(ds.x_train_rgb, ds.y_train), str(device))
    _, aug_acc_id, aug_pred_id, _ = evaluate(cnn_aug, _loader(ds.x_test_id_rgb, ds.y_test_id), str(device))
    _, aug_acc_ood, aug_pred_ood, _ = evaluate(cnn_aug, _loader(ds.x_test_ood_rgb, ds.y_test_ood), str(device))
    print(f"CNN+Aug train acc = {aug_acc_train:.4f}  ID acc = {aug_acc_id:.4f}  OOD acc = {aug_acc_ood:.4f}")
    save_confusion(ds.y_test_ood, aug_pred_ood, "CNN + ColorAug – OOD confusion", "cm_cnnaug_ood.png")
    save_misclassified(ds.x_test_ood_rgb, ds.y_test_ood, aug_pred_ood,
                       "CNN+Aug misclassified on OOD", "errors_cnnaug_ood.png")
    summary["CNN + ColorAug"] = {"train": aug_acc_train, "test_id": aug_acc_id, "test_ood": aug_acc_ood}

    # ---------- summary bar ----------
    save_comparison_bar(summary, "summary.png")

    # ---------- t-SNE feature space ----------
    print("\n== t-SNE feature visualization ==")
    feat_baseline_train = extract_features(cnn, ds.x_train_rgb[:3000], str(device))
    feat_baseline_ood   = extract_features(cnn, ds.x_test_ood_rgb[:3000], str(device))
    feat_aug_train      = extract_features(cnn_aug, ds.x_train_rgb[:3000], str(device))
    feat_aug_ood        = extract_features(cnn_aug, ds.x_test_ood_rgb[:3000], str(device))
    save_tsne(feat_baseline_train, ds.y_train[:3000], "CNN baseline – train features", "tsne_baseline_train.png")
    save_tsne(feat_baseline_ood,   ds.y_test_ood[:3000], "CNN baseline – OOD features", "tsne_baseline_ood.png")
    save_tsne(feat_aug_train,      ds.y_train[:3000], "CNN+Aug – train features", "tsne_aug_train.png")
    save_tsne(feat_aug_ood,        ds.y_test_ood[:3000], "CNN+Aug – OOD features", "tsne_aug_ood.png")

    # ---------- save metrics ----------
    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # per-class report (OOD) for the best model
    best_key = max(summary.keys(), key=lambda k: summary[k]["test_ood"])
    print(f"\nBest model on OOD: {best_key}  ({summary[best_key]['test_ood']:.4f})")
    if best_key == "CNN + ColorAug":
        best_pred = aug_pred_ood
    elif best_key == "CNN (baseline)":
        best_pred = cnn_pred_ood
    else:
        best_pred = rf_pred_ood
    report = classification_report(ds.y_test_ood, best_pred, target_names=CLASS_NAMES, digits=4)
    print(report)
    with open(os.path.join(RESULT_DIR, "best_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best model on OOD: {best_key}\n\n{report}\n")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Figures saved to: {os.path.join(os.path.dirname(__file__), 'figures')}")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="quick run for debugging")
    args = parser.parse_args()
    if args.smoke:
        main(epochs=1, rf_estimators=30, seed=args.seed)
    else:
        main(epochs=args.epochs, rf_estimators=args.rf_estimators, seed=args.seed)
