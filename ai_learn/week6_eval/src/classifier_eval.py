from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
)
from sklearn.calibration import calibration_curve

from .common import safe_div


@dataclass
class ClassifierResults:
    metrics: Dict[str, float]
    threshold: float
    threshold_curve: pd.DataFrame
    calibration: pd.DataFrame
    roc: pd.DataFrame
    confusion: Dict[str, int]


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray, objective: str) -> Tuple[float, pd.DataFrame]:
    rows = []
    best_t = 0.5
    best_val = -1e9

    # Precompute ROC if using Youden
    if objective == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        youden = tpr - fpr
        idx = int(np.argmax(youden))
        best_t = float(thr[idx])
        # Also produce grid curve for reporting consistency
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        rows.append(
            {"threshold": float(t), "f1": float(f1), "precision": float(precision), "recall": float(recall),
             "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
        )
        if objective == "f1":
            val = f1
            if val > best_val:
                best_val = val
                best_t = float(t)

    curve = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return best_t, curve


def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})


def evaluate_classifier(df: pd.DataFrame, label_col: str, prob_col: str, tune_objective: str,
                        threshold_grid: Tuple[float, float, float], calibration_bins: int) -> ClassifierResults:
    y_true = df[label_col].astype(int).to_numpy()
    y_prob = df[prob_col].astype(float).to_numpy()

    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_prob)

    start, stop, step = threshold_grid
    grid = np.arange(start, stop + 1e-9, step)

    best_t, curve = tune_threshold(y_true, y_prob, grid, tune_objective)

    y_pred = (y_prob >= best_t).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    calib = compute_calibration(y_true, y_prob, n_bins=calibration_bins)

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})

    metrics = {
        "f1_at_best_threshold": float(f1),
        "precision_at_best_threshold": float(precision),
        "recall_at_best_threshold": float(recall),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
    }
    confusion = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    return ClassifierResults(
        metrics=metrics,
        threshold=float(best_t),
        threshold_curve=curve,
        calibration=calib,
        roc=roc_df,
        confusion=confusion,
    )


def save_classifier_plots(res: ClassifierResults, out_dir: str, tag: str) -> Dict[str, str]:
    paths = {}

    # ROC
    plt.figure()
    plt.plot(res.roc["fpr"], res.roc["tpr"])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC ({tag})")
    roc_path = f"{out_dir}/roc_{tag}.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=180)
    plt.close()
    paths["roc"] = roc_path

    # Calibration
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(res.calibration["mean_pred"], res.calibration["frac_pos"])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration ({tag})")
    cal_path = f"{out_dir}/calibration_{tag}.png"
    plt.tight_layout()
    plt.savefig(cal_path, dpi=180)
    plt.close()
    paths["calibration"] = cal_path

    # Threshold curve (F1)
    plt.figure()
    plt.plot(res.threshold_curve["threshold"], res.threshold_curve["f1"])
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title(f"Threshold tuning ({tag})")
    thr_path = f"{out_dir}/threshold_f1_{tag}.png"
    plt.tight_layout()
    plt.savefig(thr_path, dpi=180)
    plt.close()
    paths["threshold_f1"] = thr_path

    return paths
