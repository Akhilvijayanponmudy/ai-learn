import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import joblib

DATA = Path("data/generated/classifier.csv")
OUT_METRICS = Path("data/generated/classifier_metrics.json")
OUT_MODEL = Path("data/generated/classifier_model.joblib")

def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    # probs: (n, C), y_true: (n,)
    # ECE computed on max-prob confidence vs correctness
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i < n_bins - 1 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(acc - avg_conf)
    return float(ece)

def tune_thresholds_ovr(y_true: np.ndarray, probs: np.ndarray, class_names):
    # One-vs-rest threshold tuning for best F1 per class
    thresholds = {}
    tuned_f1 = {}
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    for c, name in enumerate(class_names):
        best_t = 0.5
        best_f = -1
        p = probs[:, c]
        yt = y_bin[:, c]
        for t in np.linspace(0.05, 0.95, 19):
            yp = (p >= t).astype(int)
            f = f1_score(yt, yp, zero_division=0)
            if f > best_f:
                best_f = f
                best_t = float(t)
        thresholds[name] = best_t
        tuned_f1[name] = float(best_f)
    return thresholds, tuned_f1

def main():
    df = pd.read_csv(DATA)
    X = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    class_names = sorted(df["label"].unique().tolist())
    y = np.array([class_names.index(l) for l in labels], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None))
    ])
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    weighted_f1 = float(f1_score(y_test, preds, average="weighted"))

    # ROC-AUC OvR
    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
    roc_auc_ovr = float(roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr"))

    ece = expected_calibration_error(probs, y_test, n_bins=10)

    thresholds, tuned_f1 = tune_thresholds_ovr(y_test, probs, class_names)

    report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc_ovr_macro": roc_auc_ovr,
        "ece": ece,
        "thresholds_ovr_best_f1": thresholds,
        "thresholds_ovr_best_f1_scores": tuned_f1,
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": class_names
    }

    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(pipe, OUT_MODEL)

    print(f"Saved model: {OUT_MODEL}")
    print(f"Saved metrics: {OUT_METRICS}")
    print(f"macro_f1={macro_f1:.4f} roc_auc_ovr={roc_auc_ovr:.4f} ece={ece:.4f}")

if __name__ == "__main__":
    main()
