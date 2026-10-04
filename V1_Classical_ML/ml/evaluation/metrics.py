"""
NeuroPulse Model Evaluation
============================
Comprehensive evaluation of the trained XGBoost model including:
- Classification metrics (Accuracy, Precision, Recall, F1, Specificity)
- ROC curve and AUC
- Precision-Recall curve and AUC
- Confusion matrix visualization
- Patient-wise per-patient performance breakdown
- False alarm rate analysis

Usage:
    python -m ml.evaluation.metrics --patient chb01
    python -m ml.evaluation.metrics --patients chb01 chb02 chb03
"""

import os
import json
import pickle
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)

PROCESSED_DIR = "./ml/data/processed"
MODEL_DIR     = "./models/trained"
SCALER_DIR    = "./models/scalers"
META_DIR      = "./models/metadata"
FIGURES_DIR   = "./reports/figures"
RESULTS_DIR   = "./reports/model_results"

NON_FEATURE_COLS = [
    "epoch_id", "window_start", "window_end",
    "file_source", "target_label", "patient_id",
]


def load_artifacts():
    """Load trained model, scaler, and feature names."""
    with open(os.path.join(MODEL_DIR, "xgboost_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(SCALER_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(META_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


def load_all_data(patient_ids: List[str], feature_names: List[str]) -> pd.DataFrame:
    """Load and merge feature CSVs for all specified patients."""
    dfs = []
    for pid in patient_ids:
        path = os.path.join(PROCESSED_DIR, f"{pid}_labeled_features.csv")
        if not os.path.exists(path):
            print(f"  [WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["patient_id"] = pid
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "") -> str:
    """Plots and saves ROC curve."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#6C63FF", lw=2, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#6C63FF")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve {title}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "") -> str:
    """Plots and saves Precision-Recall curve."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#FF6584", lw=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#FF6584")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve {title}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "pr_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Plots and saves a styled confusion matrix."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Interictal", "Preictal"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def evaluate(df: pd.DataFrame, model, scaler, feature_names: List[str]) -> Dict:
    """Runs full evaluation and returns a metrics dictionary."""
    X = df[feature_names].values
    y = df["target_label"].values

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy":       float(accuracy_score(y, y_pred)),
        "f1":             float(f1_score(y, y_pred, zero_division=0)),
        "sensitivity":    float(tp / max(tp + fn, 1)),
        "specificity":    float(tn / max(tn + fp, 1)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "roc_auc":        float(roc_auc_score(y, y_prob)),
        "pr_auc":         float(average_precision_score(y, y_prob)),
        "n_interictal":   int((y == 0).sum()),
        "n_preictal":     int((y == 1).sum()),
        "total_samples":  len(y),
    }

    print(f"\n{'='*60}")
    print(f"  Model Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Samples    : {metrics['total_samples']} "
          f"(Interictal={metrics['n_interictal']}, Preictal={metrics['n_preictal']})")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  F1 Score   : {metrics['f1']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  FAR        : {metrics['false_alarm_rate']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC     : {metrics['pr_auc']:.4f}")
    print(f"{'='*60}\n")

    # --- Plots ---
    plot_roc_curve(y, y_prob)
    plot_pr_curve(y, y_prob)
    plot_confusion_matrix(y, y_pred)

    # --- Per-patient breakdown (if patient_id column exists) ---
    if "patient_id" in df.columns:
        print("  Patient-wise breakdown:")
        per_patient = []
        for pid in df["patient_id"].unique():
            sub = df[df["patient_id"] == pid]
            Xp = scaler.transform(sub[feature_names].values)
            yp_true = sub["target_label"].values
            yp_pred = model.predict(Xp)
            yp_prob = model.predict_proba(Xp)[:, 1]

            if len(np.unique(yp_true)) < 2:
                continue

            row = {
                "patient_id": pid,
                "roc_auc":    float(roc_auc_score(yp_true, yp_prob)),
                "f1":         float(f1_score(yp_true, yp_pred, zero_division=0)),
                "n_preictal": int((yp_true == 1).sum()),
            }
            per_patient.append(row)
            print(f"    {pid}: ROC-AUC={row['roc_auc']:.3f}, F1={row['f1']:.3f}, "
                  f"Preictal={row['n_preictal']}")

        if per_patient:
            ppdf = pd.DataFrame(per_patient)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            ppdf.to_csv(os.path.join(RESULTS_DIR, "patient_wise_metrics.csv"), index=False)
            metrics["per_patient"] = per_patient

    # Save summary metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "per_patient"}, f, indent=2)
    print(f"  Results saved to: {RESULTS_DIR}/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroPulse Model Evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", type=str, help="Single patient ID")
    group.add_argument("--patients", nargs="+", help="Multiple patient IDs")
    args = parser.parse_args()

    patient_ids = [args.patient] if args.patient else args.patients

    model, scaler, feature_names = load_artifacts()
    df = load_all_data(patient_ids, feature_names)

    if df.empty:
        print("[ERROR] No data loaded. Run ingest_dataset.py first.")
        exit(1)

    evaluate(df, model, scaler, feature_names)
