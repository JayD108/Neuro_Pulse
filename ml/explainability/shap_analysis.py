"""
NeuroPulse SHAP Explainability Engine
=======================================
Generates SHAP (SHapley Additive exPlanations) values for the trained
XGBoost model, enabling interpretable AI explanations for each prediction.

Outputs:
- Global feature importance bar chart
- SHAP waterfall plot for individual predictions
- Summary beeswarm plot
- Top-N feature attributions as JSON (consumed by the backend API)

Usage:
    python -m ml.explainability.shap_analysis --patient chb01
    python -m ml.explainability.shap_analysis --patient chb01 --n-samples 500
"""

import os
import json
import pickle
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

PROCESSED_DIR = "./ml/data/processed"
MODEL_DIR     = "./models/trained"
SCALER_DIR    = "./models/scalers"
META_DIR      = "./models/metadata"
FIGURES_DIR   = "./reports/figures"

NON_FEATURE_COLS = [
    "epoch_id", "window_start", "window_end",
    "file_source", "target_label", "patient_id",
]


def load_artifacts():
    """Loads the trained model, scaler, and feature names from disk."""
    model_path  = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    scaler_path = os.path.join(SCALER_DIR, "scaler.pkl")
    names_path  = os.path.join(META_DIR, "feature_names.json")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, names_path]):
        raise FileNotFoundError(
            "Model artifacts not found. Run ml/training/train_xgboost.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(names_path) as f:
        feature_names = json.load(f)

    return model, scaler, feature_names


def load_sample_data(
    patient_id: str,
    feature_names: List[str],
    n_samples: int = 500,
) -> pd.DataFrame:
    """
    Loads a random sample of windows from a patient's feature CSV
    for SHAP analysis (SHAP is compute-intensive on large datasets).
    """
    csv_path = os.path.join(PROCESSED_DIR, f"{patient_id}_labeled_features.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure all expected feature columns exist
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing[:5]}...")

    df = df.sample(min(n_samples, len(df)), random_state=42)
    return df[feature_names]


def generate_global_importance_plot(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
) -> str:
    """Generates and saves a bar chart of mean absolute SHAP values."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        mean_abs_shap[sorted_idx[::-1]],
        color="#6C63FF",
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Global Feature Importance (Top {top_n})", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "shap_global_importance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {save_path}")
    return save_path


def generate_beeswarm_plot(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
) -> str:
    """Generates and saves a SHAP summary beeswarm plot."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(10, 10))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title("SHAP Summary (Impact on Preictal Risk)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "shap_beeswarm.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")
    return save_path


def generate_top_features_json(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> str:
    """
    Computes global top-N features by mean absolute SHAP value and
    saves as a JSON file (consumed by the backend inference engine).
    """
    os.makedirs(META_DIR, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:top_n]

    top_features = [
        {
            "rank": i + 1,
            "feature": feature_names[idx],
            "mean_abs_shap": float(mean_abs[idx]),
        }
        for i, idx in enumerate(sorted_idx)
    ]

    save_path = os.path.join(META_DIR, "top_shap_features.json")
    with open(save_path, "w") as f:
        json.dump(top_features, f, indent=2)
    print(f"  [OK] Saved: {save_path}")
    return save_path


def explain_single_prediction(
    model,
    explainer: shap.TreeExplainer,
    scaler,
    feature_names: List[str],
    raw_features: np.ndarray,
) -> List[Dict]:
    """
    Computes SHAP values for a single feature vector and returns
    the top contributing features as a list of dicts.

    This is the function called by the backend inference engine during
    live WebSocket streaming to generate real-time XAI explanations.

    Args:
        model:         Trained XGBoost model.
        explainer:     Pre-built SHAP TreeExplainer.
        scaler:        Fitted StandardScaler.
        feature_names: Ordered list of feature names.
        raw_features:  1D numpy array of raw (unscaled) feature values.

    Returns:
        List of dicts: [{"feature": str, "contribution": float}, ...]
        Sorted by absolute contribution, descending.
    """
    scaled = scaler.transform(raw_features.reshape(1, -1))
    shap_vals = explainer.shap_values(scaled)[0]  # shape: (n_features,)

    contributions = sorted(
        [
            {"feature": feature_names[i], "contribution": float(shap_vals[i])}
            for i in range(len(feature_names))
        ],
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )
    return contributions[:5]   # Return top 5 for dashboard display


def run_shap_analysis(patient_id: str, n_samples: int = 500) -> None:
    """
    Main SHAP analysis routine:
    1. Loads trained model artifacts.
    2. Samples data for the given patient.
    3. Computes SHAP values via TreeExplainer (exact, model-native).
    4. Generates visualizations and saves JSON feature attribution.
    """
    print(f"\n{'='*60}")
    print(f"  NeuroPulse SHAP Analysis | Patient: {patient_id}")
    print(f"{'='*60}")

    model, scaler, feature_names = load_artifacts()
    print(f"  Model loaded: {model.__class__.__name__}")
    print(f"  Features    : {len(feature_names)}")

    # --- Load and scale sample data ---
    print(f"\n  Loading {n_samples} sample windows from {patient_id}...")
    X_df = load_sample_data(patient_id, feature_names, n_samples)
    X_scaled = scaler.transform(X_df.values)

    # --- Build SHAP Explainer ---
    # TreeExplainer is exact (not approximate) for XGBoost -> no sampling needed
    print(f"  Building TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    print(f"  SHAP values computed: shape {shap_values.shape}")

    # --- Generate Outputs ---
    print(f"\n  Generating plots and JSON...")
    generate_global_importance_plot(shap_values, feature_names)
    generate_beeswarm_plot(shap_values, X_scaled, feature_names)
    generate_top_features_json(shap_values, feature_names)

    print(f"\n{'='*60}")
    print(f"  SHAP Analysis Complete!")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Metadata saved to: {META_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroPulse SHAP Analysis")
    parser.add_argument("--patient", type=str, required=True,
                        help="Patient ID for sample data (e.g., chb01)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of windows to sample for SHAP analysis (default: 500)")
    args = parser.parse_args()

    run_shap_analysis(args.patient, n_samples=args.n_samples)
