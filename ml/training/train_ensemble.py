"""
NeuroPulse Classical ML Stacked Ensemble Trainer
==================================================
Final prediction layer -- 100% Classical ML.

Base Models (Level 0):
  - XGBoost (GPU-accelerated via device='cuda')
  - LightGBM
  - Random Forest
  - SVM (RBF kernel)

Meta-Learner (Level 1):
  - Logistic Regression (trained on out-of-fold predictions)

This is a 2-level stacking ensemble. Each base model sees the same feature
vector. Their probability outputs are stacked and fed into the meta-learner
which learns the optimal combination. The prediction is ALWAYS made by
Logistic Regression -- a classical model.

Usage:
    python -m ml.training.train_ensemble --patients chb01
    python -m ml.training.train_ensemble --all --lopo
"""

import os
import json
import pickle
import argparse
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = "./ml/data/processed"
MODEL_DIR     = "./models/trained"
SCALER_DIR    = "./models/scalers"
META_DIR      = "./models/metadata"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_patient_data(patient_ids: List[str]) -> pd.DataFrame:
    """Load and concatenate labeled feature CSVs for given patients."""
    frames = []
    for pid in patient_ids:
        path = os.path.join(PROCESSED_DIR, f"{pid}_labeled_features.csv")
        if not os.path.exists(path):
            print(f"  [WARN] Feature CSV not found for {pid}: {path}")
            continue
        df = pd.read_csv(path)
        df["patient_id"] = pid
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No feature CSVs found in '{PROCESSED_DIR}'. Run ml/ingest_dataset.py first."
        )
    return pd.concat(frames, ignore_index=True)


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """Extract X, y, feature names, and patient groups from dataframe."""
    # Select all numeric columns, then drop any metadata/label columns
    meta_cols = {"patient_id", "edf_file", "file_source", "window_idx",
                 "epoch_id", "window_start", "window_end", "target_label"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in meta_cols]
    X_raw = df[feature_cols].values
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X_raw, -1e30, 1e30).astype(np.float32)
    y = df["target_label"].values.astype(int)
    groups = df["patient_id"].values if "patient_id" in df.columns else None
    return X, y, feature_cols, groups


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Tuning (GPU-accelerated XGBoost)
# ---------------------------------------------------------------------------

def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50) -> dict:
    """Run Optuna hyperparameter search for XGBoost using GPU."""
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos,
            "tree_method":      "hist",
            "device":           "cuda",
            "eval_metric":      "aucpr",
            "random_state":     42,
        }
        model = XGBClassifier(**params, verbosity=0)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            prob = model.predict_proba(X_train[va_idx])[:, 1]
            scores.append(average_precision_score(y_train[va_idx], prob))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  [Optuna] Best XGB PR-AUC: {study.best_value:.4f}")
    return study.best_params


# ---------------------------------------------------------------------------
# Build Base Models
# ---------------------------------------------------------------------------

def build_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_optuna: bool = True,
    optuna_trials: int = 40,
) -> List[Tuple[str, object]]:
    """
    Build and optionally tune all base classifiers.

    Returns a list of (name, estimator) tuples compatible with sklearn's
    StackingClassifier.
    """
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"\n  Class imbalance ratio: {scale_pos:.2f}:1")

    # --- 1. XGBoost (GPU) ---
    if use_optuna:
        print("  [Optuna] Tuning XGBoost hyperparameters...")
        best_xgb_params = tune_xgboost(X_train, y_train, n_trials=optuna_trials)
        xgb_params = {
            **best_xgb_params,
            "tree_method": "hist",
            "device": "cuda",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos,
            "random_state": 42,
            "verbosity": 0,
        }
    else:
        xgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos,
            "tree_method": "hist",
            "device": "cuda",
            "eval_metric": "aucpr",
            "random_state": 42,
            "verbosity": 0,
        }
    xgb = XGBClassifier(**xgb_params)

    # --- 2. LightGBM ---
    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )

    # --- 3. Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return [
        ("xgboost",      xgb),
        ("lightgbm",     lgbm),
        ("random_forest", rf),
    ]


# ---------------------------------------------------------------------------
# Stacked Ensemble Training
# ---------------------------------------------------------------------------

def train_stacked_ensemble(
    patients: List[str],
    use_optuna: bool = True,
    optuna_trials: int = 40,
    lopo: bool = False,
) -> None:
    """
    Train the full stacked ensemble:
      Level 0: XGBoost, LightGBM, Random Forest, SVM
      Level 1: Logistic Regression meta-learner

    Args:
        patients: Patient IDs to train on.
        use_optuna: Whether to tune XGBoost with Optuna.
        optuna_trials: Number of Optuna trials for XGBoost tuning.
        lopo: Use Leave-One-Patient-Out cross-validation.
    """
    print("\n" + "=" * 65)
    print("  NeuroPulse Stacked Ensemble Training")
    print(f"  Patients : {patients}")
    print(f"  Mode     : {'LOPO' if lopo else 'Single Split'}")
    print(f"  Optuna   : {'Yes (' + str(optuna_trials) + ' trials)' if use_optuna else 'No'}")
    print("=" * 65)

    df = load_patient_data(patients)
    X, y, feature_cols, groups = prepare_features(df)

    print(f"\n  Total dataset: {len(X)} samples | Features: {X.shape[1]}")
    print(f"  Interictal={int((y == 0).sum())}, Preictal={int((y == 1).sum())}")

    # --- Normalize ---
    scaler = StandardScaler()

    if lopo and groups is not None and len(np.unique(groups)) > 1:
        # Leave-One-Patient-Out evaluation
        print("\n  Running LOPO cross-validation...")
        gkf = GroupKFold(n_splits=len(np.unique(groups)))
        lopo_results = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
            held_out = groups[va_idx][0]
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            X_tr_scaled = scaler.fit_transform(X_tr)
            X_va_scaled = scaler.transform(X_va)

            base_models = build_base_models(X_tr_scaled, y_tr, use_optuna=False)

            meta = LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=1000, solver="lbfgs", random_state=42
            )
            ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=meta,
                cv=3,
                stack_method="predict_proba",
                passthrough=False,
                n_jobs=1,
            )
            ensemble.fit(X_tr_scaled, y_tr)
            prob = ensemble.predict_proba(X_va_scaled)[:, 1]
            pred = ensemble.predict(X_va_scaled)

            roc_auc = roc_auc_score(y_va, prob)
            pr_auc  = average_precision_score(y_va, prob)
            f1      = f1_score(y_va, pred, zero_division=0)
            lopo_results.append({
                "held_out": held_out, "roc_auc": roc_auc,
                "pr_auc": pr_auc, "f1": f1
            })
            print(f"  Fold {fold} (held-out={held_out}): "
                  f"ROC={roc_auc:.4f}, PR={pr_auc:.4f}, F1={f1:.4f}")

        results_df = pd.DataFrame(lopo_results)
        print(f"\n  LOPO Mean (+/- std):")
        for col in ["roc_auc", "pr_auc", "f1"]:
            print(f"    {col}: {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")

    # --- Final Model Training on Full Data ---
    print("\n  Training final ensemble on full dataset...")
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    base_models = build_base_models(X_tr_s, y_tr, use_optuna=use_optuna, optuna_trials=optuna_trials)

    meta = LogisticRegression(
        C=1.0, class_weight="balanced",
        max_iter=1000, solver="lbfgs", random_state=42
    )
    final_ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )

    print("  Fitting StackingClassifier (this may take a few minutes)...")
    final_ensemble.fit(X_tr_s, y_tr)

    # --- Evaluation ---
    prob = final_ensemble.predict_proba(X_te_s)[:, 1]
    pred = final_ensemble.predict(X_te_s)

    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    far         = fp / max(fp + tn, 1)
    roc_auc     = roc_auc_score(y_te, prob)
    pr_auc      = average_precision_score(y_te, prob)
    f1          = f1_score(y_te, pred, zero_division=0)

    print(f"\n  --- Classification Report ---")
    print(classification_report(y_te, pred, target_names=["Interictal", "Preictal"], zero_division=0))
    print(f"  Confusion Matrix:")
    print(f"  [[{tn} {fp}]")
    print(f"   [{fn} {tp}]]")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  FAR         : {far:.4f}")
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  PR-AUC      : {pr_auc:.4f}")
    print(f"  F1 Score    : {f1:.4f}")

    # --- Save Artifacts ---
    ensemble_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    scaler_path   = os.path.join(SCALER_DIR, "ensemble_scaler.pkl")
    names_path    = os.path.join(META_DIR, "ensemble_feature_names.json")
    metrics_path  = os.path.join("./reports/model_results", "ensemble_metrics.json")
    os.makedirs("./reports/model_results", exist_ok=True)

    with open(ensemble_path, "wb") as f:
        pickle.dump(final_ensemble, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(names_path, "w") as f:
        json.dump(feature_cols, f)
    with open(metrics_path, "w") as f:
        json.dump({
            "roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1,
            "sensitivity": sensitivity, "specificity": specificity,
            "far": far, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }, f, indent=2)

    print(f"\n  [SAVE] Ensemble saved : {ensemble_path}")
    print(f"  [SAVE] Scaler saved   : {scaler_path}")
    print(f"  [SAVE] Features saved : {names_path}")
    print(f"  [SAVE] Metrics saved  : {metrics_path}")


# ---------------------------------------------------------------------------
# GPU-Native Manual Stacking (bypasses sklearn Parallel subprocess)
# ---------------------------------------------------------------------------

def _safe_scale(scaler, X, fit=False):
    """Scale X and then strip any inf/nan the scaler itself may produce
    (e.g., zero-variance columns cause StandardScaler to emit inf)."""
    if fit:
        X_s = scaler.fit_transform(X)
    else:
        X_s = scaler.transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X_s, -1e15, 1e15).astype(np.float32)


def _manual_stack_train(X_tr_s, y_tr, scale_pos, n_splits=5):

    """
    Manual OOF stacking that lets XGBoost use CUDA natively.

    sklearn StackingClassifier wraps each estimator in a joblib worker that
    does NOT inherit the CUDA device context → runs on CPU.
    This loop calls each model's .fit() directly so XGBoost's C++ backend
    talks to the NVIDIA driver with no Python intermediary.
    """
    import copy
    print(f"\n  [GPU] Manual OOF stacking — {n_splits} folds...")

    xgb_mdl = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        tree_method="hist", device="cuda",
        eval_metric="aucpr", random_state=42, verbosity=0,
    )
    lgbm_mdl = LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", n_jobs=-1, random_state=42, verbosity=-1,
    )
    rf_mdl = RandomForestClassifier(
        n_estimators=200, max_depth=12, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    blueprints = [("xgboost", xgb_mdl), ("lightgbm", lgbm_mdl), ("random_forest", rf_mdl)]

    # --- OOF predictions for meta-learner ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_models = len(blueprints)
    oof = np.zeros((len(y_tr), n_models), dtype=np.float32)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_tr_s, y_tr), 1):
        print(f"    OOF Fold {fold_i}/{n_splits}...", end=" ", flush=True)
        Xf_tr, Xf_va = X_tr_s[tr_idx], X_tr_s[va_idx]
        yf_tr = y_tr[tr_idx]
        for j, (name, blueprint) in enumerate(blueprints):
            m = copy.deepcopy(blueprint)
            m.fit(Xf_tr, yf_tr)
            oof[va_idx, j] = m.predict_proba(Xf_va)[:, 1]
        print("done")

    # --- Fit final base models on full training data ---
    print("  [GPU] Fitting final base models on full train set...")
    trained_bases = []
    for name, blueprint in blueprints:
        print(f"    {name}...", end=" ", flush=True)
        m = copy.deepcopy(blueprint)
        m.fit(X_tr_s, y_tr)
        trained_bases.append((name, m))
        print("done")

    # --- Meta-learner on OOF ---
    meta = LogisticRegression(C=1.0, class_weight="balanced",
                              max_iter=1000, solver="lbfgs", random_state=42)
    meta.fit(oof, y_tr)
    print("  [GPU] Meta-learner trained on OOF predictions.")
    return trained_bases, meta


def _manual_stack_predict(trained_bases, meta, X_s):
    base_probs = np.column_stack([
        m.predict_proba(X_s)[:, 1] for _, m in trained_bases
    ]).astype(np.float32)
    prob = meta.predict_proba(base_probs)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return prob, pred


class ManualStackingEnsemble:
    """Picklable container for the GPU-trained manual stacking ensemble."""
    def __init__(self, trained_bases, meta, scaler, feature_cols):
        self.trained_bases = trained_bases
        self.meta = meta
        self.scaler = scaler
        self.feature_cols = feature_cols

    def predict_proba(self, X_s):
        prob, _ = _manual_stack_predict(self.trained_bases, self.meta, X_s)
        out = np.zeros((len(prob), 2), dtype=np.float32)
        out[:, 1] = prob; out[:, 0] = 1 - prob
        return out

    def predict(self, X_s):
        _, pred = _manual_stack_predict(self.trained_bases, self.meta, X_s)
        return pred

    @property
    def named_estimators_(self):
        return {n: m for n, m in self.trained_bases}


# ---------------------------------------------------------------------------
# Cross-Patient Split Validation (chb01-10 vs chb11-24)
# ---------------------------------------------------------------------------

def train_split_validation(use_optuna: bool = True, optuna_trials: int = 40) -> None:
    print("\n" + "=" * 65)
    print("  NeuroPulse Split Validation (10 Train / 14 Test) + Master Model")
    print("  GPU-Native Manual Stacking — XGBoost on CUDA")
    print("=" * 65)

    train_patients = [f"chb{i:02d}" for i in range(1, 11)]
    test_patients  = [f"chb{i:02d}" for i in range(11, 25)]

    print(f"\n  Loading ALL 24 patients to align feature spaces...")
    df_all = load_patient_data(train_patients + test_patients)

    train_mask = df_all["patient_id"].isin(train_patients)
    test_mask  = df_all["patient_id"].isin(test_patients)

    print(f"  [TRAIN SET] {train_mask.sum()} windows from chb01...chb10")
    print(f"  [TEST SET]  {test_mask.sum()} windows from chb11...chb24")

    X_all, y_all, feature_cols, _ = prepare_features(df_all)

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_te, y_te = X_all[test_mask],  y_all[test_mask]

    scaler_val = StandardScaler()
    X_tr_s = _safe_scale(scaler_val, X_tr, fit=True)
    X_te_s = _safe_scale(scaler_val, X_te, fit=False)

    scale_pos = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"\n  Class imbalance (train): {scale_pos:.2f}:1")

    trained_bases, meta = _manual_stack_train(X_tr_s, y_tr, scale_pos, n_splits=5)

    print("\n  Evaluating on UNSEEN Test Set (chb11-24)...")
    prob, pred = _manual_stack_predict(trained_bases, meta, X_te_s)

    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    far         = fp / max(fp + tn, 1)
    roc_auc     = roc_auc_score(y_te, prob)
    pr_auc      = average_precision_score(y_te, prob)

    print(f"\n  --- Zero-Shot Cross-Patient Validation ---")
    print(classification_report(y_te, pred, target_names=["Interictal", "Preictal"], zero_division=0))
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  FAR         : {far:.4f}")
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  PR-AUC      : {pr_auc:.4f}")

    # --- Master Model: ALL 24 patients ---
    print("\n" + "=" * 65)
    print("  Training FINAL Master Model on ALL 24 Patients (GPU)...")
    print("=" * 65)

    scaler_final = StandardScaler()
    X_all_s = _safe_scale(scaler_final, X_all, fit=True)
    scale_pos_all = float((y_all == 0).sum() / max((y_all == 1).sum(), 1))
    print(f"  Full dataset: {len(X_all)} windows | {X_all.shape[1]} features")
    print(f"  Class imbalance (all): {scale_pos_all:.2f}:1")

    trained_bases_f, meta_f = _manual_stack_train(X_all_s, y_all, scale_pos_all, n_splits=5)
    master = ManualStackingEnsemble(trained_bases_f, meta_f, scaler_final, feature_cols)

    master_path = os.path.join(MODEL_DIR, "master_model.pkl")
    scaler_path = os.path.join(SCALER_DIR, "master_scaler.pkl")
    names_path  = os.path.join(META_DIR, "master_feature_names.json")

    with open(master_path, "wb") as f: pickle.dump(master, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler_final, f)
    with open(names_path,  "w") as f: json.dump(feature_cols, f)

    print(f"\n  [SAVE] Master Model  : {master_path}")
    print(f"  [SAVE] Master Scaler : {scaler_path}")
    print(f"  [SAVE] Feature Names : {names_path}")
    print("\n  ✅ DONE — Master model saved and ready for the Live Dashboard!")



# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser(description="Train Stacked Classical ML Ensemble")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patients", nargs="+", help="Patient IDs to train on")
    group.add_argument("--all", action="store_true", help="Use all available patients")
    group.add_argument("--split-validation", action="store_true", help="Train on chb01-10, test on chb11-24")
    parser.add_argument("--lopo", action="store_true", help="Run LOPO cross-validation")
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna tuning (faster)")
    parser.add_argument("--optuna-trials", type=int, default=40, help="Number of Optuna trials")

    args = parser.parse_args()

    if args.split_validation:
        train_split_validation(
            use_optuna=not args.no_optuna,
            optuna_trials=args.optuna_trials
        )
    else:
        if args.all:
            csvs = glob.glob(os.path.join(PROCESSED_DIR, "*_labeled_features.csv"))
            target_patients = sorted([os.path.basename(c).replace("_labeled_features.csv", "") for c in csvs])
            if not target_patients:
                print(f"[ERROR] No CSVs found in {PROCESSED_DIR}. Run ml/ingest_dataset.py first.")
                exit(1)
            print(f"Auto-detected {len(target_patients)} patients: {target_patients}")
        else:
            target_patients = args.patients

        train_stacked_ensemble(
            patients=target_patients,
            use_optuna=not args.no_optuna,
            optuna_trials=args.optuna_trials,
            lopo=args.lopo,
        )
