"""
APALA / NeuroPulse — XGBoost LOPO Training & Evaluation
======================================================
Trains an XGBClassifier on all 24 patients using Leave-One-Patient-Out (LOPO) CV,
evaluates per-patient ROC-AUC, PR-AUC, sensitivity, specificity, and saves
the final model, feature list, and metrics JSON.
"""

import os
import glob
import time
import json
import functools
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

print = functools.partial(print, flush=True)

DATA_DIR = os.path.join("V1_Classical_ML", "ml", "data", "processed")
OUTPUT_DIR = os.path.join("V1_Classical_ML", "models", "trained")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Feature List from chb01
# ---------------------------------------------------------------------------
print("=" * 65)
print("1. Identifying 590 feature columns from chb01...")
chb01_path = os.path.join(DATA_DIR, "chb01_labeled_features.csv")
df01_head = pd.read_csv(chb01_path, nrows=1)
exclude_cols = {"epoch_id", "window_start", "window_end", "file_source", "target_label"}
feature_cols = [c for c in df01_head.columns if c not in exclude_cols]
print(f"Total features extracted: {len(feature_cols)}")
assert len(feature_cols) == 590, f"Expected 590 features, got {len(feature_cols)}"

# ---------------------------------------------------------------------------
# 2. Load and Prepare All 24 Patients
# ---------------------------------------------------------------------------
print("\n2. Loading data for all 24 patients (usecols only)...")
t_data_start = time.time()
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "chb*_labeled_features.csv")))
patient_data = {}

for csv_path in csv_files:
    pid = os.path.basename(csv_path).split("_")[0]
    # Check available columns in this CSV
    file_cols = set(pd.read_csv(csv_path, nrows=0).columns)
    use_cols = [c for c in feature_cols if c in file_cols] + ["target_label"]
    
    # Read only needed columns
    df = pd.read_csv(csv_path, usecols=use_cols)
    # Keep only target_label 0 and 1
    df = df[df["target_label"].isin([0, 1])]
    
    y = df["target_label"].values.astype(np.int32)
    # Reindex to exact 590 feature columns (missing becomes NaN)
    X = df.reindex(columns=feature_cols).values.astype(np.float32)
    # Turn inf into NaN
    X[np.isinf(X)] = np.nan
    
    patient_data[pid] = (X, y)
    print(f"  {pid}: {len(y):>5} windows ({int((y==1).sum()):>4} pre-ictal, {int((y==0).sum()):>5} interictal)")

total_windows = sum(len(y) for _, y in patient_data.values())
total_preictal = sum(int((y == 1).sum()) for _, y in patient_data.values())
total_interictal = sum(int((y == 0).sum()) for _, y in patient_data.values())
print(f"Total loaded: {total_windows:,} windows ({total_preictal:,} pre-ictal, {total_interictal:,} interictal)")
print(f"Data loading took: {time.time() - t_data_start:.2f}s")

# Test device
device = "cuda"
try:
    _test = xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda")
    _test.fit(np.zeros((10, 2)), np.zeros(10))
    print("\nUsing GPU acceleration (device='cuda').")
except Exception:
    device = "cpu"
    print("\nCUDA unavailable, falling back to CPU (device='cpu').")

MODEL_SETTINGS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": device,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# 3. Leave-One-Patient-Out (LOPO) Cross-Validation
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. Starting Leave-One-Patient-Out Cross-Validation (24 folds)...")
print("=" * 65)

patient_ids = sorted(patient_data.keys())
lopo_results = []
t_lopo_start = time.time()

for fold_idx, test_pid in enumerate(patient_ids, 1):
    t_fold_start = time.time()
    
    # Train set: all other 23 patients
    train_pids = [p for p in patient_ids if p != test_pid]
    X_train = np.vstack([patient_data[p][0] for p in train_pids])
    y_train = np.concatenate([patient_data[p][1] for p in train_pids])
    
    # Held-out test patient
    X_test, y_test = patient_data[test_pid]
    
    # Calculate scale_pos_weight = (#label 0) / (#label 1)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = float(n_neg) / float(max(n_pos, 1))
    
    # Initialize and train
    clf = xgb.XGBClassifier(
        **MODEL_SETTINGS,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train, y_train)
    
    # Score held-out patient
    probs = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = float(roc_auc_score(y_test, probs))
    pr_auc = float(average_precision_score(y_test, probs))
    
    # Sensitivity and specificity at 0.5 threshold
    pred = (probs >= 0.5).astype(int)
    n_pre = int((y_test == 1).sum())
    n_int = int((y_test == 0).sum())
    
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    fold_time = time.time() - t_fold_start
    
    record = {
        "patient_id": test_pid,
        "n_windows": len(y_test),
        "n_preictal": n_pre,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "train_time_sec": round(fold_time, 2),
    }
    lopo_results.append(record)
    
    print(f"[{fold_idx:02d}/24] {test_pid}: ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | "
          f"Sens={sensitivity*100:.1f}% | Spec={specificity*100:.1f}% | "
          f"Windows={len(y_test)} (Pre={n_pre}) | Time={fold_time:.1f}s")

total_lopo_time = time.time() - t_lopo_start
roc_aucs = [r["roc_auc"] for r in lopo_results]
pr_aucs = [r["pr_auc"] for r in lopo_results]
sensitivities = [r["sensitivity"] for r in lopo_results]
specificities = [r["specificity"] for r in lopo_results]

mean_roc_auc = float(np.mean(roc_aucs))
median_roc_auc = float(np.median(roc_aucs))
mean_pr_auc = float(np.mean(pr_aucs))
median_pr_auc = float(np.median(pr_aucs))
mean_sens = float(np.mean(sensitivities))
mean_spec = float(np.mean(specificities))

print("=" * 65)
print(f"LOPO Cross-Validation Completed in: {total_lopo_time:.1f}s")
print(f"Mean ROC-AUC:   {mean_roc_auc:.4f}")
print(f"Median ROC-AUC: {median_roc_auc:.4f}")
print(f"Mean PR-AUC:    {mean_pr_auc:.4f}")
print(f"Mean Sens:      {mean_sens*100:.1f}%")
print(f"Mean Spec:      {mean_spec*100:.1f}%")
print("=" * 65)

# ---------------------------------------------------------------------------
# 4. Train Final Model on All 24 Patients
# ---------------------------------------------------------------------------
print("\n4. Training final model on all 24 patients...")
t_final_start = time.time()
X_all = np.vstack([patient_data[p][0] for p in patient_ids])
y_all = np.concatenate([patient_data[p][1] for p in patient_ids])
scale_pos_weight_all = float((y_all == 0).sum()) / float(max((y_all == 1).sum(), 1))

final_clf = xgb.XGBClassifier(
    **MODEL_SETTINGS,
    scale_pos_weight=scale_pos_weight_all,
)
final_clf.fit(X_all, y_all)
final_train_time = time.time() - t_final_start
print(f"Final model trained on {len(y_all):,} windows in {final_train_time:.1f}s.")

# ---------------------------------------------------------------------------
# 5. Save Model and Artifacts
# ---------------------------------------------------------------------------
print("\n5. Saving artifacts...")

# a) apala_xgb_v1.json (native XGBoost JSON format)
model_json_path = os.path.join(OUTPUT_DIR, "apala_xgb_v1.json")
final_clf.save_model(model_json_path)
print(f"  Saved model: {model_json_path} ({os.path.getsize(model_json_path):,} bytes)")

# b) apala_xgb_v1_features.json (590 features in order)
features_json_path = os.path.join(OUTPUT_DIR, "apala_xgb_v1_features.json")
with open(features_json_path, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"  Saved features: {features_json_path} ({len(feature_cols)} features)")

# c) apala_xgb_v1_lopo.json (all per-patient metrics + summary)
summary_output = {
    "summary": {
        "mean_roc_auc": round(mean_roc_auc, 4),
        "median_roc_auc": round(median_roc_auc, 4),
        "mean_pr_auc": round(mean_pr_auc, 4),
        "median_pr_auc": round(median_pr_auc, 4),
        "mean_sensitivity": round(mean_sens, 4),
        "mean_specificity": round(mean_spec, 4),
        "total_windows": total_windows,
        "total_preictal": total_preictal,
        "total_interictal": total_interictal,
        "total_lopo_time_sec": round(total_lopo_time, 2),
        "final_train_time_sec": round(final_train_time, 2),
    },
    "model_settings": MODEL_SETTINGS,
    "per_patient": lopo_results,
}

lopo_json_path = os.path.join(OUTPUT_DIR, "apala_xgb_v1_lopo.json")
with open(lopo_json_path, "w") as f:
    json.dump(summary_output, f, indent=2)
print(f"  Saved LOPO metrics: {lopo_json_path}")

total_elapsed = time.time() - t_data_start
print("\n" + "=" * 65)
print(f"ALL DONE! Total elapsed time: {total_elapsed:.1f}s")
print("=" * 65)
