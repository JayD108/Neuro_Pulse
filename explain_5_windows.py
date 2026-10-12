"""
APALA - how the model makes a prediction, shown on 5 real windows.

For each of 5 ten-second EEG windows from one patient, this script shows:
  1. the input: 590 features (a few of them printed)
  2. one decision tree: the exact path of yes/no questions the window takes
  3. all 500 trees: their leaf values added up into a log-odds score
  4. the sigmoid: log-odds -> probability of "pre-ictal"
  5. the explanation: SHAP contributions that add up exactly to the score
and checks every step against XGBoost's own output.

Usage (from the repository root):
    python explain_5_windows.py                 # patient chb01
    python explain_5_windows.py --patient chb08
    python explain_5_windows.py --patient chb01 --windows 120 900 4000 13100 13500

Needs: V1_Classical_ML/models/trained/apala_xgb_v1.json + apala_xgb_v1_features.json
       V1_Classical_ML/ml/data/processed/<patient>_labeled_features.csv
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "V1_Classical_ML", "models", "trained")
DATA_DIR = os.path.join(ROOT, "V1_Classical_ML", "ml", "data", "processed")
META = ["epoch_id", "window_start", "window_end", "file_source", "target_label"]
LABELS = {0: "interictal (normal)", 1: "PRE-ICTAL"}
LINE = "-" * 78


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load(patient):
    model_path = os.path.join(MODEL_DIR, "apala_xgb_v1.json")
    names_path = os.path.join(MODEL_DIR, "apala_xgb_v1_features.json")
    csv_path = os.path.join(DATA_DIR, f"{patient}_labeled_features.csv")
    for p in (model_path, names_path, csv_path):
        if not os.path.exists(p):
            sys.exit(f"Missing file: {p}\nSee the README Quick start for where to download it.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model.set_params(device="cpu")
    with open(names_path) as f:
        features = json.load(f)

    header = pd.read_csv(csv_path, nrows=0).columns
    df = pd.read_csv(csv_path, usecols=[c for c in header if c in set(META) | set(features)])
    df = df[df["target_label"].isin([0, 1])].reset_index(drop=True)
    return model, features, df


def pick_windows(df, wanted, n=5):
    """3 normal + 2 pre-ictal windows, spread out, unless --windows is given."""
    if wanted:
        return [i for i in wanted if 0 <= i < len(df)][:n]
    rng = np.random.default_rng(0)
    normal = df.index[df["target_label"] == 0].to_numpy()
    pre = df.index[df["target_label"] == 1].to_numpy()
    n_pre = min(2, len(pre))
    chosen = list(rng.choice(normal, n - n_pre, replace=False)) + list(rng.choice(pre, n_pre, replace=False))
    return sorted(int(i) for i in chosen)


def base_margin(booster):
    """The starting value (bias) of the model, in log-odds."""
    cfg = json.loads(booster.save_config())
    base = cfg["learner"]["learner_model_param"]["base_score"]
    base = float(str(base).strip("[]").split(",")[0])
    return math.log(base / (1 - base))


def tree_index(booster, features):
    """Tree structure as a table, with readable feature names."""
    t = booster.trees_to_dataframe()
    if t["Feature"].str.match(r"^f\d+$").any():
        t["Feature"] = t["Feature"].map(lambda f: features[int(f[1:])] if f.startswith("f") and f[1:].isdigit() else f)
    return t.set_index("ID")


def walk_tree(tree_df, tree_no, row, features):
    """Print the path one window takes through one tree."""
    node = f"{tree_no}-0"
    depth = 0
    while True:
        n = tree_df.loc[node]
        if n["Feature"] == "Leaf":
            print(f"      {'  ' * depth}-> leaf value {n['Gain']:+.4f}")
            return float(n["Gain"])
        value = row[features.index(n["Feature"])]
        if np.isnan(value):
            nxt, why = n["Missing"], "value is MISSING -> learned default direction"
        elif value < n["Split"]:
            nxt, why = n["Yes"], "yes"
        else:
            nxt, why = n["No"], "no"
        shown = "missing" if np.isnan(value) else f"{value:.4g}"
        print(f"      {'  ' * depth}{n['Feature']} = {shown}  <  {n['Split']:.4g} ?  {why}")
        node, depth = nxt, depth + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="chb01")
    ap.add_argument("--windows", type=int, nargs="*", help="row numbers to explain (default: 3 normal + 2 pre-ictal)")
    args = ap.parse_args()

    model, features, df = load(args.patient)
    booster = model.get_booster()
    n_trees = booster.num_boosted_rounds()
    rows = pick_windows(df, args.windows)

    X = df.reindex(columns=features).to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    Xs = X[rows]
    names = features if booster.feature_names else None        # match how the model was saved
    dm = xgb.DMatrix(Xs, missing=np.nan, feature_names=names)

    margin = booster.predict(dm, output_margin=True)          # log-odds from XGBoost
    proba = model.predict_proba(pd.DataFrame(Xs, columns=features) if names else Xs)[:, 1]
    leaves = booster.predict(dm, pred_leaf=True).astype(int)   # leaf reached in every tree
    contribs = booster.predict(dm, pred_contribs=True)         # SHAP values (+ bias in last column)
    bias = base_margin(booster)
    trees = tree_index(booster, features)
    leaf_value = trees["Gain"].to_dict()

    print(LINE)
    print(f"APALA model: XGBoost binary classifier, {n_trees} trees, {len(features)} features")
    print(f"Patient {args.patient}: {len(df):,} windows ({int((df.target_label == 1).sum()):,} pre-ictal)")
    print(f"Model bias (starting value): {bias:+.4f} log-odds  =  p {sigmoid(bias):.3f}")
    print(LINE)

    for k, r in enumerate(rows):
        x = Xs[k]
        label = int(df.at[r, "target_label"])
        print(f"\nWINDOW {k + 1} of {len(rows)}   row {r}   t {clock(df.at[r, 'window_start'])}"
              f"   file {df.at[r, 'file_source']}   true label: {LABELS[label]}")

        # 1. Input
        n_missing = int(np.isnan(x).sum())
        print(f"\n  1. INPUT  {len(features)} features, {n_missing} missing. A few of them:")
        for name in ("FP1-F7_Delta_relpow", "FP1-F7_Gamma_relpow", "FP1-F7_sample_entropy",
                     "FP1-F7_hjorth_mobility", "coh_FP1_F7_T7_P7_alpha"):
            if name in features:
                v = x[features.index(name)]
                print(f"       {name:28s} {'missing' if np.isnan(v) else f'{v:.4g}'}")

        # 2. One tree
        print(f"\n  2. ONE TREE (tree 1 of {n_trees}): the questions this window is asked")
        v0 = walk_tree(trees, 0, x, features)
        check = leaf_value[f"0-{leaves[k][0]}"]
        print(f"      (XGBoost says this window lands in a leaf worth {check:+.4f}: "
              f"{'match' if abs(check - v0) < 1e-6 else 'MISMATCH'})")

        # 3. All trees
        leaf_vals = np.array([leaf_value[f"{t}-{leaves[k][t]}"] for t in range(n_trees)])
        total = bias + leaf_vals.sum()
        print(f"\n  3. ALL {n_trees} TREES  each returns its leaf value; they are added up")
        print(f"       first 5 trees: {', '.join(f'{v:+.3f}' for v in leaf_vals[:5])} ...")
        print(f"       bias {bias:+.4f} + sum of {n_trees} leaf values {leaf_vals.sum():+.4f}"
              f" = {total:+.4f} log-odds")
        print(f"       (XGBoost's own log-odds: {margin[k]:+.4f}: "
              f"{'match' if abs(total - margin[k]) < 1e-3 else 'MISMATCH'})")

        # 4. Sigmoid
        p = sigmoid(total)
        print("\n  4. SIGMOID  p = 1 / (1 + e^(-log-odds))")
        print(f"       p = 1 / (1 + e^({-total:+.4f})) = {p:.4f}"
              f"   (XGBoost predict_proba: {proba[k]:.4f})")
        decision = 1 if p >= 0.5 else 0
        verdict = "correct" if decision == label else "wrong"
        print(f"       model decision (p >= 0.5): {LABELS[decision]}   -> {verdict}")

        # 5. Explanation
        c = contribs[k]
        order = np.argsort(np.abs(c[:-1]))[::-1][:5]
        print("\n  5. WHY  SHAP contributions (log-odds each feature added or removed):")
        for i in order:
            arrow = "UP   (towards pre-ictal)" if c[i] > 0 else "DOWN (towards normal)"
            print(f"       {c[i]:+.4f}  {features[i]:30s} {arrow}")
        print(f"       average model output {c[-1]:+.4f} + all {len(features)} contributions {c[:-1].sum():+.4f}"
              f" = {c.sum():+.4f} log-odds  "
              f"({'matches step 3' if abs(c.sum() - margin[k]) < 1e-3 else 'MISMATCH'})")
        print(LINE)

    print("\nSummary")
    print(f"  {'window':>6}  {'true label':22s} {'log-odds':>9}  {'p(pre-ictal)':>12}  decision")
    for k, r in enumerate(rows):
        label = int(df.at[r, "target_label"])
        print(f"  {k + 1:>6}  {LABELS[label]:22s} {margin[k]:>+9.3f}  {proba[k]:>12.3f}  "
              f"{LABELS[int(proba[k] >= 0.5)]}")
    print("\nNote: the final model was trained on all 24 patients, including this one, so these scores are\n"
          "optimistic. Held-out (leave-one-patient-out) results are in apala_xgb_v1_lopo.json.")


if __name__ == "__main__":
    main()
