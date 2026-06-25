"""
compare_models.py — Local model benchmarking script.

Compares multiple classifiers on the wine quality dataset using
both training phases (phase1 + phase2 combined) evaluated on eval.csv.

Usage:
    python compare_models.py

This script is for local experimentation only. It does NOT train on eval.csv.
"""

import os
import sys
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Canonical 12-feature order (must match serve.py PredictRequest)
FEATURE_COLS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]

LABEL_NAMES = {0: "low", 1: "medium", 2: "high"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    paths = {
        "train_phase1": "data/train_phase1.csv",
        "train_phase2": "data/train_phase2.csv",
        "eval":         "data/eval.csv",
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run `python generate_data.py` first.")
            sys.exit(1)

    train1  = load(paths["train_phase1"])
    train2  = load(paths["train_phase2"])
    eval_df = load(paths["eval"])

    combined = pd.concat([train1, train2], ignore_index=True)

    print("=" * 70)
    print("DATA SUMMARY")
    print(f"  train_phase1 : {len(train1):5d} samples")
    print(f"  train_phase2 : {len(train2):5d} samples")
    print(f"  combined     : {len(combined):5d} samples")
    print(f"  eval         : {len(eval_df):5d} samples  (held-out, never trained on)")
    print()
    for split_name, df in [("train_phase1+2", combined), ("eval", eval_df)]:
        dist = df["target"].value_counts(normalize=True).sort_index()
        print(f"  {split_name} class distribution:")
        for cls, pct in dist.items():
            cnt = df["target"].eq(cls).sum()
            print(f"    {cls} ({LABEL_NAMES.get(cls, '?'):6s}): {cnt:4d}  ({pct:.2%})")
    print("=" * 70)

    X_eval = eval_df[FEATURE_COLS]
    y_eval = eval_df["target"]

    # ── Models to compare ─────────────────────────────────────────────────────
    candidates = {
        "extra_trees (800, default)": ExtraTreesClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=1, random_state=42
        ),
        "extra_trees (800, balanced)": ExtraTreesClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=42
        ),
        "random_forest (800, balanced)": RandomForestClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=42
        ),
        "gradient_boosting (300, lr=0.05)": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "hist_gradient_boosting (300, lr=0.05)": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=42
        ),
        "logistic_regression (scaled)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }

    # ── Benchmark on combined training data ───────────────────────────────────
    X_train = combined[FEATURE_COLS]
    y_train = combined["target"]

    print(f"\n{'MODEL':45s}  {'ACCURACY':>8}  {'F1 (wt)':>8}")
    print("-" * 70)

    results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")
        gate  = "✓ PASS" if acc >= 0.70 else "✗ FAIL"
        print(f"{name:45s}  {acc:8.4f}  {f1:8.4f}  {gate}")
        results.append((acc, name, model, preds))

    results.sort(reverse=True)
    best_acc, best_name, best_model, best_preds = results[0]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}")
    print(f"  Accuracy : {best_acc:.4f}")
    print(f"  F1 Score : {f1_score(y_eval, best_preds, average='weighted'):.4f}")
    print()
    print("Classification Report:")
    print(classification_report(
        y_eval, best_preds,
        target_names=[f"class_{c} ({LABEL_NAMES[c]})" for c in sorted(y_eval.unique())]
    ))
    cm = confusion_matrix(y_eval, best_preds)
    classes = sorted(y_eval.unique())
    print("Confusion Matrix (rows=true, cols=pred):")
    header = "         " + "  ".join(f"pred_{c}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        print(f"  true_{cls}: {cm[i]}")
    print("=" * 70)

    if best_acc >= 0.70:
        print(f"\n✓ Best model would PASS the 0.70 gate (acc={best_acc:.4f}).")
    else:
        print(f"\n✗ No model can pass the 0.70 gate on this dataset split.")
        print("  Consider collecting more data or revisiting the class boundaries.")


if __name__ == "__main__":
    main()
