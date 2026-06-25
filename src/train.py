import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import sys
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

EVAL_THRESHOLD = 0.70

# Canonical feature order — must match serve.py PredictRequest field order
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


def _load_and_clean(path: str) -> pd.DataFrame:
    """Load a CSV and normalise column names (spaces → underscores)."""
    df = pd.read_csv(path)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def _print_class_dist(label: str, y: pd.Series) -> None:
    dist = y.value_counts(normalize=True).sort_index()
    name_map = {0: "low", 1: "medium", 2: "high"}
    print(f"\n  {label} class distribution ({len(y)} samples):")
    for cls, pct in dist.items():
        print(f"    class {cls} ({name_map.get(cls, '?'):6s}): {y.eq(cls).sum():4d}  ({pct:.2%})")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
    data_path2: str = None,
) -> float:
    df_train = _load_and_clean(data_path)

    # Optionally concatenate a second training phase
    if data_path2 and os.path.exists(data_path2):
        df_train2 = _load_and_clean(data_path2)
        df_train = pd.concat([df_train, df_train2], ignore_index=True)
        print(f"Loaded additional training data from {data_path2}")

    df_eval = _load_and_clean(eval_path)

    # Enforce canonical feature order — fail early on missing/extra columns
    missing = [c for c in FEATURE_COLS if c not in df_train.columns]
    if missing:
        raise ValueError(f"Training data is missing required features: {missing}")

    X_train = df_train[FEATURE_COLS]
    y_train = df_train["target"]
    X_eval  = df_eval[FEATURE_COLS]
    y_eval  = df_eval["target"]

    # Print class distributions for diagnostics / drift warning
    _print_class_dist("train", y_train)
    _print_class_dist("eval",  y_eval)
    print()

    label_counts = y_train.value_counts(normalize=True).to_dict()
    warnings = []
    for cls, ratio in label_counts.items():
        if ratio < 0.10:
            msg = f"WARNING: Class {cls} has low representation in training data ({ratio:.2%})"
            print(msg)
            warnings.append(msg)

    # Set up MLflow tracking URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():
        model_type   = params.get("model_type", "extra_trees")
        model_params = params.get(model_type, {})

        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        if model_type == "extra_trees":
            model = ExtraTreesClassifier(random_state=42, **model_params)
        elif model_type == "random_forest":
            model = RandomForestClassifier(random_state=42, **model_params)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=42, **model_params)
        elif model_type == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(random_state=42, **model_params)
        elif model_type == "logistic_regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, **model_params)),
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)

        preds          = model.predict(X_eval)
        acc            = accuracy_score(y_eval, preds)
        f1             = f1_score(y_eval, preds, average="weighted")
        precision_vals = precision_score(y_eval, preds, average=None, zero_division=0)
        recall_vals    = recall_score(y_eval, preds, average=None, zero_division=0)
        cm             = confusion_matrix(y_eval, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"\nConfusion Matrix (rows=true, cols=pred):")
        classes = sorted(y_eval.unique())
        header  = "       " + "  ".join(f"pred_{c}" for c in classes)
        print(header)
        for i, row_cls in enumerate(classes):
            print(f"  true_{row_cls}: {cm[i]}")

        # Metrics JSON
        os.makedirs("outputs", exist_ok=True)
        metrics_data = {
            "accuracy":           acc,
            "f1_score":           f1,
            "label_distribution": label_counts,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_data, f)

        # Report TXT
        with open("outputs/report.txt", "w") as f:
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Training samples: {len(X_train)}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Class Metrics:\n")
            for i, cls in enumerate(classes):
                p = precision_vals[i] if i < len(precision_vals) else 0.0
                r = recall_vals[i]    if i < len(recall_vals)    else 0.0
                name = {0: "low", 1: "medium", 2: "high"}.get(cls, "?")
                f.write(f"  Class {cls} ({name}): Precision={p:.4f}, Recall={r:.4f}\n")
            f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
            f.write(f"  {header}\n")
            for i, row_cls in enumerate(classes):
                f.write(f"  true_{row_cls}: {cm[i]}\n")
            if warnings:
                f.write("\nWarnings:\n")
                for w in warnings:
                    f.write(f"  - {w}\n")

        mlflow.log_artifact("outputs/metrics.json")
        mlflow.log_artifact("outputs/report.txt")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Use both training phases (phase1 + phase2) for maximum accuracy
    acc = train(
        params,
        data_path="data/train_phase1.csv",
        eval_path="data/eval.csv",
        data_path2="data/train_phase2.csv",
    )

    if acc < EVAL_THRESHOLD:
        print(f"\nFAILED: accuracy {acc:.4f} < {EVAL_THRESHOLD}. Deploy cancelled.")
        sys.exit(1)
    else:
        print(f"\nPASSED: accuracy {acc:.4f} >= {EVAL_THRESHOLD}.")
