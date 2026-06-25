import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

EVAL_THRESHOLD = 0.70

def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]
    
    # Calculate label distribution for drift warning
    label_counts = y_train.value_counts(normalize=True).to_dict()
    warnings = []
    for cls, ratio in label_counts.items():
        if ratio < 0.10:
            msg = f"WARNING: Class {cls} has low representation in training data ({ratio:.2%})"
            print(msg)
            warnings.append(msg)

    # Set up MLflow tracking URI if provided
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Local fallback
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():
        model_type = params.get("model_type", "random_forest")
        model_params = params.get(model_type, {})
        
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=42, **model_params)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=42, **model_params)
        elif model_type == "logistic_regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, **model_params))
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")
        precision_vals = precision_score(y_eval, preds, average=None, zero_division=0)
        recall_vals = recall_score(y_eval, preds, average=None, zero_division=0)
        cm = confusion_matrix(y_eval, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Metrics JSON
        os.makedirs("outputs", exist_ok=True)
        metrics_data = {
            "accuracy": acc, 
            "f1_score": f1,
            "label_distribution": label_counts
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_data, f)
            
        # Report TXT
        with open("outputs/report.txt", "w") as f:
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Class Metrics:\n")
            classes = sorted(y_eval.unique())
            for i, cls in enumerate(classes):
                # Handle case where test set might not have all classes
                p = precision_vals[i] if i < len(precision_vals) else 0.0
                r = recall_vals[i] if i < len(recall_vals) else 0.0
                f.write(f" Class {cls}: Precision = {p:.4f}, Recall = {r:.4f}\n")
            
            f.write("\nConfusion Matrix:\n")
            for row in cm:
                f.write(f" {row}\n")
                
            if warnings:
                f.write("\nWarnings:\n")
                for w in warnings:
                    f.write(f" - {w}\n")

        mlflow.log_artifact("outputs/metrics.json")
        mlflow.log_artifact("outputs/report.txt")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
