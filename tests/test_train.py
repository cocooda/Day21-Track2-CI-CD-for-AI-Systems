import os
import json
import numpy as np
import pandas as pd
from src.train import train
from fastapi.testclient import TestClient
from src.serve import app


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path  = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path,  index=False)
    
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{str(tmp_path)}/mlflow_test.db"

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 3}},
        data_path=train_path,
        eval_path=eval_path
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 3}},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "label_distribution" in metrics


def test_report_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 3}},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.txt")


def test_model_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 3}},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")

# FastAPI serve tests
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_schema_extra_forbid():
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "wine_type": 0,
        "extra_field": 123  # This should be forbidden
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
