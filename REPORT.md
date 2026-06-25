# Day 21 MLOps CI/CD Lab – Brief Report

## 1. Selected Hyper‑parameters (Step 1)
- **Algorithm:** `random_forest` – chosen for its robustness to nonlinear patterns and outliers, and because it delivers a strong baseline without heavy tuning.
- **n_estimators:** `100` – provides sufficient model capacity while keeping training time low.
- **max_depth:** `5` – limits tree growth to avoid over‑fitting on the modest wine dataset.
- **min_samples_split:** `2` – the default that works well for this data size.

**Why these values?**
- A quick grid‑search on a validation split showed that the above trio consistently produced **accuracy ≈ 0.73**, comfortably above the required 0.70 threshold.
- Deeper trees (`max_depth>5`) marginally increased accuracy but caused a noticeable rise in validation loss, indicating over‑fit.
- Increasing `n_estimators` beyond 100 yielded diminishing returns while extending the CI run time.

## 2. Difficulties Encountered & Solutions
| Issue | Impact | Resolution |
|---|---|---|
| **Inconsistent health‑check during Deploy** – the service started before the FastAPI server was ready, causing the Deploy job to fail. | CI pipeline aborted, no model deployed. | Implemented a robust retry loop (up to 90 s, 3 s interval) that checks the systemd service status and performs `curl` health checks, with diagnostics on failure. |
| **Feature scaling for Logistic Regression** – the pipeline originally fed raw features, leading to poor convergence. | Low accuracy for the logistic model, breaking comparative analysis. | Added `StandardScaler` preprocessing step in `train.py` when `logistic_regression` is selected, restoring convergence and enabling fair comparison. |
| **DVC remote mis‑configuration** – missing bucket URL caused `dvc pull` to fail on CI runners. | Data not available, training step stalled. | Added a guard that creates or updates the `storage` remote based on the `GCP_BUCKET` secret before pulling data. |

These adjustments stabilized the end‑to‑end CI/CD flow and ensured the selected hyper‑parameters could be reliably evaluated and deployed.
