# Day 21 MLOps CI/CD Lab Report

## 1. Model Selection & Best Hyperparameters
- **Default Algorithm:** `random_forest` was chosen as the default because it handles non-linear relationships well, is robust to outliers, and generally provides a high baseline accuracy without extensive tuning.
- **Hyperparameters:** `n_estimators=100`, `max_depth=5`, `min_samples_split=2`. This configuration strikes a balance between model complexity and overfitting while comfortably exceeding the `0.70` accuracy threshold.
- **Algorithm Comparison:** The lab supports `random_forest`, `gradient_boosting`, and `logistic_regression` via `params.yaml`. Logistic Regression requires scaling and more iterations to converge, while Gradient Boosting provides slightly better metrics but takes longer to train.

## 2. Continuous Training (Step 3)
To trigger the CI/CD pipeline on new data, execute the following commands locally:

```bash
# 1. Update the training dataset with phase 2 data
python add_new_data.py

# 2. Add the updated file to DVC tracking
dvc add data/train_phase1.csv
dvc push

# 3. Commit the changes to trigger GitHub Actions
git add data/train_phase1.csv.dvc
git commit -m "feat: Add phase 2 data for continuous training"
git push origin main
```
*Note: GitHub Actions will detect changes in the `.dvc` file and run the full pipeline (Test -> Train & Eval -> Deploy).*

## 3. Deployment Evidence & API Usage
**Health Check Endpoint:**
```powershell
curl http://$env:VM_HOST:8000/health
```

**Predict Endpoint:**
```powershell
curl -X POST http://$env:VM_HOST:8000/predict `
     -H "Content-Type: application/json" `
     -d '{
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
       "wine_type": 0
     }'
```

## 4. Evidence Checklist
- [ ] **MLflow UI:** Shows at least 3 runs comparing algorithms/hyperparameters, tracking parameters, metrics (`accuracy`, `f1_score`), and artifacts.
- [ ] **GitHub Actions:** Shows 3 green jobs (`test`, `train`, `deploy`).
- [ ] **Cloud Storage:** Bucket contains DVC data chunks, plus `models/latest/model.pkl`, `metrics.json`, and `report.txt`.
- [ ] **FastAPI VM:** `curl` commands return `200 OK` and a valid prediction.

## 5. Bonus Features Implemented
1. **DagsHub MLflow:** Pipeline logs directly to DagsHub via secrets `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD`.
2. **Multiple Algorithms:** `train.py` dynamically builds `random_forest`, `gradient_boosting`, or `logistic_regression` based on `params.yaml`.
3. **Performance Report:** A text report with class-wise Precision/Recall and Confusion Matrix is generated as an artifact.
4. **No-Regression Gate:** The pipeline fetches the previously deployed `metrics.json` from GCS and blocks deployment if the new accuracy is lower.
5. **Data Drift Warning:** The training pipeline computes label distribution and warns if any class makes up less than 10% of the training set.

*DagsHub Tracking URL: `<Replace with your DagsHub URL>`*
