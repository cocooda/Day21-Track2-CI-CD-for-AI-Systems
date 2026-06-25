from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import os

app = FastAPI()

MODEL_PATH = "models/model.pkl"

# Load the model locally when server starts.
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Prediction will fail.")

class PredictRequest(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    wine_type: float
    
    model_config = ConfigDict(extra="forbid")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = [
        req.fixed_acidity,
        req.volatile_acidity,
        req.citric_acid,
        req.residual_sugar,
        req.chlorides,
        req.free_sulfur_dioxide,
        req.total_sulfur_dioxide,
        req.density,
        req.pH,
        req.sulphates,
        req.alcohol,
        req.wine_type
    ]

    try:
        pred = model.predict([features])[0]
        label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
        label = label_map.get(pred, "unknown")
        
        return {"prediction": int(pred), "label": label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
