from fastapi import FastAPI, HTTPException
from app.schemas import IrisFeatures, BatchRequest
from app.model import model_service
import time

app = FastAPI(
    title="Iris Prediction API",
    version="1.0.0",
    description="API de classification d'iris — Jour 2 ML Engineer"
)

START_TIME = time.time()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_version": model_service.version
    }

@app.get("/info")
def info():
    return {
        "model": "RandomForestClassifier",
        "version": model_service.version,
        "classes": ["setosa", "versicolor", "virginica"],
        "features": ["sepal_length", "sepal_width",
                     "petal_length", "petal_width"]
    }

@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        return model_service.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if len(request.observations) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 observations par batch"
        )
    return {"predictions": model_service.predict_batch(request.observations)}