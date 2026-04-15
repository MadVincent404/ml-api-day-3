import joblib, numpy as np
from pathlib import Path
from app.schemas import IrisFeatures, PredictionResponse

LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class ModelService:
    def __init__(self):
        model_path = Path("models/model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")
        self.model = joblib.load(model_path)
        self.version = "1.0.0"

    def predict(self, features: IrisFeatures) -> PredictionResponse:
        X = np.array([[
            features.sepal_length, features.sepal_width,
            features.petal_length, features.petal_width
        ]])
        pred = int(self.model.predict(X)[0])
        proba = float(self.model.predict_proba(X)[0][pred])
        return PredictionResponse(
            prediction=pred,
            label=LABELS[pred],
            confidence=round(proba, 4)
        )

    def predict_batch(self, observations):
        return [self.predict(obs) for obs in observations]

# Singleton — chargé une seule fois au démarrage
model_service = ModelService()