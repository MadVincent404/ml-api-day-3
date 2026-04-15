from pydantic import BaseModel, Field, validator
from typing import List

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, description="cm")
    sepal_width:  float = Field(..., gt=0, lt=20)
    petal_length: float = Field(..., gt=0, lt=20)
    petal_width:  float = Field(..., gt=0, lt=20)

    # Exemple affiché dans /docs
    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    }

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float

class BatchRequest(BaseModel):
    observations: List[IrisFeatures]