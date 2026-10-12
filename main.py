
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Smart Die Care API",
    description="SVM-based diabetes severity prediction",
    version="1.0.0"
)

# Load model, scaler, and feature names
bundle = joblib.load("diabetes_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]


class PatientData(BaseModel):
    Age: float = Field(gt=0, le=120)
    Glucose: float = Field(gt=0)
    weight: float = Field(gt=0)
    height: float = Field(gt=0)
    bmi: float = Field(gt=0)
    systolic_bp: float = Field(gt=0)
    diastolic_bp: float = Field(gt=0)
    Hba1c: float = Field(gt=0)
    gender_Male: int = Field(ge=0, le=1)


@app.get("/")
def home():
    return {
        "message": "Smart Die Care API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": type(model).__name__,
        "features": len(feature_names)
    }


@app.post("/predict")
def predict(data: PatientData):
    try:
        patient_data = data.model_dump()

        # Preserve the exact training feature order
        patient_df = pd.DataFrame(
            [[patient_data[name] for name in feature_names]],
            columns=feature_names
        )

        # Use the scaler saved with the model
        scaled_data = scaler.transform(patient_df)

        prediction = model.predict(scaled_data)[0]

        result = {
            "prediction": str(prediction)
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]
            result["class_probabilities"] = {
                str(label): round(float(prob), 4)
                for label, prob in zip(model.classes_, probabilities)
            }

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
