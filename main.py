from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List

# 1. Load model and scaler
model = joblib.load("har_model.pkl")
scaler = joblib.load("scaler.pkl")

# 2. Define the translation dictionary
# Update these names to match your exact Kaggle dataset classes!
# Updated alphabetical mapping based on LabelEncoder
ACTIVITY_MAP = {
    "0": "LAYING",
    "1": "SITTING",
    "2": "STANDING",
    "3": "WALKING",
    "4": "WALKING_DOWNSTAIRS",
    "5": "WALKING_UPSTAIRS"
}
app = FastAPI()

class SensorData(BaseModel):
    features: List[float]

@app.post("/predict")
def predict_activity(data: SensorData):
    # Validate the data length
    if len(data.features) != scaler.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {scaler.n_features_in_} features but got {len(data.features)}"
        )

    # Process the data
    input_data = np.array(data.features).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    
    # Run the prediction
    prediction = model.predict(scaled_data)
    
    # Translate the result using our dictionary
    pred_str = str(prediction[0])
    english_label = ACTIVITY_MAP.get(pred_str, "Unknown Activity")

    # Return a beautifully structured JSON response
    return {
        "class_id": pred_str,
        "predicted_activity": english_label
    }