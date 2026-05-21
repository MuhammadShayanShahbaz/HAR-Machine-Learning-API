import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# 1. Load the hidden variables from the .env file
load_dotenv()

# 2. Load ML model and scaler
model = joblib.load("har_model.pkl")
scaler = joblib.load("scaler.pkl")

# 3. Define the translation dictionary
ACTIVITY_MAP = {
    "0": "LAYING",
    "1": "SITTING",
    "2": "STANDING",
    "3": "WALKING",
    "4": "WALKING_DOWNSTAIRS",
    "5": "WALKING_UPSTAIRS"
}

app = FastAPI()

# --- SECURITY PROTOCOL ---
# Fetch the key securely from the operating system environment
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Access Denied: Invalid or missing API Key"
    )
# -------------------------

class SensorData(BaseModel):
    features: List[float]

@app.get("/")
def health_check():
    return {
        "status": "Online",
        "message": "Welcome to the HAR Machine Learning API. Visit https://har-machine-learning-api.onrender.com/docs to interact with the model."
    }

# Notice the addition of the 'api_key' dependency here to lock the door!
@app.post("/predict")
def predict_activity(data: SensorData, api_key: str = Security(get_api_key)):
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