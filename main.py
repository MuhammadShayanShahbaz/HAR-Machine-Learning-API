import os
import joblib
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# --- NEW: Database Libraries ---
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# --- DATABASE SETUP (The Data Access Layer) ---
DATABASE_URL = os.getenv("DATABASE_URL")

# Connect to the database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the Blueprint for our Table
class PredictionRecord(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    predicted_class = Column(String)
    predicted_activity = Column(String)

# Automatically create the table if it doesn't exist yet!
Base.metadata.create_all(bind=engine)
# ----------------------------------------------

model = joblib.load("har_model.pkl")
scaler = joblib.load("scaler.pkl")

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
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_api_key(api_key: str = Security(api_key_header)):
    if API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Security Configuration Error"
        )
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Access Denied: Invalid API Key"
    )
# -------------------------

class SensorData(BaseModel):
    features: List[float]

@app.get("/")
def health_check():
    return {
        "status": "Online",
        "message": "Welcome to the HAR Machine Learning API. Visit /docs to interact."
    }

@app.post("/predict")
def predict_activity(data: SensorData, api_key: str = Security(get_api_key)):
    if len(data.features) != scaler.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {scaler.n_features_in_} features but got {len(data.features)}"
        )

    input_data = np.array(data.features).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    pred_str = str(prediction[0])
    english_label = ACTIVITY_MAP.get(pred_str, "Unknown Activity")

    # --- NEW: Save the memory to the Database ---
    db = SessionLocal()
    try:
        new_log = PredictionRecord(
            predicted_class=pred_str,
            predicted_activity=english_label
        )
        db.add(new_log)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
    finally:
        db.close()
    # -------------------------------------------

    return {
        "class_id": pred_str,
        "predicted_activity": english_label
    }