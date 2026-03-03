from fastapi import FastAPI
import joblib
import h3
import numpy as np
import math

app = FastAPI()

# Load model
artifact = joblib.load("eta_model_enhanced.pkl")
model = artifact["model"]
start_h3_mapping = artifact["start_h3_mapping"]
end_h3_mapping = artifact["end_h3_mapping"]

@app.post("/predict")
def predict(data: dict):
    # 1. H3 Encoding (Fixed for version 4)
    start_h3_str = h3.latlng_to_cell(data["start_lat"], data["start_lon"], 8)
    end_h3_str = h3.latlng_to_cell(data["end_lat"], data["end_lon"], 8)

    # Map to integer (fallback -1 if unknown)
    start_h3 = start_h3_mapping.get(start_h3_str, -1)
    end_h3 = end_h3_mapping.get(end_h3_str, -1)

    # 2. Compute Cyclical Features
    hour_sin = np.sin(2 * np.pi * data["hour"] / 24)
    hour_cos = np.cos(2 * np.pi * data["hour"] / 24)
    day_sin = np.sin(2 * np.pi * data["day_of_week"] / 7)
    day_cos = np.cos(2 * np.pi * data["day_of_week"] / 7)

    # 3. Compute Rush Hour (Simple logic for API)
    # Assumes data["day_of_week"] is 0-6
    is_rush_hour = 1 if (
        ((data["hour"] >= 7 and data["hour"] <= 10) or 
         (data["hour"] >= 16 and data["hour"] <= 19)) and 
        data["day_of_week"] < 5
    ) else 0

    # 4. Construct Feature Array (Order matters! Must match train.py)
    # Note: The API caller should provide weather, or we default to averages
    features = np.array([[
        data["distance"],
        data.get("passenger_count", 1),
        start_h3,
        end_h3,
        hour_sin,
        hour_cos,
        day_sin,
        day_cos,
        data.get("is_holiday", 0),  # User must provide or default 0
        is_rush_hour,
        data.get("week_of_year", 1),
        data.get("temperature", 10.0), # Default approx 10 C
        data.get("rain", 0.0),
        data.get("snow", 0.0),
        data.get("wind_speed", 10.0)
    ]])

    prediction = model.predict(features)[0]
    return {"eta_seconds": float(prediction), "eta_minutes": float(prediction)/60}