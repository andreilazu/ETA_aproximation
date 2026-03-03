# Copilot instructions for this repo

Short summary
- Small ML project: training script builds an XGBoost ETA model and the FastAPI app serves predictions.
- Artifacts: `eta_model.pkl` (contains `model`, `start_h3_mapping`, `end_h3_mapping`).

Key files
- `train.py` — data loading, feature engineering, model training, `joblib.dump(...)` to `eta_model.pkl`.
- `api.py` — FastAPI app; loads `eta_model.pkl` and exposes POST `/predict`.

Big-picture architecture
- Offline trainer (`train.py`) → produces artifact `eta_model.pkl`.
- Online serving (`api.py`) → loads artifact at startup and predicts from a JSON payload.
- Geospatial bucketing uses H3 (resolution 8). Categorical H3 values are mapped to integer codes and mappings are persisted in the artifact.

Important project conventions & patterns
- Feature order matters: `api.py` constructs the feature vector in the same order used during training. Keep `features` in `train.py` and the array in `api.py` synchronized.
- H3 handling: unseen H3 cells are mapped to `-1` in the API (`start_h3_mapping.get(..., -1)`). Training uses categorical encoding and saves mapping dicts — do not change the mapping key names in the artifact.
- Target units: training target is seconds (`trip_duration` in seconds). The API returns `eta_seconds` as float.
- Data filtering in `train.py`: trip durations kept between 60s and 10800s, and `Trip_Distance > 0`. These filters reflect production assumptions about valid trips.
- The code computes `hour_sin`/`hour_cos` but does not include them in `features`. Be careful when altering `features` — tests/validation will be needed.

Developer workflows (how to run things)
- Install dependencies (example):
```
pip install pandas numpy scikit-learn xgboost joblib fastapi uvicorn h3
```
- Train locally (requires `yellow_tripdata_2009-01.parquet` in repo root or adjust path):
```
python train.py
```
- Run the API server:
```
uvicorn api:app --reload --port 8000
```
- Predict (example):
```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"start_lat":40.7,"start_lon":-74.01,"end_lat":40.76,"end_lon":-73.98,"distance":2.1,"passenger_count":1,"hour":14,"day":2,"is_weekend":0}'
```

Integration points & external dependencies
- `h3` — used for geospatial bucketing. Keep H3 resolution consistent (`RESOLUTION = 8` in `train.py`).
- `joblib` — artifact serialization. Artifact must contain the three keys used by `api.py`: `model`, `start_h3_mapping`, `end_h3_mapping`.
- XGBoost model expects the exact feature order and types used in training.

Patterns to preserve when making changes
- If changing feature names/order, update both `train.py` `features` and `api.py` feature array simultaneously, and re-train the model.
- When changing H3 resolution or geographic bucketing, regenerate mappings and retrain; old artifacts are incompatible.
- Keep artifact filename `eta_model.pkl` or update `api.py` to match any rename.

Notes & gotchas discovered in the code
- `hour_sin`/`hour_cos` are computed but not used — this may be leftover experimentation.
- `train.py` uses `Passenger_Count` vs `passenger_count` naming conventions — when calling the API, use the lowercase names shown in the example payload (the API expects `passenger_count`).

What to ask next
- Do you want me to add a `requirements.txt` and a small `README.md` with run instructions?
- Should I adjust `api.py` to validate input fields and return clearer errors for missing keys?

If anything here is unclear or you'd like different emphasis, tell me which sections to expand or correct.
