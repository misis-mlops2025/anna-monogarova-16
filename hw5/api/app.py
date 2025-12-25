from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feast import FeatureStore

import os

# Пути внутри ДОКЕР-КОНТЕЙНЕРА
FEAST_REPO_PATH = Path(os.getenv("FEAST_REPO_PATH", "/opt/feast_repo"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", 
"/opt/models/model_from_feast.joblib"))

# Загружаем FeatureStore и модель один раз при старте сервиса
store = FeatureStore(repo_path=str(FEAST_REPO_PATH))

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

FEATURES = [
    "driver_stats:conv_rate",
    "driver_stats:acc_rate",
    "driver_stats:avg_daily_trips",
]

app = FastAPI(title="HW5 Online Inference with Feast")


class PredictRequest(BaseModel):
    driver_id: int


class PredictResponse(BaseModel):
    driver_id: int
    conv_rate: float
    acc_rate: float
    avg_daily_trips_feat: float
    prediction: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # забираем онлайн-фичи из Feast Online Store
    entity_rows = [{"driver_id": req.driver_id}]

    online_feats = store.get_online_features(
        features=FEATURES,
        entity_rows=entity_rows,
        full_feature_names=False,  # будут conv_rate, acc_rate, avg_daily_trips
    ).to_dict()

    try:
        conv_raw = online_feats["conv_rate"][0]
        acc_raw = online_feats["acc_rate"][0]
        avg_trips_raw = online_feats["avg_daily_trips"][0]
    except KeyError:
        # Если какого-то ключа нет вообще
        raise HTTPException(
            status_code=404,
            detail=(
                f"Feature keys not found for driver_id={req.driver_id}. "
                f"Did you materialize online store?"
            ),
        )

    # Если Feast вернул None (фичи не найдены в онлайн-хранилище)
    if conv_raw is None or acc_raw is None or avg_trips_raw is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Features are None for driver_id={req.driver_id}. "
                f"Probably not materialized to online store."
            ),
        )

    conv = float(conv_raw)
    acc = float(acc_raw)
    avg_trips = float(avg_trips_raw)


    X = [[conv, acc, avg_trips]]
    pred = float(model.predict(X)[0])

    return PredictResponse(
        driver_id=req.driver_id,
        conv_rate=conv,
        acc_rate=acc,
        avg_daily_trips_feat=avg_trips,
        prediction=pred,
    )

