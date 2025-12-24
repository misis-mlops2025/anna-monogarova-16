import os
from pathlib import Path

import joblib
import pandas as pd


NEW_DATA_PATH = Path("/opt/data/new_data.csv")
MODEL_PATH = Path("/opt/models/model.joblib")
OUT_DIR = Path("/opt/predictions")
OUT_PATH = OUT_DIR / "predictions.csv"


def main():
    if not NEW_DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {NEW_DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(NEW_DATA_PATH)

    # <-- FIX: если target есть (например, ты скопировала из processed.csv), убираем
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    model = joblib.load(MODEL_PATH)
    preds = model.predict(df)

    out = df.copy()
    out["prediction"] = preds
    out.to_csv(OUT_PATH, index=False)

    # по заданию: удалить исходный файл
    NEW_DATA_PATH.unlink()


if __name__ == "__main__":
    main()

