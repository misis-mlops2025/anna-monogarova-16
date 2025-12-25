from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Пути
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "driver_stats.parquet"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл с данными: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    # сортируем по времени
    df = df.sort_values("event_timestamp")

    # берём последние 10 уникальных таймстемпов
    uniq_ts = df["event_timestamp"].drop_duplicates().sort_values()
    last_10_ts = uniq_ts.tail(10)
    df_10 = df[df["event_timestamp"].isin(last_10_ts)]

    print(f"Всего строк в данных: {len(df)}")
    print(f"Строк в выборке по 10 таймстемпам: {len(df_10)}")
    print(f"Количество уникальных driver_id: {df_10['driver_id'].nunique()}")

    # Признаки и таргет по заданию
    X = df_10[["conv_rate", "acc_rate"]]
    y = df_10["avg_daily_trips"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "mse": mse,
        "r2": r2,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Модель сохранена в:", MODEL_PATH)
    print("Метрики сохранены в:", METRICS_PATH)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()

