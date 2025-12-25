from pathlib import Path
import json

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

# Пути
FEAST_REPO_PATH = ROOT / "feast_repo"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model_from_feast.joblib"
METRICS_PATH = MODEL_DIR / "metrics_from_feast.json"

# Это тот же parquet, который использует Feast как offline store
OFFLINE_DATA_PATH = ROOT / "data" / "driver_stats.parquet"


def main():
    # 1. Загружаем исходные данные, чтобы выбрать 10 таймстемпов
    if not OFFLINE_DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден parquet с данными: {OFFLINE_DATA_PATH}")

    df = pd.read_parquet(OFFLINE_DATA_PATH)
    df = df.sort_values("event_timestamp")

    # Берём последние 10 уникальных таймстемпов
    uniq_ts = df["event_timestamp"].drop_duplicates().sort_values()
    last_10_ts = uniq_ts.tail(10)

    df_10 = df[df["event_timestamp"].isin(last_10_ts)].copy()

    print(f"Всего строк в offline-данных: {len(df)}")
    print(f"Строк по последним 10 таймстемпам: {len(df_10)}")
    print(f"Уникальных driver_id: {df_10['driver_id'].nunique()}")

    # 2. Готовим entity_df для запроса в Feast
    entity_df = df_10[["driver_id", "event_timestamp"]].drop_duplicates()

    # 3. Подключаемся к Feast и вытаскиваем фичи
    store = FeatureStore(repo_path=str(FEAST_REPO_PATH))

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_stats:conv_rate",
            "driver_stats:acc_rate",
            "driver_stats:avg_daily_trips",
        ],
    ).to_df()

    print(f"Получено строк из Feast: {len(training_df)}")
    print("Колонки:", list(training_df.columns))

    # 4. Формируем X и y
    X = training_df[["conv_rate", "acc_rate"]]
    y = training_df["avg_daily_trips"]

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

    print("Модель (из Feast) сохранена в:", MODEL_PATH)
    print("Метрики (из Feast) сохранены в:", METRICS_PATH)
    print("Metrics from Feast:", metrics)


if __name__ == "__main__":
    main()

