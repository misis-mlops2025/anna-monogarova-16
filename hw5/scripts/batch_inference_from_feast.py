import argparse
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from feast import FeatureStore

# Корневой каталог проекта (hw5 или /opt внутри контейнера)
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "driver_stats.parquet"
FEAST_REPO_PATH = ROOT / "feast_repo"
MODEL_PATH = ROOT / "models" / "model_from_feast.joblib"
PREDICTIONS_DIR = ROOT / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help=(
            "Дата/время в формате ISO, например 2025-12-25T10:00:00. "
            "Если не указано – берём самое последнее время из данных."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Загружаем оффлайн-данные, чтобы понять доступные таймстемпы
    df = pd.read_parquet(DATA_PATH)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df = df.sort_values("event_timestamp")

    if args.as_of:
        as_of = datetime.fromisoformat(args.as_of)
        df = df[df["event_timestamp"] <= as_of]
    else:
        as_of = df["event_timestamp"].max()

    print(f"Using as_of={as_of.isoformat()}")
    all_ts = (
        df["event_timestamp"]
        .drop_duplicates()
        .sort_values()
    )
    last_ts = all_ts.tail(10)
    print(f"Last timestamps ({len(last_ts)}):")
    print(last_ts.to_list())

    # Собираем entity_df для Feast: driver_id + event_timestamp
    entity_df = (
        df[df["event_timestamp"].isin(last_ts)][["driver_id", 
"event_timestamp"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Entity DF shape: {entity_df.shape}")

    # Инициализируем FeatureStore
    store = FeatureStore(repo_path=str(FEAST_REPO_PATH))

    # Тянем фичи из Feast (исторические)
    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_stats:conv_rate",
            "driver_stats:acc_rate",
            "driver_stats:avg_daily_trips",
        ],
    ).to_df()

    print(f"Features shape: {features_df.shape}")

    # Признаки и таргет
    X = features_df[["conv_rate", "acc_rate"]]
    y = features_df.get("avg_daily_trips")

    # Загружаем модель и считаем предсказания
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)

    out = features_df[["driver_id", "event_timestamp"]].copy()
    out["prediction"] = preds

    if y is not None:
        out["target"] = y

    # Сохраняем предсказания
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = as_of.strftime("%Y%m%dT%H%M%S")
    out_path = PREDICTIONS_DIR / f"predictions_{ts_str}.csv"

    out.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()

