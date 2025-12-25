import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("hw5/data/driver_stats.parquet")


def main():
    rng = np.random.default_rng(42)
    driver_ids = np.arange(1, 51)  # 50 водителей
    ts = pd.date_range("2025-01-01", periods=10, freq="D")  # 10 таймстемпов

    rows = []
    for t in ts:
        for d in driver_ids:
            conv = rng.uniform(0.0, 1.0)
            acc = rng.uniform(0.0, 1.0)
            avg = 20 * conv + 10 * acc + rng.normal(0, 1.0)
            rows.append((int(d), t, float(conv), float(acc), float(avg)))

    df = pd.DataFrame(
        rows,
        columns=[
            "driver_id",
            "event_timestamp",
            "conv_rate",
            "acc_rate",
            "avg_daily_trips",
        ],
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print("Saved:", OUT, "rows:", len(df), "timestamps:", df['event_timestamp'].nunique())


if __name__ == "__main__":
    main()

