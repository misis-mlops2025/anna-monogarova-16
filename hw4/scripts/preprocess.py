import pandas as pd
from pathlib import Path

RAW_PATH = Path("/opt/data/raw.csv")
OUT_PATH = Path("/opt/data/processed.csv")

def main():
    df = pd.read_csv(RAW_PATH)

    # ожидаем колонки: f0..f19 и target
    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in raw.csv")

    # простая очистка (на будущее)
    df = df.dropna().reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved processed data to {OUT_PATH} with shape={df.shape}")

if __name__ == "__main__":
    main()
