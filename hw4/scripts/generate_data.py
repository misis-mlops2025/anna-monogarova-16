from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def main() -> None:
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    out = Path("/opt/data/raw.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved: {out} rows={len(df)} cols={df.shape[1]}")


if __name__ == "__main__":
    main()

