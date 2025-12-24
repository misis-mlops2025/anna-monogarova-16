"""Entry point for running the training pipeline with DVC."""

from pathlib import Path
import json
import pickle
import yaml

from src.models.config import DataConfig, ModelConfig, TrainConfig
from src.models.pipeline import train_pipeline


def main():
    """Train model and save artifacts."""
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    data_cfg = DataConfig(**params["data"])
    model_cfg = ModelConfig(**params["model"])
    train_cfg = TrainConfig(**params["train"])

    accuracy, model = train_pipeline(data_cfg, model_cfg, train_cfg)

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    with open("models/model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("reports/metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump({"accuracy": accuracy}, metrics_file)


if __name__ == "__main__":
    main()
