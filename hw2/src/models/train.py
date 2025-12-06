"""Entry point for running the training pipeline."""

from src.models.config import DataConfig, ModelConfig, TrainConfig
from src.models.pipeline import train_pipeline


def main():
    """Run training pipeline and print accuracy."""
    data_cfg = DataConfig()
    model_cfg = ModelConfig(model_type="forest")
    train_cfg = TrainConfig()

    acc = train_pipeline(data_cfg, model_cfg, train_cfg)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()


