"""Configuration models for data generation and training."""

from pydantic import BaseModel


class DataConfig(BaseModel):
    """Configuration for synthetic dataset generation."""

    n_samples: int = 1000
    n_features: int = 10
    n_classes: int = 2
    random_state: int = 42


class ModelConfig(BaseModel):
    """Configuration for model choice and parameters."""

    model_type: str = "logreg"
    max_depth: int | None = None
    n_estimators: int = 100


class TrainConfig(BaseModel):
    """Configuration for training process."""

    test_size: float = 0.2
