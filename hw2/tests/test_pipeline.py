from src.models.config import DataConfig, ModelConfig, TrainConfig
from src.models.pipeline import generate_data, build_model, train_pipeline


def test_generate_data():
    cfg = DataConfig(n_samples=50)
    X, y = generate_data(cfg)
    assert X.shape[0] == 50


from src.models.config import DataConfig, ModelConfig, TrainConfig
from src.models.pipeline import generate_data, build_model, train_pipeline


def test_generate_data():
    cfg = DataConfig(n_samples=50)
    X, y = generate_data(cfg)
    assert X.shape[0] == 50


def test_build_model():
    cfg = ModelConfig(model_type="logreg")
    model = build_model(cfg)
    assert model is not None


def test_train_pipeline_runs():
    acc = train_pipeline(DataConfig(n_samples=100), ModelConfig(), 
TrainConfig())
    assert 0 <= acc <= 1

