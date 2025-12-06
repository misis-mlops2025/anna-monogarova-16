from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.models.config import DataConfig, ModelConfig, TrainConfig


def generate_data(cfg: DataConfig):
    return make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_classes=cfg.n_classes,
        random_state=cfg.random_state
    )


def build_model(cfg: ModelConfig):
    if cfg.model_type == "logreg":
        return LogisticRegression(max_iter=1000)
    if cfg.model_type == "tree":
        return DecisionTreeClassifier(max_depth=cfg.max_depth)
    if cfg.model_type == "forest":
        return RandomForestClassifier(n_estimators=cfg.n_estimators)
    raise ValueError("Unknown model type")


def train_pipeline(data_cfg, model_cfg, train_cfg):
    X, y = generate_data(data_cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg.test_size
    )

    model = build_model(model_cfg)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

