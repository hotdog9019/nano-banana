from src.config import load_config
from src.data import load_dataset, make_split
from src.model import evaluate_model, fit_model
from src.predict import predict_ticket


def _bundle():
    config = load_config()
    frame = load_dataset(config["data_path"], config["text_column"], config["target_column"])
    return config, make_split(
        frame,
        config["text_column"],
        config["target_column"],
        config["test_size"],
        config["random_state"],
    )


def test_model_trains_and_returns_metrics():
    config, bundle = _bundle()
    model = fit_model(bundle, config)
    metrics = evaluate_model(model, bundle)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_prediction_contract():
    config, bundle = _bundle()
    model = fit_model(bundle, config)
    result = predict_ticket(model, "I was charged twice for my subscription")

    assert result["category"] in set(bundle.train[config["target_column"]])
    assert "confidence" in result

