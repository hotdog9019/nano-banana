from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .config import DEFAULT_CONFIG_PATH, load_config, resolve_project_path
from .data import load_dataset, make_split
from .model import compare_models, evaluate_model, fit_baseline, fit_model, make_confusion_matrix, make_prediction_table


def train(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config = load_config(config_path)
    text_column = str(config["text_column"])
    target_column = str(config["target_column"])

    frame = load_dataset(config["data_path"], text_column, target_column)
    bundle = make_split(
        frame,
        text_column=text_column,
        target_column=target_column,
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
    )

    baseline = fit_baseline(bundle)
    model = fit_model(bundle, config)

    metrics = {
        "dataset_rows": int(len(frame)),
        "classes": sorted(frame[target_column].unique().tolist()),
        "train_rows": int(len(bundle.train)),
        "test_rows": int(len(bundle.test)),
        "model_comparison": compare_models(bundle, config),
        "baseline": evaluate_model(baseline, bundle),
        "final_model": evaluate_model(model, bundle),
        "confusion_matrix": make_confusion_matrix(model, bundle),
    }

    artifacts = config["artifacts"]
    model_path = resolve_project_path(artifacts["model_path"])
    metrics_path = resolve_project_path(artifacts["metrics_path"])
    predictions_path = resolve_project_path(artifacts["predictions_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_prediction_table(model, bundle).to_csv(predictions_path, index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train support ticket classifier.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    metrics = train(args.config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
