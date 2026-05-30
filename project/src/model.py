from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import DatasetBundle


def build_model(config: dict[str, Any]) -> Pipeline:
    model_cfg = config.get("model", {})
    ngram_range = (
        int(model_cfg.get("ngram_min", 1)),
        int(model_cfg.get("ngram_max", 2)),
    )
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer=str(model_cfg.get("analyzer", "char_wb")),
                    ngram_range=ngram_range,
                    max_features=int(model_cfg.get("max_features", 3000)),
                ),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    alpha=float(model_cfg.get("alpha", 0.0001)),
                    max_iter=int(model_cfg.get("max_iter", 2000)),
                    random_state=int(config.get("random_state", 42)),
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_candidate_models(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    max_features = int(model_cfg.get("max_features", 3000))
    char_range = (
        int(model_cfg.get("ngram_min", 3)),
        int(model_cfg.get("ngram_max", 5)),
    )
    return {
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "tfidf_word_complement_nb": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)),
                ("classifier", ComplementNB()),
            ]
        ),
        "tfidf_char_linear_svc": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=char_range,
                        max_features=max_features,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=int(config.get("random_state", 42)),
                    ),
                ),
            ]
        ),
        "tfidf_char_sgd_log_loss": build_model(config),
    }


def fit_baseline(bundle: DatasetBundle) -> DummyClassifier:
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(bundle.train[bundle.text_column], bundle.train[bundle.target_column])
    return baseline


def fit_model(bundle: DatasetBundle, config: dict[str, Any]) -> Pipeline:
    model = build_model(config)
    model.fit(bundle.train[bundle.text_column], bundle.train[bundle.target_column])
    return model


def evaluate_model(model: Any, bundle: DatasetBundle) -> dict[str, Any]:
    y_true = bundle.test[bundle.target_column]
    y_pred = model.predict(bundle.test[bundle.text_column])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def compare_models(bundle: DatasetBundle, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, model in build_candidate_models(config).items():
        model.fit(bundle.train[bundle.text_column], bundle.train[bundle.target_column])
        y_true = bundle.test[bundle.target_column]
        y_pred = model.predict(bundle.test[bundle.text_column])
        results[name] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        }
    return results


def make_confusion_matrix(model: Any, bundle: DatasetBundle) -> dict[str, Any]:
    labels = sorted(bundle.test[bundle.target_column].unique().tolist())
    matrix = confusion_matrix(
        bundle.test[bundle.target_column],
        model.predict(bundle.test[bundle.text_column]),
        labels=labels,
    )
    return {"labels": labels, "matrix": matrix.tolist()}


def make_prediction_table(model: Any, bundle: DatasetBundle) -> pd.DataFrame:
    texts = bundle.test[bundle.text_column]
    predictions = model.predict(texts)
    result = bundle.test.copy()
    result["predicted_category"] = predictions
    return result
