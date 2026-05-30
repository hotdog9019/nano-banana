from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .config import resolve_project_path


def load_model(path: str | Path) -> Any:
    return joblib.load(resolve_project_path(path))


def predict_ticket(model: Any, text: str) -> dict[str, Any]:
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Field 'text' must be a non-empty string.")

    label = str(model.predict([clean_text])[0])
    response: dict[str, Any] = {"category": label, "text": clean_text}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([clean_text])[0]
        classes = [str(item) for item in model.classes_]
        response["confidence"] = round(float(max(probabilities)), 4)
        response["probabilities"] = {
            cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)
        }
    return response

