from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve_project_path


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    text_column: str
    target_column: str


def load_dataset(path: str | Path, text_column: str, target_column: str) -> pd.DataFrame:
    data_path = resolve_project_path(path)
    frame = pd.read_csv(data_path)
    required = {text_column, target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.dropna(subset=[text_column, target_column]).copy()
    frame[text_column] = frame[text_column].astype(str).str.strip()
    frame[target_column] = frame[target_column].astype(str).str.strip()
    frame = frame[frame[text_column] != ""]
    if frame[target_column].nunique() < 2:
        raise ValueError("Dataset must contain at least two target classes.")
    return frame


def make_split(
    frame: pd.DataFrame,
    text_column: str,
    target_column: str,
    test_size: float,
    random_state: int,
) -> DatasetBundle:
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[target_column],
    )
    return DatasetBundle(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        text_column=text_column,
        target_column=target_column,
    )

