from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }

def build_summary(df: pd.DataFrame):
    n_rows = len(df)
    n_cols = len(df.columns)

    # Число константных колонок
    n_const_cols = 0
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() == 1:
            n_const_cols += 1

    # Категориальные колонки
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(cat_cols) > 0:
        max_cat_card = int(df[cat_cols].nunique(dropna=True).max())
    else:
        max_cat_card = 0

    from types import SimpleNamespace
    return SimpleNamespace(
        n_rows=n_rows,
        n_cols=n_cols,
        n_const_cols=n_const_cols,
        max_cat_card=max_cat_card,
    )



def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значени.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(summary, missing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Вычисляет флаги качества данных и общий скор качества на основе:
    - доли пропущенных значений,
    - наличия постоянных (константных) столбцов,
    - высокой кардинальности признаков,
    - малого числа строк,
    - большого числа столбцов.
    """
    flags: Dict[str, Any] = {}
    if missing_df.empty:
        avg_missing = 0.0
        max_missing_share = 0.0
    else:
        avg_missing = float(missing_df["missing_share"].mean())
        max_missing_share = float(missing_df["missing_share"].max())
    flags["avg_missing_share"] = avg_missing
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100
    has_constant_columns = False
    has_high_cardinality = False
    for col in summary.columns:
        non_missing = col.non_null
        if non_missing > 0 and col.unique == 1:
            has_constant_columns = True
        dtype_clean = col.dtype.split("[")[0].lower()
        if dtype_clean in ("object", "string", "category"):
            if non_missing > 0 and col.unique > 0.5 * non_missing:
                has_high_cardinality = True
        elif dtype_clean in ("int64", "int32", "int"):
            if non_missing > 10 and col.unique == non_missing:
                has_high_cardinality = True
    flags["has_constant_columns"] = has_constant_columns
    flags["has_high_cardinality"] = has_high_cardinality
    score = 1.0
    score -= avg_missing
    if has_constant_columns:
        score -= 0.1
    if has_high_cardinality:
        score -= 0.1
    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1
    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score
    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)
