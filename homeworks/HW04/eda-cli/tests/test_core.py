from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories, 
    compute_quality_flags,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    assert "age" in corr.columns or corr.empty is False
    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_new_heuristics():
    """
    Создаём датафрейм, в котором:
    - < 100 строк → too_few_rows = True
    - > 100 столбцов → too_many_columns = True
    - один столбец с >50% пропусков → too_many_missing = True
    - один константный столбец → has_constant_columns = True
    - один int-столбец со всеми уникальными значениями (n > 10) → has_high_cardinality = True
    - один строковый столбец с >50% уникальных значений → has_high_cardinality = True
    """
    n_rows = 80  # < 100 → too_few_rows
    
    # Генерируем 105 столбцов: 
    #   - 1: id (unique int → high card)
    #   - 1: const (константа)
    #   - 1: high_card_str (уникальные строки)
    #   - 1: mostly_missing (>50% пропусков)
    #   - 101: dummy_0 … dummy_100 → чтобы было >100 столбцов
    data = {
        "id": list(range(n_rows)),                            # int, unique → high card
        "const": ["A"] * n_rows,                             # константный
        "high_card_str": [f"s_{i}" for i in range(n_rows)],  # 100% уникальных → high card
        "mostly_missing": [1.0] * (n_rows // 3) + [None] * (n_rows - n_rows // 3),  # ~66% пропусков
    }
    # Добавляем 101 "мусорный" столбец, чтобы n_cols = 105 > 100
    for i in range(101):
        data[f"dummy_{i}"] = [0] * n_rows

    df = pd.DataFrame(data)

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # === Проверяем ВСЕ флаги явно ===
    assert flags["too_few_rows"] is True, "n_rows=80 < 100"
    assert flags["too_many_columns"] is True, "n_cols=105 > 100"
    assert flags["has_constant_columns"] is True, "'const' is constant"
    assert flags["has_high_cardinality"] is True, "'id' and 'high_card_str' have high cardinality"

    # Проверяем пропуски
    max_missing = flags["max_missing_share"]
    avg_missing = flags["avg_missing_share"]
    assert max_missing > 0.5, f"'mostly_missing' has ~{max_missing:.2%} missing"
    assert flags["too_many_missing"] is True

    # Проверяем скор: база 1.0
    # - avg_missing ≈ (0.66 * 1 + 0 * 104) / 105 ≈ 0.0063
    # - штрафы: 
    #     too_few_rows: -0.2
    #     too_many_columns: -0.1
    #     has_constant_columns: -0.1
    #     has_high_cardinality: -0.1
    # Итого: ~1.0 - 0.0063 - 0.5 = ~0.4937
    expected_score = 1.0 - avg_missing - 0.2 - 0.1 - 0.1 - 0.1
    assert abs(flags["quality_score"] - expected_score) < 0.02, \
        f"Score mismatch: got {flags['quality_score']:.3f}, expected ~{expected_score:.3f}"
    assert 0.0 <= flags["quality_score"] <= 1.0

    # Дополнительно: проверим, что все ожидаемые ключи присутствуют
    required_keys = {
        "too_few_rows", "too_many_columns",
        "has_constant_columns", "has_high_cardinality",
        "too_many_missing", "max_missing_share", "avg_missing_share",
        "quality_score"
    }
    assert required_keys.issubset(flags.keys()), f"Missing keys: {required_keys - flags.keys()}"