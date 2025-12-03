"""Data loading and basic metadata helpers for Causal Playground."""

from __future__ import annotations

import logging
from typing import Iterable, List

import pandas as pd

logger = logging.getLogger("causal_playground")


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Load a CSV file or buffer into a pandas DataFrame."""
    try:
        df = pd.read_csv(path_or_buffer)
        logger.info("Loaded CSV - rows=%s cols=%s", df.shape[0], df.shape[1])
        return df
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load CSV: %s", exc)
        raise


def get_categorical_columns(df: pd.DataFrame, max_levels: int = 20) -> List[str]:
    """Return columns treated as categorical based on unique levels or dtype."""
    categorical_cols: List[str] = []
    for col in df.columns:
        series = df[col]
        # Count distinct non-null levels.
        n_unique = series.dropna().nunique()
        if n_unique <= max_levels or series.dtype == "object" or str(series.dtype).startswith(
            "category"
        ):
            categorical_cols.append(col)
    logger.info(
        "Inferred categorical columns (<=%s levels): %s",
        max_levels,
        categorical_cols,
    )
    return categorical_cols


def basic_metadata(df: pd.DataFrame, max_levels: int = 20) -> dict:
    """Infer basic dataset metadata for UI wiring."""
    categorical_cols = get_categorical_columns(df, max_levels=max_levels)
    return {
        "columns": list(df.columns),
        "categorical_columns": categorical_cols,
        "non_categorical_columns": [c for c in df.columns if c not in categorical_cols],
    }
