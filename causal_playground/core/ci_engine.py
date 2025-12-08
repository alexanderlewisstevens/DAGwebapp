"""Conditional independence computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import logging

logger = logging.getLogger("causal_playground")

def cramers_v_from_table(table: pd.DataFrame, chi2_stat: float | None = None) -> float:
    """Compute Cramér's V effect size for a contingency table."""
    n = table.values.sum()
    if n == 0:
        return float("nan")
    r, c = table.shape
    min_dim = min(r - 1, c - 1)
    if min_dim <= 0:
        return float("nan")
    if chi2_stat is None:
        chi2_stat, _, _, _ = chi2_contingency(table)
    return float(np.sqrt(chi2_stat / (n * min_dim)))


def contingency_details(sub_df: pd.DataFrame, x: str, y: str) -> Dict[str, pd.DataFrame]:
    """Return contingency counts and probability tables for a slice."""
    table = pd.crosstab(sub_df[x], sub_df[y])
    total = table.values.sum()
    joint_probs = table / total if total > 0 else table.astype(float)
    row_probs = table.div(table.sum(axis=1), axis=0).fillna(0.0)
    col_probs = table.div(table.sum(axis=0), axis=1).fillna(0.0)
    return {
        "table": table,
        "joint_probs": joint_probs,
        "row_probs": row_probs,
        "col_probs": col_probs,
    }


def _condition_label(group_key: Any, conds: Sequence[str]) -> str:
    if not conds:
        return "All"
    if len(conds) == 1:
        # Pandas may pass a scalar or a single-element tuple for single groupers.
        key_value = group_key[0] if isinstance(group_key, tuple) else group_key
        key_tuple = (key_value,)
    else:
        key_tuple = tuple(group_key)
    condition_pairs = [f"{name}={val}" for name, val in zip(conds, key_tuple)]
    return ", ".join(condition_pairs)


def conditional_ci_summary(
    df: pd.DataFrame, x: str, y: str, conds: Sequence[str] | None
) -> pd.DataFrame:
    """Compute chi-square tests of independence across conditioning slices with contingency details."""
    conds = conds or []
    rows: List[Dict[str, Any]] = []
    if conds:
        grouped = df.groupby(list(conds), dropna=False)
        try:
            group_count = len(grouped)
        except TypeError:
            group_count = None
        group_items = grouped
    else:
        group_items = [("All", df)]
        group_count = 1

    logger.debug(
        "Running conditional CI summary - x=%s y=%s conds=%s groups=%s",
        x,
        y,
        conds,
        group_count if group_count is not None else "unknown",
    )

    for group_key, sub in group_items:
        if sub.empty:
            logger.warning("Skipping empty slice for condition=%s", group_key)
            continue
        table = pd.crosstab(sub[x], sub[y])
        if table.shape[0] == 0 or table.shape[1] == 0:
            logger.warning("Skipping degenerate contingency table for condition=%s", group_key)
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        cramers_v = cramers_v_from_table(table, chi2_stat=chi2)
        counts = table.values.tolist()
        probs = (table / table.values.sum()).fillna(0.0).values.tolist()
        rows.append(
            {
                "condition": _condition_label(group_key, conds),
                "n": int(sub.shape[0]),
                "chi2": float(chi2),
                "p": float(p),
                "dof": int(dof),
                "cramers_v": float(cramers_v),
                "contingency_counts": counts,
                "contingency_probs": probs,
                "x_levels": list(table.index),
                "y_levels": list(table.columns),
            }
        )
    return pd.DataFrame(rows)


def test_independencies(
    df: pd.DataFrame, independencies: list[tuple[str, str, tuple[str, ...]]], alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run CI tests for a list of (x, y, conds) tuples and return decisions.

    For each triple we run conditional_ci_summary once and summarize the first slice.
    """
    results: list[Dict[str, Any]] = []
    for x, y, conds in independencies:
        summary = conditional_ci_summary(df, x, y, list(conds))
        if summary.empty:
            logger.warning("CI summary empty for x=%s y=%s conds=%s", x, y, conds)
            continue
        row = summary.iloc[0]
        p_val = float(row["p"])
        decision = "fail_to_reject" if p_val >= alpha else "reject"
        results.append(
            {
                "x": x,
                "y": y,
                "conds": conds,
                "p_value": p_val,
                "decision": decision,
                "n": int(row["n"]),
            }
        )
    logger.info(
        "Tested implied independencies - count=%s supported=%s violated=%s",
        len(results),
        sum(1 for r in results if r["decision"] == "fail_to_reject"),
        sum(1 for r in results if r["decision"] == "reject"),
    )
    return pd.DataFrame(results)
