"""Plotting helpers returning Matplotlib figures for Dash rendering."""

from __future__ import annotations

import matplotlib

# Non-interactive backend to prevent GUI requirements in server contexts.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_slice_countplot(sub_df: pd.DataFrame, x: str, y: str):
    """Return a countplot for the selected slice."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if sub_df.empty:
        ax.text(0.5, 0.5, "No data for selection", ha="center", va="center")
        ax.axis("off")
        return fig
    sns.countplot(data=sub_df, x=x, hue=y, ax=ax)
    ax.set_title(f"{x} vs {y}")
    fig.tight_layout()
    return fig


def plot_slice_heatmap(sub_df: pd.DataFrame, x: str, y: str):
    """Return a heatmap of the contingency table for the selected slice."""
    table = pd.crosstab(sub_df[x], sub_df[y])
    fig, ax = plt.subplots(figsize=(6, 4))
    if table.empty:
        ax.text(0.5, 0.5, "No data for selection", ha="center", va="center")
        ax.axis("off")
        return fig
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel(y)
    ax.set_ylabel(x)
    ax.set_title("Contingency counts")
    fig.tight_layout()
    return fig
