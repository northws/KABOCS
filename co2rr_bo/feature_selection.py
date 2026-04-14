"""
Phase 1: Feature Weight Evaluation & Selection.

Trains a Random Forest to evaluate nonlinear relationships between
the 19 physicochemical descriptors and the selected target product
yield, then selects the top-K most important features.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from co2rr_bo.constants import (
    ALL_FEATURE_COLUMNS,
    ALL_PRODUCT_COLUMNS,
    PRODUCT_COLUMNS,
    PRODUCT_NAMES,
    TARGET_COLUMN,
)
from co2rr_bo.utils import get_logger

logger = get_logger(__name__)


def load_and_validate_data(
    data_path: Path,
    target_column: str,
    strict_feature_schema: bool = False,
) -> pd.DataFrame:
    """Load dataset from CSV and validate required columns.

    Supports both legacy single-target format (column ``Y``) and
    multi-product format (columns ``Y_CO``, ``Y_HCOOH``, etc.).

    Parameters
    ----------
    data_path : Path
        Path to the CSV file.
    target_column : str
        Name of the target column to optimize (e.g. ``"Y_CO"``).
    strict_feature_schema : bool, optional
        If True, require all 19 descriptor columns in
        ``ALL_FEATURE_COLUMNS`` to be present (default False).

    Returns
    -------
    pd.DataFrame
        Validated dataset.

    Raises
    ------
    FileNotFoundError
        If ``data_path`` does not exist.
    ValueError
        If required columns are missing.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    n_rows, n_cols = df.shape
    logger.info("Loaded dataset: %d rows × %d columns from %s",
                n_rows, n_cols, data_path.name)

    # Validate feature columns
    available_features = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        raise ValueError(
            f"No recognized feature columns found in dataset. "
            f"Expected some of: {ALL_FEATURE_COLUMNS}"
        )
    missing_features = [c for c in ALL_FEATURE_COLUMNS if c not in df.columns]
    if strict_feature_schema and missing_features:
        raise ValueError(
            "Strict training schema is enabled, but the dataset is missing "
            f"{len(missing_features)} descriptor columns: {missing_features}. "
            "Please provide a complete 19/19 descriptor schema or disable "
            "strict mode."
        )
    if (not strict_feature_schema) and missing_features:
        logger.warning(
            "Dataset is missing %d / %d descriptor columns; continuing with "
            "available feature subset.",
            len(missing_features), len(ALL_FEATURE_COLUMNS),
        )

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Report available product columns
    available_products = [c for c in ALL_PRODUCT_COLUMNS if c in df.columns]
    if available_products:
        product_names = [PRODUCT_NAMES.get(c, c) for c in available_products]
        logger.info("Available products: %s", ", ".join(product_names))
    else:
        logger.info("Using single-target column: %s", target_column)

    logger.info("Optimization target: %s", target_column)
    logger.info("Available features: %d / %d",
                len(available_features), len(ALL_FEATURE_COLUMNS))
    return df


def train_random_forest(
    df: pd.DataFrame,
    target_column: str,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, pd.Series, list[str]]:
    """Train a Random Forest and extract feature importances.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature columns and target.
    target_column : str
        Name of the target column to train on.
    n_estimators : int, optional
        Number of trees (default 200). Automatically reduced
        for small datasets (<10 rows).
    random_state : int, optional
        Random seed for the forest bootstrap and feature subsampling
        (default 42).

    Returns
    -------
    tuple[RandomForestRegressor, pd.Series, list[str]]
        ``(rf_model, sorted_importances, available_features)``.
    """
    available_features = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    n_rows = len(df)

    # Handle edge case: very small dataset (<10 rows)
    if n_rows < 10:
        logger.warning(
            "⚠ Dataset has only %d rows (<10). Using reduced RF estimators "
            "and relaxed settings to avoid overfitting.", n_rows
        )
        n_estimators = min(n_estimators, max(10, n_rows * 2))

    X = df[available_features].values
    y = df[target_column].values

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=max(2, n_rows // 10),
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    r2_score = rf.score(X, y)
    logger.info("Random Forest R² (train): %.4f  (n_estimators=%d)",
                r2_score, n_estimators)

    # Extract and sort feature importances
    importances = pd.Series(
        rf.feature_importances_, index=available_features
    ).sort_values(ascending=False)

    logger.info("Feature importances (all) for target '%s':", target_column)
    for feat, imp in importances.items():
        logger.info("  %-35s  %.4f", feat, imp)

    return rf, importances, available_features


def select_top_k_features(
    importances: pd.Series,
    top_k: int = 10,
) -> list[str]:
    """Select the top-K most important features.

    Parameters
    ----------
    importances : pd.Series
        Sorted feature importances.
    top_k : int, optional
        Number of features to select (default 10).

    Returns
    -------
    list[str]
        Names of the top-K features.
    """
    effective_k = min(top_k, len(importances))
    if effective_k < top_k:
        logger.warning(
            "Requested top_k=%d but only %d features available. "
            "Using K=%d.", top_k, len(importances), effective_k
        )
    selected = importances.index[:effective_k].tolist()

    logger.info("Selected top %d features:", effective_k)
    for i, feat in enumerate(selected, 1):
        imp = importances[feat]
        logger.info("  [%2d] %-35s  importance=%.4f", i, feat, imp)

    return selected


def plot_feature_importances(
    importances: pd.Series,
    top_k: int,
    output_dir: Path,
    target_column: str = "",
) -> Path:
    """Generate and save a bar plot of feature importances.

    Parameters
    ----------
    importances : pd.Series
        Sorted feature importances.
    top_k : int
        Cutoff line position for top-K features.
    output_dir : Path
        Directory to save the plot.
    target_column : str, optional
        Target column name for the plot title.

    Returns
    -------
    Path
        Path to the saved plot file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(
        np.linspace(0.3, 0.9, len(importances))
    )

    bars = ax.barh(
        range(len(importances)),
        importances.values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=11)

    # Include target product in title
    target_label = PRODUCT_NAMES.get(target_column, target_column)
    ax.set_title(
        f"CO2RR Feature Importances — target: {target_label} (Random Forest)",
        fontsize=13, fontweight="bold",
    )

    # Annotate bars
    for bar, val in zip(bars, importances.values):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8,
        )

    # Mark top-K cutoff
    k = min(top_k, len(importances))
    ax.axhline(y=k - 0.5, color="red", linestyle="--", linewidth=1.2,
               label=f"Top-{k} cutoff")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plot_path = output_dir / "feature_importances.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to: %s", plot_path)
    return plot_path
