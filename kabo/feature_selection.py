"""
Phase 1: Feature Weight Evaluation & Selection.

Trains a Random Forest (or one of several alternatives) to evaluate
nonlinear relationships between the descriptors declared by the active
Task and the selected target product yield, then selects the top-K most
important features.

Supported ranking methods (v1.2):

* ``"random_forest"`` — legacy default; tree-based Gini importance.
* ``"permutation"``   — permutation importance on a fit RF; robust
                        baseline that is less biased toward high-cardinality
                        features.
* ``"mutual_info"``   — model-free; estimates I(X_j; y) with k-NN density
                        (sklearn's ``mutual_info_regression``).
* ``"shap"``          — TreeSHAP mean |φ_j| over the training set;
                        requires the optional ``shap`` package.

An optional correlation heatmap (Pearson) of the available descriptor
columns is also written so domain experts can spot multicollinearity
(e.g. catalyst-weight vs metal-loading) before the surrogate is fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:  # pragma: no cover — optional dep
    _shap = None  # type: ignore
    _HAS_SHAP = False

from kabo.utils import get_logger


FSMethod = Literal["random_forest", "permutation", "mutual_info", "shap"]
VALID_FS_METHODS: tuple[FSMethod, ...] = (
    "random_forest", "permutation", "mutual_info", "shap",
)

logger = get_logger(__name__)


def load_and_validate_data(
    data_path: Path,
    target_column: str,
    all_feature_columns: list[str],
    all_product_columns: list[str],
    product_names: dict[str, str],
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
    all_feature_columns : list[str]
        Full ordered descriptor list (supplied by the active Task).
    all_product_columns : list[str]
        Ordered list of every product yield column (supplied by Task).
    product_names : dict[str, str]
        Mapping column → display name (supplied by Task).
    strict_feature_schema : bool, optional
        If True, require every descriptor in ``all_feature_columns`` to
        be present (default False).

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
    available_features = [c for c in all_feature_columns if c in df.columns]
    if not available_features:
        raise ValueError(
            f"No recognized feature columns found in dataset. "
            f"Expected some of: {all_feature_columns}"
        )
    missing_features = [c for c in all_feature_columns if c not in df.columns]
    if strict_feature_schema and missing_features:
        raise ValueError(
            "Strict training schema is enabled, but the dataset is missing "
            f"{len(missing_features)} descriptor columns: {missing_features}. "
            "Please provide a complete descriptor schema or disable "
            "strict mode."
        )
    if (not strict_feature_schema) and missing_features:
        logger.warning(
            "Dataset is missing %d / %d descriptor columns; continuing with "
            "available feature subset.",
            len(missing_features), len(all_feature_columns),
        )

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Report available product columns
    available_products = [c for c in all_product_columns if c in df.columns]
    if available_products:
        product_name_list = [product_names.get(c, c) for c in available_products]
        logger.info("Available products: %s", ", ".join(product_name_list))
    else:
        logger.info("Using single-target column: %s", target_column)

    logger.info("Optimization target: %s", target_column)
    logger.info("Available features: %d / %d",
                len(available_features), len(all_feature_columns))
    return df


def train_random_forest(
    df: pd.DataFrame,
    target_column: str,
    all_feature_columns: list[str],
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
    all_feature_columns : list[str]
        Full ordered descriptor list (supplied by the active Task).
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
    available_features = [c for c in all_feature_columns if c in df.columns]
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


def rank_features(
    df: pd.DataFrame,
    target_column: str,
    all_feature_columns: list[str],
    method: FSMethod = "random_forest",
    n_estimators: int = 200,
    random_state: int = 42,
    permutation_repeats: int = 5,
    mutual_info_n_neighbors: int = 3,
) -> tuple[pd.Series, list[str]]:
    """Unified feature-ranking entry point.

    Delegates to the requested backend and always returns a pandas
    ``Series`` sorted in descending order of importance — the same
    shape ``train_random_forest`` has historically returned — so
    downstream code (``select_top_k_features``,
    ``plot_feature_importances``) works unchanged.

    Parameters
    ----------
    df, target_column, all_feature_columns, n_estimators, random_state
        Same semantics as ``train_random_forest``.
    method : one of ``VALID_FS_METHODS``
        Ranking backend.  ``"random_forest"`` reproduces the legacy
        Gini-importance path bit-for-bit.
    permutation_repeats : int, optional
        Number of random shuffles per feature for ``"permutation"``
        (default 5 — cheap).  Ignored for other methods.
    mutual_info_n_neighbors : int, optional
        ``n_neighbors`` for the k-NN density estimator in
        ``mutual_info_regression`` (default 3).  Ignored elsewhere.

    Returns
    -------
    tuple[pd.Series, list[str]]
        ``(sorted_importances, available_features)``.

    Raises
    ------
    ValueError
        Unknown ``method``.
    RuntimeError
        ``"shap"`` requested but the ``shap`` package is not installed.
    """
    if method not in VALID_FS_METHODS:
        raise ValueError(
            f"Unknown feature-selection method '{method}'. "
            f"Valid choices: {VALID_FS_METHODS}"
        )

    if method == "random_forest":
        _, importances, available = train_random_forest(
            df, target_column=target_column,
            all_feature_columns=all_feature_columns,
            n_estimators=n_estimators, random_state=random_state,
        )
        return importances, available

    available = [c for c in all_feature_columns if c in df.columns]
    if not available:
        raise ValueError(
            "No recognised descriptor columns in the dataset; "
            "cannot rank features."
        )
    X = df[available].to_numpy(dtype=np.float64)
    y = df[target_column].to_numpy(dtype=np.float64)

    if method == "permutation":
        # Fit a shallow RF first; permutation_importance needs a fitted
        # estimator.  The shallow depth keeps wall-clock sane on small N.
        n_rows = len(df)
        if n_rows < 10:
            logger.warning(
                "Small dataset (n=%d): reducing n_estimators for the "
                "permutation-importance RF.", n_rows,
            )
            trees = max(50, n_estimators // 4)
        else:
            trees = n_estimators
        rf = RandomForestRegressor(
            n_estimators=trees, random_state=random_state, n_jobs=-1,
        ).fit(X, y)
        result = permutation_importance(
            rf, X, y, n_repeats=permutation_repeats,
            random_state=random_state, n_jobs=-1,
        )
        scores = np.asarray(result.importances_mean, dtype=np.float64)
        # Permutation importance can produce small negatives on truly
        # noise features; clip to 0 so the downstream bar plot stays
        # sensible without losing ordering information.
        scores = np.clip(scores, a_min=0.0, a_max=None)

    elif method == "mutual_info":
        n_neighbors = max(1, min(mutual_info_n_neighbors, len(df) - 1))
        scores = mutual_info_regression(
            X, y, n_neighbors=n_neighbors, random_state=random_state,
        )
        scores = np.asarray(scores, dtype=np.float64)

    elif method == "shap":
        if not _HAS_SHAP:
            raise RuntimeError(
                "method='shap' requires the optional `shap` package. "
                "Install with `pip install shap`."
            )
        rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
        ).fit(X, y)
        explainer = _shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        scores = np.mean(np.abs(shap_values), axis=0)
        scores = np.asarray(scores, dtype=np.float64)

    else:  # pragma: no cover — unreachable thanks to the guard above
        raise ValueError(f"Unhandled method '{method}'")

    importances = pd.Series(
        scores, index=available, name=f"{method}_importance",
    ).sort_values(ascending=False)
    logger.info(
        "Feature ranking method '%s' computed for %d descriptors.",
        method, len(available),
    )
    return importances, available


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    title_suffix: str = "",
    method: str = "pearson",
) -> Optional[Path]:
    """Save a Pearson correlation heatmap for available descriptors.

    Helps spot multicollinearity before the GP surrogate is fit
    (two tightly correlated features inflate the ARD length-scale
    search and slow fitting).  Silently skips and returns ``None``
    when fewer than two descriptors are present.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    feature_columns : list[str]
        Columns to include; unknown columns are dropped with a warning.
    output_dir : Path
        Destination dir (created if missing).
    title_suffix : str, optional
        Extra text appended to the plot title (e.g. task / target name).
    method : {"pearson", "spearman", "kendall"}, optional
        Correlation method forwarded to ``pd.DataFrame.corr`` (default
        ``"pearson"``).

    Returns
    -------
    Path or None
        Path to the saved PNG, or ``None`` when skipped.
    """
    available = [c for c in feature_columns if c in df.columns]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        logger.warning(
            "Correlation heatmap: skipping %d missing columns: %s",
            len(missing), missing,
        )
    if len(available) < 2:
        logger.info(
            "Correlation heatmap: only %d usable column(s); skipping.",
            len(available),
        )
        return None

    corr = df[available].corr(method=method)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "correlation_heatmap.png"

    n = len(available)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), max(5, n * 0.3)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    # Cell annotations — only when matrix is small enough to remain readable.
    if n <= 25:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(v) > 0.5 else "black",
                )
    title = f"Feature correlation ({method})"
    if title_suffix:
        title += f"  — {title_suffix}"
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="corr")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Correlation heatmap saved to %s", out_path)
    return out_path


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
    product_names: dict[str, str] | None = None,
    task_name: str = "",
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
    product_names : dict[str, str] or None, optional
        Mapping column → display name (supplied by Task).
    task_name : str, optional
        Active task name, used as a title prefix (e.g. ``"CO2RR"``).

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
    if product_names is None:
        product_names = {}
    target_label = product_names.get(target_column, target_column)
    title_prefix = f"{task_name} " if task_name else ""
    ax.set_title(
        f"{title_prefix}Feature Importances — target: {target_label} (Random Forest)",
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
