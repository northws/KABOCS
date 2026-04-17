"""
Utility functions for the CO2RR Bayesian Optimization pipeline.

Provides data normalization, standardization, device selection,
and logging configuration.
"""

from __future__ import annotations

import logging
import random
import warnings
from typing import Optional

import matplotlib
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
_log_configured = False


def configure_logging() -> None:
    """Configure module-level logging (called once)."""
    global _log_configured
    if _log_configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Use non-interactive backend when saving plots
    matplotlib.use("Agg")
    # Suppress noisy warnings from botorch / gpytorch during fitting
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    _log_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the pipeline's configuration applied.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).

    Returns
    -------
    logging.Logger
    """
    configure_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def select_device(device: str = "auto") -> torch.device:
    """Select torch device.

    Parameters
    ----------
    device : str
        ``"auto"`` selects CUDA if available; otherwise ``"cpu"``
        or ``"cuda"`` directly.

    Returns
    -------
    torch.device
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducible BO runs.

    Parameters
    ----------
    seed : int
        Seed used for Python, NumPy and Torch random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Normalization / Standardization helpers
# ---------------------------------------------------------------------------
def compute_bounds(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute min, max, and safe range for normalization.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix, shape ``(N, K)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(x_min, x_max, x_range)`` where ``x_range`` has zeros
        replaced by 1.0 to avoid division-by-zero.
    """
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    return x_min, x_max, x_range


def normalize_x(
    X: np.ndarray,
    x_min: np.ndarray,
    x_range: np.ndarray,
) -> np.ndarray:
    """Normalize features to [0, 1].

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix, shape ``(N, K)``.
    x_min : np.ndarray
        Per-feature minimum values.
    x_range : np.ndarray
        Per-feature range (max − min).

    Returns
    -------
    np.ndarray
        Normalized features in [0, 1].
    """
    return (X - x_min) / x_range


def unnormalize_x(
    x_norm: torch.Tensor,
    bounds_raw: torch.Tensor,
) -> np.ndarray:
    """Convert normalized [0,1] features back to original scale.

    Parameters
    ----------
    x_norm : torch.Tensor
        Normalized feature vector, shape ``(K,)``.
    bounds_raw : torch.Tensor
        Raw bounds tensor, shape ``(2, K)`` — ``[min_values, max_values]``.

    Returns
    -------
    np.ndarray
        Feature vector in original physical units.
    """
    bounds_np = bounds_raw.cpu().numpy()
    x_min = bounds_np[0]
    x_max = bounds_np[1]
    x_np = x_norm.detach().cpu().numpy()
    return x_np * (x_max - x_min) + x_min


def standardize_y(Y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize target to zero mean, unit variance.

    Parameters
    ----------
    Y : np.ndarray
        Raw target values, shape ``(N,)``.

    Returns
    -------
    tuple[np.ndarray, float, float]
        ``(Y_standardized, y_mean, y_std)``.
    """
    y_mean = float(Y.mean())
    y_std = float(Y.std())
    if y_std < 1e-8:
        y_std = 1.0
    return (Y - y_mean) / y_std, y_mean, y_std


def unstandardize_y(y_std_val: float, y_mean: float, y_std: float) -> float:
    """Convert standardized Y back to original scale.

    Parameters
    ----------
    y_std_val : float
        Standardized target value.
    y_mean : float
        Original mean.
    y_std : float
        Original standard deviation.

    Returns
    -------
    float
        Target value in original scale.
    """
    return y_std_val * y_std + y_mean


# ---------------------------------------------------------------------------
# Integer / grid-snap helpers (P1 of discrete variables proposal)
# ---------------------------------------------------------------------------
def round_integer_dims_to_grid(
    X_norm: torch.Tensor,
    integer_indices: list[int],
    bounds_raw: torch.Tensor,
) -> torch.Tensor:
    """Snap integer dims in normalized [0,1] space to their nearest
    valid integer-grid point.

    For a raw integer dim with bounds ``(lo, hi)``, there are
    ``n = hi - lo`` normalized gridlines at ``{0, 1/n, 2/n, ..., 1}``
    corresponding to raw integers ``{lo, lo+1, ..., hi}``.  This function
    rounds each of the specified normalized dims to the nearest gridline
    and clamps to ``[0, 1]``.

    Does not mutate input; returns a new tensor.

    Parameters
    ----------
    X_norm : torch.Tensor
        Normalized feature tensor, shape ``(..., K)``.  Values assumed to
        lie in ``[0, 1]`` but this is not enforced.
    integer_indices : list[int]
        Dimension indices that must snap to an integer grid in raw space.
    bounds_raw : torch.Tensor
        Raw design-space bounds, shape ``(2, K)`` — row 0 is min, row 1
        is max.  Used to recover the raw integer step.

    Returns
    -------
    torch.Tensor
        New tensor of the same shape as ``X_norm`` with integer dims
        snapped to the nearest grid point.
    """
    if not integer_indices:
        return X_norm
    Y = X_norm.clone()
    for idx in integer_indices:
        lo = float(bounds_raw[0, idx].item())
        hi = float(bounds_raw[1, idx].item())
        span = hi - lo
        if span <= 0:
            # Degenerate dim (single allowed value): snap to 0.0
            Y[..., idx] = 0.0
            continue
        # Number of integer intervals; for lo=0, hi=6 -> 6 intervals, 7 values.
        n_intervals = int(round(span))
        if n_intervals <= 0:
            Y[..., idx] = 0.0
            continue
        step = 1.0 / n_intervals
        Y[..., idx] = (Y[..., idx] / step).round() * step
        Y[..., idx] = Y[..., idx].clamp(0.0, 1.0)
    return Y


def integer_indices_from_types(
    selected_features: list[str],
    feature_types: Optional[dict[str, str]],
) -> list[int]:
    """Extract integer-dim indices (within ``selected_features``).

    Parameters
    ----------
    selected_features : list[str]
        Ordered list of feature names currently used by the surrogate.
    feature_types : dict[str, str] or None
        Mapping from feature name to type label (``"continuous"`` |
        ``"integer"`` | ``"categorical"`` | ``"ordinal"``).  ``None`` is
        treated as "all continuous".

    Returns
    -------
    list[int]
        Indices (into ``selected_features``) whose declared type is
        ``"integer"``.  Empty list if no such features.
    """
    if not feature_types:
        return []
    return [
        idx
        for idx, name in enumerate(selected_features)
        if feature_types.get(name, "continuous") == "integer"
    ]


def categorical_indices_from_types(
    selected_features: list[str],
    feature_types: Optional[dict[str, str]],
) -> list[int]:
    """Extract categorical / ordinal dim indices (within
    ``selected_features``).  See :func:`integer_indices_from_types`.
    """
    if not feature_types:
        return []
    return [
        idx
        for idx, name in enumerate(selected_features)
        if feature_types.get(name, "continuous") in ("categorical", "ordinal")
    ]
