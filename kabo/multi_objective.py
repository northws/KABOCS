"""
Multi-objective Bayesian Optimization (qEHVI / qNEHVI) helpers.

This module is the entire "MO layer" that bolts on top of the existing
single-objective engine.  It provides:

* :class:`ObjectiveSpec` — a declarative description of one objective
  (column name, direction, optional reference-point override).
* :class:`MultiObjectiveSurrogate` — wraps ``botorch.models.ModelListGP``
  around one :class:`~kabo.surrogate.SurrogateModel` per objective.
* :func:`build_qnehvi` — factory for BoTorch's
  ``qNoisyExpectedHypervolumeImprovement`` acquisition.  We default to
  qNEHVI (not qEHVI) because experimental catalysis data is *always*
  noisy and qNEHVI draws Pareto fronts from the posterior rather than
  assuming a known one — this is strictly more robust.
* :func:`compute_pareto_front` — a thin wrapper around BoTorch's
  ``is_non_dominated`` for post-run analysis.
* :func:`infer_ref_point` — a conservative default for the hypervolume
  reference point when the user did not provide one.
* :func:`plot_pareto_front` — a matplotlib scatter of observed vs.
  Pareto-optimal points (2D or 3D objectives only).

Design notes
------------
* **Direction handling.**  BoTorch's HV utilities maximise every
  objective.  For ``direction="min"`` objectives we flip the sign of
  the corresponding training target *inside* this module — the rest of
  the pipeline continues to see the raw-scale numbers.
* **Independent GPs.**  Each objective owns a full SurrogateModel
  (ARD Matérn, noise term, standardisation).  This is deliberately
  simpler than a multi-output kernel (MultitaskGP); it lets us reuse
  the entire fit / log / integer-snap pipeline verbatim.  An
  informative message is logged if the user might benefit from a
  correlated surrogate, but we do not implement one here.
* **No changes to the single-objective path.**  Callers should route
  through either the single- or multi-objective API *before* entering
  phase-2 of the optimiser; no file in the legacy pipeline branches on
  an ``if multi_obj`` check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from kabo.utils import get_logger

logger = get_logger(__name__)


if TYPE_CHECKING:  # pragma: no cover
    import torch
    from botorch.acquisition.multi_objective.monte_carlo import (
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.models.model_list_gp_regression import ModelListGP


# =============================================================================
#  Public data classes
# =============================================================================
@dataclass(frozen=True)
class ObjectiveSpec:
    """Declarative description of one BO objective.

    Parameters
    ----------
    column : str
        DataFrame column that stores the observed values of this
        objective.  Must exist in the dataset.
    direction : {"max", "min"}
        Whether larger or smaller values are better.  Default ``"max"``.
    ref_point : float or None, optional
        Hypervolume reference value *on the raw scale*.  For
        ``direction="max"`` this should be **dominated by all feasible
        observations** (i.e. a worst-case lower bound).  For
        ``direction="min"`` it is an upper bound.  If ``None`` (default),
        a conservative value is inferred from the observed data by
        :func:`infer_ref_point`.
    display_name : str or None, optional
        Pretty name used in logs and plot axes; falls back to ``column``.
    """

    column: str
    direction: str = "max"
    ref_point: Optional[float] = None
    display_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.direction not in {"max", "min"}:
            raise ValueError(
                f"ObjectiveSpec(direction={self.direction!r}): must be "
                f"'max' or 'min'."
            )
        if not self.column:
            raise ValueError("ObjectiveSpec.column must be a non-empty string.")

    @property
    def sign(self) -> float:
        """+1 for maximisation, −1 for minimisation (used when flipping Y)."""
        return 1.0 if self.direction == "max" else -1.0

    @property
    def label(self) -> str:
        """Human-readable label: ``display_name`` or ``column``."""
        return self.display_name or self.column


# =============================================================================
#  Multi-objective surrogate (ModelListGP wrapper)
# =============================================================================
class MultiObjectiveSurrogate:
    """Fit one independent :class:`~kabo.surrogate.SurrogateModel` per
    objective and glue them together as a BoTorch ``ModelListGP``.

    After :meth:`fit`, the combined model exposes the standard
    ``posterior()`` API that BoTorch's multi-objective acquisitions
    expect.  Individual sub-models remain accessible via ``.models[i]``
    for diagnostics.

    Attributes
    ----------
    model : botorch.models.ModelListGP or None
        The combined multi-output surrogate; ``None`` before fit.
    submodels : list[SurrogateModel]
        Per-objective surrogates (same length and order as ``objectives``).
    objectives : list[ObjectiveSpec]
        Objective specs supplied at construction.
    signs : torch.Tensor
        Shape ``(M,)`` with +1 / −1 entries, applied before HV scoring.
    train_Y_raw : np.ndarray or None
        Raw-scale per-objective observations, shape ``(N, M)``.  Kept
        around so ``--ref-point`` inference can run after fitting.
    """

    def __init__(
        self,
        objectives: list[ObjectiveSpec],
        device,
    ) -> None:
        if not objectives:
            raise ValueError("MultiObjectiveSurrogate requires >= 1 objective.")
        # Late-imported to keep this module importable in torch-free env.
        from kabo.surrogate import SurrogateModel
        import torch

        self.objectives: list[ObjectiveSpec] = list(objectives)
        self.device = device
        self.submodels: list[SurrogateModel] = [
            SurrogateModel(device) for _ in self.objectives
        ]
        self.signs = torch.tensor(
            [s.sign for s in self.objectives],
            dtype=torch.double, device=device,
        )
        self.model: Optional["ModelListGP"] = None
        self.train_Y_raw: Optional[np.ndarray] = None
        self.bounds_raw: Optional["torch.Tensor"] = None
        # v1.2 audit: per-objective hypervolume trace (populated by optimizer)
        self.hv_trace: list[float] = []

    # ------------------------------------------------------------------
    #  Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X_raw: np.ndarray,
        Y_raw: np.ndarray,
        selected_features: list[str],
        design_bounds: dict[str, tuple[float, float]],
        kernel_type: str = "matern",
        feature_types: Optional[dict[str, str]] = None,
    ) -> "ModelListGP":
        """Fit every sub-surrogate and wrap them in a ``ModelListGP``.

        Parameters
        ----------
        X_raw : np.ndarray
            Shared feature matrix, shape ``(N, K)``.
        Y_raw : np.ndarray
            Per-objective observations, shape ``(N, M)`` where column
            order matches ``self.objectives``.  Raw scale (no sign flip
            yet).  Rows with NaN in any column are dropped jointly so
            that every sub-surrogate sees the same training set.
        selected_features, design_bounds, kernel_type, feature_types :
            Forwarded verbatim to every ``SurrogateModel.fit``.

        Returns
        -------
        botorch.models.ModelListGP
            The combined surrogate.
        """
        from botorch.models.model_list_gp_regression import ModelListGP
        import torch

        X_raw = np.asarray(X_raw, dtype=np.float64)
        Y_raw = np.asarray(Y_raw, dtype=np.float64)
        if Y_raw.ndim != 2 or Y_raw.shape[1] != len(self.objectives):
            raise ValueError(
                f"Y_raw must have shape (N, {len(self.objectives)}); "
                f"got {Y_raw.shape}."
            )

        # Drop rows with NaN in any objective column — each GP needs aligned
        # data and BoTorch's ModelListGP assumes identical train_X.
        mask = ~np.any(np.isnan(Y_raw), axis=1)
        n_dropped = int(len(Y_raw) - mask.sum())
        if n_dropped:
            logger.info(
                "MultiObjectiveSurrogate: dropping %d rows with NaN in "
                "at least one objective column.", n_dropped,
            )
        X_fit = X_raw[mask]
        Y_fit = Y_raw[mask]
        self.train_Y_raw = Y_fit.copy()

        if len(X_fit) < 2:
            raise ValueError(
                "MultiObjectiveSurrogate needs at least 2 complete "
                f"observations (got {len(X_fit)})."
            )

        botorch_models = []
        for i, obj in enumerate(self.objectives):
            logger.info(
                "  [MO %d/%d] Fitting GP for objective '%s' (direction=%s)",
                i + 1, len(self.objectives), obj.label, obj.direction,
            )
            # BoTorch maximises; flip sign for "min" objectives so the
            # qNEHVI calculation is direction-agnostic downstream.
            y_signed = obj.sign * Y_fit[:, i]
            self.submodels[i].fit(
                X_fit, y_signed, selected_features,
                design_bounds=design_bounds,
                kernel_type=kernel_type,
                feature_types=feature_types,
            )
            botorch_models.append(self.submodels[i].model)

        self.model = ModelListGP(*botorch_models)
        # All submodels share design bounds; expose one of them on the wrapper.
        self.bounds_raw = self.submodels[0].bounds_raw
        logger.info(
            "MultiObjectiveSurrogate ready: %d objective(s), %d training points.",
            len(self.objectives), len(X_fit),
        )
        return self.model

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def signed_train_Y(self) -> "torch.Tensor":
        """Return training Y flipped so every objective is maximised.

        Shape ``(N, M)``.  Useful for qNEHVI's ``ref_point`` logic and
        for hypervolume monitoring.
        """
        if self.train_Y_raw is None:
            raise RuntimeError("MultiObjectiveSurrogate not fit yet.")
        import torch

        y_signed = self.train_Y_raw * np.array(
            [s.sign for s in self.objectives], dtype=np.float64
        )
        return torch.tensor(y_signed, dtype=torch.double, device=self.device)


# =============================================================================
#  Acquisition factory
# =============================================================================
def build_qnehvi(
    mo_surrogate: MultiObjectiveSurrogate,
    ref_point: list[float],
    X_baseline: "torch.Tensor",
    mc_samples: int = 128,
) -> "qNoisyExpectedHypervolumeImprovement":
    """Construct qNEHVI around an already-fit multi-objective surrogate.

    Parameters
    ----------
    mo_surrogate : MultiObjectiveSurrogate
        Already fit; exposes ``.model`` (a ModelListGP).
    ref_point : list[float]
        Reference-point values **on the signed scale** (i.e. after the
        "max" convention has been applied — callers should pass
        ``[sign * r for r in raw_ref]``).  Length ``M``.
    X_baseline : torch.Tensor
        Normalised training inputs the qNEHVI integrand is anchored to.
        Shape ``(N, K)``.  Typically ``mo_surrogate.submodels[0].train_X``.
    mc_samples : int, optional
        Monte Carlo samples for the acquisition (default 128).

    Returns
    -------
    qNoisyExpectedHypervolumeImprovement
        A BoTorch acquisition ready to be optimised with ``optimize_acqf``.
    """
    from botorch.acquisition.multi_objective.monte_carlo import (
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.sampling.normal import SobolQMCNormalSampler
    import torch

    if mo_surrogate.model is None:
        raise RuntimeError("Call MultiObjectiveSurrogate.fit() before build_qnehvi().")
    if len(ref_point) != len(mo_surrogate.objectives):
        raise ValueError(
            f"ref_point length {len(ref_point)} does not match "
            f"{len(mo_surrogate.objectives)} objectives."
        )

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(mc_samples)]))
    acq = qNoisyExpectedHypervolumeImprovement(
        model=mo_surrogate.model,
        ref_point=torch.tensor(
            ref_point, dtype=torch.double, device=X_baseline.device,
        ),
        X_baseline=X_baseline,
        sampler=sampler,
        prune_baseline=True,  # drops dominated baseline points → faster
    )
    logger.info(
        "qNEHVI built: ref_point=%s, baseline=%s, mc_samples=%d",
        [round(float(r), 4) for r in ref_point],
        tuple(X_baseline.shape), mc_samples,
    )
    return acq


# =============================================================================
#  Post-run analysis
# =============================================================================
def compute_pareto_front(
    Y: np.ndarray,
    objectives: list[ObjectiveSpec],
) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows of ``Y``.

    Parameters
    ----------
    Y : np.ndarray
        Shape ``(N, M)`` on the **raw scale** (no sign flip).
    objectives : list[ObjectiveSpec]
        Used only to recover per-objective direction.

    Returns
    -------
    np.ndarray
        Boolean mask of length ``N``; ``True`` means non-dominated.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != len(objectives):
        raise ValueError(
            f"Y shape {Y.shape} does not match {len(objectives)} objectives."
        )
    try:
        from botorch.utils.multi_objective.pareto import is_non_dominated
        import torch
    except ImportError:  # torch unavailable → numpy fallback (O(N^2))
        return _numpy_pareto_mask(Y, [s.sign for s in objectives])
    # Drop incomplete rows — they can't be non-dominated against themselves.
    complete = ~np.any(np.isnan(Y), axis=1)
    if not complete.any():
        return complete  # all False

    signs = np.array([s.sign for s in objectives], dtype=np.float64)
    y_signed = torch.tensor(Y[complete] * signs, dtype=torch.double)
    mask_complete = is_non_dominated(y_signed).cpu().numpy()

    # Re-inflate to the original row ordering.
    out = np.zeros(len(Y), dtype=bool)
    out[np.where(complete)[0]] = mask_complete
    return out


def _numpy_pareto_mask(Y: np.ndarray, signs: list[float]) -> np.ndarray:
    """Pure-numpy Pareto mask (used only when torch is unavailable)."""
    Y = np.asarray(Y, dtype=np.float64)
    s = np.asarray(signs, dtype=np.float64)
    y = Y * s  # now every objective is maximise
    n = len(y)
    mask = np.ones(n, dtype=bool)
    complete = ~np.any(np.isnan(y), axis=1)
    mask &= complete
    for i in range(n):
        if not mask[i]:
            continue
        # point i is dominated if some other j has y[j] >= y[i] elementwise AND strict somewhere
        diff = y - y[i]
        dominates = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        dominates[i] = False
        dominates &= complete
        if dominates.any():
            mask[i] = False
    return mask


def infer_ref_point(
    Y: np.ndarray,
    objectives: list[ObjectiveSpec],
    margin: float = 0.1,
) -> list[float]:
    """Heuristic reference-point inference.

    For a "max" objective the reference must be *worse* (smaller) than
    every feasible point; we use ``y_min − margin * |y_range|``.  For a
    "min" objective it is the symmetric upper bound.

    The returned list is on the **signed** scale (i.e. ready to feed into
    :func:`build_qnehvi`), matching BoTorch's HV conventions.

    Parameters
    ----------
    Y : np.ndarray
        Per-objective training values, shape ``(N, M)``, raw scale.
    objectives : list[ObjectiveSpec]
    margin : float, optional
        Fractional padding below the worst observed value (default 0.1).

    Returns
    -------
    list[float]
        Length ``M`` reference point on the signed / maximisation scale.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}.")
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != len(objectives):
        raise ValueError(
            f"Y shape {Y.shape} does not match {len(objectives)} objectives."
        )

    ref: list[float] = []
    for i, obj in enumerate(objectives):
        if obj.ref_point is not None:
            ref.append(obj.sign * float(obj.ref_point))
            continue
        col = Y[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            ref.append(obj.sign * 0.0)
            continue
        y_signed = obj.sign * col  # now bigger == better
        lo = float(y_signed.min())
        hi = float(y_signed.max())
        span = max(abs(hi - lo), 1e-9)
        ref.append(lo - margin * span)
    return ref


def hypervolume(
    Y: np.ndarray,
    objectives: list[ObjectiveSpec],
    ref_point_signed: list[float],
) -> float:
    """Dominated hypervolume w.r.t. ``ref_point_signed`` on the signed scale.

    Used by the optimizer's early-stopping check when running in
    multi-objective mode.
    """
    try:
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        import torch
    except ImportError:  # pragma: no cover - torch optional
        return float("nan")

    Y = np.asarray(Y, dtype=np.float64)
    complete = ~np.any(np.isnan(Y), axis=1)
    if not complete.any():
        return 0.0
    signs = np.array([s.sign for s in objectives], dtype=np.float64)
    y_signed = torch.tensor(Y[complete] * signs, dtype=torch.double)
    hv = Hypervolume(ref_point=torch.tensor(ref_point_signed, dtype=torch.double))
    return float(hv.compute(y_signed))


# =============================================================================
#  Plotting
# =============================================================================
def plot_pareto_front(
    df: pd.DataFrame,
    objectives: list[ObjectiveSpec],
    save_path: str | Path,
    title: str = "Pareto front",
) -> Optional[Path]:
    """Save a PNG scatter of observations with the Pareto front highlighted.

    Supports 2D (M=2) and 3D (M=3) projections; for M > 3, silently
    skips the plot and returns ``None`` (the CSV is still written).

    Parameters
    ----------
    df : pd.DataFrame
        Full observed dataset including all objective columns.
    objectives : list[ObjectiveSpec]
    save_path : str or Path
        Destination PNG path.  Parent directories are created.
    title : str, optional
        Figure title.

    Returns
    -------
    Path or None
        Actual path written, or ``None`` if plotting was skipped.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    m = len(objectives)
    if m not in (2, 3):
        logger.info(
            "plot_pareto_front: skipping (only 2D / 3D objective spaces "
            "are rendered, got M=%d).", m,
        )
        return None

    cols = [o.column for o in objectives]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(
            "plot_pareto_front: missing columns %s — skipping.", missing,
        )
        return None

    Y = df[cols].values
    mask = compute_pareto_front(Y, objectives)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6), dpi=120)
    if m == 2:
        ax = fig.add_subplot(111)
        ax.scatter(
            Y[~mask, 0], Y[~mask, 1],
            c="lightgray", s=40, alpha=0.7, label="Dominated",
        )
        # Sort Pareto points along objective 0 for a cleaner line overlay
        pareto = Y[mask]
        if len(pareto):
            order = np.argsort(pareto[:, 0])
            ax.plot(
                pareto[order, 0], pareto[order, 1],
                "o-", color="tab:red", linewidth=1.5, markersize=7,
                label=f"Pareto front ({int(mask.sum())} pt)",
            )
        ax.set_xlabel(f"{objectives[0].label}  ({objectives[0].direction})")
        ax.set_ylabel(f"{objectives[1].label}  ({objectives[1].direction})")
        ax.grid(alpha=0.3)
        ax.legend()
    else:  # m == 3
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            Y[~mask, 0], Y[~mask, 1], Y[~mask, 2],
            c="lightgray", s=25, alpha=0.6, label="Dominated",
        )
        ax.scatter(
            Y[mask, 0], Y[mask, 1], Y[mask, 2],
            c="tab:red", s=55, label=f"Pareto ({int(mask.sum())})",
        )
        ax.set_xlabel(objectives[0].label)
        ax.set_ylabel(objectives[1].label)
        ax.set_zlabel(objectives[2].label)
        ax.legend()

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Pareto-front plot saved to: %s", save_path)
    return save_path


__all__ = [
    "ObjectiveSpec",
    "MultiObjectiveSurrogate",
    "build_qnehvi",
    "compute_pareto_front",
    "infer_ref_point",
    "hypervolume",
    "plot_pareto_front",
]
