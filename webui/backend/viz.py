"""
Real-time visualization helpers for the WebUI.

Given a live ``KABOOptimizer`` + its per-iteration acquisition function,
produce small PNG snapshots of:

* GP posterior **mean** and **variance** heatmaps over the two most
  sensitive (shortest-ARD-lengthscale) dims — other dims pinned at the
  normalized midpoint ``0.5``.
* **Acquisition** function landscape over the same two dims.
* A 2-D **PCA** projection of training data, discrete candidate pool and
  current top-N recommendations.

Everything is best-effort: any failure short-circuits to ``None`` so a
viz error never poisons the optimization run.

Images are returned as base64 (``data:image/png;base64,...``) strings,
so the frontend can inline them with a simple ``<img src>``.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Matplotlib is already pinned in requirements.txt; force Agg so the
# worker thread never tries to pop a GUI window.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fig_to_base64(fig, dpi: int = 92) -> str:
    """Serialize ``fig`` to a base64-encoded PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _extract_lengthscales(model) -> Optional[np.ndarray]:
    """Pull the ARD lengthscales out of a fitted BoTorch/GPyTorch GP.

    Supports ``SingleTaskGP``, ``SingleTaskVariationalGP`` (wraps the
    real model one level deeper) and ``MixedSingleTaskGP``.  Returns
    ``None`` when the kernel does not expose a lengthscale vector
    (e.g. ``SpectralMixtureKernel`` — ARD sensitivity is not a single
    number there).
    """
    try:
        # SVGP keeps the real GP at model.model
        covar = getattr(model, "model", None)
        covar = covar.covar_module if covar is not None else model.covar_module
    except Exception:
        return None
    try:
        base = getattr(covar, "base_kernel", covar)
        ls = getattr(base, "lengthscale", None)
        if ls is None:
            return None
        arr = ls.detach().cpu().numpy().squeeze()
        if arr.ndim == 0:
            return None
        return np.asarray(arr, dtype=float).ravel()
    except Exception:
        return None


def _pick_two_dims(model, K: int) -> Optional[tuple[int, int]]:
    """Pick the two dims with smallest ARD lengthscales.

    Returns ``None`` when the model has fewer than two dims or the
    kernel is not ARD-Matern (we fall back to "first two" in that
    case to keep the panel useful).
    """
    if K < 2:
        return None
    ls = _extract_lengthscales(model)
    if ls is None or ls.size != K:
        return 0, 1
    idx = np.argsort(ls)
    return int(idx[0]), int(idx[1])


def _posterior_mean_var_acq(
    model,
    acq_func,
    K: int,
    dim_a: int,
    dim_b: int,
    grid_size: int = 60,
) -> Optional[dict[str, np.ndarray]]:
    """Evaluate GP mean/variance and acquisition on a 2-D grid.

    All non-plotted dims are pinned to ``0.5``.  Returns a dict with
    ``mean``, ``var``, ``acq`` as 2-D ``(grid, grid)`` arrays, and
    ``a_grid`` / ``b_grid`` 1-D coordinates (in normalized [0,1]).
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    a = torch.linspace(0.0, 1.0, grid_size, dtype=torch.double, device=device)
    b = torch.linspace(0.0, 1.0, grid_size, dtype=torch.double, device=device)
    A, B = torch.meshgrid(a, b, indexing="xy")  # (G, G)
    X = torch.full(
        (grid_size * grid_size, K), 0.5, dtype=torch.double, device=device,
    )
    X[:, dim_a] = A.reshape(-1)
    X[:, dim_b] = B.reshape(-1)

    try:
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X)
            mean = posterior.mean.detach().cpu().numpy().reshape(grid_size, grid_size)
            var = (
                posterior.variance.detach()
                .cpu()
                .numpy()
                .reshape(grid_size, grid_size)
            )
    except Exception as exc:
        logger.debug("posterior evaluation failed: %s", exc)
        return None

    acq_grid: Optional[np.ndarray] = None
    if acq_func is not None:
        try:
            with torch.no_grad():
                # BoTorch acquisitions expect (batch, q=1, d).
                acq_vals = acq_func(X.unsqueeze(1)).detach().cpu().numpy()
            acq_grid = acq_vals.reshape(grid_size, grid_size)
        except Exception as exc:
            logger.debug("acquisition evaluation failed: %s", exc)
            acq_grid = None

    return {
        "mean": mean,
        "var": var,
        "acq": acq_grid,
        "a": a.detach().cpu().numpy(),
        "b": b.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Public entry: GP / acquisition heatmaps
# ---------------------------------------------------------------------------
def render_gp_landscape(
    model,
    acq_func,
    selected_features: list[str],
    iteration: int,
    target_name: str,
    train_X_norm: Optional[np.ndarray] = None,
    grid_size: int = 60,
) -> Optional[dict[str, Any]]:
    """Render a combined GP mean / variance / acquisition figure.

    Returns a dict ``{"image": <data:image/png;base64,...>, "dims":
    [a_name, b_name]}`` or ``None`` when the surrogate shape is
    unsupported (e.g. fewer than 2 dims, multi-output, categorical-only).
    """
    if model is None:
        return None
    K = len(selected_features)
    picked = _pick_two_dims(model, K)
    if picked is None:
        return None
    dim_a, dim_b = picked

    data = _posterior_mean_var_acq(
        model, acq_func, K=K, dim_a=dim_a, dim_b=dim_b, grid_size=grid_size,
    )
    if data is None:
        return None

    ncols = 3 if data["acq"] is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.4))
    if ncols == 1:
        axes = [axes]

    feat_a = selected_features[dim_a]
    feat_b = selected_features[dim_b]
    extent = [0.0, 1.0, 0.0, 1.0]

    ax = axes[0]
    im = ax.imshow(
        data["mean"], origin="lower", extent=extent,
        aspect="auto", cmap="viridis",
    )
    ax.set_title(f"GP posterior mean\n{target_name} (iter {iteration})", fontsize=9)
    ax.set_xlabel(feat_a, fontsize=8)
    ax.set_ylabel(feat_b, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1]
    im = ax.imshow(
        data["var"], origin="lower", extent=extent,
        aspect="auto", cmap="magma",
    )
    ax.set_title("GP posterior variance", fontsize=9)
    ax.set_xlabel(feat_a, fontsize=8)
    ax.set_ylabel(feat_b, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)

    if data["acq"] is not None:
        ax = axes[2]
        im = ax.imshow(
            data["acq"], origin="lower", extent=extent,
            aspect="auto", cmap="plasma",
        )
        ax.set_title("Acquisition", fontsize=9)
        ax.set_xlabel(feat_a, fontsize=8)
        ax.set_ylabel(feat_b, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)

    # Overlay training data projected to the chosen 2 dims.
    if train_X_norm is not None and train_X_norm.ndim == 2 and train_X_norm.shape[1] >= 2:
        try:
            xs = train_X_norm[:, dim_a]
            ys = train_X_norm[:, dim_b]
            for ax in axes:
                ax.scatter(
                    xs, ys, s=20, c="white", edgecolors="black",
                    linewidths=0.6, alpha=0.85, zorder=3,
                )
        except Exception:
            pass

    try:
        fig.tight_layout()
    except Exception:
        pass
    image = _fig_to_base64(fig)
    return {
        "image": image,
        "dims": [feat_a, feat_b],
        "iteration": int(iteration),
    }


# ---------------------------------------------------------------------------
# Public entry: PCA projection of design-space exploration
# ---------------------------------------------------------------------------
def render_pca_projection(
    train_X_norm: np.ndarray,
    selected_features: list[str],
    iteration: int,
    candidates_norm: Optional[np.ndarray] = None,
    top_indices: Optional[list[int]] = None,
) -> Optional[dict[str, Any]]:
    """Render a PCA (n=2) scatter of training points / candidate pool /
    top-N picks to illustrate design-space exploration.

    Returns ``None`` when the data is too small (<2 points) or sklearn
    is unavailable at runtime.
    """
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        logger.debug("sklearn PCA unavailable: %s", exc)
        return None

    try:
        train_X_norm = np.asarray(train_X_norm, dtype=float)
    except Exception:
        return None
    if train_X_norm.ndim != 2 or train_X_norm.shape[0] < 2 or train_X_norm.shape[1] < 2:
        return None

    pieces: list[np.ndarray] = [train_X_norm]
    cand_arr: Optional[np.ndarray] = None
    if candidates_norm is not None:
        cand_arr = np.asarray(candidates_norm, dtype=float)
        if cand_arr.ndim == 2 and cand_arr.shape[1] == train_X_norm.shape[1]:
            pieces.append(cand_arr)
        else:
            cand_arr = None

    combined = np.vstack(pieces) if len(pieces) > 1 else train_X_norm
    try:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(combined)
    except Exception as exc:
        logger.debug("PCA fit failed: %s", exc)
        return None

    n_train = train_X_norm.shape[0]
    train_pc = proj[:n_train]
    cand_pc = proj[n_train:] if cand_arr is not None else None

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    if cand_pc is not None and cand_pc.shape[0] > 0:
        ax.scatter(
            cand_pc[:, 0], cand_pc[:, 1], s=10, c="#cbd5e1",
            edgecolors="none", alpha=0.6, label="candidate pool", zorder=1,
        )
    ax.scatter(
        train_pc[:, 0], train_pc[:, 1], s=38, c="#2563eb",
        edgecolors="white", linewidths=0.6, alpha=0.9,
        label=f"observations (N={n_train})", zorder=2,
    )
    if (
        top_indices
        and cand_pc is not None
        and cand_pc.shape[0] > 0
    ):
        highlights: list[tuple[float, float]] = []
        for i in top_indices:
            if 0 <= i < cand_pc.shape[0]:
                highlights.append((cand_pc[i, 0], cand_pc[i, 1]))
        if highlights:
            hx = [p[0] for p in highlights]
            hy = [p[1] for p in highlights]
            ax.scatter(
                hx, hy, s=120, marker="*", c="#dc2626",
                edgecolors="white", linewidths=0.8, alpha=0.95,
                label=f"top-{len(highlights)} recs", zorder=3,
            )

    ax.set_title(
        f"Design-space exploration (PCA, iter {iteration})", fontsize=9,
    )
    try:
        var_ratio = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)", fontsize=8)
    except Exception:
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
    ax.legend(loc="best", fontsize=8, frameon=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    image = _fig_to_base64(fig)
    return {
        "image": image,
        "iteration": int(iteration),
        "n_train": int(n_train),
        "n_candidates": int(cand_pc.shape[0]) if cand_pc is not None else 0,
    }


# ---------------------------------------------------------------------------
# Convenience: build the full visualization payload for a single iter
# ---------------------------------------------------------------------------
def build_iteration_visualization(
    *,
    optimizer: Any,
    acq_func: Any,
    iteration: int,
    candidates_norm: Optional[list[torch.Tensor]] = None,
    top_indices: Optional[list[int]] = None,
) -> Optional[dict[str, Any]]:
    """Compose a ``visualization`` event payload for iteration ``iteration``.

    Returns ``None`` (silently) if the run has any trait we can't
    visualize (e.g. multi-objective, no surrogate yet, fewer than 2
    selected features).  Callers should treat ``None`` as "skip emit".
    """
    if optimizer is None:
        return None
    engine = getattr(optimizer, "engine", None)
    if engine is None:
        return None
    if bool(getattr(engine, "is_multi_objective", False)):
        return None
    surrogate = getattr(engine, "surrogate", None)
    if surrogate is None or getattr(surrogate, "model", None) is None:
        return None
    # Categorical dims are skipped for now — the lengthscale-based pick
    # and the continuous-grid eval don't respect the cat-dim encoding.
    if getattr(surrogate, "categorical_indices", None):
        return None

    selected_features = list(getattr(optimizer, "selected_features", []) or [])
    if len(selected_features) < 2:
        return None

    target_column = str(getattr(optimizer, "target_column", "") or "")
    try:
        task = getattr(optimizer, "task", None)
        product_names = task.product_names() if task is not None else {}
    except Exception:
        product_names = {}
    target_name = product_names.get(target_column, target_column)

    train_X_norm: Optional[np.ndarray] = None
    try:
        train_t = surrogate.train_X
        if train_t is not None:
            train_X_norm = train_t.detach().cpu().numpy()
    except Exception:
        train_X_norm = None

    cands_np: Optional[np.ndarray] = None
    if candidates_norm:
        try:
            cands_np = np.stack(
                [c.detach().cpu().numpy().reshape(-1) for c in candidates_norm],
                axis=0,
            )
        except Exception:
            cands_np = None

    gp_panel = None
    try:
        gp_panel = render_gp_landscape(
            surrogate.model,
            acq_func,
            selected_features=selected_features,
            iteration=iteration,
            target_name=target_name,
            train_X_norm=train_X_norm,
        )
    except Exception as exc:
        logger.debug("render_gp_landscape failed: %s", exc)

    pca_panel = None
    if train_X_norm is not None:
        try:
            pca_panel = render_pca_projection(
                train_X_norm=train_X_norm,
                selected_features=selected_features,
                iteration=iteration,
                candidates_norm=cands_np,
                top_indices=top_indices,
            )
        except Exception as exc:
            logger.debug("render_pca_projection failed: %s", exc)

    if gp_panel is None and pca_panel is None:
        return None

    return {
        "iteration": int(iteration),
        "target_column": target_column,
        "target_name": target_name,
        "selected_features": selected_features,
        "gp_landscape": gp_panel,
        "pca_projection": pca_panel,
    }
