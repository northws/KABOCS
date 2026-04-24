"""
Phase 2: BoTorch Surrogate Model Setup.

Builds and fits a SingleTaskGP with an ARD Matérn 2.5 kernel.
ARD (Automatic Relevance Determination) assigns each feature its own
learned length-scale, which is essential since CO2RR descriptors span
different physical quantities.

Normalization is based on explicit design-space bounds (not training
data min/max) so that the GP can model and explore the full
experimental design space.

Follows the paper's methodology on:
- Gaussian Process surrogate models (Part II)
- Hyperparameter Adaptation via marginal likelihood maximization
- ScaleKernel(MaternKernel) kernel design
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel, SpectralMixtureKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from kabo.utils import (
    categorical_indices_from_types,
    get_logger,
    integer_indices_from_types,
    normalize_x,
    standardize_y,
)

try:
    from botorch.models import MixedSingleTaskGP
    _HAS_MIXED_GP = True
except ImportError:  # pragma: no cover — BoTorch too old
    MixedSingleTaskGP = None  # type: ignore
    _HAS_MIXED_GP = False

# v1.2: stochastic variational GP for large N (O(N m^2) vs ExactGP's O(N^3)).
try:
    from botorch.models import SingleTaskVariationalGP
    from gpytorch.mlls import VariationalELBO
    _HAS_SVGP = True
except ImportError:  # pragma: no cover — BoTorch too old
    SingleTaskVariationalGP = None  # type: ignore
    VariationalELBO = None  # type: ignore
    _HAS_SVGP = False

# Training-size heuristic for the "auto" routing.  Under ~200 points the
# Cholesky factor in ExactGP is cheap and more accurate; above ~200 points
# SVGP starts to pay off in wall-clock time for a small accuracy hit.
_SVGP_AUTO_THRESHOLD = 200

# Default inducing-point count when the user does not specify one.  Capped
# at ``min(N, this)`` inside ``fit()``.
_SVGP_DEFAULT_INDUCING = 100

# Default number of Adam epochs for SVGP.
_SVGP_DEFAULT_EPOCHS = 200

logger = get_logger(__name__)


class SurrogateModel:
    """Wrapper for the BoTorch SingleTaskGP surrogate model.

    Handles data normalization, GP construction with ARD Matérn kernel,
    and hyperparameter fitting via marginal likelihood.

    Normalization uses explicit design-space bounds rather than
    training-data min/max, ensuring the GP can explore beyond
    observed data.

    Parameters
    ----------
    device : torch.device
        Torch device for tensor operations.

    Attributes
    ----------
    model : SingleTaskGP or None
        The fitted GP model.
    mll : ExactMarginalLogLikelihood or None
        Marginal log-likelihood object.
    train_X : torch.Tensor or None
        Normalized training features.
    train_Y : torch.Tensor or None
        Standardized training targets.
    bounds_raw : torch.Tensor or None
        Design-space bounds ``(2, K)`` for un-normalization.
    y_mean : float
        Mean of raw target values.
    y_std : float
        Std of raw target values.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None
        self.bounds_raw: Optional[torch.Tensor] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        # P1 of discrete variables proposal: type-aware dim routing.
        # Populated by fit(); consumed by suggest_continuous / evaluate_discrete.
        self.integer_indices: list[int] = []
        self.categorical_indices: list[int] = []
        self.feature_types: Optional[dict[str, str]] = None
        # v1.2: which backend was actually used — "exact" or "variational".
        # Populated by fit(); surfaced in run_metadata.json.
        self.gp_model_type: str = "exact"
        self.num_inducing_points: Optional[int] = None

    def fit(
        self,
        X_raw: np.ndarray,
        Y_raw: np.ndarray,
        selected_features: list[str],
        design_bounds: dict[str, tuple[float, float]],
        kernel_type: str = "matern",
        feature_types: Optional[dict[str, str]] = None,
        gp_model_type: str = "auto",
        num_inducing_points: Optional[int] = None,
        svgp_epochs: int = _SVGP_DEFAULT_EPOCHS,
        svgp_lr: float = 1e-2,
    ) -> SingleTaskGP:
        """Build and fit the GP surrogate model.

        Steps:
        1. Normalize X to [0, 1] using explicit design-space bounds.
        2. Standardize Y (zero mean, unit variance).
        3. Define the GP:
           * ``gp_model_type="exact"`` (legacy) → ``SingleTaskGP`` with
             the chosen kernel (Matern / SpectralMixture), or
             ``MixedSingleTaskGP`` when categorical dims are declared.
           * ``gp_model_type="variational"`` → ``SingleTaskVariationalGP``
             with ``num_inducing_points`` inducing locations
             (Cholesky variational distribution).
           * ``gp_model_type="auto"`` → ``"variational"`` when
             ``N >= _SVGP_AUTO_THRESHOLD`` (currently 200), else
             ``"exact"``.  Categorical Tasks always route to exact
             because ``MixedSingleTaskGP`` has no variational twin.
        4. Fit hyperparameters via ``ExactMarginalLogLikelihood +
           fit_gpytorch_mll`` (exact) or ``VariationalELBO`` + Adam
           (variational).

        Parameters
        ----------
        X_raw : np.ndarray
            Raw feature matrix, shape ``(N, K)``.
        Y_raw : np.ndarray
            Raw target values, shape ``(N,)``.
        selected_features : list[str]
            Names of the selected features (for logging and bounds lookup).
        design_bounds : dict[str, tuple[float, float]]
            Explicit design-space bounds per feature,
            e.g. ``{"A_pI": (2.7, 10.8), ...}``.
        kernel_type : str, optional
            Type of kernel to use ("matern" or "spectral_mixture"), default "matern".
        feature_types : dict[str, str] or None, optional
            Per-feature type labels from ``TaskBase.feature_types()``.
            When the mapping contains ``"categorical"`` or ``"ordinal"``
            labels, the surrogate switches to ``MixedSingleTaskGP`` with
            ``cat_dims`` routed automatically.  ``"integer"`` labels are
            stored for acquisition-time grid snapping (see
            :func:`kabo.utils.round_integer_dims_to_grid`).  ``None`` (the
            default) preserves legacy behaviour (all continuous).
        gp_model_type : {"exact", "variational", "auto"}, optional
            See docstring above.  Default ``"auto"``.
        num_inducing_points : int or None, optional
            Number of inducing points for the variational path.
            Defaults to ``min(N, 100)``.  Capped at ``N``.
        svgp_epochs : int, optional
            Adam epochs for the variational ELBO loop (default 200).
        svgp_lr : float, optional
            Learning rate for Adam in the variational loop (default 1e-2).

        Returns
        -------
        SingleTaskGP
            The fitted GP model.
        """
        K = X_raw.shape[1]
        logger.info("Building GP with K=%d selected features", K)

        # ----- Type metadata routing (P1) -----
        self.feature_types = feature_types
        self.integer_indices = integer_indices_from_types(
            selected_features, feature_types,
        )
        self.categorical_indices = categorical_indices_from_types(
            selected_features, feature_types,
        )
        if self.integer_indices or self.categorical_indices:
            logger.info(
                "Feature types: %d integer dim(s), %d categorical dim(s), "
                "%d continuous dim(s).",
                len(self.integer_indices),
                len(self.categorical_indices),
                K - len(self.integer_indices) - len(self.categorical_indices),
            )
            if self.integer_indices:
                int_names = [selected_features[i] for i in self.integer_indices]
                logger.info("  Integer dims: %s", int_names)
            if self.categorical_indices:
                cat_names = [selected_features[i] for i in self.categorical_indices]
                logger.info("  Categorical dims: %s", cat_names)

        # ----- 1. Compute normalization bounds from design space -----
        x_min = np.array([design_bounds[f][0] for f in selected_features])
        x_max = np.array([design_bounds[f][1] for f in selected_features])
        x_range = x_max - x_min
        x_range[x_range == 0] = 1.0

        self.bounds_raw = torch.tensor(
            np.stack([x_min, x_max]), dtype=torch.double, device=self.device
        )  # shape (2, K)

        # Normalize X to [0, 1] using design-space bounds
        X_norm = normalize_x(X_raw, x_min, x_range)

        # Warn if training data falls outside design bounds
        n_below = np.sum(X_norm < -0.01)
        n_above = np.sum(X_norm > 1.01)
        if n_below + n_above > 0:
            logger.warning(
                "⚠ %d training data values fall outside design-space bounds "
                "(below: %d, above: %d). Consider expanding DESIGN_SPACE_BOUNDS.",
                n_below + n_above, n_below, n_above,
            )

        # Clip to [0, 1] for GP stability (data slightly outside is OK)
        X_norm = np.clip(X_norm, 0.0, 1.0)

        self.train_X = torch.tensor(
            X_norm, dtype=torch.double, device=self.device
        )

        # ----- 2. Standardize Y -----
        Y_std, self.y_mean, self.y_std = standardize_y(Y_raw)
        self.train_Y = torch.tensor(
            Y_std, dtype=torch.double, device=self.device
        ).unsqueeze(-1)  # shape (N, 1) for BoTorch

        logger.info("Training data: X shape=%s, Y shape=%s",
                     tuple(self.train_X.shape), tuple(self.train_Y.shape))
        logger.info("Y statistics: mean=%.4f, std=%.4f",
                     self.y_mean, self.y_std)
        logger.info("Design-space bounds (not data-derived):")
        for feat, (lo, hi) in zip(selected_features,
                                   self.bounds_raw.cpu().numpy().T):
            logger.info("  %-35s  [%.4f, %.4f]", feat, lo, hi)

        # ----- 3. Resolve gp_model_type / routing -----
        N = int(self.train_X.shape[0])
        resolved_model_type = self._resolve_gp_model_type(
            gp_model_type=gp_model_type,
            N=N,
            has_categorical=bool(self.categorical_indices),
        )
        self.gp_model_type = resolved_model_type  # audit field

        # ----- 4. Define the GP model -----
        if resolved_model_type == "variational":
            self._build_variational_gp(
                K=K, N=N,
                num_inducing_points=num_inducing_points,
                kernel_type=kernel_type,
            )
            self._fit_variational(epochs=svgp_epochs, lr=svgp_lr)
        else:
            # Route to MixedSingleTaskGP when Task declared categorical dims.
            use_mixed = bool(self.categorical_indices) and _HAS_MIXED_GP

            if use_mixed:
                if kernel_type != "matern":
                    logger.warning(
                        "MixedSingleTaskGP ignores kernel_type='%s' for its "
                        "continuous factor; using BoTorch defaults.",
                        kernel_type,
                    )
                self.model = MixedSingleTaskGP(
                    train_X=self.train_X,
                    train_Y=self.train_Y,
                    cat_dims=list(self.categorical_indices),
                ).to(self.device)
                logger.info(
                    "Using MixedSingleTaskGP (cat_dims=%s, %d continuous dim(s)).",
                    self.categorical_indices,
                    K - len(self.categorical_indices),
                )
            else:
                if self.categorical_indices and not _HAS_MIXED_GP:
                    logger.warning(
                        "MixedSingleTaskGP unavailable in this BoTorch build — "
                        "falling back to SingleTaskGP; categorical dims will be "
                        "treated as continuous (less accurate).",
                    )
                if kernel_type == "matern":
                    covar_module = ScaleKernel(
                        MaternKernel(nu=2.5, ard_num_dims=K)
                    )
                elif kernel_type == "spectral_mixture":
                    logger.info("Using SpectralMixtureKernel (derived from CatBOX / catalysis literature).")
                    # 4 mixtures is a reasonable default for standard datasets
                    covar_module = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=K)
                else:
                    raise ValueError(f"Unknown kernel_type '{kernel_type}'. Choose 'matern' or 'spectral_mixture'.")

                self.model = SingleTaskGP(
                    train_X=self.train_X,
                    train_Y=self.train_Y,
                    covar_module=covar_module,
                ).to(self.device)

                # For SpectralMixtureKernel, we must initialize parameters from data
                if kernel_type == "spectral_mixture":
                    covar_module.initialize_from_data(self.train_X, self.train_Y)

            # ----- 5. Fit hyperparameters (exact MLL) -----
            self.mll = ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )

            logger.info("Fitting GP hyperparameters via marginal likelihood...")
            try:
                fit_gpytorch_mll(self.mll)
            except Exception as e:
                logger.warning(
                    "fit_gpytorch_mll raised an exception (may still have "
                    "partially converged): %s", e
                )

        # Log learned hyperparameters
        self._log_hyperparameters(selected_features)

        logger.info("GP surrogate model fitted (type=%s).", resolved_model_type)
        return self.model

    def _log_hyperparameters(self, selected_features: list[str]) -> None:
        """Log the learned GP kernel hyperparameters."""
        if self.model is None:
            return

        try:
            # SVGP tucks the real kernel one level deeper: model.model.covar_module.
            if self.gp_model_type == "variational" and hasattr(self.model, "model"):
                covar = self.model.model.covar_module
            else:
                covar = self.model.covar_module
            logger.info("GP Hyperparameters:")
            logger.info("  Noise variance: %.6f",
                         self.model.likelihood.noise.item())
            if self.gp_model_type == "variational":
                logger.info("  Backend: SingleTaskVariationalGP (m=%s inducing points)",
                             self.num_inducing_points)

            if isinstance(covar, ScaleKernel) and hasattr(covar, "base_kernel"):
                outputscale = covar.outputscale.item()
                lengthscales = covar.base_kernel.lengthscale.detach().cpu().squeeze()

                logger.info("  Output scale:  %.4f", outputscale)
                logger.info("  ARD length-scales (per feature):")
                for i, feat in enumerate(selected_features):
                    ls = (lengthscales[i].item()
                          if lengthscales.dim() > 0
                          else lengthscales.item())
                    logger.info("    %-35s  ℓ=%.4f", feat, ls)
            elif isinstance(covar, SpectralMixtureKernel):
                logger.info("  Kernel: SpectralMixtureKernel")
                logger.info("  Num mixtures: %d", covar.num_mixtures)
                logger.info(
                    "  Mixture means shape: %s",
                    tuple(covar.mixture_means.shape),
                )
                logger.info(
                    "  Mixture scales shape: %s",
                    tuple(covar.mixture_scales.shape),
                )
            else:
                logger.info("  Kernel type: %s", covar.__class__.__name__)
        except Exception as e:
            logger.debug("Could not log hyperparameters: %s", e)

    # ------------------------------------------------------------------
    #  v1.2 — Variational GP helpers
    # ------------------------------------------------------------------
    def _resolve_gp_model_type(
        self,
        gp_model_type: str,
        N: int,
        has_categorical: bool,
    ) -> str:
        """Pick ``"exact"`` vs ``"variational"`` given the user request.

        Rules
        -----
        * ``"exact"`` / ``"variational"`` → honoured verbatim (with
          safety fallbacks documented below).
        * ``"auto"`` → ``"variational"`` when ``N >= 200`` **and** no
          categorical dims are declared, else ``"exact"``.

        Fallbacks
        ---------
        * Variational requested but ``SingleTaskVariationalGP`` not
          available → logs a warning and falls back to ``"exact"``.
        * Variational + categorical → logs a warning and falls back to
          ``"exact"`` (no SVGP twin of ``MixedSingleTaskGP`` exists).
        """
        if gp_model_type not in {"exact", "variational", "auto"}:
            raise ValueError(
                f"gp_model_type must be 'exact', 'variational', or 'auto'; "
                f"got {gp_model_type!r}."
            )

        if gp_model_type == "auto":
            resolved = (
                "variational"
                if (N >= _SVGP_AUTO_THRESHOLD and not has_categorical)
                else "exact"
            )
            if resolved == "variational":
                logger.info(
                    "gp_model_type='auto' resolved to 'variational' "
                    "(N=%d ≥ %d, no categorical dims).",
                    N, _SVGP_AUTO_THRESHOLD,
                )
            return resolved

        if gp_model_type == "variational":
            if not _HAS_SVGP:
                logger.warning(
                    "gp_model_type='variational' requested but "
                    "SingleTaskVariationalGP is unavailable in this BoTorch "
                    "install — falling back to 'exact'."
                )
                return "exact"
            if has_categorical:
                logger.warning(
                    "gp_model_type='variational' requested but task declared "
                    "categorical dims; MixedSingleTaskGP has no SVGP twin — "
                    "falling back to 'exact' (MixedSingleTaskGP)."
                )
                return "exact"
        return gp_model_type

    def _build_variational_gp(
        self,
        K: int,
        N: int,
        num_inducing_points: Optional[int],
        kernel_type: str,
    ) -> None:
        """Construct a ``SingleTaskVariationalGP`` and park it on ``self.model``."""
        if not _HAS_SVGP:  # pragma: no cover — already guarded in _resolve_
            raise RuntimeError("SingleTaskVariationalGP unavailable.")

        # Resolve inducing count.  Hard-capped at N so we never pick more
        # inducing points than we have training data.
        m = (
            int(num_inducing_points)
            if num_inducing_points is not None
            else min(N, _SVGP_DEFAULT_INDUCING)
        )
        m = max(1, min(m, N))
        self.num_inducing_points = m

        if kernel_type == "matern":
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=K)
            )
        elif kernel_type == "spectral_mixture":
            # SpectralMixture + VariationalELBO has documented instability;
            # warn and fall back to a matern kernel for the SVGP path.
            logger.warning(
                "SpectralMixtureKernel + variational GP is unstable; "
                "using Matern 2.5 instead for the SVGP path."
            )
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=K)
            )
        else:
            raise ValueError(
                f"Unknown kernel_type '{kernel_type}'. "
                "Choose 'matern' or 'spectral_mixture'."
            )

        # Subsample m rows from train_X as initial inducing locations.
        if m < N:
            idx = torch.randperm(N, device=self.device)[:m]
            inducing_points = self.train_X[idx].clone()
        else:
            inducing_points = self.train_X.clone()

        self.model = SingleTaskVariationalGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            inducing_points=inducing_points,
            covar_module=covar_module,
            learn_inducing_points=True,
        ).to(self.device)
        logger.info(
            "Using SingleTaskVariationalGP (m=%d inducing points, N=%d, K=%d).",
            m, N, K,
        )

    def _fit_variational(self, epochs: int, lr: float) -> None:
        """Train the SVGP via Adam + VariationalELBO (full-batch)."""
        if self.model is None or self.train_Y is None or self.train_X is None:
            raise RuntimeError("Variational GP is not initialised.")
        assert VariationalELBO is not None  # narrowed by _HAS_SVGP

        # BoTorch's SingleTaskVariationalGP wraps the actual ApproximateGP at
        # ``.model`` and the likelihood at ``.likelihood`` — VariationalELBO
        # binds to that inner pair.
        inner_gp = self.model.model
        likelihood = self.model.likelihood
        self.mll = VariationalELBO(
            likelihood, inner_gp, num_data=self.train_X.shape[0],
        ).to(self.device)

        optimizer = torch.optim.Adam(
            [{"params": self.model.parameters()}], lr=lr,
        )

        # Flatten train_Y to (N,) — that is what VariationalELBO expects
        # for single-output regression.
        y_flat = self.train_Y.squeeze(-1)

        self.model.train()
        likelihood.train()
        logger.info(
            "Fitting SVGP via Adam (epochs=%d, lr=%.1e)...", epochs, lr,
        )
        last_loss = float("nan")
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = inner_gp(self.train_X)
            loss = -self.mll(output, y_flat)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            if epoch == 0 or (epoch + 1) % max(epochs // 4, 1) == 0:
                logger.info(
                    "  SVGP epoch %4d/%d  ELBO=%.4f", epoch + 1, epochs, -last_loss,
                )
        self.model.eval()
        likelihood.eval()
        logger.info("SVGP training done. Final ELBO=%.4f", -last_loss)
