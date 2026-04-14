"""
Unified candidate data structure for CO2RR Bayesian Optimization.

Provides a single ``CandidateRecord`` dataclass that holds the complete
state of every candidate recommendation — raw values for all 19
features, normalised values for selected features, provenance metadata,
and audit fields required for high-fidelity paper reproduction.

This directly addresses REVIEW_REPORT §P1-2:
  - Unified candidate object with ``raw_values``, ``normalized_values``,
    ``source``, and ``is_valid_full_feature``.
  - Audit fields: ``expert_rank``, ``overridden_fields``,
    ``oob_confirmation_count``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateRecord:
    """Unified representation of a single BO candidate.

    Attributes
    ----------
    raw_values : dict[str, float]
        Raw (un-normalised) values for all 19 features.  For discrete
        candidates this comes from the original CSV row; for continuous
        candidates it combines GP-optimised selected features with
        expert-supplied non-selected features.
    normalized_values : dict[str, float]
        Normalised [0, 1] values for the *selected* features only.
    source : str
        Provenance label — ``"continuous"`` or ``"discrete_<N>"``.
    is_valid_full_feature : bool
        Whether *every* one of the 19 features passes boundary validation
        (not just the selected subset).
    acq_value : float
        UCB acquisition function value.
    orig_row_idx : int
        Row index in the discrete candidates CSV (−1 for continuous).

    # --- audit fields ---
    expert_rank : int
        Rank assigned by the expert when selecting this candidate
        (1-based; −1 if not yet selected).
    overridden_fields : list[str]
        Feature names whose values were manually overridden by the expert.
    oob_confirmation_count : int
        Number of out-of-bounds confirmations the expert accepted for
        this candidate's non-selected features.
    """

    raw_values: dict[str, float]
    normalized_values: dict[str, float]
    source: str
    is_valid_full_feature: bool
    acq_value: float
    orig_row_idx: int

    # Audit fields
    expert_rank: int = -1
    overridden_fields: list[str] = field(default_factory=list)
    oob_confirmation_count: int = 0
