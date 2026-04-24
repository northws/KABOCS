"""Unit tests for ``PreferenceModel.generate_pe_queries`` — v1.2 vectorisation.

Covers:
  * Edge cases that should return an empty list (n<2 / n_queries<=0).
  * ``strategy="random"`` cold-start path (works without a fit model).
  * Pool-cap honouring + returned indices refer back to the original pool.
  * Parameter validation.
  * Determinism under ``random_state``.

Uncertainty-path numeric behaviour requires a fitted PairwiseGP and is
therefore covered by the torch-gated e2e stack rather than in this
torch-free unit file.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_torch


@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")


def _make_preference_model(torch):
    from kabo.preference import PreferenceModel

    return PreferenceModel(device=torch.device("cpu"))


def _tensor_pool(torch, n: int, d: int = 3):
    """``n`` distinct normalised candidates in ``[0, 1]^d``."""
    rng = torch.Generator().manual_seed(0)
    return [torch.rand(d, generator=rng, dtype=torch.double) for _ in range(n)]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_pool_returns_empty(self, torch_mod):
        pm = _make_preference_model(torch_mod)
        assert pm.generate_pe_queries([], n_queries=3) == []

    def test_singleton_pool_returns_empty(self, torch_mod):
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=1)
        assert pm.generate_pe_queries(pool, n_queries=3) == []

    def test_zero_budget_returns_empty(self, torch_mod):
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=5)
        assert pm.generate_pe_queries(pool, n_queries=0) == []

    def test_unknown_strategy_raises(self, torch_mod):
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=5)
        with pytest.raises(ValueError, match="Unknown PE strategy"):
            pm.generate_pe_queries(pool, n_queries=1, strategy="information_gain")


# ---------------------------------------------------------------------------
# Random / cold-start strategy (works without a fit model)
# ---------------------------------------------------------------------------
class TestRandomStrategy:
    def test_random_without_model_returns_distinct_pairs(self, torch_mod):
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=6)
        pairs = pm.generate_pe_queries(pool, n_queries=3, strategy="random")
        assert len(pairs) == 3
        # All pairs must be distinct and reference distinct pool indices.
        seen = set()
        for a, b in pairs:
            assert 0 <= a < 6 and 0 <= b < 6
            assert a != b
            key = tuple(sorted((a, b)))
            assert key not in seen
            seen.add(key)

    def test_random_caps_at_all_pairs(self, torch_mod):
        """With n=3 the upper-triangle has only 3 pairs; asking for 10 → 3."""
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=3)
        pairs = pm.generate_pe_queries(pool, n_queries=10, strategy="random")
        assert len(pairs) == 3
        keys = {tuple(sorted(p)) for p in pairs}
        assert keys == {(0, 1), (0, 2), (1, 2)}

    def test_random_state_is_deterministic(self, torch_mod):
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=8)
        out_a = pm.generate_pe_queries(
            pool, n_queries=4, strategy="random", random_state=42,
        )
        out_b = pm.generate_pe_queries(
            pool, n_queries=4, strategy="random", random_state=42,
        )
        assert out_a == out_b

    def test_no_model_falls_back_to_random_path(self, torch_mod):
        """Default strategy='uncertainty' without a fit model → random path."""
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=5)
        pairs = pm.generate_pe_queries(
            pool, n_queries=2, random_state=7,  # still 'uncertainty'
        )
        assert len(pairs) == 2
        for a, b in pairs:
            assert a != b


# ---------------------------------------------------------------------------
# Pool cap semantics
# ---------------------------------------------------------------------------
class TestPoolCap:
    def test_cap_smaller_than_pool_limits_search_space(self, torch_mod):
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=50)
        pairs = pm.generate_pe_queries(
            pool, n_queries=6,
            max_pool_size=5,
            strategy="random",
            random_state=0,
        )
        assert len(pairs) == 6
        # All returned indices must lie inside the chosen subsample of size 5.
        used = set()
        for a, b in pairs:
            used.add(a)
            used.add(b)
        assert len(used) <= 5
        # Indices still address the *original* pool.
        for a, b in pairs:
            assert 0 <= a < 50 and 0 <= b < 50

    def test_cap_larger_than_pool_is_noop(self, torch_mod):
        pm = _make_preference_model(torch_mod)
        pool = _tensor_pool(torch_mod, n=4)
        pairs = pm.generate_pe_queries(
            pool, n_queries=10,
            max_pool_size=100, strategy="random", random_state=0,
        )
        # All 6 upper-triangle pairs should be returned (n=4 → C(4,2)=6).
        assert len(pairs) == 6
