"""Unit tests for the ``CandidateRecord`` dataclass."""

from __future__ import annotations

from kabo.candidate import CandidateRecord


def _make(**overrides) -> CandidateRecord:
    base = dict(
        raw_values={"x1": 0.5, "x2": 0.3, "x3": 0.9},
        normalized_values={"x1": 0.5, "x2": 0.3},
        source="continuous",
        is_valid_full_feature=True,
        acq_value=1.23,
        orig_row_idx=-1,
    )
    base.update(overrides)
    return CandidateRecord(**base)


class TestCandidateRecord:
    def test_defaults_for_audit_fields(self):
        rec = _make()
        assert rec.expert_rank == -1
        assert rec.overridden_fields == []
        assert rec.oob_confirmation_count == 0

    def test_overridden_fields_are_independent_between_instances(self):
        """Mutable default factory prevents the classic shared-list footgun."""
        a = _make()
        b = _make()
        a.overridden_fields.append("x1")
        assert b.overridden_fields == []

    def test_discrete_source_label(self):
        rec = _make(source="discrete_7", orig_row_idx=7)
        assert rec.source.startswith("discrete_")
        assert rec.orig_row_idx == 7

    def test_manual_override_audit_trail(self):
        rec = _make(
            source="manual_override",
            is_valid_full_feature=False,
            overridden_fields=["x1", "x3"],
            oob_confirmation_count=2,
            expert_rank=0,
        )
        assert rec.source == "manual_override"
        assert rec.expert_rank == 0
        assert len(rec.overridden_fields) == 2
        assert rec.oob_confirmation_count == 2
        assert not rec.is_valid_full_feature
