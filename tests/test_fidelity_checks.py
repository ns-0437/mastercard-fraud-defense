"""
Unit tests for generate/validate_fidelity.py's individual check functions, using small
hand-built DataFrames instead of the full 6.36M-row backbone -- fast, deterministic,
and specifically targets the exact bug classes this project actually hit in production
(not aspirational coverage written after the fact for its own sake).

Run: pytest tests/test_fidelity_checks.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "generate"))
from validate_fidelity import (  # noqa: E402
    check_schema, check_sanity, check_conservation, check_non_degenerate,
    check_family_semantics, check_type_validity, STRUCTURING_THRESHOLD,
)

REQUIRED_COLS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def make_row(**overrides):
    row = {
        "step": 1, "type": "TRANSFER", "amount": 100.0, "nameOrig": "A", "oldbalanceOrg": 200.0,
        "newbalanceOrig": 100.0, "nameDest": "B", "oldbalanceDest": 0.0, "newbalanceDest": 100.0,
        "isFraud": 1, "isFlaggedFraud": 0, "attack_family": "test_family", "is_synthetic": 1,
    }
    row.update(overrides)
    return row


class TestSchema:
    def test_passes_with_all_required_columns(self):
        df = pd.DataFrame([make_row()])
        ok, detail = check_schema(df)
        assert ok, detail

    def test_fails_when_column_missing(self):
        df = pd.DataFrame([make_row()]).drop(columns=["oldbalanceOrg"])
        ok, detail = check_schema(df)
        assert not ok
        assert "oldbalanceOrg" in detail

    def test_fails_on_unexpected_nan(self):
        df = pd.DataFrame([make_row(amount=float("nan"))])
        ok, detail = check_schema(df)
        assert not ok


class TestSanity:
    def test_fails_on_zero_amount(self):
        df = pd.DataFrame([make_row(amount=0.0)])
        ok, _ = check_sanity(df)
        assert not ok

    def test_fails_on_negative_balance(self):
        df = pd.DataFrame([make_row(oldbalanceOrg=-5.0)])
        ok, _ = check_sanity(df)
        assert not ok

    def test_passes_on_realistic_row(self):
        df = pd.DataFrame([make_row()])
        ok, _ = check_sanity(df)
        assert ok


class TestConservation:
    def test_passes_when_balances_conserved(self):
        df = pd.DataFrame([make_row(oldbalanceOrg=200.0, amount=100.0, newbalanceOrig=100.0)])
        ok, detail = check_conservation(df)
        assert ok, detail

    def test_fails_when_balances_broken(self):
        # oldbalance - amount should be ~100, not 500 -- this is the shape of bug that
        # a broken simulator (or the units bug) would actually produce.
        df = pd.DataFrame([make_row(oldbalanceOrg=200.0, amount=100.0, newbalanceOrig=500.0)])
        ok, detail = check_conservation(df)
        assert not ok, detail

    def test_ignores_non_transfer_types(self):
        df = pd.DataFrame([make_row(type="PAYMENT", oldbalanceOrg=200.0, amount=100.0, newbalanceOrig=999.0)])
        ok, _ = check_conservation(df)
        assert ok  # PAYMENT isn't in the checked type list, so a "broken" balance here shouldn't fail


class TestFamilySemantics:
    """Direct regression test for the real $94.9M structuring bug this project hit:
    a units mismatch made the LLM's fractional threshold value get treated as an
    absolute dollar amount, producing structuring transactions far above the reporting
    threshold they're supposed to stay under."""

    def test_structuring_below_threshold_passes(self):
        df = pd.DataFrame([make_row(attack_family="structuring_smurfing", amount=8500.0)])
        ok, detail = check_family_semantics(df)
        assert ok, detail

    def test_structuring_above_threshold_fails(self):
        # This is exactly the bug: a $94.9M "structuring" transaction is numerically
        # plausible against the real backbone's envelope (PaySim's real max is ~$92M)
        # but semantically nonsensical for an attack whose entire premise is staying
        # UNDER a $10,000 threshold.
        df = pd.DataFrame([make_row(attack_family="structuring_smurfing", amount=94_900_000.0)])
        ok, detail = check_family_semantics(df)
        assert not ok, "must catch amounts at/above the reporting threshold"
        assert str(int(STRUCTURING_THRESHOLD)) in detail or "threshold" in detail.lower()

    def test_other_families_have_no_constraint(self):
        df = pd.DataFrame([make_row(attack_family="ato_rapid_drain", amount=94_900_000.0)])
        ok, _ = check_family_semantics(df)
        assert ok  # no semantic constraint defined for this family -- shouldn't fail


class TestTypeValidity:
    def test_fails_on_type_not_in_backbone(self):
        df = pd.DataFrame([make_row(type="TOTALLY_MADE_UP_TYPE")])
        backbone = pd.DataFrame({"type": ["PAYMENT", "TRANSFER", "CASH_OUT"]})
        ok, detail = check_type_validity(df, backbone)
        assert not ok
        assert "TOTALLY_MADE_UP_TYPE" in detail


class TestNonDegenerate:
    def test_fails_when_all_amounts_identical(self):
        df = pd.DataFrame([make_row(amount=100.0), make_row(amount=100.0), make_row(amount=100.0)])
        ok, _ = check_non_degenerate(df)
        assert not ok

    def test_passes_with_varied_amounts(self):
        df = pd.DataFrame([make_row(amount=a) for a in [100.0, 250.0, 75.0]])
        ok, _ = check_non_degenerate(df)
        assert ok


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
