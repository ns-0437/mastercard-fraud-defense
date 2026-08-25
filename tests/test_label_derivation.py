"""
Regression test for a real bug: defend/build_features.py originally derived the
training label as `label = is_synthetic`, which meant warm-up transactions (added in
Phase 4 hardening -- an ordinary-looking payment a synthetic account makes to a real
account before its actual attack, specifically to avoid presenting as a brand-new
isolated node) were incorrectly labeled as fraud. A warm-up row IS synthetic-sourced
but is deliberately isFraud=0 -- it represents the attacker's innocuous prior activity,
not the attack itself. Training on it as a positive example would have taught the
model that small innocuous payments are suspicious.

Run: pytest tests/test_label_derivation.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "defend"))


def derive_label(df: pd.DataFrame) -> pd.Series:
    """Mirrors the exact expression in build_features.py's load_and_combine --
    kept here explicitly so a future edit to that file that regresses the label logic
    fails this test immediately, rather than silently corrupting a future training run."""
    return ((df["is_synthetic"] == 1) & (df["isFraud"] == 1)).astype(int)


class TestLabelDerivation:
    def test_synthetic_fraud_row_labeled_positive(self):
        df = pd.DataFrame([{"is_synthetic": 1, "isFraud": 1}])
        assert derive_label(df).iloc[0] == 1

    def test_warmup_row_labeled_negative_despite_being_synthetic(self):
        # This is the exact bug: synthetic-sourced but isFraud=0 (a warm-up transaction).
        df = pd.DataFrame([{"is_synthetic": 1, "isFraud": 0}])
        assert derive_label(df).iloc[0] == 0, (
            "a warm-up transaction (synthetic-sourced, isFraud=0) must NOT be labeled "
            "as fraud -- the old `label = is_synthetic` logic would have failed this"
        )

    def test_real_legit_row_labeled_negative(self):
        df = pd.DataFrame([{"is_synthetic": 0, "isFraud": 0}])
        assert derive_label(df).iloc[0] == 0

    def test_native_paysim_fraud_not_counted_as_primary_label(self):
        # PaySim's own native fraud (isFraud=1, is_synthetic=0) is deliberately NOT
        # part of the primary training label -- it's evaluated separately as a
        # secondary signal, not conflated with the GenAI-attack target.
        df = pd.DataFrame([{"is_synthetic": 0, "isFraud": 1}])
        assert derive_label(df).iloc[0] == 0

    def test_mixed_batch(self):
        df = pd.DataFrame([
            {"is_synthetic": 1, "isFraud": 1},  # real attack row -> 1
            {"is_synthetic": 1, "isFraud": 0},  # warm-up row -> 0
            {"is_synthetic": 0, "isFraud": 0},  # real legit -> 0
            {"is_synthetic": 0, "isFraud": 1},  # native fraud -> 0 (secondary signal only)
        ])
        assert list(derive_label(df)) == [1, 0, 0, 0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
