"""
Regression test for a real bug: orchestrator/pipeline.py's Step B (Phase 4's actual
closed loop -- see the module docstring's "single most important gate in the project")
called build_working_set(), which computed graph features over the full v1+v2 combined
set, and only THEN called time_split() on the result. Same bug as build_features.py's
main() (see tests/test_build_features.py) -- a training row's graph feature columns
(orig_component_size, shared_neighbor_count, ...) could be built from a graph containing
test-period transactions that, at that row's own step, hadn't happened yet.

Run: pytest tests/test_pipeline_no_leakage.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "generate"))
sys.path.insert(0, str(REPO_ROOT / "defend"))

from pipeline import assemble_working_set, build_working_set  # noqa: E402
from build_features import add_graph_features_without_future_leakage, time_split  # noqa: E402


def _frames():
    """Mirrors the shape build_working_set/assemble_working_set expect: separate
    legit/native_fraud/synthetic frames with the raw PaySim-style columns
    add_tabular_features needs, plus a future-only cross-account link identical in
    spirit to tests/test_build_features.py's fixture."""
    cols = dict(oldbalanceOrg=100.0, newbalanceOrig=90.0, oldbalanceDest=0.0,
                newbalanceDest=10.0, type="TRANSFER")
    legit = pd.DataFrame([
        {"nameOrig": "A", "nameDest": "M1", "amount": 10.0, "step": 1, "isFraud": 0, **cols},
        {"nameOrig": "B", "nameDest": "M2", "amount": 10.0, "step": 1, "isFraud": 0, **cols},
        {"nameOrig": "C", "nameDest": "D", "amount": 10.0, "step": 1, "isFraud": 0, **cols},
    ])
    legit["is_synthetic"] = 0
    native_fraud = legit.iloc[0:0].copy()  # empty, same columns
    synthetic = pd.DataFrame([
        {"nameOrig": "A", "nameDest": "M2", "amount": 10.0, "step": 100, "isFraud": 1, **cols},
    ])
    synthetic["is_synthetic"] = 1
    return legit, native_fraud, synthetic


class TestPipelineStepBDoesNotLeakFutureTransactionsIntoTraining:
    def test_build_working_set_alone_still_leaks_the_future_link(self):
        """Confirms build_working_set is still correct for its OWN documented use
        (Step A: evaluation only, no split) -- and incidentally confirms the fixture
        reproduces the leak when graph features are computed as one shared graph."""
        legit, native_fraud, synthetic = _frames()
        working = build_working_set(legit, native_fraud, synthetic)
        train_row = working[(working["step"] == 1) & (working["nameOrig"] == "A")].iloc[0]
        assert train_row["orig_component_size"] == 4

    def test_step_b_path_does_not_leak_the_future_link_into_a_train_row(self):
        legit, native_fraud, synthetic = _frames()
        assembled = assemble_working_set(legit, native_fraud, synthetic)
        combined_all = add_graph_features_without_future_leakage(assembled, split_frac=0.20)
        train, _test = time_split(combined_all, split_frac=0.20)
        train_row = train[(train["step"] == 1) & (train["nameOrig"] == "A")].iloc[0]
        assert train_row["orig_component_size"] == 2, (
            "a Step B train row must not see the future (step=100) link that joins "
            "A's component to B's -- that's exactly what Phase 4's own docstring calls "
            "the single most important gate in the project, and it was leaking"
        )
