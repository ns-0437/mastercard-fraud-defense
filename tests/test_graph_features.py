"""
Unit tests for defend/graph_features.py against a small, hand-built transaction set
with known graph structure -- verifies degree/component computations are correct, and
specifically guards against reintroducing the leakage risk the module's own docstring
warns about (deriving a feature from the literal SYNTH_ account-ID string, which would
trivially "detect fraud" by reading this project's own naming convention rather than
learning real structural signal).

Run: pytest tests/test_graph_features.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "defend"))
from graph_features import build_transaction_graph, compute_node_features, add_graph_features  # noqa: E402


@pytest.fixture
def star_topology():
    """One hub account (A) transacting with 3 distinct destinations -- the exact
    structural signature of a card-testing burst (one origin, high fanout)."""
    return pd.DataFrame([
        {"nameOrig": "A", "nameDest": "M1", "amount": 2.0},
        {"nameOrig": "A", "nameDest": "M2", "amount": 3.0},
        {"nameOrig": "A", "nameDest": "M3", "amount": 4.0},
    ])


@pytest.fixture
def isolated_pair():
    """A single isolated transaction pair, disconnected from everything else --
    the exact structural signature of a brand-new synthetic account before hardening."""
    return pd.DataFrame([{"nameOrig": "SYNTH_X", "nameDest": "SYNTH_Y", "amount": 100.0}])


class TestBuildTransactionGraph:
    def test_star_topology_has_correct_degree(self, star_topology):
        g = build_transaction_graph(star_topology)
        assert g.out_degree("A") == 3
        assert g.in_degree("A") == 0
        for dest in ["M1", "M2", "M3"]:
            assert g.in_degree(dest) == 1

    def test_repeated_edge_increments_weight_not_a_new_edge(self):
        df = pd.DataFrame([
            {"nameOrig": "A", "nameDest": "B", "amount": 10.0},
            {"nameOrig": "A", "nameDest": "B", "amount": 20.0},
        ])
        g = build_transaction_graph(df)
        assert g.number_of_edges() == 1
        assert g["A"]["B"]["weight"] == 2
        assert g["A"]["B"]["total_amount"] == 30.0


class TestComputeNodeFeatures:
    def test_hub_has_high_degree_leaf_has_low(self, star_topology):
        g = build_transaction_graph(star_topology)
        feats = compute_node_features(g)
        assert feats.loc["A", "out_degree"] == 3
        assert feats.loc["M1", "in_degree"] == 1

    def test_star_topology_forms_one_component(self, star_topology):
        g = build_transaction_graph(star_topology)
        feats = compute_node_features(g)
        # all 4 nodes (A + 3 merchants) are in the same connected component
        assert (feats["component_size"] == 4).all()

    def test_isolated_pair_forms_its_own_small_component(self, isolated_pair):
        g = build_transaction_graph(isolated_pair)
        feats = compute_node_features(g)
        assert feats.loc["SYNTH_X", "component_size"] == 2
        assert feats.loc["SYNTH_Y", "component_size"] == 2


class TestNoAccountIdLeakage:
    """The module's own docstring warns against this exact failure mode: a feature
    that reads the literal account-ID string (e.g. checking for a 'SYNTH_' prefix)
    would trivially separate synthetic from real accounts in THIS project's data, but
    would be worthless -- or actively misleading -- against real account IDs in a
    live system, since real accounts never carry that prefix."""

    def test_isolated_synthetic_account_gets_no_special_treatment_by_name(self, isolated_pair):
        g = build_transaction_graph(isolated_pair)
        feats = compute_node_features(g)
        # A real account with the exact same structural position (isolated pair) must
        # get IDENTICAL feature values to a "SYNTH_"-prefixed one -- proving the
        # features are purely structural, not name-derived.
        real_pair = pd.DataFrame([{"nameOrig": "C1234567", "nameDest": "C7654321", "amount": 100.0}])
        g_real = build_transaction_graph(real_pair)
        feats_real = compute_node_features(g_real)
        assert feats.loc["SYNTH_X", "out_degree"] == feats_real.loc["C1234567", "out_degree"]
        assert feats.loc["SYNTH_X", "component_size"] == feats_real.loc["C1234567", "component_size"]

    def test_add_graph_features_output_columns_never_include_raw_account_id(self, star_topology):
        star_topology = star_topology.assign(isFraud=0, is_synthetic=0)
        out = add_graph_features(star_topology)
        # nameOrig/nameDest may still exist as passthrough columns (needed elsewhere in
        # the pipeline), but no *_FEATURE_ column should be a copy of the raw ID string.
        feature_cols = [c for c in out.columns if c not in star_topology.columns]
        for col in feature_cols:
            assert out[col].dtype != object, (
                f"{col} is a non-numeric graph feature -- likely leaking the raw "
                f"account ID string instead of a structural property"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
