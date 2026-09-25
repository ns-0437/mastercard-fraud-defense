"""
Regression test for a real bug: defend/build_features.py's main() called
add_graph_features(combined) ONCE over the full train+test set, before the time-based
split. Every graph feature column (orig_component_size, orig_out_degree,
shared_neighbor_count, ...) was therefore built from a graph containing every
test-period (future) transaction too -- so a TRAINING row's structural features could
reflect accounts and connections that, at that row's own step, had not happened yet.
That is exactly the "a real deployed detector only ever sees the past" invariant this
module's own docstring gives as the reason for a time-based (not random) split in the
first place -- the split protected the tabular features but not the graph ones.

Run: pytest tests/test_build_features.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "defend"))
from build_features import add_graph_features_without_future_leakage  # noqa: E402
from graph_features import add_graph_features  # noqa: E402


@pytest.fixture
def txns_with_a_future_only_link():
    """Account A transacts with M1 in the train period (step=1). A's component stays
    small (A, M1) throughout training. Only in the TEST period (step=100) does A
    transact with M2, which joins A's component to B's and C-D's -- a link that must
    not be visible to any training-period row's features."""
    return pd.DataFrame([
        {"nameOrig": "A", "nameDest": "M1", "amount": 10.0, "step": 1},
        {"nameOrig": "B", "nameDest": "M2", "amount": 10.0, "step": 1},
        {"nameOrig": "C", "nameDest": "D", "amount": 10.0, "step": 1},
        {"nameOrig": "A", "nameDest": "M2", "amount": 10.0, "step": 100},
    ])


class TestNoFutureLeakageIntoTrainingGraphFeatures:
    def test_naive_single_graph_over_combined_set_leaks_the_future_link(
        self, txns_with_a_future_only_link
    ):
        """Confirms the bug actually existed: the old code path (add_graph_features
        called once over the whole set) DOES let the future-period link change a
        train-period row's component_size."""
        naive = add_graph_features(txns_with_a_future_only_link)
        train_row = naive[naive["step"] == 1].iloc[0]
        assert train_row["nameOrig"] == "A"
        assert train_row["orig_component_size"] == 4, (
            "sanity check on the fixture: without the fix, A's train-period row sees "
            "the merged 4-account component (A, M1, B, M2) even though M2 only joins "
            "via a transaction 99 steps in the future"
        )

    def test_fixed_function_does_not_leak_the_future_link_into_a_train_row(
        self, txns_with_a_future_only_link
    ):
        fixed = add_graph_features_without_future_leakage(
            txns_with_a_future_only_link, split_frac=0.20
        )
        train_row = fixed[(fixed["step"] == 1) & (fixed["nameOrig"] == "A")].iloc[0]
        assert train_row["orig_component_size"] == 2, (
            "A's train-period row must reflect only what had happened by step 1 "
            "(A and M1, component size 2) -- not the future step=100 link to M2's "
            "and B's component"
        )

    def test_test_period_rows_still_see_the_full_observed_graph(
        self, txns_with_a_future_only_link
    ):
        """The fix must not become overly conservative: a test-period row legitimately
        sees the merged component, since by the time it's being (batch) evaluated,
        every transaction in the set has already occurred."""
        fixed = add_graph_features_without_future_leakage(
            txns_with_a_future_only_link, split_frac=0.20
        )
        test_row = fixed[fixed["step"] == 100].iloc[0]
        assert test_row["orig_component_size"] == 4

    def test_row_count_and_columns_are_unchanged_by_the_fix(
        self, txns_with_a_future_only_link
    ):
        naive = add_graph_features(txns_with_a_future_only_link)
        fixed = add_graph_features_without_future_leakage(
            txns_with_a_future_only_link, split_frac=0.20
        )
        assert len(fixed) == len(naive) == len(txns_with_a_future_only_link)
        assert set(fixed.columns) == set(naive.columns)
