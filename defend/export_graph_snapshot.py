"""
Exports a static snapshot of the account-graph node features (out/in degree, component
size) built from the same working set used for Phase 4 training (legit sample + v1 +
v2 attacks), so the live API (webapp/backend) can look up graph context for an incoming
transaction without rebuilding a NetworkX graph over hundreds of thousands of rows on
every request.

This is a deliberate simplification of the training-time pipeline, and webapp/backend
is explicit about it (see scoring.py): the snapshot has per-account DEGREE and
COMPONENT SIZE, but not full adjacency, so the live API approximates shared_neighbor_count
as 0 for any pair not already connected in the snapshot, instead of computing it exactly
as build_features.py does in batch. A real production deployment would query a live
graph store (e.g. a graph database or streaming feature store) instead of a static
snapshot -- this is called out explicitly in the docx's real-world feasibility section,
not hidden as if the demo were doing full-fidelity scoring.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "generate"))
sys.path.insert(0, str(Path(__file__).parent))

from simulate import load_backbone  # noqa: E402
from graph_features import build_transaction_graph, compute_node_features  # noqa: E402
from build_features import WORKING_SET_LEGIT, SEED  # noqa: E402

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"


def main():
    print("Rebuilding the Phase 4 working set to snapshot its graph node features...")
    real = load_backbone()
    legit = real[real["isFraud"] == 0].sample(n=min(WORKING_SET_LEGIT, (real["isFraud"] == 0).sum()), random_state=SEED)
    native_fraud = real[real["isFraud"] == 1]
    v1 = pd.read_csv(SYNTHETIC_DIR / "combined.csv")
    v2_path = SYNTHETIC_DIR / "combined_v2.csv"
    frames = [legit, native_fraud, v1]
    if v2_path.exists():
        frames.append(pd.read_csv(v2_path))
    combined = pd.concat(frames, ignore_index=True)

    g = build_transaction_graph(combined)
    node_feats = compute_node_features(g)
    out_path = ARTIFACTS_DIR / "graph_node_snapshot.csv"
    node_feats.to_csv(out_path)
    print(f"Wrote {len(node_feats):,} account rows -> {out_path}")


if __name__ == "__main__":
    main()
