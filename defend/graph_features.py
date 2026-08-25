"""
Transaction/entity graph features — the second of the three "graph engineering" pieces
described in CLAUDE.md. Accounts are nodes, transactions are directed edges. Built with
NetworkX over a bounded working set (full backbone is 6.3M rows; a directed multigraph
over all of it is unnecessary for feature engineering and slow to query per-row, so we
build it over whatever transaction set is passed in — callers are responsible for
scoping that set, see build_features.py).

Deliberately excludes any feature derived from the literal account ID string (e.g. a
"starts with SYNTH_" check) — every synthetic account in this project is prefixed that
way for traceability, and a model trained on that string would trivially "detect fraud"
by reading our own naming convention instead of learning real structural signal. That
would be worthless the moment this ran against real account IDs. Only structural graph
properties (degree, component size, shared neighbors) are used.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_transaction_graph(txns: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for orig, dest, amount in zip(txns["nameOrig"], txns["nameDest"], txns["amount"]):
        if g.has_edge(orig, dest):
            g[orig][dest]["weight"] += 1
            g[orig][dest]["total_amount"] += amount
        else:
            g.add_edge(orig, dest, weight=1, total_amount=amount)
    return g


def compute_node_features(g: nx.DiGraph) -> pd.DataFrame:
    """One row per account with structural features. Computed once per graph build,
    then joined onto the transaction table by nameOrig/nameDest."""
    undirected = g.to_undirected()
    # Connected components: a tiny isolated component (e.g. size 2-10) touching an
    # otherwise sparse region of the graph is the classic mule-ring signature —
    # legitimate accounts sit in one or a few giant components via shared merchants.
    component_of = {}
    component_sizes = {}
    for comp in nx.connected_components(undirected):
        size = len(comp)
        for node in comp:
            component_of[node] = size
    rows = []
    for node in g.nodes():
        rows.append({
            "account": node,
            "out_degree": g.out_degree(node),
            "in_degree": g.in_degree(node),
            "total_degree": undirected.degree(node),
            "component_size": component_of.get(node, 1),
        })
    return pd.DataFrame(rows).set_index("account")


def shared_neighbor_count(g: nx.DiGraph, u: str, v: str) -> int:
    if u not in g or v not in g:
        return 0
    u_neighbors = set(g.predecessors(u)) | set(g.successors(u))
    v_neighbors = set(g.predecessors(v)) | set(g.successors(v))
    return len(u_neighbors & v_neighbors)


def add_graph_features(txns: pd.DataFrame) -> pd.DataFrame:
    """Returns txns with graph feature columns appended. Builds the graph once over
    the full input set, so callers should pass exactly the transaction set they want
    graph context from (see build_features.py's WORKING_SET_SIZE)."""
    g = build_transaction_graph(txns)
    node_feats = compute_node_features(g)

    out = txns.copy()
    orig_feats = node_feats.reindex(out["nameOrig"]).reset_index(drop=True)
    dest_feats = node_feats.reindex(out["nameDest"]).reset_index(drop=True)

    out["orig_out_degree"] = orig_feats["out_degree"].values
    out["orig_in_degree"] = orig_feats["in_degree"].values
    out["orig_component_size"] = orig_feats["component_size"].values
    out["dest_out_degree"] = dest_feats["out_degree"].values
    out["dest_in_degree"] = dest_feats["in_degree"].values
    out["dest_component_size"] = dest_feats["component_size"].values

    out["shared_neighbor_count"] = [
        shared_neighbor_count(g, o, d) for o, d in zip(out["nameOrig"], out["nameDest"])
    ]
    # A brand-new account transacting for the first time (degree 1, tiny component) is
    # a structural tell shared across ATO drains, structuring, and mule layering alike.
    out["orig_is_low_activity"] = (out["orig_component_size"] <= 3).astype(int)
    return out
