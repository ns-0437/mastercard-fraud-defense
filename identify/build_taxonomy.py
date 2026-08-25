"""
Builds the attack taxonomy graph from identify/taxonomy_data.py and writes
identify/taxonomy_graph.json.

Nodes = attack vectors. Edges = shared technique tags (two attacks that rely on the
same underlying capability, e.g. deepfake_audio, are connected). This is what the
Generate pillar queries to pick attack families and blend techniques across channels,
and what the web prototype visualizes to demonstrate diversity.
"""
import json
import math
from itertools import combinations
from pathlib import Path

import networkx as nx

from taxonomy_data import ATTACK_TAXONOMY, TECHNIQUE_DESCRIPTIONS

OUT_PATH = Path(__file__).parent / "taxonomy_graph.json"


def build_graph() -> nx.Graph:
    g = nx.Graph()
    for node in ATTACK_TAXONOMY:
        g.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

    # Edge = shared technique tag between two distinct nodes.
    by_technique: dict[str, list[str]] = {}
    for node in ATTACK_TAXONOMY:
        for tech in node["techniques"]:
            by_technique.setdefault(tech, []).append(node["id"])

    for tech, node_ids in by_technique.items():
        for a, b in combinations(sorted(set(node_ids)), 2):
            if g.has_edge(a, b):
                g[a][b]["shared_techniques"].append(tech)
            else:
                g.add_edge(a, b, shared_techniques=[tech])

    return g


def _clustered_layout(g: nx.Graph) -> dict:
    """A plain spring_layout over all 21 nodes at once renders as an undifferentiated
    hairball once 83 cross-cutting edges are drawn -- it makes real diversity look like
    clutter instead of structure. Placing each channel's nodes in their own region
    (a mini spring_layout run per channel, positioned around a shared center) makes the
    8-channel structure visible at a glance before a viewer reads a single label, and
    the cross-channel edges that remain read as genuine "shared technique" bridges
    rather than background noise.
    """
    channels: dict[str, list[str]] = {}
    for n, data in g.nodes(data=True):
        channels.setdefault(data["channel"], []).append(n)

    n_channels = len(channels)
    cluster_radius = 1.0
    local_spread = 0.32
    layout = {}
    for i, (channel, node_ids) in enumerate(sorted(channels.items())):
        angle = 2 * math.pi * i / n_channels
        cx, cy = cluster_radius * math.cos(angle), cluster_radius * math.sin(angle)
        subgraph = g.subgraph(node_ids)
        if len(node_ids) == 1:
            local = {node_ids[0]: (0.0, 0.0)}
        else:
            local = nx.spring_layout(subgraph, seed=42, k=0.9)
        for n, (lx, ly) in local.items():
            layout[n] = (cx + lx * local_spread, cy + ly * local_spread)
    return layout


def to_json(g: nx.Graph) -> dict:
    # Precomputed layout so the web frontend can render a plain SVG with static
    # coordinates -- no client-side force-layout library/dependency needed.
    layout = _clustered_layout(g)
    return {
        "nodes": [
            {"id": n, "x": float(layout[n][0]), "y": float(layout[n][1]), **data}
            for n, data in g.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "shared_techniques": data["shared_techniques"]}
            for u, v, data in g.edges(data=True)
        ],
        "technique_descriptions": TECHNIQUE_DESCRIPTIONS,
    }


def main():
    g = build_graph()
    payload = to_json(g)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload['nodes'])} nodes, {len(payload['edges'])} edges -> {OUT_PATH}")


if __name__ == "__main__":
    main()
