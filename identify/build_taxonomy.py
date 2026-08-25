"""
Builds the attack taxonomy graph from identify/taxonomy_data.py and writes
identify/taxonomy_graph.json.

Nodes = attack vectors. Edges = shared technique tags (two attacks that rely on the
same underlying capability, e.g. deepfake_audio, are connected). This is what the
Generate pillar queries to pick attack families and blend techniques across channels,
and what the web prototype visualizes to demonstrate diversity.
"""
import json
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


def to_json(g: nx.Graph) -> dict:
    return {
        "nodes": [
            {"id": n, **data} for n, data in g.nodes(data=True)
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
