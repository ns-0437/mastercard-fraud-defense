"""
Query/validate the built taxonomy graph. Run with --stats to check the Phase 1 gate
from docs/PHASES.md before moving on to Phase 2.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx

GRAPH_PATH = Path(__file__).parent / "taxonomy_graph.json"


def load_graph() -> nx.Graph:
    payload = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    g = nx.Graph()
    for node in payload["nodes"]:
        g.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in payload["edges"]:
        g.add_edge(edge["source"], edge["target"], shared_techniques=edge["shared_techniques"])
    return g


def print_stats(g: nx.Graph):
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    channels = Counter(nx.get_node_attributes(g, "channel").values())
    ungrounded = [n for n, d in g.nodes(data=True) if not d.get("grounding", "").strip()]
    connected_nodes = sum(1 for n in g.nodes if g.degree(n) > 0)
    components = list(nx.connected_components(g))

    print(f"Nodes (attack vectors): {n_nodes}")
    print(f"Edges (shared-technique links): {n_edges}")
    print(f"Connected components: {len(components)} (largest: {len(max(components, key=len))} nodes)")
    print(f"Nodes with >=1 edge: {connected_nodes}/{n_nodes} ({100*connected_nodes/n_nodes:.0f}%)")
    print(f"Nodes missing 'grounding': {len(ungrounded)} {ungrounded if ungrounded else ''}")
    print("\nCoverage by channel:")
    for channel, count in sorted(channels.items(), key=lambda kv: -kv[1]):
        print(f"  {channel:28s} {count}")

    gate_pass = (
        n_nodes >= 15
        and len(ungrounded) == 0
        and connected_nodes / n_nodes >= 0.60
    )
    print(f"\nPHASE 1 GATE: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print("  Needs: >=15 nodes, 0 ungrounded nodes, >=60% of nodes with an edge.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if not GRAPH_PATH.exists():
        raise SystemExit("taxonomy_graph.json not found — run build_taxonomy.py first.")

    g = load_graph()
    if args.stats:
        print_stats(g)
    else:
        print(f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges. Use --stats for detail.")


if __name__ == "__main__":
    main()
