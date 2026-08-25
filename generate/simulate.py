"""
Phase 2 driver: loads the PaySim backbone, gets a (possibly LLM-refined) config per
attack family, runs the matching simulator, and writes synthetic transactions to
data/synthetic/. Run validate_fidelity.py afterward before treating this output as
usable — per SKILL.md, unvalidated synthetic data must not reach defend/.

Usage:
    python simulate.py --attack-family ato_rapid_drain --n 400 --out ../data/synthetic
    python simulate.py --all --out ../data/synthetic          # runs all 4 families
    python simulate.py --all --no-llm --out ../data/synthetic # skip LLM, use hand defaults
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from attack_configs import DEFAULT_CONFIGS
from llm_config_generator import generate_attack_config
from simulators import SIMULATORS

REPO_ROOT = Path(__file__).parent.parent
TAXONOMY_PATH = REPO_ROOT / "identify" / "taxonomy_graph.json"
BACKBONE_CANDIDATES = list((REPO_ROOT / "data" / "raw").glob("PS_*.csv"))


def load_taxonomy_mechanism(taxonomy_ref: str) -> str:
    if not TAXONOMY_PATH.exists():
        return ""
    payload = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    for node in payload["nodes"]:
        if node["id"] == taxonomy_ref:
            return node["mechanism"]
    return ""


def load_backbone() -> pd.DataFrame:
    if not BACKBONE_CANDIDATES:
        print("WARNING: no PaySim CSV found in data/raw/ (expected PS_*.csv). "
              "Using a small synthetic stand-in backbone for a code smoke test only — "
              "this is NOT a fidelity-validated run. Download the real dataset per "
              "data/raw/README.md before trusting any output.", file=sys.stderr)
        rng = np.random.default_rng(0)
        n = 5000
        return pd.DataFrame({
            "step": rng.integers(1, 744, n),
            "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n),
            "amount": rng.lognormal(7, 1.5, n),
            "nameOrig": [f"C{i}" for i in rng.integers(0, 3000, n)],
            "oldbalanceOrg": rng.lognormal(8, 1.5, n),
            "newbalanceOrig": rng.lognormal(8, 1.5, n),
            "nameDest": [f"C{i}" for i in rng.integers(3000, 6000, n)],
            "oldbalanceDest": rng.lognormal(7, 1.5, n),
            "newbalanceDest": rng.lognormal(7, 1.5, n),
            "isFraud": 0,
            "isFlaggedFraud": 0,
        })
    path = BACKBONE_CANDIDATES[0]
    print(f"Loading backbone from {path}")
    return pd.read_csv(path)


def compute_backbone_stats(backbone: pd.DataFrame) -> dict:
    amt = backbone["amount"]
    return {
        "amount_p50": float(amt.median()),
        "amount_p90": float(amt.quantile(0.90)),
        "amount_p99": float(amt.quantile(0.99)),
        "amount_max": float(amt.max()),
        "step_max": int(backbone["step"].max()) if "step" in backbone else 743,
        "fraud_rate": float(backbone["isFraud"].mean()) if "isFraud" in backbone else None,
        "type_distribution": backbone["type"].value_counts(normalize=True).round(4).to_dict()
        if "type" in backbone else {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-family", choices=list(SIMULATORS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n", type=int, default=None, help="override n_instances")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.attack_family and not args.all:
        parser.error("pass --attack-family <name> or --all")

    families = list(SIMULATORS.keys()) if args.all else [args.attack_family]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_backbone()
    stats = compute_backbone_stats(backbone)
    rng = np.random.default_rng(args.seed)

    all_frames = []
    for family in families:
        default = DEFAULT_CONFIGS[family]
        taxonomy_node = {
            "attack_family": family,
            "mechanism": load_taxonomy_mechanism(default.taxonomy_ref),
        }
        config = generate_attack_config(taxonomy_node, stats, use_llm=not args.no_llm)
        if args.n:
            config.n_instances = args.n

        df = SIMULATORS[family](config, backbone, rng)
        out_path = out_dir / f"{family}.csv"
        df.to_csv(out_path, index=False)
        print(f"{family}: wrote {len(df)} rows -> {out_path}")
        all_frames.append(df)

    if len(all_frames) > 1:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = out_dir / "combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"combined: {len(combined)} rows -> {combined_path}")


if __name__ == "__main__":
    main()
