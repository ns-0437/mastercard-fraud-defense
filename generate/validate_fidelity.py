"""
Phase 2 gate: validates that synthetic attack transactions are numerically and
distributionally plausible before they're allowed into defend/. Per CLAUDE.md, this
must be a real check that can fail and force a fix to the simulator — not a rubber
stamp.

Deliberately NOT a "does synthetic fraud match PaySim's native fraud distribution"
test: three of our four attack families (structuring, card-testing, mule-layering) are
*designed* to look different from PaySim's native TRANSFER->CASH_OUT fraud pattern
(that's the point — see attack_configs.py docstring). Instead each check validates
plausibility against the real backbone's overall per-type distribution and internal
numeric consistency, which is what "does this look like a real transaction" actually
means.

Checks per attack family:
  1. schema      - required PaySim columns present, no unexpected NaNs
  2. sanity      - no negative/zero amounts, no negative balances
  3. conservation- oldbalance - amount ~= newbalance for TRANSFER/CASH_OUT rows
                   (PaySim's own data has this property; synthetic data lacking it
                   is an obvious tell)
  4. plausibility- KS-test + range check of synthetic amount vs. real backbone's
                   amount distribution for the SAME transaction type
  5. type_validity - synthetic `type` values are a subset of real backbone's types
  6. non_degenerate - amounts aren't all identical (a common lazy-generator tell)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from simulate import load_backbone, REPO_ROOT

REQUIRED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]


def check_schema(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"missing columns: {missing}"
    nan_counts = df[REQUIRED_COLUMNS].isna().sum()
    bad = nan_counts[nan_counts > 0]
    if len(bad):
        return False, f"unexpected NaNs: {bad.to_dict()}"
    return True, "ok"


def check_sanity(df: pd.DataFrame) -> tuple[bool, str]:
    if (df["amount"] <= 0).any():
        return False, f"{(df['amount'] <= 0).sum()} rows with amount <= 0"
    for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if (df[col] < -1e-6).any():
            return False, f"{(df[col] < -1e-6).sum()} rows with negative {col}"
    return True, "ok"


def check_conservation(df: pd.DataFrame, rel_tol: float = 0.02) -> tuple[bool, str]:
    relevant = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    if relevant.empty:
        return True, "no TRANSFER/CASH_OUT rows to check"
    expected = relevant["oldbalanceOrg"] - relevant["amount"]
    diff = (expected - relevant["newbalanceOrig"]).abs()
    tol = np.maximum(relevant["amount"] * rel_tol, 1.0)
    violations = (diff > tol).sum()
    frac = violations / len(relevant)
    ok = frac < 0.05
    return ok, f"{violations}/{len(relevant)} rows ({frac:.1%}) violate balance conservation"


def check_plausibility(df: pd.DataFrame, backbone: pd.DataFrame) -> tuple[bool, str]:
    results = []
    overall_ok = True
    for txn_type in df["type"].unique():
        synth_amt = df.loc[df["type"] == txn_type, "amount"]
        real_amt = backbone.loc[backbone["type"] == txn_type, "amount"] if "type" in backbone else pd.Series(dtype=float)
        if real_amt.empty or len(synth_amt) < 2:
            results.append(f"{txn_type}: no real comparison data, skipped")
            continue
        stat, pvalue = ks_2samp(synth_amt, real_amt)
        real_p99 = real_amt.quantile(0.99)
        synth_p99 = synth_amt.quantile(0.99)
        # Not-absurd bound: synthetic p99 shouldn't dwarf real p99 by more than 20x,
        # nor be a rounding error of it (both would signal a broken simulator, not
        # just "different fraud pattern than average legit traffic").
        range_ok = synth_p99 < real_p99 * 20 and synth_p99 > real_p99 * 0.0001
        overall_ok = overall_ok and range_ok
        results.append(
            f"{txn_type}: KS stat={stat:.3f} p={pvalue:.4f} | synth_p99=${synth_p99:,.2f} "
            f"vs real_p99=${real_p99:,.2f} | range_ok={range_ok}"
        )
    return overall_ok, "; ".join(results)


def check_type_validity(df: pd.DataFrame, backbone: pd.DataFrame) -> tuple[bool, str]:
    real_types = set(backbone["type"].unique()) if "type" in backbone else set()
    synth_types = set(df["type"].unique())
    invalid = synth_types - real_types
    if invalid:
        return False, f"invalid types not seen in real backbone: {invalid}"
    return True, "ok"


def check_non_degenerate(df: pd.DataFrame) -> tuple[bool, str]:
    std = df["amount"].std()
    if std == 0 or pd.isna(std):
        return False, "all synthetic amounts identical"
    return True, f"amount std=${std:,.2f}"


CHECKS = {
    "schema": check_schema,
    "sanity": check_sanity,
    "conservation": check_conservation,
    "non_degenerate": check_non_degenerate,
}
BACKBONE_CHECKS = {
    "plausibility": check_plausibility,
    "type_validity": check_type_validity,
}


def validate_family(df: pd.DataFrame, backbone: pd.DataFrame) -> dict:
    results = {}
    for name, fn in CHECKS.items():
        ok, detail = fn(df)
        results[name] = {"pass": bool(ok), "detail": detail}
    for name, fn in BACKBONE_CHECKS.items():
        ok, detail = fn(df, backbone)
        results[name] = {"pass": bool(ok), "detail": detail}
    results["_overall_pass"] = bool(all(v["pass"] for v in results.values()))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=str(REPO_ROOT / "data" / "synthetic"))
    parser.add_argument("--report-out", default=str(Path(__file__).parent / "fidelity_report.json"))
    args = parser.parse_args()

    backbone = load_backbone()
    synth_dir = Path(args.synthetic)
    family_files = sorted(p for p in synth_dir.glob("*.csv") if p.stem != "combined")

    if not family_files:
        raise SystemExit(f"no per-family synthetic CSVs found in {synth_dir} — run simulate.py first")

    report = {}
    n_pass = 0
    for path in family_files:
        df = pd.read_csv(path)
        result = validate_family(df, backbone)
        report[path.stem] = result
        status = "PASS" if result["_overall_pass"] else "FAIL"
        n_pass += result["_overall_pass"]
        print(f"\n=== {path.stem} ({len(df)} rows): {status} ===")
        for check_name, r in result.items():
            if check_name == "_overall_pass":
                continue
            print(f"  [{'OK ' if r['pass'] else 'FAIL'}] {check_name}: {r['detail']}")

    Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.report_out}")

    gate_pass = n_pass >= 4
    print(f"\nPHASE 2 GATE ({n_pass}/{len(family_files)} families pass all checks, need >=4): "
          f"{'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
