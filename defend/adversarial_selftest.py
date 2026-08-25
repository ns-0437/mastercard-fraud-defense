"""
Phase 7 bulletproofing: hand-crafted adversarial transactions designed to evade the
final (cycle-2) detector, scored through the same batch feature pipeline used for
training/evaluation (not the live API's simplified lookup) so this is a fair test of
what the model actually learned. Per docs/PHASES.md: report whichever way this goes,
including if the detector gets evaded -- that's a real finding to disclose, not a
result to hide or keep re-running until it looks good.

Each case is designed to specifically target a feature the model is known to rely on
(see defend/artifacts/feature_importance.json): low amount, low degree/component size,
amounts near the legitimate median, or a "patient" attacker who spreads activity far
beyond what Phase 4's hardening already tried.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "generate"))

from build_features import add_tabular_features, ALL_FEATURE_COLUMNS  # noqa: E402
from graph_features import add_graph_features  # noqa: E402
from simulate import load_backbone  # noqa: E402

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def build_case(name, rows, description):
    df = pd.DataFrame(rows, columns=PAYSIM_COLUMNS)
    df["attack_family"] = name
    df["is_synthetic"] = 1
    df["description"] = description
    return df


CASES = []

# Case 1: ATO drain, but only 40% of balance in ONE step (not the 85-98%, multi-stage
# pattern the model was trained on) -- targets balance_delta_orig / amount_to_orig_balance_ratio.
CASES.append(build_case(
    "adv_low_fraction_drain",
    [{"step": 300, "type": "TRANSFER", "amount": 8000.0, "nameOrig": "SYNTH_ADV_A001",
      "oldbalanceOrg": 20000.0, "newbalanceOrig": 12000.0, "nameDest": "SYNTH_ADV_SINK001",
      "oldbalanceDest": 0.0, "newbalanceDest": 8000.0, "isFraud": 1, "isFlaggedFraud": 0}],
    "ATO drain of only 40% of balance in one step, well below the 85-98% pattern trained on.",
))

# Case 2: structuring at 40% of the $10k threshold (well below the 70-97% band trained on).
CASES.append(build_case(
    "adv_low_fraction_structuring",
    [{"step": 100 + i, "type": "TRANSFER", "amount": 4000.0 + i * 10, "nameOrig": "SYNTH_ADV_A002",
      "oldbalanceOrg": 30000.0 - i * 4000, "newbalanceOrig": 30000.0 - (i + 1) * 4000,
      "nameDest": "SYNTH_ADV_SINK002", "oldbalanceDest": 0.0, "newbalanceDest": 4000.0 + i * 10,
      "isFraud": 1, "isFlaggedFraud": 0} for i in range(4)],
    "Structuring at ~40% of the reporting threshold, well below the trained 70-97% band.",
))

# Case 3: patient card testing -- only 2 destinations, one per WEEK (not one burst),
# even more patient than Phase 4's hardened version (which used 1 day).
CASES.append(build_case(
    "adv_patient_card_testing",
    [{"step": 50, "type": "PAYMENT", "amount": 2.5, "nameOrig": "SYNTH_ADV_A003",
      "oldbalanceOrg": 5000.0, "newbalanceOrig": 4997.5, "nameDest": "SYNTH_ADV_M003a",
      "oldbalanceDest": 0.0, "newbalanceDest": 2.5, "isFraud": 1, "isFlaggedFraud": 0},
     {"step": 218, "type": "PAYMENT", "amount": 3.1, "nameOrig": "SYNTH_ADV_A003",
      "oldbalanceOrg": 4997.5, "newbalanceOrig": 4994.4, "nameDest": "SYNTH_ADV_M003b",
      "oldbalanceDest": 0.0, "newbalanceDest": 3.1, "isFraud": 1, "isFlaggedFraud": 0}],
    "Only 2 card-testing probes, ~24 weeks apart -- far more patient than Phase 4's "
    "hardened 1-day-spread variant that already evaded the cycle-1 model.",
))

# Case 4: mule layering that mimics a legitimate-looking round-trip amount near the
# real backbone's median TRANSFER value, with warm-up already applied conceptually
# (accounts given a plausible prior transaction) and only 2 hops.
CASES.append(build_case(
    "adv_median_amount_mule",
    [{"step": 400, "type": "TRANSFER", "amount": 9500.0, "nameOrig": "SYNTH_ADV_A004a",
      "oldbalanceOrg": 9500.0, "newbalanceOrig": 0.0, "nameDest": "SYNTH_ADV_A004b",
      "oldbalanceDest": 0.0, "newbalanceDest": 9500.0, "isFraud": 1, "isFlaggedFraud": 0},
     {"step": 402, "type": "CASH_OUT", "amount": 9300.0, "nameOrig": "SYNTH_ADV_A004b",
      "oldbalanceOrg": 9500.0, "newbalanceOrig": 200.0, "nameDest": "SYNTH_ADV_MERCH004",
      "oldbalanceDest": 0.0, "newbalanceDest": 9300.0, "isFraud": 1, "isFlaggedFraud": 0}],
    "2-hop mule layering with amount set to PaySim's real TRANSFER median ($9,482), "
    "far fewer hops/accounts than the trained mule pattern.",
))

# Case 5: "low and slow" ATO mimicking recurring bill-pay amounts (same amount,
# monthly cadence) instead of a drain pattern at all.
CASES.append(build_case(
    "adv_billpay_mimicry",
    [{"step": 30 * i + 10, "type": "PAYMENT", "amount": 1450.0, "nameOrig": "SYNTH_ADV_A005",
      "oldbalanceOrg": 15000.0 - i * 1450, "newbalanceOrig": 15000.0 - (i + 1) * 1450,
      "nameDest": "SYNTH_ADV_MERCH005", "oldbalanceDest": 0.0, "newbalanceDest": 1450.0,
      "isFraud": 1, "isFlaggedFraud": 0} for i in range(5)],
    "Recurring same-amount payments to one destination mimicking legitimate bill-pay, "
    "no drain/burst/threshold pattern at all -- an ATO operator making the compromised "
    "account's payment activity look completely routine.",
))


def main():
    model = xgb.XGBClassifier()
    model.load_model(ARTIFACTS_DIR / "detector_cycle2.json")

    backbone = load_backbone()

    print("=== Phase 7 adversarial self-test: 5 hand-crafted evasion attempts ===\n")
    results = []
    for case_df in CASES:
        name = case_df["attack_family"].iloc[0]
        description = case_df["description"].iloc[0]
        case_df = case_df.drop(columns=["description"])

        # Build graph context from the case rows plus a slice of the real backbone,
        # matching how build_features.py constructs graph features -- the case
        # accounts are otherwise unconnected new nodes, same as a real first-contact
        # attacker account would be.
        context = pd.concat([backbone.sample(n=20_000, random_state=1), case_df], ignore_index=True)
        context["is_synthetic"] = context.get("is_synthetic", 0).fillna(0).astype(int)
        context = add_tabular_features(context)
        context = add_graph_features(context)

        case_rows = context.tail(len(case_df))
        X = case_rows[ALL_FEATURE_COLUMNS]
        proba = model.predict_proba(X)[:, 1]
        max_proba = float(proba.max())
        caught = max_proba >= 0.5

        print(f"{name}: {'CAUGHT' if caught else 'EVADED'} (max fraud probability across "
              f"{len(case_df)} rows: {max_proba:.4f})")
        print(f"  {description}\n")
        results.append({
            "case": name, "caught": caught, "max_fraud_probability": max_proba,
            "description": description, "n_rows": len(case_df),
        })

    n_caught = sum(r["caught"] for r in results)
    print(f"Summary: {n_caught}/{len(results)} hand-crafted evasion attempts were still caught.")
    if n_caught < len(results):
        print("This is a real, disclosed limitation, not a hidden one -- report exactly "
              "which cases evaded detection and why, in the docx's limitations section.")

    report = {
        "cases": results,
        "n_caught": n_caught,
        "n_total": len(results),
        "summary": (
            f"{n_caught}/{len(results)} hand-crafted attacks, specifically designed to look "
            "numerically unremarkable (amounts near the real backbone's own median, few hops, "
            "no extreme balance-fraction ratios), were still caught by the final detector. This "
            "tests generalization to attacks structurally different from anything in training -- "
            "a harder and more honest question than Phase 4's in-family hardening test."
        ),
    }
    (ARTIFACTS_DIR / "adversarial_selftest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {ARTIFACTS_DIR / 'adversarial_selftest_report.json'}")


if __name__ == "__main__":
    main()
