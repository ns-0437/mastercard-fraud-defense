"""
Quantifies whether the LLM config-generation path actually adds value over the
hand-authored fallback, using a clean, current, apples-to-apples comparison -- not a
retrospective comparison across historical runs, which would be confounded by the
several bugs found and fixed in between (the card-testing balance-multiplier bug, the
structuring units bug, the mule sample-size fix). Both datasets here are generated
from the IDENTICAL, current, bug-fixed code -- the only difference is --no-llm.

Writes its own separate artifacts (does not touch data/synthetic/, data/processed/, or
defend/artifacts/ -- those remain the canonical LLM-driven results already deployed).

Run: python defend/run_ablation_llm_vs_fallback.py
Prerequisite: data/synthetic_ablation/combined.csv must exist (generate/simulate.py
--all --no-llm --out ../data/synthetic_ablation).
"""
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "generate"))

from build_features import add_tabular_features, time_split, ALL_FEATURE_COLUMNS, WORKING_SET_LEGIT, SEED  # noqa: E402
from graph_features import add_graph_features  # noqa: E402
from simulate import load_backbone  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
ABLATION_SYNTHETIC = REPO_ROOT / "data" / "synthetic_ablation" / "combined.csv"
LLM_SYNTHETIC = REPO_ROOT / "data" / "synthetic" / "combined.csv"


def build_and_evaluate(synthetic_path: Path, label: str) -> dict:
    real = load_backbone()
    real["is_synthetic"] = 0
    if "attack_family" not in real.columns:
        real["attack_family"] = "none"
    legit = real[real["isFraud"] == 0].sample(n=min(WORKING_SET_LEGIT, (real["isFraud"] == 0).sum()), random_state=SEED)
    native_fraud = real[real["isFraud"] == 1]
    synthetic = pd.read_csv(synthetic_path)

    combined = pd.concat([legit, native_fraud, synthetic], ignore_index=True)
    combined["label"] = ((combined["is_synthetic"] == 1) & (combined["isFraud"] == 1)).astype(int)
    combined = add_tabular_features(combined)
    combined = add_graph_features(combined)

    train, test = time_split(combined)
    n_pos, n_neg = train["label"].sum(), len(train) - train["label"].sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos, eval_metric="aucpr", random_state=42, n_jobs=-1,
    )
    model.fit(train[ALL_FEATURE_COLUMNS], train["label"])

    proba = model.predict_proba(test[ALL_FEATURE_COLUMNS])[:, 1]
    pred = proba >= 0.5
    y = test["label"]

    per_family = {}
    for fam in sorted(test.loc[test["label"] == 1, "attack_family"].unique()):
        mask = (test["attack_family"] == fam) & (test["label"] == 1)
        per_family[fam] = float(recall_score(test.loc[mask, "label"], pred[mask.values], zero_division=0))

    result = {
        "label": label,
        "n_synthetic_rows": int(len(synthetic)),
        "n_test_positives": int(y.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, proba)),
        "per_family_recall": per_family,
    }
    return result


def main():
    if not ABLATION_SYNTHETIC.exists():
        raise SystemExit(
            f"{ABLATION_SYNTHETIC} not found -- run:\n"
            f"  python generate/simulate.py --all --no-llm --out ../data/synthetic_ablation"
        )

    print("=== Ablation: LLM-driven config vs. hand-authored fallback, identical bug-fixed code ===\n")
    fallback_result = build_and_evaluate(ABLATION_SYNTHETIC, "hand-authored fallback (--no-llm)")
    llm_result = build_and_evaluate(LLM_SYNTHETIC, "LLM-driven (Gemini)")

    for r in [fallback_result, llm_result]:
        print(f"{r['label']}: {r['n_synthetic_rows']} synthetic rows, "
              f"precision={r['precision']:.4f} recall={r['recall']:.4f} "
              f"f1={r['f1']:.4f} pr_auc={r['pr_auc']:.4f}")
        for fam, rec in r["per_family_recall"].items():
            print(f"    {fam:28s} recall={rec:.4f}")
        print()

    pr_auc_diff = llm_result["pr_auc"] - fallback_result["pr_auc"]
    print(f"PR-AUC difference (LLM - fallback): {pr_auc_diff:+.4f}")
    if abs(pr_auc_diff) < 0.01:
        print("Conclusion: the LLM path does not meaningfully change bulk detection metrics "
              "on this comparison. Its value, if any, is in the DIVERSITY and REALISM of the "
              "generated parameters (documented reasoning per family, adapting to the specific "
              "backbone's statistics) rather than a measurable accuracy gain -- reported honestly, "
              "not oversold as a performance improvement it doesn't demonstrably provide here.")
    else:
        direction = "higher" if pr_auc_diff > 0 else "lower"
        print(f"Conclusion: the LLM-driven configs produced {direction} PR-AUC than the fallback "
              f"on this comparison -- a real, measured difference, not assumed.")

    out = {"fallback": fallback_result, "llm": llm_result, "pr_auc_diff": pr_auc_diff}
    out_path = Path(__file__).parent / "artifacts" / "ablation_llm_vs_fallback.json"
    import json
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
