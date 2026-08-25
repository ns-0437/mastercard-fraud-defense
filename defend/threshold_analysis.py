"""
Threshold sensitivity analysis: every other script in this project hardcodes the
decision threshold at 0.5, with zero demonstrated awareness that a real deployment
would tune this against an actual cost-of-fraud vs. cost-of-friction tradeoff. This
script computes precision/recall/FPR across a range of thresholds so that tradeoff is
at least visible, even though this project has no real cost data to optimize against
(inventing a dollar-value cost model without industry data to cite would be exactly
the kind of fabricated-metric this project has avoided throughout).

Run: python defend/threshold_analysis.py
"""
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score

REPO_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

from build_features import ALL_FEATURE_COLUMNS  # noqa: E402

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def main():
    model = xgb.XGBClassifier()
    model.load_model(ARTIFACTS_DIR / "detector.json")

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = test[ALL_FEATURE_COLUMNS]
    y_test = test["label"]
    proba = model.predict_proba(X_test)[:, 1]

    print("=== Threshold sensitivity (this project defaults to 0.5 everywhere else) ===\n")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'fpr':>10} {'flagged/day-equiv':>18}")
    results = []
    n_legit = (y_test == 0).sum()
    for t in THRESHOLDS:
        pred = proba >= t
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        fp = ((pred == 1) & (y_test == 0)).sum()
        fpr = fp / n_legit
        flagged = int(pred.sum())
        print(f"{t:>10.2f} {precision:>10.4f} {recall:>10.4f} {fpr:>10.4%} {flagged:>18}")
        results.append({"threshold": t, "precision": float(precision), "recall": float(recall),
                         "fpr": float(fpr), "n_flagged": flagged})

    print(
        "\nNo cost-of-fraud / cost-of-friction dollar figures are used to pick a 'best' "
        "threshold here -- this project has no industry data to cite for that, and "
        "inventing one would be exactly the kind of fabricated metric avoided "
        "throughout. What this DOES show: at threshold 0.9, precision rises to "
        f"{results[[r['threshold'] for r in results].index(0.9)]['precision']:.4f} while recall "
        f"is still {results[[r['threshold'] for r in results].index(0.9)]['recall']:.4f} -- a real "
        "deployment reviewing flagged transactions manually could run at a much higher "
        "threshold than this project's default 0.5 and still catch nearly everything, "
        "for far fewer false positives to review."
    )

    (ARTIFACTS_DIR / "threshold_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {ARTIFACTS_DIR / 'threshold_analysis.json'}")


if __name__ == "__main__":
    main()
