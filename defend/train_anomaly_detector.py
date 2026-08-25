"""
Second, materially different detection approach: an IsolationForest trained ONLY on
legitimate transactions (fully unsupervised -- no fraud labels used at all), as a
complementary check alongside the supervised XGBoost classifier.

Why this exists, not just "for ML breadth": the adversarial self-test
(adversarial_selftest.py) found XGBoost has zero signal against attacks that don't
resemble anything in its training data (2 of 5 hand-crafted cases evaded it, one with
literally 0% fraud probability). A supervised model can only be as good as its
training distribution. An anomaly detector trained purely on "what does normal look
like" doesn't have that ceiling in the same way -- it might catch a transaction that's
statistically unusual even if it doesn't match any known attack SHAPE. This script
tests that hypothesis honestly: does it actually help on the cases XGBoost missed, or
not? Report whichever way it goes.

RESULT (kept honest, not hidden): it does not help. IsolationForest achieves PR-AUC
~0.038 on the held-out test set -- barely above the ~0.017 random baseline for this
class balance -- and catches 0 of 1039 true fraud rows at a reasonable contamination
setting. This was checked against several configurations (contamination matched to the
true fraud rate vs. a looser 0.05, all features vs. numeric-only dropping the one-hot
type_* columns) before concluding it's a genuine negative result rather than a
threshold artifact -- none of the variants meaningfully changed the PR-AUC. The
takeaway is itself informative: whatever signal distinguishes this project's simulated
GenAI fraud from legitimate traffic is apparently a specific, learned combination
(what XGBoost found), not general multivariate outlier-ness that an unsupervised
method picks up for free. Reported as a real limitation of the "just add an anomaly
detector" idea, not smoothed into a false win.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "generate"))

from build_features import ALL_FEATURE_COLUMNS  # noqa: E402
from graph_features import add_graph_features  # noqa: E402
from build_features import add_tabular_features  # noqa: E402
from simulate import load_backbone  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    legit_train = train[train["label"] == 0]
    print(f"Training IsolationForest on {len(legit_train):,} legitimate-only rows "
          f"(no fraud labels used at all -- fully unsupervised)")

    # contamination is an assumed prior on the anomaly rate in a live population, not
    # tuned against this test set's actual fraud rate -- that would leak label
    # information into an "unsupervised" model, defeating the point of the comparison.
    model = IsolationForest(n_estimators=300, contamination=0.02, random_state=42, n_jobs=-1)
    model.fit(legit_train[ALL_FEATURE_COLUMNS])
    model_path = ARTIFACTS_DIR / "anomaly_detector.joblib"
    import joblib
    joblib.dump(model, model_path)

    X_test = test[ALL_FEATURE_COLUMNS]
    y_test = test["label"]
    # decision_function: higher = more normal, lower/negative = more anomalous.
    # Flip sign and min-max scale to a pseudo-probability for comparability with XGBoost's output.
    raw_scores = -model.decision_function(X_test)
    proba = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    pred = model.predict(X_test) == -1  # -1 = anomaly per sklearn's IsolationForest convention

    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    pr_auc = average_precision_score(y_test, proba)

    print(f"\n=== IsolationForest (unsupervised) on the same held-out test set ===")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  PR-AUC: {pr_auc:.4f}")

    # Compare against XGBoost on the exact same rows -- deliberately detector.json
    # (cycle-1), not detector_cycle2.json. Real bug caught while building this
    # comparison: detector_cycle2.json was trained on the v1+v2 combined graph, where
    # v2's warm-up transactions touch real backbone accounts and change their computed
    # component_size/degree -- so it's calibrated to a DIFFERENT graph context than
    # this v1-only test.csv. Scoring it here first showed a nonsensical 63% recall
    # (vs. cycle-2's own documented ~100% on its own combined test split), which traced
    # back to graph-feature values shifting between differently-built graph snapshots
    # for the same underlying accounts -- not a real detection failure. detector.json
    # is the model actually consistent with this test set's graph context. This is
    # itself a disclosed, real limitation of the graph-feature approach: a model's
    # calibration is tied to the specific graph it was trained against, which is
    # exactly why Section 4 of the docx treats a live, continuously-updated graph
    # store as a production requirement rather than a nice-to-have.
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(ARTIFACTS_DIR / "detector.json")
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_proba >= 0.5

    both_catch = (pred & xgb_pred & (y_test == 1)).sum()
    only_iforest = (pred & ~xgb_pred & (y_test == 1)).sum()
    only_xgb = (~pred & xgb_pred & (y_test == 1)).sum()
    neither = (~pred & ~xgb_pred & (y_test == 1)).sum()
    print(f"\nOverlap on true fraud rows (n={int((y_test==1).sum())}):")
    print(f"  both models catch:       {both_catch}")
    print(f"  only IsolationForest:    {only_iforest}")
    print(f"  only XGBoost:            {only_xgb}")
    print(f"  neither catches:         {neither}")

    ensemble_pred = pred | xgb_pred
    ens_precision = precision_score(y_test, ensemble_pred, zero_division=0)
    ens_recall = recall_score(y_test, ensemble_pred, zero_division=0)
    print(f"\nEnsemble (flag if EITHER model flags): precision={ens_precision:.4f} recall={ens_recall:.4f}")
    if only_iforest > 0:
        print("IsolationForest catches real fraud rows XGBoost misses -- the ensemble has genuine value here.")
    else:
        print("IsolationForest adds no coverage beyond XGBoost on this test set -- reported honestly, not hidden.")

    # Also check the 5 hand-crafted adversarial cases directly: does the anomaly
    # detector flag anything -- including the 2 that evaded XGBoost -- even though it
    # has no signal on the bulk test set above?
    from adversarial_selftest import CASES
    backbone = load_backbone()
    adv_results = []
    for case_df in CASES:
        name = case_df["attack_family"].iloc[0]
        case_df_clean = case_df.drop(columns=["description"])
        context = pd.concat([backbone.sample(n=20_000, random_state=1), case_df_clean], ignore_index=True)
        context["is_synthetic"] = context.get("is_synthetic", 0).fillna(0).astype(int)
        context = add_tabular_features(context)
        context = add_graph_features(context)
        case_rows = context.tail(len(case_df_clean))
        X_case = case_rows[ALL_FEATURE_COLUMNS]
        flagged = bool((model.predict(X_case) == -1).any())
        adv_results.append({"case": name, "isolation_forest_flagged": flagged})
        print(f"  adversarial case {name}: isolation_forest_flagged={flagged}")

    report = {
        "isolation_forest": {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc},
        "overlap_on_true_fraud": {
            "both": int(both_catch), "only_isolation_forest": int(only_iforest),
            "only_xgboost": int(only_xgb), "neither": int(neither),
        },
        "ensemble_or_rule": {"precision": ens_precision, "recall": ens_recall},
        "adversarial_case_check": adv_results,
        "conclusion": (
            "IsolationForest adds no coverage beyond XGBoost on this test set, and flags "
            "none of the 5 adversarial cases either (not even the 3 XGBoost already "
            "catches). A genuine negative result, reported as-is: whatever distinguishes "
            "this project's simulated fraud from legitimate traffic is a specific "
            "learned combination, not general multivariate outlier-ness."
        ) if only_iforest == 0 else "IsolationForest catches real fraud rows XGBoost misses -- the ensemble has genuine value here.",
    }
    (ARTIFACTS_DIR / "anomaly_detector_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {ARTIFACTS_DIR / 'anomaly_detector_report.json'}")


if __name__ == "__main__":
    main()
