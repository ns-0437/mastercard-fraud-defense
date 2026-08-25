"""
Phase 3 step 3: evaluates the trained detector on the held-out time-split test set.
Per CLAUDE.md/SKILL.md gate: reports precision/recall/F1/PR-AUC (not accuracy alone --
meaningless on data this imbalanced), explicit false positive rate on legitimate
transactions, and a per-attack-family recall breakdown (a detector that catches 3 of 4
families and silently misses the 4th is not "detection efficacy," it's a diversity gap
hiding inside an averaged metric).

Also reports, as a clearly-labeled SECONDARY signal: recall on PaySim's own native
fraud rows, which the model was never trained to detect (see build_features.py -- only
`is_synthetic` rows are the training label). This is not a "generalization test" in the
holdout sense -- it's an honest note on incidental transfer to a known fraud pattern.
"""
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, confusion_matrix,
)

from build_features import ALL_FEATURE_COLUMNS

REPO_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DECISION_THRESHOLD = 0.5


def main():
    model = xgb.XGBClassifier()
    model.load_model(ARTIFACTS_DIR / "detector.json")

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = test[ALL_FEATURE_COLUMNS]
    y_test = test["label"]

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)

    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr_on_legit = fp / (fp + tn)

    print("=== PRIMARY: detection of synthetic GenAI-simulated attacks (held-out, time-split test set) ===")
    print(f"Test set: {len(test):,} rows, {y_test.sum():,} positives ({y_test.mean():.3%})")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1:                   {f1:.4f}")
    print(f"PR-AUC:               {pr_auc:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"False positive rate on legitimate transactions: {fpr_on_legit:.4%} ({fp:,}/{fp+tn:,})")
    print(f"Confusion matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

    print("\n=== Per-attack-family recall (diversity of detection, not just averaged recall) ===")
    family_results = {}
    for family in test.loc[test["label"] == 1, "attack_family"].unique():
        mask = (test["attack_family"] == family) & (test["label"] == 1)
        fam_recall = recall_score(test.loc[mask, "label"], pred[mask.values], zero_division=0)
        family_results[family] = fam_recall
        print(f"  {family:28s} recall={fam_recall:.4f} (n={mask.sum()})")

    print("\n=== SECONDARY (not a generalization test, see module docstring): "
          "recall on PaySim's own native fraud, never trained on ===")
    native_mask = test["is_native_fraud"] == 1
    if native_mask.sum() > 0:
        native_recall = recall_score(test.loc[native_mask, "is_native_fraud"], pred[native_mask.values], zero_division=0)
        print(f"  native fraud rows in test: {native_mask.sum():,}, recall={native_recall:.4f}")
    else:
        native_recall = None
        print("  no native fraud rows in this test split")

    report = {
        "primary": {
            "precision": precision, "recall": recall, "f1": f1,
            "pr_auc": pr_auc, "roc_auc": roc_auc,
            "false_positive_rate_on_legit": fpr_on_legit,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "test_set_size": len(test), "test_set_positives": int(y_test.sum()),
        },
        "per_family_recall": family_results,
        "secondary_native_fraud_recall": native_recall,
        "decision_threshold": DECISION_THRESHOLD,
    }
    report_path = ARTIFACTS_DIR / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    gate_pass = pr_auc > 0.5 and all(r > 0 for r in family_results.values()) and fpr_on_legit < 0.05
    print(f"\nPHASE 3 GATE (PR-AUC>0.5, all families recall>0, FPR<5%): {'PASS' if gate_pass else 'FAIL'}")

    if pr_auc > 0.99:
        print(
            "\nCAVEAT -- do not quote this PR-AUC in the docx without this context: a "
            "near-perfect score here is expected to be partly an easy-benchmark effect, "
            "not proof of production-grade detection. Two concrete reasons: (1) every "
            "synthetic attack account is brand-new by construction, and the working set "
            "is a 300K-row RANDOM SUBSAMPLE of a 6.3M-row backbone, so most real accounts "
            "show degree=1 by subsampling accident, exaggerating the real-vs-synthetic "
            "degree gap beyond what a full-history graph would show; (2) card-testing and "
            "mule-layering are deliberately high-velocity/high-fanout IN THIS FIRST "
            "GENERATION, which a basic hand-coded velocity rule would likely also catch -- "
            "this run doesn't yet prove ML adds value over simpler heuristics. The real "
            "test is Phase 4: whether recall holds up once the closed loop generates "
            "harder, lower-velocity variants specifically designed to evade this model. "
            "Report this run's numbers as a baseline, and report Phase 4's numbers as "
            "the actual claim of detection efficacy."
        )


if __name__ == "__main__":
    main()
