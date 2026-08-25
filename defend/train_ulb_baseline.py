"""
Phase 3 secondary run: trains and evaluates a methodologically-comparable XGBoost model
on the ULB Credit Card Fraud dataset, entirely independently of the PaySim-trained
detector.

Why this exists instead of a literal "holdout" run of the PaySim model (see the Aug 25
correction in docs/PHASES.md): ULB has no account/counterparty fields at all -- just
`Time`, PCA-anonymized `V1`-`V28`, `Amount`, and `Class`. There is no graph to build and
none of build_features.py's tabular features (balances, transaction type) exist here.
Running the PaySim model on this schema isn't possible without inventing features that
don't correspond to anything real, which would be a fabricated metric dressed up as a
generalization test -- exactly what CLAUDE.md prohibits.

This script instead applies the SAME METHODOLOGY (XGBoost, time-based split, imbalance
handling, PR-AUC as the headline metric) to a completely independent real fraud
dataset. It answers "does this approach work on real-world card fraud in general,"
not "does the specific PaySim model transfer." Report both results side by side in the
docx with that distinction stated explicitly.
"""
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix,
)

REPO_ROOT = Path(__file__).parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "creditcard.csv"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
SPLIT_FRAC = 0.20
DECISION_THRESHOLD = 0.5

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Loading {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"{len(df):,} rows, {df['Class'].sum():,} fraud ({df['Class'].mean():.4%})")

    cutoff = df["Time"].quantile(1 - SPLIT_FRAC)
    train = df[df["Time"] < cutoff]
    test = df[df["Time"] >= cutoff]
    print(f"Time-based split: train={len(train):,} ({train['Class'].sum()} fraud), "
          f"test={len(test):,} ({test['Class'].sum()} fraud)")

    X_train, y_train = train[FEATURE_COLUMNS], train["Class"]
    X_test, y_test = test[FEATURE_COLUMNS], test["Class"]

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    model.save_model(ARTIFACTS_DIR / "ulb_baseline_detector.json")

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)

    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr = fp / (fp + tn)

    print("\n=== ULB independent baseline (same methodology, unrelated real dataset) ===")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}  ROC-AUC: {roc_auc:.4f}")
    print(f"False positive rate on legit: {fpr:.4%} ({fp}/{fp+tn})")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print("\nNote: no graph features exist here (no account IDs in this dataset) and "
          "no GenAI-specific attack signal is being tested -- this is general card-"
          "present/card-not-present fraud. Lower or different performance than the "
          "PaySim result is expected and should be reported as a scoped comparison, "
          "not a failure.")

    report = {
        "precision": precision, "recall": recall, "f1": f1,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "false_positive_rate": fpr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "note": "Independent methodology validation, not a literal transfer/holdout test "
                "of the PaySim-trained model -- see module docstring.",
    }
    (ARTIFACTS_DIR / "ulb_baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {ARTIFACTS_DIR / 'ulb_baseline_report.json'}")


if __name__ == "__main__":
    main()
