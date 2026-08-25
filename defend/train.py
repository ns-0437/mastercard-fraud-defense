"""
Phase 3 step 2: trains the primary detector (XGBoost, tabular + graph features) on
data/processed/train.csv, produced by build_features.py's time-based split.

class imbalance handled via scale_pos_weight (ratio of negatives to positives in the
TRAIN set only -- computing it from train+test combined would leak test-set class
balance information into a training hyperparameter).
"""
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from build_features import ALL_FEATURE_COLUMNS

REPO_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(PROCESSED_DIR / "train.csv")

    X_train = train[ALL_FEATURE_COLUMNS]
    y_train = train["label"]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"Train set: {len(train):,} rows, {n_pos:,} positive ({n_pos/len(train):.3%}), "
          f"scale_pos_weight={scale_pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    model_path = ARTIFACTS_DIR / "detector.json"
    model.save_model(model_path)

    importances = dict(zip(ALL_FEATURE_COLUMNS, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
    (ARTIFACTS_DIR / "feature_importance.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")

    print(f"\nModel saved to {model_path}")
    print("\nTop 10 features by importance:")
    for feat, imp in list(importances.items())[:10]:
        print(f"  {feat:35s} {imp:.4f}")


if __name__ == "__main__":
    main()
