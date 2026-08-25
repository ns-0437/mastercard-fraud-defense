"""
Phase 3 step 1: assembles the labeled, feature-engineered dataset the detector trains
on, and writes a time-based train/test split to data/processed/.

Dataset composition (see docs/PHASES.md for why):
  - A random subsample of real legitimate PaySim transactions (isFraud==0) — full
    6.3M-row backbone isn't needed for feature engineering and makes the transaction
    graph unwieldy to build per-run; WORKING_SET_LEGIT controls the subsample size.
  - ALL of PaySim's own native fraud rows (isFraud==1, not synthetic) — kept out of
    the primary training target, used only as a secondary "does this incidentally
    catch fraud patterns it wasn't built for" check in evaluate.py.
  - ALL synthetic GenAI-attack rows from data/synthetic/combined.csv — this is the
    actual `label=1` target Defend is built to catch.

Split: by `step` (PaySim's time unit), NOT randomly shuffled — the last SPLIT_FRAC of
the time range is held out as test. Random shuffling would leak future account balance
patterns into training and inflate every metric; a real deployed detector only ever
sees the past.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from graph_features import add_graph_features

REPO_ROOT = Path(__file__).parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
SYNTHETIC_PATH = REPO_ROOT / "data" / "synthetic" / "combined.csv"
OUT_DIR = REPO_ROOT / "data" / "processed"

WORKING_SET_LEGIT = 300_000   # subsample size of real legit rows for graph tractability
SPLIT_FRAC = 0.20             # last 20% of the step range held out as test
SEED = 42

TABULAR_FEATURE_COLUMNS = [
    "amount", "step_of_day", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "balance_delta_orig",
    "amount_to_orig_balance_ratio", "dest_balance_was_zero",
    "type_PAYMENT", "type_TRANSFER", "type_CASH_OUT", "type_CASH_IN", "type_DEBIT",
]
GRAPH_FEATURE_COLUMNS = [
    "orig_out_degree", "orig_in_degree", "orig_component_size",
    "dest_out_degree", "dest_in_degree", "dest_component_size",
    "shared_neighbor_count", "orig_is_low_activity",
]
ALL_FEATURE_COLUMNS = TABULAR_FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS


def load_and_combine() -> pd.DataFrame:
    print(f"Loading real backbone from {RAW_PATH}")
    real = pd.read_csv(RAW_PATH)
    real["is_synthetic"] = 0
    if "attack_family" not in real.columns:
        real["attack_family"] = "none"

    legit = real[real["isFraud"] == 0]
    native_fraud = real[real["isFraud"] == 1]
    print(f"Real backbone: {len(real):,} rows total, {len(legit):,} legit, {len(native_fraud):,} native fraud")

    rng = np.random.default_rng(SEED)
    legit_sample = legit.sample(n=min(WORKING_SET_LEGIT, len(legit)), random_state=SEED)
    print(f"Subsampled legit to {len(legit_sample):,} rows (WORKING_SET_LEGIT={WORKING_SET_LEGIT:,})")

    print(f"Loading synthetic attacks from {SYNTHETIC_PATH}")
    synthetic = pd.read_csv(SYNTHETIC_PATH)
    print(f"Synthetic attacks: {len(synthetic):,} rows across {synthetic['attack_family'].nunique()} families")

    combined = pd.concat([legit_sample, native_fraud, synthetic], ignore_index=True)
    # Primary detection target: is this a GenAI-simulated attack row specifically
    # (isFraud==1 AND synthetic-sourced) -- NOT just "came from the synthetic
    # generator." Warm-up transactions (Phase 4 hardening, see simulators.py) are
    # synthetic-sourced but deliberately isFraud==0: they represent an attacker
    # account's ordinary-looking prior activity and must not be trained/scored as
    # fraud, or the model would learn that small innocuous payments are suspicious.
    # Native PaySim fraud is deliberately NOT part of this label (see module
    # docstring) -- it's evaluated separately in evaluate.py as a secondary signal.
    combined["label"] = ((combined["is_synthetic"] == 1) & (combined["isFraud"] == 1)).astype(int)
    combined["is_native_fraud"] = ((combined["isFraud"] == 1) & (combined["is_synthetic"] == 0)).astype(int)
    return combined


def add_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["step_of_day"] = out["step"] % 24
    out["balance_delta_orig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["amount_to_orig_balance_ratio"] = out["amount"] / (out["oldbalanceOrg"] + 1.0)
    out["dest_balance_was_zero"] = (out["oldbalanceDest"] == 0).astype(int)
    for t in ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]:
        out[f"type_{t}"] = (out["type"] == t).astype(int)
    return out


def time_split(df: pd.DataFrame, split_frac: float = SPLIT_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["step"].quantile(1 - split_frac)
    train = df[df["step"] < cutoff]
    test = df[df["step"] >= cutoff]
    return train, test


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = load_and_combine()
    combined = add_tabular_features(combined)

    print("Building transaction graph and computing graph features "
          f"over {len(combined):,} rows (this is the working set, not the full 6.3M-row backbone)...")
    combined = add_graph_features(combined)

    train, test = time_split(combined)
    print(f"\nTime-based split at step cutoff (last {SPLIT_FRAC:.0%} of step range held out):")
    print(f"  train: {len(train):,} rows, {train['label'].sum():,} synthetic-attack positives")
    print(f"  test:  {len(test):,} rows, {test['label'].sum():,} synthetic-attack positives")

    keep_cols = ALL_FEATURE_COLUMNS + ["label", "is_native_fraud", "attack_family", "step"]
    train[keep_cols].to_csv(OUT_DIR / "train.csv", index=False)
    test[keep_cols].to_csv(OUT_DIR / "test.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'train.csv'} and {OUT_DIR / 'test.csv'}")


if __name__ == "__main__":
    main()
