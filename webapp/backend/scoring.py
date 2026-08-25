"""
Live single-transaction scoring for the web prototype's demo endpoint.

Deliberately NOT a call into defend/graph_features.py's NetworkX pipeline -- that
pipeline builds a full graph over the whole batch it's given, which is right for
training/evaluation but wrong for a live API that needs to score one transaction in
milliseconds. Instead this does an O(1) dictionary lookup against a precomputed
snapshot (defend/export_graph_snapshot.py) of per-account degree/component-size.

Known, disclosed simplification (say this in the docx, don't let it be an accidental
discrepancy): shared_neighbor_count, which build_features.py computes exactly from full
adjacency, is approximated here as 0 whenever either account isn't in the snapshot's
edge list -- this demo doesn't ship a full adjacency index. A real production system
would query a live graph/feature store (e.g. a graph database updated in near-real-time)
instead of a static CSV snapshot; this is a demo-scoped simplification, not a claim
about how this would run in production.
"""
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "defend" / "artifacts"

FEATURE_COLUMNS = [
    "amount", "step_of_day", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "balance_delta_orig",
    "amount_to_orig_balance_ratio", "dest_balance_was_zero",
    "type_PAYMENT", "type_TRANSFER", "type_CASH_OUT", "type_CASH_IN", "type_DEBIT",
    "orig_out_degree", "orig_in_degree", "orig_component_size",
    "dest_out_degree", "dest_in_degree", "dest_component_size",
    "shared_neighbor_count", "orig_is_low_activity",
]

DEFAULT_NODE = {"out_degree": 0, "in_degree": 0, "component_size": 1}


class Scorer:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(ARTIFACTS_DIR / "detector_cycle2.json")

        snapshot = pd.read_csv(ARTIFACTS_DIR / "graph_node_snapshot.csv", index_col="account")
        self.node_lookup = snapshot.to_dict(orient="index")

        importances = json.loads((ARTIFACTS_DIR / "feature_importance.json").read_text(encoding="utf-8"))
        self.top_features = list(importances.keys())[:6]

    def _node(self, account: str) -> dict:
        return self.node_lookup.get(account, DEFAULT_NODE)

    def build_features(self, txn: dict) -> dict:
        orig_node = self._node(txn.get("nameOrig", ""))
        dest_node = self._node(txn.get("nameDest", ""))
        known_pair = txn.get("nameOrig") in self.node_lookup and txn.get("nameDest") in self.node_lookup

        amount = float(txn["amount"])
        old_bal_orig = float(txn.get("oldbalanceOrg", 0.0))
        new_bal_orig = float(txn.get("newbalanceOrig", max(0.0, old_bal_orig - amount)))
        old_bal_dest = float(txn.get("oldbalanceDest", 0.0))
        new_bal_dest = float(txn.get("newbalanceDest", old_bal_dest + amount))
        txn_type = txn.get("type", "PAYMENT")

        feats = {
            "amount": amount,
            "step_of_day": int(txn.get("step", 1)) % 24,
            "oldbalanceOrg": old_bal_orig,
            "newbalanceOrig": new_bal_orig,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "balance_delta_orig": old_bal_orig - new_bal_orig,
            "amount_to_orig_balance_ratio": amount / (old_bal_orig + 1.0),
            "dest_balance_was_zero": int(old_bal_dest == 0),
            "type_PAYMENT": int(txn_type == "PAYMENT"),
            "type_TRANSFER": int(txn_type == "TRANSFER"),
            "type_CASH_OUT": int(txn_type == "CASH_OUT"),
            "type_CASH_IN": int(txn_type == "CASH_IN"),
            "type_DEBIT": int(txn_type == "DEBIT"),
            "orig_out_degree": orig_node["out_degree"],
            "orig_in_degree": orig_node["in_degree"],
            "orig_component_size": orig_node["component_size"],
            "dest_out_degree": dest_node["out_degree"],
            "dest_in_degree": dest_node["in_degree"],
            "dest_component_size": dest_node["component_size"],
            "shared_neighbor_count": 0,  # see module docstring
            "orig_is_low_activity": int(orig_node["component_size"] <= 3),
        }
        feats["_known_pair_in_snapshot"] = known_pair
        return feats

    def score(self, txn: dict) -> dict:
        feats = self.build_features(txn)
        row = pd.DataFrame([{k: feats[k] for k in FEATURE_COLUMNS}])
        proba = float(self.model.predict_proba(row)[0, 1])
        return {
            "fraud_probability": proba,
            "prediction": "fraud" if proba >= 0.5 else "legitimate",
            "features": feats,
            "top_model_signals": self.top_features,
        }
