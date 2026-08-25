"""
FastAPI backend for the web prototype. Exposes the taxonomy graph, closed-loop cycle
results, and a live single-transaction detection demo, all reading directly from the
artifacts this repo's Identify/Generate/Defend/Orchestrator scripts already produced --
this API does not retrain or resimulate anything, it only serves what's on disk.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from scoring import Scorer  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
TAXONOMY_PATH = REPO_ROOT / "identify" / "taxonomy_graph.json"
FIDELITY_PATH = REPO_ROOT / "generate" / "fidelity_report.json"
EVAL_PATH = REPO_ROOT / "defend" / "artifacts" / "evaluation_report.json"
ULB_PATH = REPO_ROOT / "defend" / "artifacts" / "ulb_baseline_report.json"
CYCLE_PATH = REPO_ROOT / "orchestrator" / "cycle_report.json"
ADVERSARIAL_PATH = REPO_ROOT / "defend" / "artifacts" / "adversarial_selftest_report.json"
CONFIGS_USED_PATH = REPO_ROOT / "generate" / "attack_configs_used.json"
FEATURE_IMPORTANCE_PATH = REPO_ROOT / "defend" / "artifacts" / "feature_importance.json"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
# A small pre-extracted sample of real legit PaySim rows, NOT the full 493MB raw file --
# the deployed container doesn't ship the full backbone dataset, only what the API
# actually needs to demo (see webapp/backend/README.md for how this was generated).
LEGIT_SAMPLES_PATH = Path(__file__).parent / "legit_samples.csv"

app = FastAPI(title="Mastercard Innovation Challenge - AI Defense Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo scope; a real deployment would scope this to the actual frontend origin
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer = Scorer()


class TransactionIn(BaseModel):
    amount: float
    type: str = "PAYMENT"
    step: int = 1
    nameOrig: str = ""
    nameDest: str = ""
    oldbalanceOrg: float = 0.0
    newbalanceOrig: float | None = None
    oldbalanceDest: float = 0.0
    newbalanceDest: float | None = None


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/taxonomy")
def taxonomy():
    data = _read_json(TAXONOMY_PATH)
    if data is None:
        raise HTTPException(404, "taxonomy graph not built yet -- run identify/build_taxonomy.py")
    return data


@app.get("/api/cycles")
def cycles():
    return {
        "fidelity": _read_json(FIDELITY_PATH),
        "primary_evaluation": _read_json(EVAL_PATH),
        "ulb_baseline": _read_json(ULB_PATH),
        "closed_loop": _read_json(CYCLE_PATH),
        "adversarial_selftest": _read_json(ADVERSARIAL_PATH),
    }


@app.get("/api/generate")
def generate_pillar():
    """Everything that demonstrates the Generate pillar's actual output: which LLM
    provider produced each attack family's config and what it said, plus the fidelity
    validation results that gate whether that output was trustworthy enough to train
    on. Both already exist as artifacts (attack_configs_used.json, fidelity_report.json)
    -- this just serves them together for the web app's Generate & Fidelity tab."""
    return {
        "configs_used": _read_json(CONFIGS_USED_PATH),
        "fidelity": _read_json(FIDELITY_PATH),
        "feature_importance": _read_json(FEATURE_IMPORTANCE_PATH),
    }


@app.get("/api/samples")
def samples():
    out = []
    v2_path = SYNTHETIC_DIR / "combined_v2.csv"
    v1_path = SYNTHETIC_DIR / "combined.csv"
    for path, tag in [(v2_path, "hardened"), (v1_path, "original")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # isFraud==1 filter matters here: hardened (v2) families include deliberately
        # non-fraud warm-up transactions (see generate/simulators.py) mixed in with the
        # actual attack rows -- without this filter, .iloc[0] can grab a warm-up row
        # and the demo would show a "hardened attack" that's actually a legitimate-
        # looking setup transaction, scoring as legitimate and looking like a bug.
        attack_rows = df[df["isFraud"] == 1]
        for family in attack_rows["attack_family"].unique():
            row = attack_rows[attack_rows["attack_family"] == family].iloc[0]
            out.append({
                "label": f"{family} ({tag})",
                "attack_family": family,
                "transaction": {
                    "amount": float(row["amount"]), "type": row["type"], "step": int(row["step"]),
                    "nameOrig": row["nameOrig"], "nameDest": row["nameDest"],
                    "oldbalanceOrg": float(row["oldbalanceOrg"]), "newbalanceOrig": float(row["newbalanceOrig"]),
                    "oldbalanceDest": float(row["oldbalanceDest"]), "newbalanceDest": float(row["newbalanceDest"]),
                },
            })
    if LEGIT_SAMPLES_PATH.exists():
        legit_df = pd.read_csv(LEGIT_SAMPLES_PATH)
        legit_row = legit_df.iloc[0]
        out.insert(0, {
            "label": "legitimate transaction (real PaySim data)",
            "attack_family": "none",
            "transaction": {
                "amount": float(legit_row["amount"]), "type": legit_row["type"], "step": int(legit_row["step"]),
                "nameOrig": legit_row["nameOrig"], "nameDest": legit_row["nameDest"],
                "oldbalanceOrg": float(legit_row["oldbalanceOrg"]), "newbalanceOrig": float(legit_row["newbalanceOrig"]),
                "oldbalanceDest": float(legit_row["oldbalanceDest"]), "newbalanceDest": float(legit_row["newbalanceDest"]),
            },
        })
    return out


@app.post("/api/detect")
def detect(txn: TransactionIn):
    return scorer.score(txn.model_dump())
