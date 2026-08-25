---
name: mastercard-fraud-defense
description: Operate the Mastercard Innovation Challenge closed-loop fraud system — build/query the attack taxonomy graph, run attack generation, train and evaluate the detector with graph features, run the full closed-loop pipeline, and run the pre-submission bulletproofing checklist. Use when working anywhere in this repo.
---

# Mastercard Fraud Defense — operating skill

This repo builds a closed-loop Identify → Generate → Defend system. See
[CLAUDE.md](CLAUDE.md) for architecture and constraints, [docs/PHASES.md](docs/PHASES.md)
for the phase plan. This file is the "how do I run X" reference — keep it in sync as
scripts get added; a command listed here that doesn't exist yet is a bug, fix the doc
or write the script.

## Core commands

### Identify — attack taxonomy graph
```bash
python identify/build_taxonomy.py          # (re)generates identify/taxonomy_graph.json
python identify/query_taxonomy.py --stats  # prints node/edge counts, coverage by channel
```
Gate: every node must cite a real-world mechanism (a payment rail, a social-engineering
surface, or a documented fraud pattern) in its `grounding` field — no invented attacks
with no real-world anchor.

### Generate — attack simulation
```bash
python generate/simulate.py --attack-family <name> --n 5000 --out data/synthetic/
python generate/validate_fidelity.py --synthetic data/synthetic/ --real data/raw/paysim.csv
```
Gate: `validate_fidelity.py` must pass distribution-similarity checks (KS-test per
numeric feature, categorical frequency comparison) before the output is allowed into
`defend/`. If it fails, fix the simulator — do not hand-tune the check.

### Defend — detector training and evaluation
```bash
python defend/build_features.py     # tabular + graph features from data/processed
python defend/train.py              # trains classifier, writes model + metrics to defend/artifacts/
python defend/evaluate.py --holdout ulb_creditcard   # generalization check, run once
```
Gate: report precision/recall/F1/PR-AUC on (a) held-out synthetic-attack test set and
(b) the ULB holdout, side by side. Both numbers go in the docx — never just (a).

### Orchestrator — full closed loop
```bash
python orchestrator/pipeline.py --cycles 2
```
Runs Identify → Generate → Defend → Feedback for N cycles; each cycle's missed-detection
cases get re-fed into Generate as harder variants. Gate: cycle 2's detection metrics on
the *original* attack families must not regress versus cycle 1 — if they do, the
feedback loop is overfitting to new variants at the expense of old ones, and that's a
real bug to fix before demoing it.

### Webapp
```bash
uvicorn webapp.backend.main:app --reload   # backend, http://localhost:8000
npm --prefix webapp/frontend run dev       # frontend, http://localhost:5173
```

### Pre-submission bulletproofing
```bash
git clone <repo> /tmp/fresh-clone && cd /tmp/fresh-clone && <setup + run instructions from README>
```
Run this literally, from a clean clone, before submitting. If it doesn't work from
scratch it doesn't work — "it works on my machine" is not a passing gate.

## Working rules for whoever (human or Claude) touches this repo
- Never commit `data/raw/*.csv` (large, redistributable-license-restricted) — `.gitignore`
  covers this. Document the download source in `data/raw/README.md` instead.
- Never put a metric in `docs/` or the .docx source that isn't the literal stdout of a
  script in this repo.
- Every new script gets a corresponding gate check added to this file in the same commit.
