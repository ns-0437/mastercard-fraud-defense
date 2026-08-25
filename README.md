# AI Defense Lab for Payment Security

Submission for the Mastercard Innovation Challenge 2026 @ GFF Mumbai. A closed-loop
red-team/blue-team AI system for GenAI-powered payment fraud: **Identify** emerging
attack vectors as a knowledge graph, **Generate** high-fidelity simulated attacks at
scale, **Defend** with a graph-feature-augmented classifier, wired into a feedback loop
so detector blind spots drive the next round of generation.

See [CLAUDE.md](CLAUDE.md) for architecture/constraints, [docs/PHASES.md](docs/PHASES.md)
for the build plan, [SKILL.md](SKILL.md) for how to run each piece.

## Status
Phases 0-5 built, gate-checked, and deployed (see docs/PHASES.md for the full record,
including an earlier FAIL on one small-sample regression in Phase 4 that was kept in
the log rather than deleted once it was superseded). Remaining: Phase 6 (.docx
walkthrough), Phase 7 (bulletproofing pass), Phase 8 (submission).

**Live demo**: https://mastercard-frontend-612620013489.us-central1.run.app
(backend: https://mastercard-backend-612620013489.us-central1.run.app) — both on GCP
Cloud Run, project `mastercard-fraud-defense`.

**Repo**: https://github.com/ns-0437/mastercard-fraud-defense

## Quickstart
Requires Python 3.11+, Node 20+, and both real datasets in `data/raw/` (see
`data/raw/README.md`). Optionally set `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` /
`OPENAI_API_KEY` in `.env` (repo root) for LLM-driven attack config generation — every
step below runs without any of them too, falling back to hand-authored configs.

```bash
pip install -r requirements.txt

# Phase 1 -- Identify: build the attack taxonomy graph
python identify/build_taxonomy.py
python identify/query_taxonomy.py --stats

# Phase 2 -- Generate: simulate all 4 attack families, validate fidelity
python generate/simulate.py --all --out data/synthetic
python generate/validate_fidelity.py

# Phase 3 -- Defend: features, train, evaluate (primary + independent ULB baseline)
python defend/build_features.py
python defend/train.py
python defend/evaluate.py
python defend/train_ulb_baseline.py

# Phase 4 -- Closed loop: generate hardened variants, test cycle-1 model, retrain
python defend/export_graph_snapshot.py   # needed once before the orchestrator
python orchestrator/pipeline.py

# Phase 5 -- Web prototype
uvicorn webapp.backend.main:app --port 8000    # backend, http://localhost:8000
npm --prefix webapp/frontend install
npm --prefix webapp/frontend run dev           # frontend, http://localhost:5173
```

Run order matters: each phase reads artifacts the previous one wrote (taxonomy graph ->
synthetic attacks -> processed features -> trained model -> graph snapshot -> web API).

**Reproducing without any LLM API key**: everything above runs and every gate still
mostly passes, but Phase 4's regression-check gate reproduces the FAIL documented in
docs/PHASES.md (the mule_network_layering regression), not the later PASS — the
hand-authored fallback configs and the Gemini-generated configs aren't numerically
identical, so exact figures shift slightly. The qualitative finding (cycle-1 detector
is nearly blind to hardened card-testing; retraining recovers it) reproduces either way
— confirmed by re-running this exact fresh-clone test during Phase 7 bulletproofing.

## Datasets
- **PaySim** (primary backbone): https://www.kaggle.com/datasets/ealaxi/paysim1
- **Credit Card Fraud Detection (ULB)** (generalization holdout):
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download both into `data/raw/` (not committed — see `data/raw/README.md`).

## License / disclosure
Built for a security research / defensive-fraud competition. All simulated attacks are
synthetic data generated for training and stress-testing a defensive classifier; no real
payment data, real cardholders, or real accounts are used or targeted.
