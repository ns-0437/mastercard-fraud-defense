---
name: mastercard-fraud-defense
description: Operate the Mastercard Innovation Challenge closed-loop fraud system — build/query the attack taxonomy graph, run attack generation, train and evaluate the detector with graph features, run the closed-loop pipeline, the adversarial self-test, the anomaly-detector comparison, and the pytest suite. Use when working anywhere in this repo.
---

# Mastercard Fraud Defense — operating skill

This repo builds a closed-loop Identify → Generate → Defend system. See
[CLAUDE.md](CLAUDE.md) for architecture and constraints, [docs/PHASES.md](docs/PHASES.md)
for the full build history. This file is the "how do I run X" reference, checked
against the actual scripts and their actual CLI args as of the mule-network
sample-size fix — keep it in sync as scripts change; a command listed here that
doesn't match reality is a bug in this file, fix it in the same commit that changes
the script.

Run order matters: each step reads artifacts the previous one wrote. Full order:
`identify/build_taxonomy.py` → `generate/simulate.py` → `generate/validate_fidelity.py`
→ `defend/build_features.py` → `defend/train.py` → `defend/evaluate.py` →
`defend/train_ulb_baseline.py` (independent, no dependency on the above) →
`defend/export_graph_snapshot.py` → `orchestrator/pipeline.py` →
`defend/adversarial_selftest.py` → `defend/train_anomaly_detector.py` → webapp.

## Core commands

### Identify — attack taxonomy graph
```bash
python identify/build_taxonomy.py          # (re)generates identify/taxonomy_graph.json
python identify/query_taxonomy.py --stats  # prints node/edge counts, coverage by channel, gate PASS/FAIL
```
Gate (enforced by query_taxonomy.py): >=15 nodes, every node has a non-empty
`grounding` field, >=60% of nodes have at least one shared-technique edge.

### Generate — attack simulation
```bash
python generate/simulate.py --all --out ../data/synthetic     # all 4 families, LLM-driven if a key is set
python generate/simulate.py --all --no-llm --out ../data/synthetic   # deterministic fallback, no API calls
python generate/validate_fidelity.py       # validates whatever's currently in data/synthetic/
```
Requires `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` in `.env` (repo
root) for the LLM path — tries providers in that order, falls back to hand-authored
defaults if none succeed. Gate (enforced by validate_fidelity.py): schema, sanity,
balance conservation, per-type value envelope, and family-specific semantic checks
(e.g. structuring must stay under the $10K threshold) must all pass for at least 4
families. **Non-determinism note**: re-running with a live LLM key produces different
numeric results each time — this is expected and part of the honest story documented
in docs/PHASES.md, not a bug to chase away.

### Defend — detector training and evaluation
```bash
python defend/build_features.py            # tabular + graph features -> data/processed/{train,test}.csv
python defend/train.py                     # trains XGBoost -> defend/artifacts/detector.json
python defend/evaluate.py                  # precision/recall/F1/PR-AUC + per-family recall + gate
python defend/train_ulb_baseline.py        # independent methodology check on ULB (run once, no dependency on the above)
```
Gate (evaluate.py): PR-AUC > 0.5, every attack family has recall > 0, FPR on legit
< 5%. `evaluate.py` also prints an automatic caveat if PR-AUC > 0.99 — a near-perfect
score is expected to be partly an easy-benchmark effect, don't quote it without that
context (Phase 4's closed loop is the real test).

### Orchestrator — the closed loop
```bash
python defend/export_graph_snapshot.py     # must run before the orchestrator -- builds the live-API graph snapshot
python orchestrator/pipeline.py            # no args; generates hardened (v2) attacks, retrains, checks regression
```
Gate: cycle-2 (retrained) recall on the *original* attack families must not regress
vs. cycle-1. **This experiment is non-deterministic and has been run 4 times across
this project's history with genuinely different outcomes (2 found a severe blind
spot, 2 did not)** — report the actual run's result, never re-run to chase a more
dramatic or more flattering number. See docs/PHASES.md Phase 4 for the full history.

### Adversarial self-test and second detection model
```bash
python defend/adversarial_selftest.py      # 5 hand-crafted evasion attempts against detector.json
python defend/train_anomaly_detector.py    # IsolationForest comparison + adversarial-case check
```
No fixed "gate" — the point is to report whatever happens, including if it's bad.
`train_anomaly_detector.py` deliberately scores against `detector.json` (cycle-1),
NOT `detector_cycle2.json` — the latter is calibrated to a different graph context
(the closed loop's v1+v2 combined graph) and will silently produce nonsensical numbers
if scored against this test set. See the module docstring for the incident this
caused.

### Tests
```bash
python -m pytest tests/ -v                 # 27 tests as of the last run, all passing
```
Regression tests for actual bugs this project hit (the $94.9M structuring units bug,
the warm-up-row mislabeling bug, graph-feature account-ID leakage) — not aspirational
coverage. Run this before trusting any change to `generate/`, `defend/graph_features.py`,
or `defend/build_features.py`.

### Webapp
```bash
uvicorn webapp.backend.main:app --reload   # backend, http://localhost:8000
npm --prefix webapp/frontend run dev       # frontend, http://localhost:5173 (proxies /api to :8000 in dev)
```
Production build needs `VITE_API_BASE_URL` set at build time (Vite env vars are
compile-time) — see `Dockerfile.frontend`'s `--build-arg`. CORS on the backend is an
explicit origin allowlist (`webapp/backend/main.py::ALLOWED_ORIGINS`) — add a new
frontend origin there before it can call the API from a browser.

### Deploying (GCP Cloud Run, project `mastercard-fraud-defense`)
```bash
docker build -f Dockerfile.backend -t us-central1-docker.pkg.dev/mastercard-fraud-defense/mastercard-fraud-defense/backend:vN .
docker push us-central1-docker.pkg.dev/mastercard-fraud-defense/mastercard-fraud-defense/backend:vN
gcloud run deploy aidefenselab-backend --image=...backend:vN --region=us-central1 --allow-unauthenticated --memory=1Gi --port=8080 --min-instances=1
# same pattern for frontend, with --build-arg VITE_API_BASE_URL=<backend's .a.run.app URL>
```
**Use the `.a.run.app` URL format, not `<region>.run.app`** — some ISP DNS resolvers
(confirmed: Reliance Jio) block the newer format. `gcloud run services describe
<service> --region=us-central1 --format="json(status.url)"` returns the working
`.a.run.app` URL for an existing service. If `gcloud config get-value project` doesn't
say `mastercard-fraud-defense`, `gcloud config set project mastercard-fraud-defense`
first — it can drift to another project on this machine between sessions.

### Pre-submission bulletproofing
```bash
git clone <repo> /tmp/fresh-clone && cd /tmp/fresh-clone
# copy data/raw/*.csv in per data/raw/README.md, then run every command above in order
```
Done twice already (see docs/PHASES.md Phase 7) — both times worked end to end. Worth
knowing: a fresh clone with no `.env`/API key reproduces the FALLBACK-generated
results, which may differ numerically from whatever's currently documented since the
LLM path is non-deterministic.

## Working rules for whoever (human or Claude) touches this repo
- `data/raw/*.csv` stays gitignored (large, license-restricted) — download per
  `data/raw/README.md`. `data/synthetic/` and `defend/artifacts/` are DELIBERATELY
  committed (not gitignored) — they're the actual output of a specific run, not just
  regenerable-in-principle code. If you regenerate them, commit the new versions.
- Never put a metric in `docs/` or the .docx source that isn't the literal stdout of a
  script in this repo, from a run you can point to.
- Never re-run an experiment specifically to get a more flattering number, and never
  quote only the best of several runs — report the run, including which one and when.
- Every new script gets a corresponding entry added to this file in the same commit.
