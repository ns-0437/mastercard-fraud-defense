# CLAUDE.md — Mastercard Innovation Challenge 2026 (AI Defense Lab)

## What this is
Closed-loop red-team/blue-team system for GenAI-powered payment fraud, submitted to
Mastercard Innovation Challenge 2026 @ GFF Mumbai. Three judged pillars:
**Identify → Generate → Defend**, plus a deployed web prototype and a .docx writeup.

Competition page: https://www.kaggle.com/competitions/mastercard-innovation-challenge-2026
Live demo: https://aidefenselab-frontend-gwezm4pj4a-uc.a.run.app
Repo: https://github.com/ns-0437/mastercard-fraud-defense

## Status (updated as of the mule-network sample-size fix, not the original plan)
Submission not made yet -- deadline is 31 Aug 2026, 11:59 PM IST, and there's still
runway. Phases 0-7 done and gate-checked; see docs/PHASES.md for the full history,
including results that were superseded by later, more rigorous runs (kept in the log,
not deleted). Do not treat this file's numbers as necessarily current -- re-run
`defend/evaluate.py`, `orchestrator/pipeline.py`, and `defend/adversarial_selftest.py`
and check their fresh output before quoting a metric anywhere.

## Working agreement (the user asked for a ruthless mentor — hold to this)
- No fabricated metrics, ever. Every number that goes in the .docx must come from a
  script's actual printed output, reproducible by re-running it.
- No self-graded exams. Report the ULB independent baseline, not just PaySim.
- Call out weak spots instead of hiding them, INCLUDING when a fix makes something else
  worse. The mule_network_layering sample-size fix improved that metric's reliability
  but made the adversarial self-test result worse (1/5 caught, down from 3/5) --
  reported honestly, not smoothed over, because "we fixed X" doesn't mean "everything
  got better."
- Don't re-run an experiment to chase a more flattering number. If a result changes
  between runs, report the run, don't cherry-pick.
- Test at every phase boundary. Each phase in docs/PHASES.md has an explicit gate.
- Don't add speculative abstraction. Every line of infra must serve one of the five
  judged criteria (diversity, fidelity, detection efficacy, novelty, real-world
  feasibility) or a concrete bug found along the way.

## Architecture — the three graphs
"Graph engineering" shows up in three distinct places:

1. **Attack taxonomy graph** (`identify/taxonomy_graph.json`, built by
   `identify/build_taxonomy.py` from `identify/taxonomy_data.py`) — 33 nodes across 12
   channels, edges are shared techniques. Rendered in the web app with a per-channel
   clustered layout (a single spring_layout over all nodes was an undifferentiated
   hairball -- fixed in `webapp/frontend/src/components/TaxonomyGraph.jsx`).

2. **Transaction/entity graph** (`defend/graph_features.py`) — accounts as nodes,
   transactions as edges, built with NetworkX. Features: out/in-degree, connected-
   component size, low-activity flag. `orig_out_degree` is consistently the #2 most
   important model feature (see `defend/artifacts/feature_importance.json`) -- not
   decorative. Known limitation, found the hard way: a model's calibration is tied to
   the SPECIFIC graph it was trained against -- `detector_cycle2.json` (trained on the
   closed loop's v1+v2 combined graph) scores nonsensically when run against a
   differently-built graph snapshot, because warm-up transactions shift real accounts'
   computed degree/component values between graph builds. Always match the model to
   the graph context it was trained on; see `defend/train_anomaly_detector.py`'s
   docstring for the full incident.

3. **Pipeline DAG** (`orchestrator/pipeline.py`) — Identify → Generate → Defend →
   Feedback, runnable end-to-end with one command (`python orchestrator/pipeline.py`).

## Data
- **PaySim** (primary backbone), **ULB Credit Card Fraud** (independent methodology
  check, not a literal holdout of the same model -- ULB has no account fields at all,
  see docs/PHASES.md Phase 3 for why the original "holdout" plan was wrong).
- LLM output in Generate is behavioral *config* only (amount shape, timing, account-
  graph size) -- never raw transaction numbers. A deterministic simulator turns config
  into rows. **Real bug this caught**: the LLM once returned an absolute dollar value
  where a FRACTION of a $10,000 threshold was expected, producing a $94.9M
  "structuring" transaction. Fixed via per-family bound keys and explicit unit
  instructions in the prompt (`generate/llm_config_generator.py`), plus an independent
  semantic fidelity check (`generate/validate_fidelity.py::check_family_semantics`).
- `data/synthetic/` and `defend/artifacts/` (including the trained models) ARE
  committed to git, deliberately -- they were gitignored early on, which meant the repo
  contained only code that could regenerate SOME version of the results, not the
  specific run the docx was written from. Given LLM-driven generation is non-
  deterministic, that's a real reproducibility gap; committing the actual artifacts
  closes it.

## Stack (as actually deployed, not the original plan)
- Python 3.11, FastAPI backend, XGBoost + NetworkX + scikit-learn (IsolationForest) for
  Defend, Google Gemini (with Anthropic/OpenAI fallback) for Generate.
- React + Vite + Tailwind frontend.
- **Deployed to GCP Cloud Run** (not Vercel/Render as originally planned), project
  `mastercard-fraud-defense`, both services as Docker containers. Uses the OLDER
  `.a.run.app` URL format, not the newer `<region>.run.app` one -- some ISP DNS
  resolvers (confirmed: Reliance Jio) block the newer format specifically. See
  README.md's Status section for the full diagnosis if this ever needs re-deploying
  under a different domain.
- CORS is scoped to an explicit origin allowlist (`webapp/backend/main.py`), not `*`.

## Directory layout
```
identify/       attack taxonomy graph + builder
generate/       LLM-driven attack config generation + programmatic simulators + fidelity checks
defend/         feature engineering, XGBoost training/eval, IsolationForest comparison,
                adversarial self-test, graph snapshot export for the live API
orchestrator/   closed-loop pipeline (hardened-variant generation + retrain + regression check)
webapp/backend  FastAPI serving all of the above to the frontend, plus live single-transaction scoring
webapp/frontend React app: Overview, Attack Taxonomy, Generate & Fidelity, Live Detection, Closed-Loop Results
data/           raw (gitignored, download per data/raw/README.md) / processed (gitignored,
                regenerable) / synthetic (committed)
tests/          pytest suite -- regression tests for the actual bugs this project hit
                (units bug, label-derivation bug, graph-feature leakage), not aspirational coverage
docs/           PHASES.md (full build history + gates), PRESENTATION_SCRIPT.md (GFF pitch)
```

## See also
- [docs/PHASES.md](docs/PHASES.md) — the full phase-by-phase build history with gates,
  including results superseded by later runs (kept, not deleted).
- [SKILL.md](SKILL.md) — the actual commands to run each piece, kept in sync with the
  scripts that exist (if you add a script, update SKILL.md in the same commit).
- [docs/PRESENTATION_SCRIPT.md](docs/PRESENTATION_SCRIPT.md) — the GFF pitch script.
