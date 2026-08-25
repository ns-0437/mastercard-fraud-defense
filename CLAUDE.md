# CLAUDE.md — Mastercard Innovation Challenge 2026 (AI Defense Lab)

## What this is
Closed-loop red-team/blue-team system for GenAI-powered payment fraud, submitted to
Mastercard Innovation Challenge 2026 @ GFF Mumbai. Three judged pillars:
**Identify → Generate → Defend**, plus a deployed web prototype and a .docx writeup.

Competition page: https://www.kaggle.com/competitions/mastercard-innovation-challenge-2026

## Hard constraints — do not forget these
- **Deadline: 31 Aug 2026, 11:59 PM GMT+5:30.** Today's date is tracked in the session
  context. Target internal submission (not literal deadline) is the evening of Aug 30 —
  the last day is buffer for Kaggle upload issues, not for finishing features.
- **Solo builder.** No parallelization across people. Every phase must be sized for one
  person working alone. If a phase can't be finished by one person in its allotted day,
  cut scope, don't extend the timeline.
- **Registered already** — submission mechanics (writeups section) are the only Kaggle
  step left, don't re-verify registration.
- Judged criteria, verbatim from the brief: diversity of attacks identified, fidelity of
  simulated attacks, detection efficacy, novelty, real-world feasibility in live payments.
  Every phase gate below maps back to one of these five — if a piece of work doesn't move
  one of them, it's scope creep, cut it.

## Working agreement (the user asked for a ruthless mentor — hold to this)
- No fabricated metrics, ever. Every number that goes in the .docx must come from a
  script's actual printed output, reproducible by re-running it. If a number can't be
  regenerated on demand, it doesn't go in the writeup.
- No self-graded exams. The detector (Defend) must never be evaluated *only* on the
  fraud its own generator produced. It must also be tested on the ULB Credit Card Fraud
  dataset (a distribution it was never tuned on) as a generalization check. Report both
  numbers, not just the flattering one.
- Call out weak spots instead of hiding them. If fidelity validation shows the generated
  data is unrealistic, say so in the docx and explain the mitigation — judges trust an
  honest limitations section more than a suspiciously perfect one.
- Test at every phase boundary, not just at the end. Each phase below has an explicit
  gate. Do not start the next phase until the current gate passes.
- Don't add speculative abstraction (plugin systems, config DSLs, multi-model
  orchestration frameworks) — this is a 6-day solo build. Every line of infra must
  directly serve one of the five judged criteria.

## Architecture — the three graphs
"Graph engineering" is a real requirement here, used in three distinct places. Don't
let any of these become decorative:

1. **Attack taxonomy graph** (`identify/taxonomy_graph.json` + builder script) — nodes
   are individual GenAI-powered fraud attack vectors, edges are shared techniques /
   precursor relationships (e.g. "deepfake voice auth" and "synthetic KYC" share the
   "identity spoofing" precursor). Used to prove *diversity* and to derive the attack
   configs that Generate consumes — this is not a static list, it's queried
   programmatically by the generator to pick attack families and blend techniques.

2. **Transaction/entity graph** (`defend/graph_features.py`) — accounts, devices, and
   IPs as nodes, transactions as edges, built with NetworkX from PaySim's sender/
   receiver structure. Features: shared-device/IP degree, community detection (mule
   ring candidates), account age vs. degree anomalies. These feed the classifier
   alongside tabular features — this is the main lever for detection *novelty* over a
   plain XGBoost-on-tabular-features baseline.

3. **Pipeline DAG** (`orchestrator/pipeline.py`) — Identify → Generate → Defend →
   Feedback modeled as an explicit directed graph of stages, not a linear script. The
   feedback edge is real: cases the detector misses get routed back to Generate to
   produce harder adversarial variants of that attack family, which get re-injected and
   re-evaluated. This IS the "closed loop" the brief asks for — it must be runnable
   end-to-end with one command, not just described in prose.

## Data
- **PaySim** (primary backbone) — mobile-money transfers, has `nameOrig`/`nameDest`
  account IDs → the only reason it's usable for graph construction. Subsample to a
  workable size (~1-2M rows) rather than loading all 6.3M into memory-constrained dev.
- **ULB Credit Card Fraud (Kaggle)** (secondary, holdout) — real, PCA-anonymized,
  never touched during training or threshold tuning. Used exactly once, at the end of
  Phase 3, as a generalization check. Touching it earlier invalidates the check.
- Do not train or validate on LLM-hallucinated transaction *numbers*. LLM output in
  Generate is for attack narratives, KYC document text, and behavioral *config*
  (velocity, amount distribution parameters, timing patterns) that a programmatic
  simulator then uses to produce numeric transaction data. Numbers come from code, not
  from the model's imagination — this is the difference between fidelity and fiction.

## Stack decisions (locked, don't re-litigate mid-build)
- Python 3.11, FastAPI backend, XGBoost/LightGBM + NetworkX for Defend.
- Claude API (user has budget) for Generate's narrative/config layer.
- React + Vite + Tailwind frontend, deployed to Vercel; backend on Render/Railway.
- SQLite for demo persistence — no need for anything heavier at this scale.

## Directory layout
```
identify/       attack taxonomy graph + builder + research notes
generate/       LLM-driven attack config generation + programmatic transaction simulator
defend/         feature engineering (incl. graph features), model training, evaluation
orchestrator/   pipeline DAG wiring the closed loop + feedback logic
webapp/backend  FastAPI serving pipeline results to the frontend
webapp/frontend React dashboard: taxonomy graph viz, live detection demo, feedback loop viz
data/           raw/processed/synthetic — raw real datasets, processed features, generated attacks
notebooks/      throwaway exploration only, nothing load-bearing lives here
tests/          fidelity validation, detector eval, end-to-end pipeline tests
docs/           PHASES.md, the .docx source material, research citations
```

## See also
- [PHASES.md](docs/PHASES.md) — the 8-phase build plan with gates and deliverables.
- [SKILL.md](SKILL.md) — repeatable commands for operating this repo.
