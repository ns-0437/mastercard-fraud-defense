# Build plan — 8 phases, 6 days, solo

Deadline: Aug 31 2026, 11:59 PM IST. Internal target: submitted by evening Aug 30.
Each phase has a **gate** — do not start the next phase until it passes. If you're
behind schedule, cut scope *within* a phase (fewer attack families, simpler model);
do not skip gates.

| Phase | Day | Deliverable | Judged criterion it serves |
|---|---|---|---|
| 0 | Aug 25 AM | Repo scaffold, decisions locked | (enables everything) |
| 1 | Aug 25 | Attack taxonomy graph | Diversity |
| 2 | Aug 26 | Attack generator + fidelity validation | Fidelity |
| 3 | Aug 27 | Detector + graph features + generalization test | Detection efficacy |
| 4 | Aug 28 | Closed-loop orchestrator (feedback) | Novelty, real-world feasibility |
| 5 | Aug 29 | Web prototype, deployed | Real-world feasibility (demo) |
| 6 | Aug 30 AM | .docx walkthrough | Written evaluation |
| 7 | Aug 30 PM | Bulletproofing pass | (protects everything above) |
| 8 | Aug 31 | Submit with buffer | — |

---

## Phase 0 — Scaffold & decisions (done same day as kickoff)
- Repo structure, CLAUDE.md, SKILL.md (this file's siblings) — done.
- Datasets picked: PaySim (primary), ULB Credit Card Fraud (holdout). Download both now,
  put real download instructions in `data/raw/README.md`.
- **Gate**: `git log` shows an initial commit with scaffold + a README that a stranger
  could read and know what this project is and how to run it (even if scripts don't
  exist yet — README describes the plan).

## Phase 1 — Identify: attack taxonomy graph
- Enumerate GenAI-powered payment fraud attack vectors across: card-not-present /
  e-commerce, account takeover, KYC/onboarding, agentic commerce (AI shopping/payment
  agents), authorized-push-payment social engineering, mule networks, and adversarial
  attacks on ML defenses themselves. Target 15-25 distinct, real-grounded vectors — not
  padding with rephrased duplicates.
- Each node gets: name, channel, mechanism description, real-world grounding
  (what makes this plausible given how payment rails/KYC/support flows actually work),
  and a rough severity/likelihood estimate.
- Build edges for shared techniques (e.g. multiple attacks share "voice cloning" or
  "LLM-written pretext") — this is what makes it a graph, not a list.
- **Gate**: `query_taxonomy.py --stats` runs, taxonomy has 15+ nodes each with a
  non-empty `grounding` field, graph is connected enough that at least 60% of nodes
  share an edge with another node (proves it's a real graph, not disconnected trivia).

## Phase 2 — Generate: simulate attacks at scale
- Load PaySim as the "normal + baseline fraud" backbone.
- For each taxonomy node selected for simulation, use the LLM to produce an **attack
  config** (behavioral parameters: transaction velocity, amount distribution shift,
  timing pattern, account-graph structure for mule rings) and narrative artifacts
  (e.g. synthetic phishing pretext text, fake KYC doc text) — never raw transaction
  numbers directly from the LLM.
- A programmatic simulator consumes that config to inject realistic synthetic
  transactions/accounts into the backbone, preserving PaySim's real statistical shape
  elsewhere.
- Build `validate_fidelity.py`: KS-test per numeric feature (synthetic-fraud vs.
  real-fraud-in-PaySim), categorical distribution comparison, and a manual spot-check
  writeup of 3 generated cases.
- **Gate**: fidelity validation passes for at least 4 distinct attack families; failures
  are fixed by adjusting the simulator, not by lowering the bar in the validator.

## Phase 3 — Defend: detection model
- Feature engineering: tabular (amount, velocity, time-of-day, account age) + graph
  features from `defend/graph_features.py` (shared-device/IP degree, community
  detection output, account-degree anomaly score).
- Train XGBoost/LightGBM. Time-based train/test split (no shuffling across time —
  leakage kills credibility with judges who know fraud ML).
- Report precision/recall/F1/PR-AUC on synthetic-attack test set.
- **Correction made after inspecting both real files (Aug 25)**: ULB Credit Card Fraud
  has no account/counterparty fields at all — just PCA-anonymized `V1`-`V28` + `Amount`
  + `Time`. There is no graph to build and no way to literally run the PaySim-trained
  model on it. The original plan below ("run evaluate.py --holdout ulb_creditcard" on
  the same model) was wrong and would have produced a meaningless number dressed up as
  a generalization test. The honest version: train a SEPARATE, methodologically
  comparable XGBoost model on ULB (same time-based split discipline, same imbalance
  handling) and report both results side by side as "same approach validated
  independently on a genuinely different real fraud dataset" — this proves the
  *methodology* isn't a PaySim-specific trick, not that one model transfers across
  domains. Say this distinction explicitly in the docx; don't blur it.
- Run `train_ulb_baseline.py` **once** on ULB, report its metrics even if worse than
  the PaySim model's — that's expected (different fraud type, no graph signal
  available) and is a scoped limitation to state plainly, not a hidden flaw.
- **Gate**: PR-AUC reported (not just accuracy — accuracy is meaningless on imbalanced
  fraud data and using it alone signals to judges you don't know that). False positive
  rate on legitimate transactions explicitly stated.

## Phase 4 — Closed loop: orchestrator + feedback
- `orchestrator/pipeline.py` wires Identify → Generate → Defend as a real DAG (not a
  shell script calling three files in sequence with no shared state).
- Feedback: after Defend runs, missed-detection cases are grouped by attack family and
  routed back to Generate with an instruction to produce harder variants (e.g. lower
  velocity, more realistic timing) of that specific family. Re-inject, re-evaluate.
- **Gate**: run 2 full cycles. Metrics on the *original* attack set must not regress in
  cycle 2 (see SKILL.md gate). This is the single most important gate in the whole
  project — it's the actual proof of the "closed loop," not a diagram.

## Phase 5 — Web prototype
- FastAPI backend exposing: taxonomy graph (for viz), a "run detection on this
  transaction/batch" endpoint, and pipeline-cycle history/metrics.
- React frontend: taxonomy graph visualization, a live detection demo (paste/generate a
  transaction, see the model's verdict + which graph features drove it), and a view of
  the feedback loop's metric history across cycles.
- Deploy: frontend to Vercel, backend to Render/Railway. Must be reachable by a judge
  clicking a link — a `localhost` screenshot does not satisfy "working web-based
  prototype."
- **Gate**: deployed URL loads cold (not just from your warmed dev cache), the live
  detection demo produces a real model verdict end-to-end, no console errors.

## Phase 6 — Solution walkthrough (.docx)
- Sections: attack landscape (pull straight from the taxonomy graph, don't
  re-write it from memory), generation approach + fidelity results, detection approach +
  efficacy results (both synthetic and ULB holdout numbers), real-world feasibility
  discussion (deployment latency, false-positive cost tradeoffs, what would need to
  change to run this against live rails), limitations stated honestly.
- Every number pasted in must be copy-pasted from an actual script run that day —
  re-run everything fresh before writing this, don't reuse numbers from Phase 3's dev
  runs if the model changed since.
- **Gate**: every metric in the doc has a corresponding script + timestamp you can point
  to. If you can't say which command produced a number, delete the number and re-run.

## Phase 7 — Bulletproofing pass (this is the "test until it's bulletproof" phase)
- Fresh clone, fresh environment, follow only the README — does it run end to end?
- Try to break your own detector: hand-craft 5 adversarial transactions designed to
  evade it (e.g. split a large transfer into many small ones just under your velocity
  threshold) — report whether it catches them. If it doesn't, that's a real limitation
  to disclose, not to hide — judges respect a red-teamer who red-teams their own defense.
- Check all three submission artifacts exist and satisfy the literal checklist: repo
  (runnable, documented), .docx (all four required sections present), deployed web
  prototype (link works, cold-loads).
- Skim the taxonomy and docx once more for anything that reads as filler/padding rather
  than grounded — cut it, diversity is judged on real distinctness, not node count.
- **Gate**: a person who has never seen this project could clone the repo, read the
  README, and reproduce your headline metric within 20 minutes.

## Phase 8 — Submit
- Submit from the Kaggle Writeups section with all three artifacts linked, at least
  several hours before the literal deadline — not at 11:58 PM.
- Keep the deployed backend/frontend alive through Sep 11 (GFF dates) in case judges or
  the presentation slot need the live link again.
