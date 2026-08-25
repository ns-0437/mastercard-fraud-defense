# AI Defense Lab for Payment Security

Submission for the Mastercard Innovation Challenge 2026 @ GFF Mumbai. A closed-loop
red-team/blue-team AI system for GenAI-powered payment fraud: **Identify** emerging
attack vectors as a knowledge graph, **Generate** high-fidelity simulated attacks at
scale, **Defend** with a graph-feature-augmented classifier, wired into a feedback loop
so detector blind spots drive the next round of generation.

See [CLAUDE.md](CLAUDE.md) for architecture/constraints, [docs/PHASES.md](docs/PHASES.md)
for the build plan, [SKILL.md](SKILL.md) for how to run each piece.

## Status
Build in progress — see docs/PHASES.md for current phase. This section gets updated
as phases complete; treat it as the single source of truth for "does X actually work
yet," not the phase plan (which is the plan, not the log).

## Quickstart
_(filled in as each component lands — Phase 0 scaffold only so far)_

## Datasets
- **PaySim** (primary backbone): https://www.kaggle.com/datasets/ealaxi/paysim1
- **Credit Card Fraud Detection (ULB)** (generalization holdout):
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download both into `data/raw/` (not committed — see `data/raw/README.md`).

## License / disclosure
Built for a security research / defensive-fraud competition. All simulated attacks are
synthetic data generated for training and stress-testing a defensive classifier; no real
payment data, real cardholders, or real accounts are used or targeted.
