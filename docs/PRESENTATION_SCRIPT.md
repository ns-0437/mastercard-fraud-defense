# GFF presentation script — ~4 minutes

For live judging / the GFF Mumbai presentation slot. Written to be read almost verbatim,
timed at a natural pace. Swap in the live demo (aidefenselab-frontend URL) at the
marked points instead of describing screenshots — it's faster and more convincing live.

Every number below is pulled from the actual eval artifacts as of the last deploy
(README.md's Status section has the exact live URL and latest headline numbers) —
re-check `defend/artifacts/evaluation_report.json` and
`defend/artifacts/adversarial_selftest_report.json` before presenting if more than a
day has passed, in case the pipeline was re-run.

---

**[0:00–0:20] Open with the finding, not the architecture.**

"Most fraud-defense submissions here will show you a taxonomy, a classifier, some
metrics. We're going to show you something different: we built a detector, then we
red-teamed it against ourselves, found it was nearly blind to an attack we thought we'd
already covered, fixed it — and then found the edge of that fix too. All three of those
findings are in this submission. Nobody hides the third one."

**[0:20–0:50] Identify — show the taxonomy tab.**

"We modeled the GenAI fraud landscape as a graph, not a list — 33 attack vectors across
12 payment channels, connected by 239 shared-technique edges. [Click a node — e.g. the
vendor payment redirection one.] Every node cites a real, documented fraud pattern —
this one's grounded in FBI IC3's business email compromise loss data. The graph
structure matters: [click through 2-3 connected nodes] these three attacks share a
'deepfake_audio' technique — a defense against one should be tested against the
others, which is exactly why we built it this way instead of a flat spreadsheet."

**[0:50–1:30] Generate — show the Generate & Fidelity tab.**

"For each attack family, an LLM — Google Gemini in this run — proposes the behavioral
parameters: amount distribution, timing, account-graph shape. [Point to a reasoning
quote.] It never invents the actual transaction numbers — a deterministic simulator
does that, validated against the real 6.36-million-row PaySim backbone. We're
comfortable showing you a bug we found here: our fidelity check once caught the LLM
returning a units mismatch that would have produced a $94.9 million 'structuring'
transaction. We fixed it and added a permanent guard against it happening again. That's
not a footnote — it's the validation pipeline actually working."

**[1:30–2:15] Defend — show Live Detection, then Closed-Loop Results.**

"The detector combines tabular features with account-graph features — degree,
component size — computed from the transaction network itself. [Run a live detection
on a sample.] On its own generated attacks, this scores a 0.995 PR-AUC. That number
alone would be the whole pitch for most teams. We don't think it should be, because a
classifier that's only ever tested against attacks shaped like its own training data
isn't proving much. So we ran two harder tests."

**[2:15–3:00] The closed loop — the actual differentiator.**

"[Switch to Closed-Loop Results tab, scroll to Step A.] We generated hardened variants
of the same four attacks — spread over time, lower volume — and scored them with the
existing detector. We ran this experiment three separate times across our build
history. Two of three times, the detector was nearly blind to a hardened card-testing
pattern — under half a percent recall. We retrained on the harder attacks and recovered
full detection with zero regression on the originals. We're showing you all three runs,
not just the most dramatic one, because a closed loop that only worked once isn't a
mechanism, it's a coincidence."

**[3:00–3:40] The honest limit.**

"[Scroll to the adversarial self-test section.] Then we asked a harder question: does
this generalize to attacks that don't just vary a known pattern, but are structurally
different? Five hand-crafted transactions, built to look numerically unremarkable. The
detector caught three. Two evaded it — including one where an attacker made a
compromised account's activity look like routine bill payments, and the model assigned
it a 0% fraud probability. We're telling you that on stage, in front of judges, because
a fraud team that doesn't know where its own model breaks is more dangerous than one
that does."

**[3:40–4:00] Close.**

"Everything you just saw is live — [gesture at the deployed URL] — not slides. The code,
the .docx, and the running system are all linked in our submission. The differentiator
isn't a metric. It's that we built a system that finds its own mistakes, and we're
showing you the ones it hasn't fixed yet, not just the ones it has."

---

## If judges ask (anticipated Q&A)

**"Why does the closed-loop result change between runs?"**
Attack generation is LLM-driven, so it's not deterministic — 2 of 3 runs found a severe
gap in card-testing, the third found a milder one in mule-network layering instead. We
report all three rather than cherry-picking the most dramatic, because a system that
reliably finds *some* real gap every time is a stronger claim than a system that found
exactly one specific vulnerability once.

**"Why is your PaySim recall so much higher than your ULB baseline?"**
They're not directly comparable — ULB has no account graph at all, so that model is
tabular-only against completely different fraud (general card-present/CNP fraud, not
GenAI-specific attacks). We report it as a methodology check, not an apples-to-apples
benchmark.

**"What would you need to actually deploy this?"**
A live graph/feature store instead of a precomputed snapshot (Section 4 of the
walkthrough covers this in detail), a threshold calibrated against real cost-of-fraud
vs. cost-of-friction data, and — given the adversarial self-test — continuous
retraining against novel attack shapes, not a one-time model.

**"Isn't a 0% fraud probability on an evasion case a serious flaw?"**
Yes, and we're not going to pretend otherwise. It's evidence this specific detector,
scored transaction-by-transaction, has no signal against an attacker who makes
compromised-account activity look completely routine — that would need account-level
behavioral baselining, which is out of scope for what we built but explicitly named as
a next step.
