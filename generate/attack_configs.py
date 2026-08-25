"""
Attack config schema + hand-authored fallback defaults for the four attack families
selected for numeric simulation in Phase 2 (see docs/PHASES.md).

Why only 4 of the 21 taxonomy nodes get numeric simulation: most of the taxonomy
(voice cloning, deepfake video, phishing text, prompt injection) produces its damage
through a *channel* PaySim/ULB don't model (a phone call, a video stream, a chat
conversation) — the attack's *outcome* is what shows up in transaction data. These 4
were chosen because they have a distinct, well-documented transaction-level signature
that is NOT the same as PaySim's native TRANSFER->CASH_OUT fraud pattern, which is the
point: if we only regenerated PaySim's existing fraud signature we'd be proving nothing
about GenAI-era fraud specifically.

  ato_rapid_drain        <- ato_llm_support_pretexting / ato_deepfake_voice_ivr
                            (account takeover: staged drain, less flat than PaySim's
                            native single-shot pattern because an LLM-guided operator
                            paces withdrawals to look organic)
  structuring_smurfing   <- mule_adaptive_layering_vs_aml
                            (many transfers just under a reporting/velocity threshold)
  mule_network_layering  <- mule_synthetic_network_generation
                            (funds hop across a small synthetic account graph before
                            reaching a cash-out sink — this is what defend/graph_features.py
                            is built to catch)
  card_testing_burst     <- cnp_ai_card_testing
                            (rapid very-small payments from one origin to many distinct
                            destinations in a short window)

The LLM's job (see llm_config_generator.py) is to propose *parameters* within the
bounds below, grounded in the attack's real-world logic and the real backbone's own
statistics — never to invent transaction numbers directly.
"""
from dataclasses import dataclass, field


@dataclass
class AmountDistParams:
    distribution: str = "lognormal"   # lognormal | uniform
    low: float = 0.0
    high: float = 0.0
    mean_log: float = 0.0
    sigma_log: float = 0.3


@dataclass
class TimingParams:
    burst_window_steps: int = 1        # how many PaySim "step" units the attack spans
    inter_event_jitter: float = 0.2    # fractional jitter on spacing between events


@dataclass
class GraphParams:
    n_accounts: int = 1
    hops: int = 1
    fanout: int = 1


@dataclass
class AttackConfig:
    attack_family: str
    taxonomy_ref: str
    n_instances: int
    amount: AmountDistParams
    timing: TimingParams
    graph: GraphParams
    notes: str = ""
    # Phase 4 (closed-loop hardening) lever: if True, each synthetic account gets one
    # small transaction with a real backbone account before the attack pattern, so it
    # doesn't present as a brand-new isolated node — directly targets the
    # orig_component_size / orig_is_low_activity graph features the detector leans on.
    warm_up: bool = False
    # Which provider actually produced this config -- "hand-authored-fallback" if no
    # LLM call succeeded. Surfaced in the web app so the "Generate" pillar's LLM-driven
    # claim is verifiable, not just asserted.
    provider: str = "hand-authored-fallback"


# Hand-authored, domain-grounded fallback configs. Used when no ANTHROPIC_API_KEY is
# set, or when the LLM call fails/returns out-of-bounds values — the pipeline must be
# runnable end-to-end without an API key; the LLM only refines these, it isn't load
# bearing for the pipeline to function at all.
DEFAULT_CONFIGS: dict[str, AttackConfig] = {
    "ato_rapid_drain": AttackConfig(
        attack_family="ato_rapid_drain",
        taxonomy_ref="ato_llm_support_pretexting",
        n_instances=400,
        amount=AmountDistParams(distribution="lognormal", mean_log=9.5, sigma_log=0.6),
        timing=TimingParams(burst_window_steps=3, inter_event_jitter=0.3),
        graph=GraphParams(n_accounts=1, hops=1, fanout=1),
        notes="2-3 staged transfers draining ~85-98% of account balance within a few "
              "hours of a support-channel contact event, amounts non-round to avoid "
              "naive round-number heuristics.",
    ),
    "structuring_smurfing": AttackConfig(
        attack_family="structuring_smurfing",
        taxonomy_ref="mule_adaptive_layering_vs_aml",
        n_instances=600,
        # Widened from 0.70-0.97 after the adversarial self-test found a ~40%-of-
        # threshold variant evaded detection entirely -- structuring at a more
        # cautious fraction (sacrificing per-transaction efficiency for lower
        # suspicion) is a real operator choice, not an edge case. Training only on
        # the "textbook" near-threshold shape taught that specific range, not
        # "structuring" as a general behavior.
        amount=AmountDistParams(distribution="uniform", low=0.30, high=0.97),  # fraction of threshold
        timing=TimingParams(burst_window_steps=48, inter_event_jitter=0.5),
        graph=GraphParams(n_accounts=1, hops=1, fanout=3),
        notes="Many transfers at 30-97% of a $10,000-equivalent reporting threshold, "
              "spread over ~48 steps to avoid naive velocity windows.",
    ),
    "mule_network_layering": AttackConfig(
        attack_family="mule_network_layering",
        taxonomy_ref="mule_synthetic_network_generation",
        n_instances=320,
        amount=AmountDistParams(distribution="lognormal", mean_log=8.5, sigma_log=0.4),
        timing=TimingParams(burst_window_steps=12, inter_event_jitter=0.4),
        graph=GraphParams(n_accounts=8, hops=3, fanout=2),
        notes="8-account synthetic layering network, funds hop 3 times before reaching "
              "a sink account, amount shrinks slightly per hop (simulated fee/skim).",
    ),
    "card_testing_burst": AttackConfig(
        attack_family="card_testing_burst",
        taxonomy_ref="cnp_ai_card_testing",
        n_instances=800,
        amount=AmountDistParams(distribution="uniform", low=1.0, high=5.0),
        timing=TimingParams(burst_window_steps=1, inter_event_jitter=0.1),
        graph=GraphParams(n_accounts=1, hops=1, fanout=20),
        notes="One origin probing 20+ distinct destinations with $1-5 payments within "
              "a single time step, adaptive-looking but not literally adaptive in v1.",
    ),
    "ato_lowandslow_exfiltration": AttackConfig(
        # Added after the adversarial self-test found this exact shape (recurring
        # similar-amount payments, no drain/burst signature) evaded a detector trained
        # only on ato_rapid_drain's fast-drain and structuring_smurfing's near-
        # threshold shapes. Same real-world vector as ato_rapid_drain (account
        # takeover), a genuinely different exfiltration pattern -- see
        # simulate_ato_lowandslow's docstring.
        attack_family="ato_lowandslow_exfiltration",
        taxonomy_ref="ato_llm_support_pretexting",
        n_instances=300,
        amount=AmountDistParams(distribution="lognormal", mean_log=7.5, sigma_log=0.6),
        timing=TimingParams(burst_window_steps=200, inter_event_jitter=0.3),
        graph=GraphParams(n_accounts=1, hops=1, fanout=1),
        notes="4-7 similar-amount payments to one destination, spaced ~15-40 steps "
              "apart, mimicking legitimate recurring bill-pay instead of draining the "
              "account quickly.",
    ),
}
