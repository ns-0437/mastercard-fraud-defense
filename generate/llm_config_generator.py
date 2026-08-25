"""
Uses Claude to refine an attack's numeric parameters against the real backbone's own
statistics, grounded in the taxonomy node's real-world mechanism. Never trusts the
model's numeric output blindly: every returned value is clamped to a hard sanity range
before use, and any parse/validation failure falls back to attack_configs.DEFAULT_CONFIGS.

Requires ANTHROPIC_API_KEY in the environment. If it's absent, generate_attack_config
returns the hand-authored default immediately (logged, not silently) — the pipeline
must keep working without an API key.
"""
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from attack_configs import AttackConfig, AmountDistParams, TimingParams, GraphParams, DEFAULT_CONFIGS

# Loads ANTHROPIC_API_KEY from the repo-root .env if it's not already in the environment.
load_dotenv(Path(__file__).parent.parent / ".env")

# Hard bounds — the LLM can tune within these, never outside. Prevents a bad/hallucinated
# response from producing an unusable or unrealistic simulation (e.g. negative amounts,
# a 500-account graph that takes forever to simulate).
BOUNDS = {
    "n_instances": (10, 2000),
    "mean_log": (3.0, 12.0),
    "sigma_log": (0.05, 1.5),
    "low_frac_or_abs": (0.0, 1_000_000.0),
    "high_frac_or_abs": (0.0, 1_000_000.0),
    "burst_window_steps": (1, 200),
    "inter_event_jitter": (0.0, 1.0),
    "n_accounts": (1, 30),
    "hops": (1, 6),
    "fanout": (1, 50),
}


def _clamp(value, key):
    lo, hi = BOUNDS[key]
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric value for {key}: {value!r}")
    return max(lo, min(hi, v))


PROMPT_TEMPLATE = """You are helping parameterize a defensive fraud-simulation tool used to \
train a payment fraud detector. This is for a security research competition (red-team/blue-team \
fraud defense), not to commit fraud.

Attack family: {attack_family}
Real-world mechanism (from our research taxonomy): {mechanism}

Real backbone transaction statistics (from actual normal transaction data), for calibration:
{backbone_stats}

Propose numeric simulation parameters as a JSON object with exactly these keys:
{{
  "n_instances": <int, number of synthetic fraud cases to generate>,
  "amount_distribution": "lognormal" or "uniform",
  "mean_log": <float, only if lognormal>,
  "sigma_log": <float, only if lognormal>,
  "low": <float, only if uniform>,
  "high": <float, only if uniform>,
  "burst_window_steps": <int, how many time steps the attack pattern spans>,
  "inter_event_jitter": <float 0-1>,
  "n_accounts": <int, size of any account network involved>,
  "hops": <int, layering hops if relevant, else 1>,
  "fanout": <int, number of distinct counterparties if relevant, else 1>,
  "reasoning": "<one sentence on why these values reflect the real attack mechanism>"
}}

Ground every value in how this attack actually behaves against real payment systems and in \
the backbone statistics above. Respond with ONLY the JSON object, no other text."""


def _build_backbone_stats_summary(backbone_stats: dict) -> str:
    return json.dumps(backbone_stats, indent=2)


def generate_attack_config(taxonomy_node: dict, backbone_stats: dict, use_llm: bool = True) -> AttackConfig:
    family = taxonomy_node["attack_family"]
    default = DEFAULT_CONFIGS[family]

    if not use_llm or not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"[llm_config_generator] no ANTHROPIC_API_KEY / use_llm=False -> using default config for {family}")
        return default

    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = PROMPT_TEMPLATE.format(
            attack_family=family,
            mechanism=taxonomy_node.get("mechanism", ""),
            backbone_stats=_build_backbone_stats_summary(backbone_stats),
        )
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        raw = json.loads(text)

        amount_dist = raw.get("amount_distribution", default.amount.distribution)
        if amount_dist == "lognormal":
            amount = AmountDistParams(
                distribution="lognormal",
                mean_log=_clamp(raw.get("mean_log", default.amount.mean_log), "mean_log"),
                sigma_log=_clamp(raw.get("sigma_log", default.amount.sigma_log), "sigma_log"),
            )
        else:
            amount = AmountDistParams(
                distribution="uniform",
                low=_clamp(raw.get("low", default.amount.low), "low_frac_or_abs"),
                high=_clamp(raw.get("high", default.amount.high), "high_frac_or_abs"),
            )

        config = AttackConfig(
            attack_family=family,
            taxonomy_ref=default.taxonomy_ref,
            n_instances=int(_clamp(raw.get("n_instances", default.n_instances), "n_instances")),
            amount=amount,
            timing=TimingParams(
                burst_window_steps=int(_clamp(raw.get("burst_window_steps", default.timing.burst_window_steps), "burst_window_steps")),
                inter_event_jitter=_clamp(raw.get("inter_event_jitter", default.timing.inter_event_jitter), "inter_event_jitter"),
            ),
            graph=GraphParams(
                n_accounts=int(_clamp(raw.get("n_accounts", default.graph.n_accounts), "n_accounts")),
                hops=int(_clamp(raw.get("hops", default.graph.hops), "hops")),
                fanout=int(_clamp(raw.get("fanout", default.graph.fanout), "fanout")),
            ),
            notes=raw.get("reasoning", default.notes),
        )
        print(f"[llm_config_generator] LLM-refined config for {family}: {raw.get('reasoning', '')}")
        return config

    except Exception as e:
        print(f"[llm_config_generator] LLM call/parse failed for {family} ({e}) -> using default config")
        return default
