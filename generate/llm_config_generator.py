"""
Uses an LLM to refine an attack's numeric parameters against the real backbone's own
statistics, grounded in the taxonomy node's real-world mechanism. Never trusts the
model's numeric output blindly: every returned value is clamped to a hard sanity range
before use, and any parse/validation failure falls back to attack_configs.DEFAULT_CONFIGS.

Tries providers in this order, using whichever has a configured API key AND actually
succeeds: Anthropic (ANTHROPIC_API_KEY) -> Gemini (GEMINI_API_KEY) -> OpenAI
(OPENAI_API_KEY) -> hand-authored default. This project's own money is on Anthropic;
Gemini/OpenAI support exists so the Generate pillar keeps working on whichever
legitimate key is actually funded, without the pipeline ever depending on one specific
vendor being available. If none are configured or all fail, generate_attack_config
returns the hand-authored default (logged, not silently) — the pipeline must keep
working without any API key at all.
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


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5", max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_gemini(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return response.text


def _call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


PROVIDERS = [
    ("ANTHROPIC_API_KEY", "anthropic", _call_anthropic),
    ("GEMINI_API_KEY", "gemini", _call_gemini),
    ("OPENAI_API_KEY", "openai", _call_openai),
]


def _get_llm_response(prompt: str) -> tuple[str, str] | tuple[None, None]:
    """Tries each configured provider in order; returns (raw_text, provider_name) from
    the first one that succeeds, or (None, None) if none are configured or all fail."""
    for env_var, name, call_fn in PROVIDERS:
        if not os.environ.get(env_var):
            continue
        try:
            text = call_fn(prompt)
            return text, name
        except Exception as e:
            print(f"[llm_config_generator] {name} call failed ({e}) -> trying next provider")
    return None, None


def generate_attack_config(taxonomy_node: dict, backbone_stats: dict, use_llm: bool = True) -> AttackConfig:
    family = taxonomy_node["attack_family"]
    default = DEFAULT_CONFIGS[family]

    if not use_llm:
        print(f"[llm_config_generator] use_llm=False -> using default config for {family}")
        return default

    prompt = PROMPT_TEMPLATE.format(
        attack_family=family,
        mechanism=taxonomy_node.get("mechanism", ""),
        backbone_stats=_build_backbone_stats_summary(backbone_stats),
    )
    text, provider = _get_llm_response(prompt)
    if text is None:
        print(f"[llm_config_generator] no configured provider succeeded -> using default config for {family}")
        return default

    try:
        raw = json.loads(_strip_code_fence(text))

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
        print(f"[llm_config_generator] {provider}-refined config for {family}: {raw.get('reasoning', '')}")
        return config

    except Exception as e:
        print(f"[llm_config_generator] {provider} response failed to parse for {family} ({e}) -> using default config")
        return default
