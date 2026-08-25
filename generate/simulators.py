"""
Programmatic transaction simulators — one per attack family. Each takes an AttackConfig
(numeric parameters, from attack_configs.py or LLM-refined via llm_config_generator.py)
plus the real backbone DataFrame (for realistic step-range/balance sampling) and returns
synthetic rows in PaySim's own schema, so they can be concatenated straight into the
backbone for training/evaluation.

This is deliberately all deterministic numpy/pandas code — no LLM calls happen here.
The LLM's contribution stopped at picking the AttackConfig parameters; turning those
parameters into actual transaction rows is ordinary simulation code, which is what
makes the output numerically trustworthy rather than a model's guess at what a number
should look like.

PaySim schema: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from attack_configs import AttackConfig

PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def _draw_amounts(config: AttackConfig, n: int, rng: np.random.Generator, scale_ref: float = 1.0) -> np.ndarray:
    a = config.amount
    if a.distribution == "lognormal":
        return rng.lognormal(mean=a.mean_log, sigma=a.sigma_log, size=n)
    # uniform: for structuring, low/high are FRACTIONS of scale_ref (e.g. a threshold);
    # for card testing, low/high are absolute dollar bounds already.
    return rng.uniform(a.low, a.high, size=n) * (scale_ref if scale_ref != 1.0 else 1.0)


def _new_account_id(prefix: str, i: int) -> str:
    return f"SYNTH_{prefix}_{i:07d}"


def _sample_steps(backbone: pd.DataFrame, n: int, window: int, rng: np.random.Generator) -> np.ndarray:
    max_step = int(backbone["step"].max()) if "step" in backbone.columns and len(backbone) else 743
    starts = rng.integers(1, max(2, max_step - window), size=n)
    jitter = rng.integers(0, max(1, window), size=n)
    return np.clip(starts + jitter, 1, max_step)


def _real_account_pool(backbone: pd.DataFrame, cap: int = 50_000) -> np.ndarray:
    """A sample of real account IDs a synthetic account can transact with during a
    warm-up step, so it inherits a real edge into the real graph's component instead
    of presenting as an isolated new node. Capped for speed on the full 6.3M-row set."""
    if "nameOrig" not in backbone.columns or len(backbone) == 0:
        return np.array(["C0"])
    pool = backbone["nameOrig"].to_numpy()
    if len(pool) > cap:
        pool = np.random.default_rng(0).choice(pool, size=cap, replace=False)
    return pool


def _warmup_row(orig: str, real_pool: np.ndarray, rng: np.random.Generator, attack_step: int) -> dict:
    """One small, ordinary-looking transaction with a real account, a few steps before
    the attack proper, so the synthetic account has out_degree>=2 and shares a
    component with the real graph instead of being a fresh isolated pair."""
    partner = str(rng.choice(real_pool))
    amt = round(float(rng.uniform(5.0, 250.0)), 2)
    warmup_step = max(1, attack_step - int(rng.integers(1, 6)))
    bal = amt * float(rng.uniform(3.0, 15.0))
    return {
        "step": warmup_step, "type": "PAYMENT", "amount": amt,
        "nameOrig": orig, "oldbalanceOrg": round(bal, 2), "newbalanceOrig": round(bal - amt, 2),
        "nameDest": partner, "oldbalanceDest": 0.0, "newbalanceDest": amt,
        "isFraud": 0, "isFlaggedFraud": 0,
    }


def simulate_ato_rapid_drain(config: AttackConfig, backbone: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_instances
    rows = []
    steps = _sample_steps(backbone, n, config.timing.burst_window_steps, rng)
    starting_balance = rng.lognormal(mean=config.amount.mean_log + 0.3, sigma=config.amount.sigma_log, size=n)
    drain_fraction = rng.uniform(0.85, 0.98, size=n)
    real_pool = _real_account_pool(backbone) if config.warm_up else None

    for i in range(n):
        orig = _new_account_id("ATO", i)
        dest = _new_account_id("ATOSINK", i)
        if config.warm_up:
            rows.append(_warmup_row(orig, real_pool, rng, int(steps[i])))
        bal = starting_balance[i]
        drain_amt = bal * drain_fraction[i]
        n_stages = rng.integers(2, 4)
        stage_amts = np.sort(rng.dirichlet(np.ones(n_stages)) * drain_amt)[::-1]
        cur_bal = bal
        for s, amt in enumerate(stage_amts):
            new_bal = max(0.0, cur_bal - amt)
            rows.append({
                "step": int(steps[i]) + s, "type": "TRANSFER", "amount": round(float(amt), 2),
                "nameOrig": orig, "oldbalanceOrg": round(float(cur_bal), 2),
                "newbalanceOrig": round(float(new_bal), 2), "nameDest": dest,
                "oldbalanceDest": 0.0, "newbalanceDest": round(float(amt), 2),
                "isFraud": 1, "isFlaggedFraud": 0,
            })
            cur_bal = new_bal

    df = pd.DataFrame(rows, columns=PAYSIM_COLUMNS)
    df["attack_family"] = config.attack_family
    df["is_synthetic"] = 1
    return df


def simulate_structuring(config: AttackConfig, backbone: pd.DataFrame, rng: np.random.Generator, threshold: float = 10_000.0) -> pd.DataFrame:
    n = config.n_instances
    n_actors = max(1, n // 6)
    rows = []
    real_pool = _real_account_pool(backbone) if config.warm_up else None
    for actor_i in range(n_actors):
        orig = _new_account_id("SMURF", actor_i)
        dest = _new_account_id("SMURFSINK", actor_i)
        n_txns = max(1, n // n_actors)
        amounts = _draw_amounts(config, n_txns, rng, scale_ref=threshold)
        steps = _sample_steps(backbone, n_txns, config.timing.burst_window_steps, rng)
        if config.warm_up:
            rows.append(_warmup_row(orig, real_pool, rng, int(steps[0])))
        bal = float(np.sum(amounts) * rng.uniform(1.05, 1.3))
        cur_bal = bal
        for amt, step in zip(amounts, steps):
            new_bal = max(0.0, cur_bal - amt)
            rows.append({
                "step": int(step), "type": "TRANSFER", "amount": round(float(amt), 2),
                "nameOrig": orig, "oldbalanceOrg": round(float(cur_bal), 2),
                "newbalanceOrig": round(float(new_bal), 2), "nameDest": dest,
                "oldbalanceDest": 0.0, "newbalanceDest": round(float(amt), 2),
                "isFraud": 1, "isFlaggedFraud": 0,
            })
            cur_bal = new_bal

    df = pd.DataFrame(rows, columns=PAYSIM_COLUMNS)
    df["attack_family"] = config.attack_family
    df["is_synthetic"] = 1
    return df


def simulate_mule_network(config: AttackConfig, backbone: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_networks = max(1, config.n_instances // config.graph.n_accounts)
    rows = []
    real_pool = _real_account_pool(backbone) if config.warm_up else None
    for net_i in range(n_networks):
        accounts = [_new_account_id(f"MULE{net_i}", i) for i in range(config.graph.n_accounts)]
        sink = _new_account_id(f"MULESINK{net_i}", 0)
        entry_amount = float(rng.lognormal(mean=config.amount.mean_log, sigma=config.amount.sigma_log))
        step = int(_sample_steps(backbone, 1, config.timing.burst_window_steps, rng)[0])

        if config.warm_up:
            # Warm up every mule account in the chain, not just the entry point --
            # each one is an isolated fresh node by construction otherwise, which is
            # exactly the tell the cycle-1 detector leaned on (orig_component_size).
            for acc in accounts + [sink]:
                rows.append(_warmup_row(acc, real_pool, rng, step))

        cur_amount = entry_amount
        prev_account = accounts[0]
        # entry hop: assume funds first land in accounts[0] from an untracked external source
        for hop in range(config.graph.hops):
            next_account = accounts[(hop + 1) % len(accounts)] if hop + 1 < len(accounts) else sink
            skim = rng.uniform(0.02, 0.08)  # simulated fee/skim per hop
            out_amount = cur_amount * (1 - skim)
            rows.append({
                "step": step + hop, "type": "TRANSFER", "amount": round(float(cur_amount), 2),
                "nameOrig": prev_account, "oldbalanceOrg": round(float(cur_amount), 2),
                "newbalanceOrig": 0.0, "nameDest": next_account,
                "oldbalanceDest": 0.0, "newbalanceDest": round(float(out_amount), 2),
                "isFraud": 1, "isFlaggedFraud": 0,
            })
            cur_amount = out_amount
            prev_account = next_account

        # cash-out at sink
        rows.append({
            "step": step + config.graph.hops, "type": "CASH_OUT", "amount": round(float(cur_amount), 2),
            "nameOrig": sink, "oldbalanceOrg": round(float(cur_amount), 2), "newbalanceOrig": 0.0,
            "nameDest": _new_account_id(f"MERCHANT{net_i}", 0), "oldbalanceDest": 0.0,
            "newbalanceDest": round(float(cur_amount), 2), "isFraud": 1, "isFlaggedFraud": 0,
        })

    df = pd.DataFrame(rows, columns=PAYSIM_COLUMNS)
    df["attack_family"] = config.attack_family
    df["is_synthetic"] = 1
    return df


def simulate_card_testing_burst(config: AttackConfig, backbone: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_actors = max(1, config.n_instances // config.graph.fanout)
    rows = []
    real_pool = _real_account_pool(backbone) if config.warm_up else None
    for actor_i in range(n_actors):
        orig = _new_account_id("CARDTEST", actor_i)
        step = int(_sample_steps(backbone, 1, config.timing.burst_window_steps, rng)[0])
        if config.warm_up:
            rows.append(_warmup_row(orig, real_pool, rng, step))
        amounts = _draw_amounts(config, config.graph.fanout, rng)
        # Origin balance drawn independently of probe amount (real accounts' balances
        # don't scale with a single transaction's size) -- a fixed amount*k multiplier
        # here would make amount_to_orig_balance_ratio a near-constant, trivially
        # learnable artifact of the simulator rather than a real fraud signal.
        cur_bal = float(rng.lognormal(mean=8.5, sigma=1.3))
        # Probes spread across the burst window rather than all landing on one step --
        # a hardened config widens burst_window_steps specifically to lower per-step
        # fanout (the dominant graph signal from cycle 1) without reducing the total
        # number of probes.
        window = max(1, config.timing.burst_window_steps)
        probe_steps = step + rng.integers(0, window, size=len(amounts))
        for j, (amt, p_step) in enumerate(zip(amounts, probe_steps)):
            dest = _new_account_id(f"MERCH{actor_i}", j)
            new_bal = max(0.0, cur_bal - amt)
            rows.append({
                "step": int(p_step), "type": "PAYMENT", "amount": round(float(amt), 2),
                "nameOrig": orig, "oldbalanceOrg": round(cur_bal, 2),
                "newbalanceOrig": round(new_bal, 2), "nameDest": dest,
                "oldbalanceDest": 0.0, "newbalanceDest": round(float(amt), 2),
                "isFraud": 1, "isFlaggedFraud": 0,
            })
            cur_bal = new_bal

    df = pd.DataFrame(rows, columns=PAYSIM_COLUMNS)
    df["attack_family"] = config.attack_family
    df["is_synthetic"] = 1
    return df


SIMULATORS = {
    "ato_rapid_drain": simulate_ato_rapid_drain,
    "structuring_smurfing": simulate_structuring,
    "mule_network_layering": simulate_mule_network,
    "card_testing_burst": simulate_card_testing_burst,
}
