"""Prespecified paired stress scenarios for the repair mechanism."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from config import CONFIG
from experiments.output import capture_provenance, create_run_directory, save_json, save_rows
from experiments.paired_repair import _initial_coverage, _summarize
from model import StorageModel


STRESS_SCENARIOS: dict[str, dict[str, Any]] = {
    "normal": {"description": "No scheduled shock."},
    "mass_miner_exit": {
        "description": "30% of active miners exit once at month 6.",
        "forced_exit": {"month": 6, "rate": 0.30},
    },
    "failure_spike": {
        "description": "Monthly miner failure probability is 10% for months 6-8.",
        "stochastic": {"failure_shock": {"month": 6, "duration": 3, "probability": 0.10}},
    },
    "yield_drop": {
        "description": "Effective yield is reduced 75% for six months from month 6.",
        "stochastic": {"yield_process": {"shock_month": 6, "shock_duration": 6, "shock_multiplier": 0.25}},
    },
    "storage_cost_increase": {
        "description": "Miner cost level rises 50% for six months from month 6.",
        "stochastic": {"cost_process": {"shock_month": 6, "shock_duration": 6, "shock_multiplier": 1.50}},
    },
    "slippage_increase": {
        "description": "Realized slippage doubles for six months from month 6.",
        "stochastic": {"slippage": {"shock_month": 6, "shock_duration": 6, "shock_multiplier": 2.0}},
    },
    "demand_increase": {
        "description": "Dataset arrival intensity doubles for six months from month 6.",
        "stochastic": {"demand": {"shock_month": 6, "shock_duration": 6, "shock_multiplier": 2.0}},
    },
    "token_crash_90pct": {
        "description": "Token price falls 90% once at month 6.",
        "interpretation": "settlement_only",
        "stochastic": {"token_price": {"crash_month": 6, "crash_multiplier": 0.10}},
    },
}


def _merge_nested(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"cannot merge scenario mapping into {key}")
            _merge_nested(child, value)
        else:
            target[key] = value


def run_stress_experiments(
    config: dict[str, Any] | None = None,
    pair_count: int = 20,
    months: int = 24,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    scenario_names: Iterable[str] = tuple(STRESS_SCENARIOS),
    seed_start: int = 30_000,
) -> Path:
    """Run fixed/dynamic policies under the same seed for each stress scenario."""
    if pair_count <= 0 or months < 0:
        raise ValueError("pair_count must be positive and months nonnegative")
    names = tuple(scenario_names)
    unknown = set(names) - set(STRESS_SCENARIOS)
    if not names or unknown:
        raise ValueError(f"unknown or empty scenario list: {sorted(unknown)}")
    base = deepcopy(CONFIG if config is None else config)
    dynamic_alpha = float(base.get("repair", {}).get("alpha", 2.0))
    if dynamic_alpha <= 0:
        raise ValueError("dynamic repair requires repair.alpha > 0")

    run_dir = create_run_directory("stress_tests", output_root, run_id)
    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scenario_name in names:
        scenario = STRESS_SCENARIOS[scenario_name]
        for pair_id in range(pair_count):
            model_seed = seed_start + pair_id
            for policy, alpha in (("fixed", 0.0), ("dynamic", dynamic_alpha)):
                policy_config = deepcopy(base)
                policy_config.setdefault("simulation", {})["seed"] = model_seed
                policy_config["simulation"]["months"] = months
                policy_config.setdefault("endowment", {})["insolvency_policy"] = "stop"
                policy_config.setdefault("repair", {})["alpha"] = alpha
                _merge_nested(
                    policy_config.setdefault("stochastic", {}),
                    scenario.get("stochastic", {}),
                )
                model = StorageModel(policy_config, seed=model_seed)
                coverage = _initial_coverage(model)
                rows: list[dict[str, Any]] = []
                forced = scenario.get("forced_exit")
                for _ in range(months):
                    exit_rate = (
                        float(forced["rate"])
                        if forced and model.time == int(forced["month"])
                        else 0.0
                    )
                    rows.append(model.step(forced_exit_rate=exit_rate))
                    if model.insolvent:
                        break
                ids = {
                    "scenario": scenario_name,
                    "pair_id": pair_id,
                    "policy": policy,
                    "model_seed": model_seed,
                    "environment_seed": model_seed + 1_000_003,
                }
                monthly_rows.extend({**ids, **row} for row in rows)
                summary_rows.append(_summarize(model, rows, ids, months, coverage))

    save_rows(run_dir / "monthly.csv", monthly_rows)
    save_rows(run_dir / "summary.csv", summary_rows)
    save_json(run_dir / "config.json", base)
    save_json(
        run_dir / "metadata.json",
        {
            "experiment": "stress_tests",
            "pair_count": pair_count,
            "horizon_months": months,
            "seed_start": seed_start,
            "scenarios": {name: STRESS_SCENARIOS[name] for name in names},
            "token_crash_interpretation": (
                "settlement-only: USD miner profit and USD storage obligation are unchanged"
            ),
            "provenance": capture_provenance(),
        },
    )
    return run_dir


def main() -> None:
    print(run_stress_experiments())


if __name__ == "__main__":
    main()
