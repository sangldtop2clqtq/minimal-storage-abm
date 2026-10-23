"""One-factor-at-a-time exploratory sensitivity screening."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from config import CONFIG
from experiments.output import capture_provenance, create_run_directory, save_json, save_rows
from experiments.paired_repair import _initial_coverage, _summarize
from model import StorageModel


FACTOR_LEVELS: dict[str, dict[str, Any]] = {
    "initial_endowment": {"path": ("endowment", "initial"), "low": 5_000.0, "base": 10_000.0, "high": 20_000.0, "unit": "USD"},
    "monthly_yield": {"path": ("stochastic", "yield_process", "initial"), "low": 0.004, "base": 0.008, "high": 0.012, "unit": "fraction/month"},
    "initial_dataset_count": {"path": ("datasets", "count"), "low": 5, "base": 10, "high": 20, "unit": "datasets (utilization proxy)"},
    "target_replicas": {"path": ("datasets", "target_replicas"), "low": 3, "base": 5, "high": 7, "unit": "replicas/dataset"},
    "base_repair_reward": {"path": ("repair", "base_reward"), "low": 0.5, "base": 1.0, "high": 2.0, "unit": "USD/completed repair"},
    "repair_alpha": {"path": ("repair", "alpha"), "low": 1.0, "base": 2.0, "high": 4.0, "unit": "dimensionless"},
    "repair_latency": {"path": ("repair", "latency_months"), "low": 1, "base": 2, "high": 3, "unit": "months"},
    "exit_sensitivity": {"path": ("miner_behavior", "beta_leave"), "low": 1.0, "base": 2.0, "high": 4.0, "unit": "1/USD"},
    "cost_volatility": {"path": ("stochastic", "cost_process", "volatility"), "low": 0.02, "base": 0.04, "high": 0.08, "unit": "per sqrt(month)"},
    "failure_probability": {"path": ("stochastic", "miner_failure_probability"), "low": 0.0, "base": 0.001, "high": 0.005, "unit": "probability/miner/month"},
}


def _set_path(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def build_oat_design(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return one normalized base point plus low/high for ten factors."""
    normalized = deepcopy(CONFIG if config is None else config)
    for spec in FACTOR_LEVELS.values():
        _set_path(normalized, spec["path"], spec["base"])
    design = [{
        "setting_id": "base",
        "factor": "base",
        "level": "base",
        "requested_value": "normalized base configuration",
        "unit": "mixed",
        "config": deepcopy(normalized),
    }]
    for factor, spec in FACTOR_LEVELS.items():
        for level in ("low", "high"):
            point = deepcopy(normalized)
            _set_path(point, spec["path"], spec[level])
            design.append({
                "setting_id": f"{factor}-{level}",
                "factor": factor,
                "level": level,
                "requested_value": spec[level],
                "unit": spec["unit"],
                "config": point,
            })
    return design


def run_sensitivity_experiment(
    config: dict[str, Any] | None = None,
    runs_per_setting: int = 10,
    months: int = 24,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    max_settings: int | None = None,
    seed_start: int = 40_000,
) -> Path:
    if runs_per_setting <= 0 or months < 0:
        raise ValueError("runs_per_setting must be positive and months nonnegative")
    design = build_oat_design(config)
    if max_settings is not None:
        if max_settings <= 0:
            raise ValueError("max_settings must be positive")
        design = design[:max_settings]
    run_dir = create_run_directory("sensitivity", output_root, run_id)
    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for setting_index, point in enumerate(design):
        for run_index in range(runs_per_setting):
            model_seed = seed_start + setting_index * runs_per_setting + run_index
            point_config = deepcopy(point["config"])
            point_config.setdefault("simulation", {})["seed"] = model_seed
            point_config["simulation"]["months"] = months
            point_config.setdefault("endowment", {})["insolvency_policy"] = "stop"
            model = StorageModel(point_config, seed=model_seed)
            realized_initial_utilization = model.calculate_utilization()
            coverage = _initial_coverage(model)
            rows = model.run(months)
            ids = {
                "setting_id": point["setting_id"],
                "factor": point["factor"],
                "level": point["level"],
                "requested_setting": f"{point['requested_value']} {point['unit']}",
                "requested_value": point["requested_value"],
                "unit": point["unit"],
                "run_index": run_index,
                "model_seed": model_seed,
                "environment_seed": model_seed + 1_000_003,
                "realized_initial_utilization": realized_initial_utilization,
            }
            monthly_rows.extend({**ids, **row} for row in rows)
            summary_rows.append(_summarize(model, rows, ids, months, coverage))

    save_rows(run_dir / "monthly.csv", monthly_rows)
    save_rows(run_dir / "summary.csv", summary_rows)
    save_json(run_dir / "config.json", deepcopy(CONFIG if config is None else config))
    save_json(
        run_dir / "metadata.json",
        {
            "experiment": "sensitivity_oat",
            "runs_per_setting": runs_per_setting,
            "horizon_months": months,
            "total_settings": len(design),
            "planned_trajectories": len(design) * runs_per_setting,
            "seed_start": seed_start,
            "design": [{k: v for k, v in p.items() if k != "config"} for p in design],
            "provenance": capture_provenance(),
        },
    )
    return run_dir


def main() -> None:
    print(run_sensitivity_experiment())


if __name__ == "__main__":
    main()
