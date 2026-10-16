"""Run independent stochastic ABM realizations and summarize uncertainty."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np

from config import CONFIG
from experiments.common import PROJECT_ROOT
from model import StorageModel


def run_monte_carlo(config: dict | None = None) -> list[dict[str, float | int]]:
    """Run the configured number of seeds and return one summary per run."""
    cfg = deepcopy(CONFIG if config is None else config)
    mc_cfg = cfg.get("monte_carlo", {})
    run_count = int(mc_cfg.get("runs", 100))
    seed_start = int(mc_cfg.get("seed_start", 1_000))
    months = int(cfg.get("simulation", {}).get("months", 12))
    if run_count <= 0 or months < 0:
        raise ValueError("Monte Carlo cần runs > 0 và months >= 0")

    summaries: list[dict[str, float | int]] = []
    for run_index in range(run_count):
        seed = seed_start + run_index
        model = StorageModel(config=cfg, seed=seed)
        metrics = model.run(months=months)
        final = metrics[-1] if metrics else {}
        endowment_values = [float(row["endowment"]) for row in metrics]
        availability_values = [float(row["data_availability"]) for row in metrics]
        summaries.append({
            "run": run_index,
            "seed": seed,
            "months": months,
            "final_endowment": float(final.get("endowment", model.endowment)),
            "minimum_endowment": min(endowment_values, default=model.endowment),
            "endowment_survival": int(all(value >= 0 for value in endowment_values)),
            "mean_data_availability": (
                sum(availability_values) / len(availability_values)
                if availability_values else 1.0
            ),
            "minimum_data_availability": min(availability_values, default=1.0),
            "final_active_miners": int(final.get("active_miners", len(model.miners))),
            "total_random_failures": sum(int(row["random_failures"]) for row in metrics),
            "total_voluntary_exits": sum(int(row["voluntary_exits"]) for row in metrics),
            "total_successful_repairs": sum(int(row["successful_repairs"]) for row in metrics),
            "total_demand_arrivals": sum(int(row["demand_arrivals"]) for row in metrics),
            "mean_token_price_usd": (
                sum(float(row["token_price_usd"]) for row in metrics) / len(metrics)
                if metrics else float(cfg.get("stochastic", {}).get("token_price", {}).get("initial", 1.0))
            ),
        })

    raw_dir = PROJECT_ROOT / "results" / "raw" / "monte_carlo"
    metadata_dir = PROJECT_ROOT / "results" / "metadata" / "monte_carlo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"mc_{run_count}_runs_seed_{seed_start}"
    csv_path = raw_dir / f"{run_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    survival_probability = sum(int(row["endowment_survival"]) for row in summaries) / run_count
    metadata = {
        "experiment": "monte_carlo",
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": run_count,
        "seed_start": seed_start,
        "months": months,
        "endowment_survival_probability": survival_probability,
        "final_endowment_mean": float(np.mean([r["final_endowment"] for r in summaries])),
        "final_endowment_p05": float(np.quantile([r["final_endowment"] for r in summaries], 0.05)),
        "final_endowment_p95": float(np.quantile([r["final_endowment"] for r in summaries], 0.95)),
        "mean_availability_mean": float(np.mean([r["mean_data_availability"] for r in summaries])),
        "mean_availability_p05": float(np.quantile([r["mean_data_availability"] for r in summaries], 0.05)),
        "mean_availability_p95": float(np.quantile([r["mean_data_availability"] for r in summaries], 0.95)),
        "config": cfg,
    }
    metadata_path = metadata_dir / f"{run_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Monte Carlo completed: {run_count} runs × {months} months")
    print(f"Endowment survival probability: {survival_probability:.1%}")
    print(f"Run summaries: {csv_path}")
    print(f"Metadata: {metadata_path}")
    return summaries


def main() -> None:
    run_monte_carlo()


if __name__ == "__main__":
    main()
