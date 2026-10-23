"""Paired fixed-versus-dynamic repair experiments.

The two policies in each pair share a model seed. Because model initialization,
the exogenous environment, and agent decisions use isolated deterministic
streams, this gives common random numbers without coupling draw order.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from config import CONFIG
from experiments.output import (
    capture_provenance,
    create_run_directory,
    save_json,
    save_rows,
)
from model import StorageModel


VALID_COMPARISONS = {"equal_base_reward", "equal_repair_budget"}


def _average(rows: list[dict[str, Any]], field: str, default: float) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else default


def _initial_coverage(model: StorageModel) -> float | None:
    used = model.calculate_used_storage()
    price = model.update_storage_price(model.calculate_utilization())
    obligation = used * price / (1.0 - model.slippage)
    return model.endowment / obligation if obligation > 0 else None


def _summarize(
    model: StorageModel,
    rows: list[dict[str, Any]],
    identifiers: dict[str, Any],
    horizon: int,
    initial_coverage: float | None,
) -> dict[str, Any]:
    final = rows[-1] if rows else {}
    end_reason = model.end_reason or "HORIZON"
    return {
        **identifiers,
        "horizon_months": horizon,
        "months_observed": len(rows),
        "end_reason": end_reason,
        "is_insolvent": int(model.insolvent),
        "first_insolvency_month": model.first_insolvency_month,
        "initial_obligation_coverage_ratio": initial_coverage,
        "final_endowment": float(final.get("endowment", model.endowment)),
        "minimum_endowment": min(
            (float(row["endowment"]) for row in rows), default=model.endowment
        ),
        "mean_replication_compliance_rate": _average(
            rows, "replication_compliance_rate", 1.0
        ),
        "mean_accessible_dataset_fraction": _average(
            rows, "accessible_dataset_fraction", 1.0
        ),
        "mean_replica_sufficiency": _average(
            rows, "mean_replica_sufficiency", 1.0
        ),
        "permanent_loss_count": len(model.permanent_loss_ids),
        "any_permanent_loss": int(bool(model.permanent_loss_ids)),
        "mean_completed_recovery_time": (
            mean(model.recovery_durations) if model.recovery_durations else None
        ),
        "unrecovered_episode_count": len(model.dataset_degraded_since),
        "total_repair_expenditure": sum(float(r["repair_cost"]) for r in rows),
        "total_storage_expenditure": sum(
            float(r["storage_obligation"]) for r in rows
        ),
        "total_obligation_shortfall": sum(
            float(r["obligation_shortfall"]) for r in rows
        ),
        "total_repairs_initiated": sum(int(r["repair_initiated"]) for r in rows),
        "total_repairs_completed": sum(int(r["repair_completed"]) for r in rows),
        "total_repairs_cancelled": sum(int(r["repair_cancelled"]) for r in rows),
        "final_pending_repairs": int(final.get("repair_pending", 0)),
        "final_active_miners": int(final.get("active_miners", len(model.miners))),
    }


def run_paired_experiment(
    config: dict[str, Any] | None = None,
    pair_count: int = 100,
    months: int = 120,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    comparisons: Iterable[str] = ("equal_base_reward",),
) -> Path:
    """Run paired repair policies and save an immutable artifact directory."""
    if pair_count <= 0 or months < 0:
        raise ValueError("pair_count must be positive and months nonnegative")
    comparison_names = tuple(comparisons)
    if not comparison_names or len(set(comparison_names)) != len(comparison_names):
        raise ValueError("comparisons must be a nonempty sequence without duplicates")
    unknown = set(comparison_names) - VALID_COMPARISONS
    if unknown:
        raise ValueError(f"unknown comparisons: {sorted(unknown)}")

    base = deepcopy(CONFIG if config is None else config)
    dynamic_alpha = float(base.get("repair", {}).get("alpha", 2.0))
    if dynamic_alpha <= 0:
        raise ValueError("dynamic repair requires repair.alpha > 0")
    if "equal_repair_budget" in comparison_names:
        budget = base.get("repair", {}).get("monthly_budget")
        if budget is None or float(budget) < 0:
            raise ValueError(
                "equal_repair_budget requires nonnegative repair.monthly_budget"
            )

    run_dir = create_run_directory(
        "paired_repair", output_root=output_root, run_id=run_id
    )
    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seed_start = int(base.get("monte_carlo", {}).get(
        "seed_start", base.get("simulation", {}).get("seed", 42)
    ))

    for comparison in comparison_names:
        for pair_id in range(pair_count):
            model_seed = seed_start + pair_id
            for policy, alpha in (("fixed", 0.0), ("dynamic", dynamic_alpha)):
                policy_config = deepcopy(base)
                policy_config.setdefault("simulation", {})["months"] = months
                policy_config["simulation"]["seed"] = model_seed
                policy_config.setdefault("endowment", {})["insolvency_policy"] = "stop"
                policy_config.setdefault("repair", {})["alpha"] = alpha
                identifiers = {
                    "comparison": comparison,
                    "pair_id": pair_id,
                    "policy": policy,
                    "model_seed": model_seed,
                    "environment_seed": model_seed + 1_000_003,
                }
                model = StorageModel(policy_config, seed=model_seed)
                initial_coverage = _initial_coverage(model)
                rows = model.run(months=months)
                monthly_rows.extend({**identifiers, **row} for row in rows)
                summary_rows.append(
                    _summarize(
                        model, rows, identifiers, months, initial_coverage
                    )
                )

    comparison_metadata: dict[str, dict[str, Any]] = {
        "equal_base_reward": {
            "base_reward": float(base.get("repair", {}).get("base_reward", 1.0)),
            "fixed_alpha": 0.0,
            "dynamic_alpha": dynamic_alpha,
        }
    }
    if "equal_repair_budget" in comparison_names:
        comparison_metadata["equal_repair_budget"] = {
            "monthly_budget": float(base["repair"]["monthly_budget"]),
            "fixed_alpha": 0.0,
            "dynamic_alpha": dynamic_alpha,
        }

    save_rows(run_dir / "monthly.csv", monthly_rows)
    save_rows(run_dir / "summary.csv", summary_rows)
    save_json(run_dir / "config.json", base)
    save_json(
        run_dir / "metadata.json",
        {
            "experiment": "paired_repair",
            "pair_count": pair_count,
            "policies": ["fixed", "dynamic"],
            "comparisons": {
                name: comparison_metadata[name] for name in comparison_names
            },
            "horizon_months": months,
            "seed_mapping": {
                "model_seed_start": seed_start,
                "model_seed_rule": "model_seed_start + pair_id",
                "environment_seed_rule": "model_seed + 1000003",
            },
            "randomness": (
                "Paired policies share initialization and exogenous seeds; "
                "endogenous draws are keyed by stream, month, and entity."
            ),
            "provenance": capture_provenance(),
        },
    )
    return run_dir


def main() -> None:
    run_dir = run_paired_experiment()
    print(f"Paired repair experiment saved to {run_dir}")


if __name__ == "__main__":
    main()
