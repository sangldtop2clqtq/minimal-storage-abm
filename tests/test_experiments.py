import csv
import json

from experiments.paired_repair import run_paired_experiment
from tests.test_model import small_config


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_paired_runner_shares_exogenous_path_and_writes_complete_artifacts(tmp_path):
    config = small_config()
    config["simulation"].update({"months": 3, "seed": 800})
    config["repair"].update({"alpha": 2.0, "latency_months": 1})
    config["endowment"]["insolvency_policy"] = "stop"
    config["stochastic"] = {
        "enabled": True,
        "dt": 1.0,
        "yield_process": {
            "initial": 0.01, "long_run": 0.01, "kappa": 0.2,
            "volatility": 0.001,
        },
        "cost_process": {"drift": 0.0, "volatility": 0.01},
        "token_price": {
            "initial": 1.0, "drift": 0.0, "volatility": 0.01,
            "crash_probability": 0.0, "crash_multiplier": 0.1,
        },
        "slippage": {"alpha": 2.0, "beta": 98.0},
        "demand": {"arrival_rate": 0.2, "size_min": 1.0, "size_max": 1.0},
        "miner_failure_probability": 0.0,
        "correlation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }

    run_dir = run_paired_experiment(
        config, pair_count=2, months=3, output_root=tmp_path,
        run_id="smoke", comparisons=("equal_base_reward",),
    )

    assert {path.name for path in run_dir.iterdir()} == {
        "monthly.csv", "summary.csv", "config.json", "metadata.json"
    }
    monthly = _read_csv(run_dir / "monthly.csv")
    summary = _read_csv(run_dir / "summary.csv")
    assert len(summary) == 4
    assert {(row["pair_id"], row["policy"]) for row in summary} == {
        ("0", "fixed"), ("0", "dynamic"),
        ("1", "fixed"), ("1", "dynamic"),
    }
    for pair_id in ("0", "1"):
        fixed = [r for r in monthly if r["pair_id"] == pair_id and r["policy"] == "fixed"]
        dynamic = [r for r in monthly if r["pair_id"] == pair_id and r["policy"] == "dynamic"]
        assert [r["yield_rate"] for r in fixed] == [r["yield_rate"] for r in dynamic]
        assert [r["token_price_usd"] for r in fixed] == [r["token_price_usd"] for r in dynamic]
        assert [r["slippage"] for r in fixed] == [r["slippage"] for r in dynamic]
        assert [r["demand_arrivals"] for r in fixed] == [r["demand_arrivals"] for r in dynamic]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pair_count"] == 2
    assert metadata["seed_mapping"]["environment_seed_rule"] == "model_seed + 1000003"
    assert metadata["provenance"]["python_version"]


def test_budget_matched_comparison_sets_same_monthly_cap(tmp_path):
    config = small_config()
    config["repair"].update({"alpha": 2.0, "monthly_budget": 7.5})

    run_dir = run_paired_experiment(
        config, pair_count=1, months=1, output_root=tmp_path,
        run_id="budget", comparisons=("equal_repair_budget",),
    )

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["comparisons"]["equal_repair_budget"]["monthly_budget"] == 7.5
