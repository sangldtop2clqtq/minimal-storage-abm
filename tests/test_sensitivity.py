import csv

from experiments.sensitivity import build_oat_design, run_sensitivity_experiment
from tests.test_scenarios import deterministic_environment
from tests.test_model import small_config


def test_oat_design_has_one_base_and_two_levels_for_ten_factors():
    config = small_config()
    config["stochastic"] = deterministic_environment()

    design = build_oat_design(config)

    assert len(design) == 21
    assert design[0]["factor"] == "base"
    assert {point["factor"] for point in design[1:]} == {
        "initial_endowment", "monthly_yield", "initial_dataset_count",
        "target_replicas", "base_repair_reward", "repair_alpha",
        "repair_latency", "exit_sensitivity", "cost_volatility",
        "failure_probability",
    }
    assert all(point["level"] in {"base", "low", "high"} for point in design)


def test_sensitivity_smoke_records_requested_and_realized_utilization(tmp_path):
    config = small_config()
    config["stochastic"] = deterministic_environment()
    run_dir = run_sensitivity_experiment(
        config, runs_per_setting=1, months=1, output_root=tmp_path,
        run_id="sensitivity-smoke", max_settings=3,
    )

    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert all(row["requested_setting"] for row in rows)
    assert all(float(row["realized_initial_utilization"]) >= 0 for row in rows)
