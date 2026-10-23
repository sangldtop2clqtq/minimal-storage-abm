import csv
import json

from experiments.stress_tests import STRESS_SCENARIOS, run_stress_experiments
from tests.test_scenarios import deterministic_environment
from tests.test_model import small_config


def test_stress_catalog_contains_all_prespecified_channels():
    assert set(STRESS_SCENARIOS) == {
        "normal", "mass_miner_exit", "failure_spike", "yield_drop",
        "storage_cost_increase", "slippage_increase", "demand_increase",
        "token_crash_90pct",
    }
    assert STRESS_SCENARIOS["token_crash_90pct"]["interpretation"] == "settlement_only"


def test_stress_smoke_writes_balanced_policy_scenario_rows(tmp_path):
    config = small_config()
    config["stochastic"] = deterministic_environment()
    config["repair"]["alpha"] = 2.0
    run_dir = run_stress_experiments(
        config, pair_count=1, months=7, output_root=tmp_path,
        run_id="stress-smoke", scenario_names=("normal", "token_crash_90pct"),
    )

    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    assert len(summary) == 4
    assert {(r["scenario"], r["policy"]) for r in summary} == {
        ("normal", "fixed"), ("normal", "dynamic"),
        ("token_crash_90pct", "fixed"),
        ("token_crash_90pct", "dynamic"),
    }
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["token_crash_interpretation"] == (
        "settlement-only: USD miner profit and USD storage obligation are unchanged"
    )
    with (run_dir / "monthly.csv").open(newline="", encoding="utf-8") as file:
        monthly = list(csv.DictReader(file))
    normal_month_6 = next(
        r for r in monthly
        if r["scenario"] == "normal" and r["policy"] == "fixed" and r["time"] == "6"
    )
    crash_month_6 = next(
        r for r in monthly
        if r["scenario"] == "token_crash_90pct"
        and r["policy"] == "fixed" and r["time"] == "6"
    )
    assert float(normal_month_6["token_price_usd"]) == 1.0
    assert float(crash_month_6["token_price_usd"]) == 0.1
