import pytest

from model import StorageModel
from tests.test_model import small_config


def insolvent_config(policy):
    config = small_config()
    config["miners"].update(
        {"count": 1, "capacity_min": 10.0, "capacity_max": 10.0,
         "cost_min": 0.0, "cost_max": 0.0}
    )
    config["datasets"].update(
        {"count": 1, "size_min": 1.0, "size_max": 1.0,
         "target_replicas": 1}
    )
    config["bonding_curve"].update({"p_min": 10.0, "p_max": 10.0})
    config["miner_behavior"]["beta_leave"] = 100.0
    config["endowment"] = {
        "initial": 5.0,
        "yield_rate": 0.0,
        "insolvency_policy": policy,
    }
    return config


def test_stop_policy_records_payment_shortfall_and_ends_run_early():
    model = StorageModel(insolvent_config("stop"))

    rows = model.run(months=3)

    assert len(rows) == 1
    row = rows[0]
    assert row["total_obligations"] == pytest.approx(10.0)
    assert row["paid_obligations"] == pytest.approx(5.0)
    assert row["obligation_shortfall"] == pytest.approx(5.0)
    assert row["endowment"] == 0.0
    assert row["is_insolvent"] == 1
    assert row["first_insolvency_month"] == 0
    assert row["end_reason"] == "INSOLVENT"


def test_allow_negative_policy_distinguishes_negative_balance_from_shortfall():
    model = StorageModel(insolvent_config("allow_negative"))

    rows = model.run(months=2)

    assert len(rows) == 2
    assert rows[0]["endowment"] == pytest.approx(-5.0)
    assert rows[0]["obligation_shortfall"] == 0.0
    assert rows[0]["is_insolvent"] == 0


def test_recovery_time_is_recorded_when_degraded_dataset_becomes_healthy():
    config = small_config()
    config["repair"].update(
        {"base_reward": 10.0, "alpha": 1.0, "beta_repair": 100.0,
         "cost_per_replica": 0.0, "latency_months": 1}
    )
    model = StorageModel(config)
    dataset = model.datasets[0]
    model._remove_replica(next(iter(dataset.miners)), dataset)
    model.update_dataset_health_tracking()
    model.run_repair_game()
    model.time = 1
    model.complete_pending_repairs()

    recoveries = model.update_dataset_health_tracking()

    assert recoveries == [1]
