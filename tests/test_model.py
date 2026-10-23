from copy import deepcopy

import pytest

from model import StorageModel


def small_config():
    return {
        "simulation": {"months": 3, "seed": 17},
        "miners": {
            "count": 10,
            "capacity_min": 100.0,
            "capacity_max": 100.0,
            "cost_min": 0.0,
            "cost_max": 0.0,
            "monthly_cost_decline": 0.03,
        },
        "datasets": {
            "count": 2,
            "size_min": 1.0,
            "size_max": 1.0,
            "target_replicas": 2,
        },
        "bonding_curve": {"p_min": 0.5, "p_max": 1.0, "rho0": 0.75, "k": 10.0},
        "miner_behavior": {"beta_leave": 1.0},
        "repair": {
            "base_reward": 1.0,
            "alpha": 1.0,
            "beta_repair": 1.0,
            "cost_per_replica": 0.5,
        },
        "endowment": {"initial": 1_000.0, "yield_rate": 0.0},
    }


def test_initial_replica_links_and_capacity_invariants():
    model = StorageModel(small_config())
    for miner in model.miners:
        assert miner.used_capacity <= miner.capacity
        for dataset in miner.datasets:
            assert miner in dataset.miners
    for dataset in model.datasets:
        for miner in dataset.miners:
            assert dataset in miner.datasets


def test_forced_exit_removes_miners_and_their_replicas():
    model = StorageModel(small_config())
    leaving = model.process_forced_miner_exits(1.0)
    assert len(leaving) == 10
    assert all(miner.status == "OFFLINE" for miner in leaving)
    assert all(dataset.replicas == 0 for dataset in model.datasets)


def test_forced_exit_rate_must_be_between_zero_and_one():
    model = StorageModel(small_config())
    with pytest.raises(ValueError):
        model.process_forced_miner_exits(1.1)


def test_same_seed_and_config_reproduce_metrics():
    config1 = small_config()
    config2 = deepcopy(config1)
    first = StorageModel(config1).run()
    second = StorageModel(config2).run()
    assert first == second


def test_monthly_cost_decline_is_applied_for_next_step():
    model = StorageModel(small_config())
    model.miners[0].unit_cost = 2.0
    initial_cost = model.miners[0].unit_cost
    model.update_miner_costs()
    assert model.miners[0].unit_cost == pytest.approx(initial_cost * 0.97)


def test_exit_draws_are_independent_of_miner_iteration_order():
    first = StorageModel(small_config())
    second = StorageModel(deepcopy(small_config()))
    for miner in first.miners + second.miners:
        miner.profit = 0.0
    second.miners.reverse()

    first_leavers = {miner.miner_id for miner in first.process_miner_exits()}
    second_leavers = {miner.miner_id for miner in second.process_miner_exits()}

    assert first_leavers == second_leavers


def test_end_of_period_utilization_matches_end_of_period_capacity():
    config = small_config()
    config["miner_behavior"]["beta_leave"] = 0.0
    model = StorageModel(config)

    row = model.step()

    assert "pricing_utilization" in row
    if row["total_capacity"] > 0:
        assert row["utilization"] == pytest.approx(
            row["used_capacity"] / row["total_capacity"]
        )
    else:
        assert row["used_capacity"] == 0.0
        assert row["utilization"] == 0.0


def test_voluntary_leaver_is_paid_for_current_month_service():
    config = small_config()
    config["miners"].update(
        {"count": 1, "capacity_min": 10.0, "capacity_max": 10.0,
         "cost_min": 2.0, "cost_max": 2.0}
    )
    config["datasets"].update(
        {"count": 1, "size_min": 1.0, "size_max": 1.0,
         "target_replicas": 1}
    )
    config["bonding_curve"].update({"p_min": 1.0, "p_max": 1.0})
    config["miner_behavior"]["beta_leave"] = 100.0
    config["repair"].update({"base_reward": 0.0, "alpha": 0.0})
    config["endowment"] = {"initial": 100.0, "yield_rate": 0.0}
    model = StorageModel(config)

    row = model.step()

    assert row["voluntary_exits"] == 1
    assert row["storage_reward"] == pytest.approx(1.0)
    assert row["endowment"] == pytest.approx(99.0)
