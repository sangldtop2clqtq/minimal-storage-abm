import pytest

from model import StorageModel
from tests.test_model import small_config


def repair_config():
    config = small_config()
    config["repair"].update(
        {
            "base_reward": 10.0,
            "alpha": 1.0,
            "beta_repair": 100.0,
            "cost_per_replica": 0.0,
            "latency_months": 1,
        }
    )
    return config


def make_one_replica_deficit(model):
    dataset = model.datasets[0]
    removed = next(iter(dataset.miners))
    model._remove_replica(removed, dataset)
    return dataset


def test_repair_initiation_reserves_capacity_without_activating_replica():
    model = StorageModel(repair_config())
    dataset = make_one_replica_deficit(model)

    requests, initiated, promised = model.run_repair_game()

    assert requests == 1
    assert initiated == 1
    assert promised == pytest.approx(15.0)
    assert dataset.replicas == 1
    assert len(model.pending_repairs) == 1
    pending = model.pending_repairs[0]
    target = next(m for m in model.miners if m.miner_id == pending.target_miner_id)
    assert target.reserved_capacity == pytest.approx(dataset.size)
    assert pending.completes_at == 1


def test_due_repair_completes_and_releases_reserved_capacity():
    model = StorageModel(repair_config())
    dataset = make_one_replica_deficit(model)
    model.run_repair_game()
    pending = model.pending_repairs[0]
    target = next(m for m in model.miners if m.miner_id == pending.target_miner_id)
    model.time = 1

    completed, cancelled, payment, latencies = model.complete_pending_repairs()

    assert (completed, cancelled) == (1, 0)
    assert payment == pytest.approx(pending.promised_reward)
    assert latencies == [1]
    assert dataset.replicas == dataset.target_replicas
    assert target.reserved_capacity == 0.0
    assert model.pending_repairs == []


@pytest.mark.parametrize("failed_role", ["source", "target"])
def test_repair_is_cancelled_when_endpoint_goes_offline(failed_role):
    model = StorageModel(repair_config())
    dataset = make_one_replica_deficit(model)
    model.run_repair_game()
    pending = model.pending_repairs[0]
    miner_id = (
        pending.source_miner_id if failed_role == "source" else pending.target_miner_id
    )
    miner = next(m for m in model.miners if m.miner_id == miner_id)
    for stored in list(miner.datasets):
        model._remove_replica(miner, stored)
    miner.go_offline()
    model.time = 1

    completed, cancelled, payment, latencies = model.complete_pending_repairs()

    assert (completed, cancelled) == (0, 1)
    assert payment == 0.0
    assert latencies == []
    assert model.pending_repairs == []
    assert all(m.reserved_capacity == 0.0 for m in model.miners)
    assert dataset.replicas < dataset.target_replicas


def test_reserved_capacity_counts_against_free_capacity():
    model = StorageModel(repair_config())
    dataset = make_one_replica_deficit(model)
    model.run_repair_game()
    pending = model.pending_repairs[0]
    target = next(m for m in model.miners if m.miner_id == pending.target_miner_id)
    assert target.used_capacity + target.reserved_capacity <= target.capacity
    assert target.free_capacity == pytest.approx(
        target.capacity - target.used_capacity - target.reserved_capacity
    )


def test_monthly_repair_budget_caps_new_promised_rewards():
    config = repair_config()
    config["repair"]["monthly_budget"] = 14.0
    model = StorageModel(config)
    dataset = make_one_replica_deficit(model)

    requests, initiated, promised = model.run_repair_game()

    assert requests == 1
    assert initiated == 0
    assert promised == 0.0
    assert model.repair_budget_rejections_this_step == 1
    assert dataset.replicas == 1
