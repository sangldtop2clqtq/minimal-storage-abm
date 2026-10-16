from agent import MinerAgent
from dataset import Dataset


def test_dataset_replica_states_update_with_miners():
    dataset = Dataset("d1", size=10.0, target_replicas=2)
    first = MinerAgent(1, 20.0, 1.0)
    second = MinerAgent(2, 20.0, 1.0)

    assert dataset.is_lost
    dataset.add_miner(first)
    assert dataset.is_degraded
    dataset.add_miner(second)
    assert dataset.is_healthy
    assert dataset.replica_deficit == 0


def test_duplicate_replica_is_not_added():
    dataset = Dataset("d1", size=10.0, target_replicas=2)
    miner = MinerAgent(1, 20.0, 1.0)
    assert dataset.add_miner(miner)
    assert not dataset.add_miner(miner)
    assert dataset.replicas == 1
