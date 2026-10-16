from agent import MinerAgent
from dataset import Dataset


def test_leave_probability_decreases_as_profit_increases():
    miner = MinerAgent(miner_id=1, capacity=100.0, unit_cost=1.0)
    miner.profit = -2.0
    probability_at_loss = miner.leave_probability(beta=1.0)
    miner.profit = 2.0
    probability_at_profit = miner.leave_probability(beta=1.0)
    assert probability_at_loss > probability_at_profit


def test_cost_and_profit_use_stored_capacity():
    miner = MinerAgent(miner_id=1, capacity=100.0, unit_cost=0.5)
    dataset = Dataset("dataset-1", size=10.0, target_replicas=1)
    assert miner.add_dataset(dataset)
    assert miner.calculate_profit(storage_price=1.0) == 5.0
