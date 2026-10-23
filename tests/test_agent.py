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


def test_leave_probability_at_zero_profit_is_one_half():
    miner = MinerAgent(miner_id=1, capacity=100.0, unit_cost=1.0)
    miner.profit = 0.0
    assert miner.leave_probability(beta=2.0) == 0.5


def test_leave_probability_is_stable_for_extreme_profit():
    miner = MinerAgent(miner_id=1, capacity=100.0, unit_cost=1.0)
    miner.profit = -1_000_000.0
    assert miner.leave_probability(beta=2.0) == 1.0
    miner.profit = 1_000_000.0
    assert miner.leave_probability(beta=2.0) == 0.0


def test_releasing_multiple_float_reservations_tolerates_rounding_noise():
    miner = MinerAgent(miner_id=1, capacity=200.0, unit_cost=1.0)
    first = 30.242888476223023
    second = 46.972006341425356
    miner.reserve_capacity(first)
    miner.reserve_capacity(second)

    miner.release_capacity(first)
    miner.release_capacity(second)

    assert miner.reserved_capacity == 0.0
