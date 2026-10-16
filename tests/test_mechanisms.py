import pytest

from mechanisms import bonding_curve, repair_probability, repair_reward


def test_bonding_curve_increases_with_utilization():
    low = bonding_curve(0.2, 0.75, 0.5, 2.0, 15.0)
    high = bonding_curve(0.8, 0.75, 0.5, 2.0, 15.0)
    assert low < high


def test_bonding_curve_price_is_bounded():
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        price = bonding_curve(rho, 0.75, 0.5, 2.0, 15.0)
        assert 0.5 <= price <= 2.0


def test_repair_reward_increases_with_deficit():
    low = repair_reward(1, 5, 1.0, 2.0)
    high = repair_reward(4, 5, 1.0, 2.0)
    assert low < high


def test_repair_probability_increases_with_utility():
    assert repair_probability(-1.0, 2.0) < repair_probability(1.0, 2.0)


@pytest.mark.parametrize("utility", [-1000.0, -1.0, 0.0, 1.0, 1000.0])
def test_repair_probability_is_bounded(utility):
    probability = repair_probability(utility, 2.0)
    assert 0.0 <= probability <= 1.0
