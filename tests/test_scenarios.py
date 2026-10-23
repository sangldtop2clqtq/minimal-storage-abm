from copy import deepcopy

import pytest

from model import StorageModel
from stochastic_environment import StochasticEnvironment
from tests.test_model import small_config


def deterministic_environment():
    return {
        "enabled": True,
        "dt": 1.0,
        "yield_process": {
            "initial": 0.01, "long_run": 0.01, "kappa": 0.2,
            "volatility": 0.0,
        },
        "cost_process": {"drift": 0.0, "volatility": 0.0},
        "token_price": {
            "initial": 1.0, "drift": 0.0, "volatility": 0.0,
            "crash_probability": 0.0, "crash_multiplier": 0.1,
        },
        "slippage": {"alpha": 2.0, "beta": 98.0},
        "demand": {"arrival_rate": 0.0, "size_min": 1.0, "size_max": 1.0},
        "miner_failure_probability": 0.0,
        "correlation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def test_yield_reduction_is_active_only_in_scheduled_window():
    config = deterministic_environment()
    config["yield_process"].update(
        {"shock_month": 1, "shock_duration": 2, "shock_multiplier": 0.5}
    )
    env = StochasticEnvironment(config, seed=1)

    observed = [env.advance(0)["yield_rate"] for _ in range(4)]

    assert observed == pytest.approx([0.01, 0.005, 0.005, 0.01])


def test_cost_level_shock_starts_and_reverses_at_window_boundaries():
    config = deterministic_environment()
    config["cost_process"].update(
        {"shock_month": 1, "shock_duration": 2, "shock_multiplier": 1.5}
    )
    env = StochasticEnvironment(config, seed=1)

    states = [env.advance(0) for _ in range(4)]

    assert [state["cost_level"] for state in states] == pytest.approx(
        [1.0, 1.5, 1.5, 1.0]
    )


def test_slippage_shock_multiplies_only_the_scheduled_window():
    base = deterministic_environment()
    shocked = deepcopy(base)
    shocked["slippage"].update(
        {"shock_month": 1, "shock_duration": 1, "shock_multiplier": 2.0}
    )
    control_env = StochasticEnvironment(base, seed=7)
    shock_env = StochasticEnvironment(shocked, seed=7)

    control = [control_env.advance(0)["slippage"] for _ in range(3)]
    treatment = [shock_env.advance(0)["slippage"] for _ in range(3)]

    assert treatment[0] == pytest.approx(control[0])
    assert treatment[1] == pytest.approx(2.0 * control[1])
    assert treatment[2] == pytest.approx(control[2])


def test_failure_probability_shock_reaches_failure_draws():
    config = deterministic_environment()
    config.update({
        "miner_failure_probability": 0.0,
        "failure_shock": {
            "month": 1, "duration": 1, "probability": 1.0,
        },
    })
    env = StochasticEnvironment(config, seed=3)

    first = env.advance(3)["miner_failures"]
    second = env.advance(3)["miner_failures"]
    third = env.advance(3)["miner_failures"]

    assert first == [False, False, False]
    assert second == [True, True, True]
    assert third == [False, False, False]


def test_token_crash_changes_token_settlement_but_not_usd_miner_profit():
    base = small_config()
    base["stochastic"] = deterministic_environment()
    shocked = deepcopy(base)
    shocked["stochastic"]["token_price"]["crash_month"] = 0
    control = StorageModel(base, seed=12).step()
    crash = StorageModel(shocked, seed=12).step()

    assert crash["token_price_usd"] == pytest.approx(0.1)
    assert crash["storage_reward_tokens"] == pytest.approx(
        10.0 * control["storage_reward_tokens"]
    )
    assert crash["average_miner_profit"] == pytest.approx(
        control["average_miner_profit"]
    )
    assert crash["storage_obligation"] == pytest.approx(control["storage_obligation"])


def test_demand_shock_does_not_shift_other_exogenous_streams():
    base = deterministic_environment()
    base["demand"]["arrival_rate"] = 2.0
    shocked = deepcopy(base)
    shocked["demand"].update(
        {"shock_month": 1, "shock_duration": 2, "shock_multiplier": 5.0}
    )
    control_env = StochasticEnvironment(base, seed=99)
    shock_env = StochasticEnvironment(shocked, seed=99)

    control = [control_env.advance(5) for _ in range(5)]
    treatment = [shock_env.advance(5) for _ in range(5)]

    for left, right in zip(control, treatment):
        for field in (
            "yield_rate", "cost_growth", "token_price", "slippage",
            "miner_failures",
        ):
            assert left[field] == pytest.approx(right[field])
