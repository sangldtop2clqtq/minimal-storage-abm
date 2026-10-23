from copy import deepcopy
import math

import numpy as np
import pytest

from model import StorageModel
from stochastic_environment import StochasticEnvironment
from tests.test_model import small_config


def environment_config(**overrides):
    config = {
        "enabled": True,
        "dt": 1.0,
        "yield_process": {
            "initial": 0.01,
            "long_run": 0.03,
            "kappa": 0.5,
            "volatility": 0.0,
        },
        "cost_process": {"drift": math.log(0.9), "volatility": 0.0},
        "token_price": {
            "initial": 2.0,
            "drift": math.log(1.1),
            "volatility": 0.0,
            "crash_probability": 0.0,
            "crash_multiplier": 0.1,
        },
        "slippage": {"alpha": 2.0, "beta": 98.0},
        "demand": {"arrival_rate": 0.0, "size_min": 1.0, "size_max": 2.0},
        "miner_failure_probability": 0.0,
        "correlation": np.eye(3).tolist(),
    }
    config.update(overrides)
    return config


def test_zero_volatility_cir_and_gbm_follow_deterministic_steps():
    env = StochasticEnvironment(environment_config(), seed=1)
    state = env.advance(miner_count=2)

    assert state["yield_rate"] == pytest.approx(0.02)
    assert state["cost_growth"] == pytest.approx(0.9)
    assert state["token_price"] == pytest.approx(2.2)
    assert state["miner_failures"] == [False, False]
    assert state["demand_sizes"] == []


def test_cir_dt_scales_deterministic_mean_reversion_step():
    env = StochasticEnvironment(environment_config(dt=0.5), seed=1)
    state = env.advance(miner_count=0)
    assert state["yield_rate"] == pytest.approx(0.015)


@pytest.mark.parametrize(
    "correlation",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, float("nan"), 0.0], [float("nan"), 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 1.1, 0.0], [1.1, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ],
)
def test_invalid_correlation_matrix_is_rejected(correlation):
    with pytest.raises(ValueError):
        StochasticEnvironment(environment_config(correlation=correlation), seed=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("miner_failure_probability", -0.1),
        ("miner_failure_probability", 1.1),
    ],
)
def test_failure_probability_must_be_in_unit_interval(field, value):
    with pytest.raises(ValueError):
        StochasticEnvironment(environment_config(**{field: value}), seed=1)


def test_stochastic_mode_does_not_also_apply_deterministic_cost_decline():
    config = small_config()
    config["miners"].update({"cost_min": 2.0, "cost_max": 2.0})
    config["stochastic"] = environment_config()
    model = StorageModel(config)

    model.step()

    assert all(miner.unit_cost == pytest.approx(1.8) for miner in model.miners)


def test_storage_model_requires_one_month_stochastic_step():
    config = deepcopy(small_config())
    config["stochastic"] = environment_config(dt=0.5)
    with pytest.raises(ValueError, match="one month"):
        StorageModel(config)
