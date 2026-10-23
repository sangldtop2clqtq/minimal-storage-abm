"""Monthly stochastic paths and scheduled exogenous shocks for the ABM."""

from __future__ import annotations

from typing import Any

import numpy as np


class StochasticEnvironment:
    """Generate reproducible economic and demand states from one local RNG."""

    def __init__(self, config: dict[str, Any], seed: int):
        self.config = config
        child_seeds = np.random.SeedSequence(seed).spawn(6)
        self.economic_rng = np.random.default_rng(child_seeds[0])
        self.crash_rng = np.random.default_rng(child_seeds[1])
        self.slippage_rng = np.random.default_rng(child_seeds[2])
        self.demand_count_rng = np.random.default_rng(child_seeds[3])
        self.demand_size_rng = np.random.default_rng(child_seeds[4])
        self.failure_rng = np.random.default_rng(child_seeds[5])
        self.dt = float(config.get("dt", 1.0))
        if self.dt <= 0:
            raise ValueError("stochastic.dt must be positive")

        self.yield_cfg = config.get("yield_process", {})
        self.cost_cfg = config.get("cost_process", {})
        self.token_cfg = config.get("token_price", {})
        self.slippage_cfg = config.get("slippage", {})
        self.demand_cfg = config.get("demand", {})
        self.failure_probability = float(config.get("miner_failure_probability", 0.0))
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("miner_failure_probability must be in [0, 1]")

        self.yield_rate = float(self.yield_cfg.get("initial", 0.008))
        self.cost_level = 1.0
        self._cost_shock_factor = 1.0
        self.token_price = float(self.token_cfg.get("initial", 1.0))
        self.step_count = 0
        if self.token_price <= 0:
            raise ValueError("token_price.initial must be positive")
        if self.yield_rate < 0 or float(self.yield_cfg.get("long_run", 0.008)) < 0:
            raise ValueError("CIR initial and long_run cannot be negative")

        correlation = np.asarray(
            config.get("correlation", np.eye(3).tolist()), dtype=float
        )
        if correlation.shape != (3, 3):
            raise ValueError("stochastic.correlation must be a 3x3 matrix")
        if not np.all(np.isfinite(correlation)):
            raise ValueError("stochastic.correlation must contain finite values")
        if not np.allclose(correlation, correlation.T):
            raise ValueError("stochastic.correlation must be symmetric")
        if not np.allclose(np.diag(correlation), np.ones(3)):
            raise ValueError("stochastic.correlation diagonal must equal 1")
        try:
            self.cholesky = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as exc:
            raise ValueError("stochastic.correlation must be positive definite") from exc

    def _in_shock_window(
        self, config: dict[str, Any], month_key: str = "shock_month"
    ) -> bool:
        start = config.get(month_key)
        if start is None:
            return False
        start = int(start)
        duration = int(config.get("shock_duration", config.get("duration", 1)))
        if start < 0 or duration < 1:
            raise ValueError("shock month must be nonnegative and duration at least 1")
        return start <= self.step_count < start + duration

    def advance(self, miner_count: int) -> dict[str, Any]:
        """Advance one model month and return the environmental state."""
        z_yield, z_cost, z_token = (
            self.cholesky @ self.economic_rng.standard_normal(3)
        )

        # CIR full-truncation Euler. The shock changes the effective output,
        # while the unshocked latent process continues to evolve.
        kappa = float(self.yield_cfg.get("kappa", 0.3))
        theta = float(self.yield_cfg.get("long_run", 0.008))
        sigma_y = float(self.yield_cfg.get("volatility", 0.001))
        if kappa <= 0 or theta < 0 or sigma_y < 0:
            raise ValueError("invalid CIR parameters")
        current = max(0.0, self.yield_rate)
        self.yield_rate = max(
            0.0,
            current
            + kappa * (theta - current) * self.dt
            + sigma_y * np.sqrt(current) * np.sqrt(self.dt) * z_yield,
        )
        effective_yield = self.yield_rate
        if self._in_shock_window(self.yield_cfg):
            multiplier = float(self.yield_cfg.get("shock_multiplier", 1.0))
            if multiplier < 0:
                raise ValueError("yield_process.shock_multiplier must be nonnegative")
            effective_yield *= multiplier

        # Exact GBM. A scheduled cost shock is a temporary level multiplier:
        # apply it at window start and reverse it immediately after the window.
        mu_cost = float(self.cost_cfg.get("drift", 0.0))
        sigma_cost = float(self.cost_cfg.get("volatility", 0.0))
        if sigma_cost < 0:
            raise ValueError("cost_process.volatility cannot be negative")
        cost_growth = float(
            np.exp(
                (mu_cost - 0.5 * sigma_cost**2) * self.dt
                + sigma_cost * np.sqrt(self.dt) * z_cost
            )
        )
        desired_cost_shock = 1.0
        if self._in_shock_window(self.cost_cfg):
            desired_cost_shock = float(self.cost_cfg.get("shock_multiplier", 1.0))
            if desired_cost_shock <= 0:
                raise ValueError("cost_process.shock_multiplier must be positive")
        cost_growth *= desired_cost_shock / self._cost_shock_factor
        self._cost_shock_factor = desired_cost_shock
        self.cost_level *= cost_growth
        if not np.isfinite(cost_growth) or not np.isfinite(self.cost_level) or cost_growth <= 0:
            raise FloatingPointError("GBM cost produced a nonfinite/nonpositive value")

        mu_token = float(self.token_cfg.get("drift", 0.0))
        sigma_token = float(self.token_cfg.get("volatility", 0.0))
        if sigma_token < 0:
            raise ValueError("token_price.volatility cannot be negative")
        token_growth = float(
            np.exp(
                (mu_token - 0.5 * sigma_token**2) * self.dt
                + sigma_token * np.sqrt(self.dt) * z_token
            )
        )
        crash_probability = float(self.token_cfg.get("crash_probability", 0.0))
        crash_multiplier = float(self.token_cfg.get("crash_multiplier", 0.1))
        if not 0.0 <= crash_probability <= 1.0 or not 0.0 < crash_multiplier <= 1.0:
            raise ValueError("invalid token crash configuration")
        scheduled_crash = self.token_cfg.get("crash_month") == self.step_count
        # Always consume the random crash draw so enabling a scheduled crash
        # cannot shift later slippage, demand, or failure draws.
        random_crash = self.crash_rng.random() < crash_probability
        token_crash = bool(scheduled_crash or random_crash)
        if token_crash:
            token_growth *= crash_multiplier
        self.token_price *= token_growth
        if not np.isfinite(self.token_price) or self.token_price <= 0:
            raise FloatingPointError("token price became nonfinite or nonpositive")

        slippage_alpha = float(self.slippage_cfg.get("alpha", 2.0))
        slippage_beta = float(self.slippage_cfg.get("beta", 98.0))
        if slippage_alpha <= 0 or slippage_beta <= 0:
            raise ValueError("slippage Beta parameters must be positive")
        slippage = float(self.slippage_rng.beta(slippage_alpha, slippage_beta))
        if self._in_shock_window(self.slippage_cfg):
            slippage *= float(self.slippage_cfg.get("shock_multiplier", 1.0))
        if not 0.0 <= slippage < 1.0:
            raise ValueError("realized slippage must remain in [0, 1)")

        arrival_rate = float(self.demand_cfg.get("arrival_rate", 0.0))
        if self._in_shock_window(self.demand_cfg):
            arrival_rate *= float(self.demand_cfg.get("shock_multiplier", 1.0))
        if arrival_rate < 0:
            raise ValueError("demand.arrival_rate cannot be negative")
        arrivals = int(self.demand_count_rng.poisson(arrival_rate))
        size_min = float(self.demand_cfg.get("size_min", 1.0))
        size_max = float(self.demand_cfg.get("size_max", size_min))
        if size_min <= 0 or size_max < size_min:
            raise ValueError("invalid arriving dataset size range")
        demand_sizes = self.demand_size_rng.uniform(
            size_min, size_max, size=arrivals
        ).tolist()

        failure_probability = self.failure_probability
        failure_shock = self.config.get("failure_shock", {})
        if self._in_shock_window(failure_shock, month_key="month"):
            failure_probability = float(failure_shock.get("probability", failure_probability))
        if not 0.0 <= failure_probability <= 1.0:
            raise ValueError("effective miner failure probability must be in [0, 1]")
        failures = self.failure_rng.random(miner_count) < failure_probability

        state = {
            "yield_rate": effective_yield,
            "cost_growth": cost_growth,
            "cost_level": self.cost_level,
            "token_price": self.token_price,
            "slippage": slippage,
            "token_crash": int(token_crash),
            "demand_sizes": demand_sizes,
            "miner_failures": failures.tolist(),
        }
        self.step_count += 1
        return state
