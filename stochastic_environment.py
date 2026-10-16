"""Monthly stochastic paths for the storage ABM.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class StochasticEnvironment:
    #Generate reproducible economic and demand shocks from one local RNG.

    def __init__(self, config: dict[str, Any], seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.dt = float(config.get("dt", 1.0))
        if self.dt <= 0:
            raise ValueError("stochastic.dt phải lớn hơn 0")

        self.yield_cfg = config.get("yield_process", {})
        self.cost_cfg = config.get("cost_process", {})
        self.token_cfg = config.get("token_price", {})
        self.slippage_cfg = config.get("slippage", {})
        self.demand_cfg = config.get("demand", {})
        self.failure_probability = float(config.get("miner_failure_probability", 0.0))
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("miner_failure_probability phải nằm trong [0, 1]")

        self.yield_rate = float(self.yield_cfg.get("initial", 0.008))
        self.cost_level = 1.0
        self.token_price = float(self.token_cfg.get("initial", 1.0))
        self.step_count = 0
        if self.token_price <= 0:
            raise ValueError("token_price.initial phải lớn hơn 0")
        if self.yield_rate < 0 or float(self.yield_cfg.get("long_run", 0.008)) < 0:
            raise ValueError("CIR initial và long_run không được âm")

        correlation = np.asarray(
            config.get("correlation", np.eye(3).tolist()), dtype=float
        )
        if correlation.shape != (3, 3) or not np.allclose(correlation, correlation.T):
            raise ValueError("stochastic.correlation phải là ma trận đối xứng 3x3")
        try:
            self.cholesky = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as exc:
            raise ValueError("stochastic.correlation phải xác định dương") from exc

    def advance(self, miner_count: int) -> dict[str, Any]:
        """Advance one month and return the environmental state for that month."""
        z_yield, z_cost, z_token = self.cholesky @ self.rng.standard_normal(3)

        # Full-truncation Euler keeps the CIR yield nonnegative.
        kappa = float(self.yield_cfg.get("kappa", 0.3))
        theta = float(self.yield_cfg.get("long_run", 0.008))
        sigma_y = float(self.yield_cfg.get("volatility", 0.001))
        if kappa <= 0 or theta < 0 or sigma_y < 0:
            raise ValueError("Tham số CIR không hợp lệ")
        current = max(0.0, self.yield_rate)
        self.yield_rate = max(
            0.0,
            current
            + kappa * (theta - current) * self.dt
            + sigma_y * np.sqrt(current) * np.sqrt(self.dt) * z_yield,
        )

        # Exact GBM step; drift is the arithmetic drift parameter mu.
        mu_cost = float(self.cost_cfg.get("drift", 0.0))
        sigma_cost = float(self.cost_cfg.get("volatility", 0.0))
        if sigma_cost < 0:
            raise ValueError("cost_process.volatility không được âm")
        cost_growth = float(np.exp(
            (mu_cost - 0.5 * sigma_cost**2) * self.dt
            + sigma_cost * np.sqrt(self.dt) * z_cost
        ))
        if not np.isfinite(cost_growth) or cost_growth <= 0:
            raise FloatingPointError("GBM cost tạo ra growth factor không hữu hạn/dương")
        self.cost_level *= cost_growth

        mu_token = float(self.token_cfg.get("drift", 0.0))
        sigma_token = float(self.token_cfg.get("volatility", 0.0))
        if sigma_token < 0:
            raise ValueError("token_price.volatility không được âm")
        token_growth = float(np.exp(
            (mu_token - 0.5 * sigma_token**2) * self.dt
            + sigma_token * np.sqrt(self.dt) * z_token
        ))
        crash_probability = float(self.token_cfg.get("crash_probability", 0.0))
        crash_multiplier = float(self.token_cfg.get("crash_multiplier", 0.1))
        if not 0.0 <= crash_probability <= 1.0 or not 0.0 < crash_multiplier <= 1.0:
            raise ValueError("Cấu hình token crash không hợp lệ")
        scheduled_crash = self.token_cfg.get("crash_month") == self.step_count
        token_crash = bool(scheduled_crash or self.rng.random() < crash_probability)
        if token_crash:
            token_growth *= crash_multiplier
        self.token_price *= token_growth
        if not np.isfinite(self.token_price) or self.token_price <= 0:
            raise FloatingPointError("Token price đã tràn số hoặc giảm về 0")

        if self.cost_cfg.get("shock_month") == self.step_count:
            shock_multiplier = float(self.cost_cfg.get("shock_multiplier", 1.0))
            if shock_multiplier <= 0:
                raise ValueError("cost_process.shock_multiplier phải lớn hơn 0")
            cost_growth *= shock_multiplier
            self.cost_level *= shock_multiplier
            if not np.isfinite(cost_growth) or not np.isfinite(self.cost_level):
                raise FloatingPointError("Storage cost shock tạo giá trị không hữu hạn")

        slippage_alpha = float(self.slippage_cfg.get("alpha", 2.0))
        slippage_beta = float(self.slippage_cfg.get("beta", 98.0))
        if slippage_alpha <= 0 or slippage_beta <= 0:
            raise ValueError("Tham số Beta của slippage phải lớn hơn 0")
        slippage = float(self.rng.beta(slippage_alpha, slippage_beta))

        arrival_rate = float(self.demand_cfg.get("arrival_rate", 0.0))
        if self.demand_cfg.get("shock_month") == self.step_count:
            arrival_rate *= float(self.demand_cfg.get("shock_multiplier", 1.0))
        if arrival_rate < 0:
            raise ValueError("demand.arrival_rate không được âm")
        arrivals = int(self.rng.poisson(arrival_rate))
        size_min = float(self.demand_cfg.get("size_min", 1.0))
        size_max = float(self.demand_cfg.get("size_max", size_min))
        if size_min <= 0 or size_max < size_min:
            raise ValueError("Khoảng kích thước dataset mới không hợp lệ")
        demand_sizes = self.rng.uniform(size_min, size_max, size=arrivals).tolist()

        failures = self.rng.random(miner_count) < self.failure_probability
        state = {
            "yield_rate": self.yield_rate,
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
