"""Agent-based model for perpetual storage with optional stochastic inputs.

The model coordinates miners, datasets, pricing, exits, repairs, and the
endowment. Random choices use a model-local ``random.Random`` instance.
"""

from __future__ import annotations

import random
from typing import Any

from agent import MinerAgent
from dataset import Dataset
from mechanisms import bonding_curve, repair_probability, repair_reward
from stochastic_environment import StochasticEnvironment


class StorageModel:
    """Run the monthly storage-network simulation.

    ``config`` is a nested dictionary with ``simulation``, ``miners``,
    ``datasets``, ``bonding_curve``, ``miner_behavior``, ``repair``, and
    ``endowment`` sections. Missing sections use the defaults below.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int | None = None):
        self.config = config or {}
        simulation = self.config.get("simulation", {})
        self.seed = simulation.get("seed", 42) if seed is None else seed
        self.rng = random.Random(self.seed)
        self.time = 0
        self.miners: list[MinerAgent] = []
        self.datasets: list[Dataset] = []
        self.metrics: list[dict[str, float | int]] = []
        self.stochastic_cfg = self.config.get("stochastic", {})
        self.stochastic_enabled = bool(self.stochastic_cfg.get("enabled", False))
        self.environment = (
            StochasticEnvironment(self.stochastic_cfg, int(self.seed) + 1_000_003)
            if self.stochastic_enabled
            else None
        )
        self.token_price = float(
            self.stochastic_cfg.get("token_price", {}).get("initial", 1.0)
        )
        self.slippage = 0.0
        self._demand_arrivals = 0
        self._unserved_demand_gb = 0.0
        self._random_failures = 0

        endowment_cfg = self.config.get("endowment", {})
        self.endowment = float(endowment_cfg.get("initial", 10_000.0))
        self.yield_rate = float(endowment_cfg.get("yield_rate", 0.003))
        miner_cfg = self.config.get("miners", {})
        self.monthly_cost_decline = float(
            miner_cfg.get("monthly_cost_decline", 0.03)
        )
        if not 0.0 <= self.monthly_cost_decline < 1.0:
            raise ValueError("monthly_cost_decline phải nằm trong [0, 1)")

        self.create_miners()
        self.create_datasets()
        self.initialize_replication()

    def create_miners(self) -> None:
        """Create heterogeneous miners from the configured uniform ranges."""
        cfg = self.config.get("miners", {})
        count = int(cfg.get("count", 100))
        capacity_min = float(cfg.get("capacity_min", 100.0))
        capacity_max = float(cfg.get("capacity_max", 500.0))
        cost_min = float(cfg.get("cost_min", 0.5))
        cost_max = float(cfg.get("cost_max", 1.5))
        if count < 0 or capacity_min <= 0 or capacity_max < capacity_min:
            raise ValueError("Cấu hình miners không hợp lệ")
        if cost_min < 0 or cost_max < cost_min:
            raise ValueError("Khoảng unit_cost không hợp lệ")

        self.miners = [
            MinerAgent(
                miner_id=i,
                capacity=self.rng.uniform(capacity_min, capacity_max),
                unit_cost=self.rng.uniform(cost_min, cost_max),
            )
            for i in range(count)
        ]

    def create_datasets(self) -> None:
        """Create datasets with sizes sampled from configured uniform bounds."""
        cfg = self.config.get("datasets", {})
        count = int(cfg.get("count", 10))
        size_min = float(cfg.get("size_min", 10.0))
        size_max = float(cfg.get("size_max", 50.0))
        target_replicas = int(cfg.get("target_replicas", 5))
        if count < 0 or size_min <= 0 or size_max < size_min:
            raise ValueError("Cấu hình datasets không hợp lệ")
        if target_replicas <= 0:
            raise ValueError("target_replicas phải lớn hơn 0")

        self.datasets = [
            Dataset(
                dataset_id=f"dataset-{i}",
                size=self.rng.uniform(size_min, size_max),
                target_replicas=target_replicas,
            )
            for i in range(count)
        ]

    def _assign_replica(self, miner: MinerAgent, dataset: Dataset) -> None:
        """Update both sides of the miner-dataset relationship."""
        if dataset in miner.datasets:
            raise ValueError("Miner không thể giữ hai replica của cùng dataset")
        if dataset.size > miner.free_capacity:
            raise ValueError("Không đủ capacity để gán replica")
        miner.datasets.add(dataset)
        dataset.miners.add(miner)

    def _remove_replica(self, miner: MinerAgent, dataset: Dataset) -> None:
        """Remove a replica from both sides; reject inconsistent state."""
        if dataset not in miner.datasets or miner not in dataset.miners:
            raise ValueError("Quan hệ miner-dataset không đồng bộ")
        miner.datasets.remove(dataset)
        dataset.miners.remove(miner)

    def initialize_replication(self) -> None:
        """Assign each dataset its target replicas or fail clearly."""
        for dataset in self.datasets:
            candidates = [
                miner
                for miner in self.miners
                if miner.status == MinerAgent.ACTIVE
                and dataset not in miner.datasets
                and miner.free_capacity >= dataset.size
            ]
            self.rng.shuffle(candidates)
            if len(candidates) < dataset.target_replicas:
                raise ValueError(
                    f"Không đủ miner/capacity để khởi tạo {dataset.dataset_id}: "
                    f"cần {dataset.target_replicas} replica, chỉ có {len(candidates)} miner phù hợp"
                )
            for miner in candidates[: dataset.target_replicas]:
                self._assign_replica(miner, dataset)

    def calculate_total_capacity(self) -> float:
        return sum(
            miner.capacity
            for miner in self.miners
            if miner.status == MinerAgent.ACTIVE
        )

    def calculate_used_storage(self) -> float:
        return sum(dataset.size * dataset.replicas for dataset in self.datasets)

    def calculate_utilization(self) -> float:
        capacity = self.calculate_total_capacity()
        used = self.calculate_used_storage()
        if capacity == 0:
            if used > 0:
                raise ValueError("Còn dữ liệu nhưng tổng capacity hoạt động bằng 0")
            return 0.0
        utilization = used / capacity
        if not 0.0 <= utilization <= 1.0:
            raise ValueError(
                f"Utilization ngoài [0, 1] ({utilization:.4f}); mạng vượt capacity"
            )
        return utilization

    def update_storage_price(self, utilization: float) -> float:
        cfg = self.config.get("bonding_curve", {})
        return bonding_curve(
            rho=utilization,
            rho0=float(cfg.get("rho0", 0.75)),
            p_min=float(cfg.get("p_min", 0.5)),
            p_max=float(cfg.get("p_max", 2.0)),
            k=float(cfg.get("k", 15.0)),
        )

    def calculate_miner_economics(self, storage_price: float) -> None:
        for miner in self.miners:
            if miner.status == MinerAgent.ACTIVE:
                miner.calculate_profit(storage_price)

    def update_miner_costs(self) -> None:
        """Apply C_(t+1) = C_t * (1 - monthly_cost_decline)."""
        reduction_factor = 1.0 - self.monthly_cost_decline
        for miner in self.miners:
            miner.unit_cost *= reduction_factor

    def _apply_random_failures(self, draws: list[bool]) -> int:
        failed = 0
        for miner, should_fail in zip(self.miners, draws):
            if should_fail and miner.status == MinerAgent.ACTIVE:
                for dataset in list(miner.datasets):
                    self._remove_replica(miner, dataset)
                miner.go_offline()
                failed += 1
        return failed

    def _admit_demand(self, sizes: list[float]) -> tuple[int, float]:
        """Add arriving datasets and place as many replicas as capacity allows."""
        admitted = 0
        unserved_gb = 0.0
        cfg = self.config.get("datasets", {})
        target = int(cfg.get("target_replicas", 5))
        for size in sizes:
            dataset = Dataset(
                dataset_id=f"dataset-demand-{self.time}-{len(self.datasets)}",
                size=float(size),
                target_replicas=target,
            )
            self.datasets.append(dataset)
            candidates = [
                miner for miner in self.miners
                if miner.status == MinerAgent.ACTIVE
                and dataset not in miner.datasets
                and miner.free_capacity >= dataset.size
            ]
            self.rng.shuffle(candidates)
            for miner in candidates[:target]:
                self._assign_replica(miner, dataset)
            missing = dataset.replica_deficit
            if missing == 0:
                admitted += 1
            unserved_gb += missing * dataset.size
        return admitted, unserved_gb

    def process_miner_exits(self) -> list[MinerAgent]:
        """Decide exits from one shared snapshot, then apply them together."""
        cfg = self.config.get("miner_behavior", {})
        beta = float(cfg.get("beta_leave", 2.0))
        leaving = [
            miner
            for miner in self.miners
            if miner.status == MinerAgent.ACTIVE
            and self.rng.random() < miner.leave_probability(beta)
        ]

        for miner in leaving:
            for dataset in list(miner.datasets):
                self._remove_replica(miner, dataset)
            miner.go_offline()
        return leaving

    def process_forced_miner_exits(self, exit_rate: float) -> list[MinerAgent]:
        """Force a fraction of active miners offline and remove their replicas."""
        if not 0.0 <= exit_rate <= 1.0:
            raise ValueError("exit_rate phải nằm trong [0, 1]")

        active = [miner for miner in self.miners if miner.status == MinerAgent.ACTIVE]
        exit_count = round(len(active) * exit_rate)
        leaving = self.rng.sample(active, exit_count) if exit_count else []
        for miner in leaving:
            for dataset in list(miner.datasets):
                self._remove_replica(miner, dataset)
            miner.go_offline()
        return leaving

    def run_repair_game(self) -> tuple[int, int, float]:
        """Attempt repairs in shuffled eligible-miner order.

        Repair payments are represented as repair expenditure from the
        endowment. A successful replica receives the reward amount.
        """
        cfg = self.config.get("repair", {})
        base_reward = float(cfg.get("base_reward", 1.0))
        alpha = float(cfg.get("alpha", 2.0))
        beta = float(cfg.get("beta_repair", 2.0))
        repair_cost_per_replica = float(cfg.get("cost_per_replica", base_reward))
        if repair_cost_per_replica < 0:
            raise ValueError("repair.cost_per_replica không được âm")

        requests = sum(dataset.replica_deficit for dataset in self.datasets)
        successes = 0
        expenditure = 0.0

        for dataset in self.datasets:
            while dataset.replica_deficit > 0:
                candidates = [
                    miner
                    for miner in self.miners
                    if miner.status == MinerAgent.ACTIVE
                    and dataset not in miner.datasets
                    and miner.free_capacity >= dataset.size
                ]
                self.rng.shuffle(candidates)
                repaired = False
                reward = repair_reward(
                    deficit=dataset.replica_deficit,
                    target_replicas=dataset.target_replicas,
                    base_reward=base_reward,
                    alpha=alpha,
                )

                for miner in candidates:
                    utility = reward - repair_cost_per_replica
                    probability = repair_probability(utility, beta)
                    if self.rng.random() < probability:
                        self._assign_replica(miner, dataset)
                        successes += 1
                        expenditure += reward
                        repaired = True
                        break

                # No eligible miner or no miner accepts: avoid an infinite loop.
                if not repaired:
                    break

        return requests, successes, expenditure

    def update_endowment(self, storage_reward: float, repair_cost: float) -> None:
        if self.yield_rate < -1.0:
            raise ValueError("yield_rate không thể nhỏ hơn -1")
        self.endowment = (
            self.endowment * (1.0 + self.yield_rate)
            - storage_reward
            - repair_cost
        )

    def collect_metrics(
        self,
        utilization: float,
        storage_price: float,
        repair_requests: int,
        successful_repairs: int,
        repair_cost: float,
        storage_reward: float,
        forced_exits: int,
        voluntary_exits: int,
        random_failures: int = 0,
        environment_state: dict[str, Any] | None = None,
    ) -> dict[str, float | int]:
        active = [m for m in self.miners if m.status == MinerAgent.ACTIVE]
        replicas = [d.replicas for d in self.datasets]
        metric: dict[str, float | int] = {
            "time": self.time,
            "active_miners": len(active),
            "forced_exits": forced_exits,
            "voluntary_exits": voluntary_exits,
            "random_failures": random_failures,
            "total_capacity": self.calculate_total_capacity(),
            "used_capacity": self.calculate_used_storage(),
            "utilization": utilization,
            "storage_price": storage_price,
            "yield_rate": self.yield_rate,
            "token_price_usd": self.token_price,
            "token_crash": int(environment_state.get("token_crash", 0)) if environment_state else 0,
            "storage_reward_tokens": storage_reward / self.token_price,
            "slippage": self.slippage,
            "settlement_cost": storage_reward / (1.0 - self.slippage),
            "average_miner_profit": (
                sum(m.profit for m in active) / len(active) if active else 0.0
            ),
            "average_unit_cost": (
                sum(m.unit_cost for m in self.miners) / len(self.miners)
                if self.miners
                else 0.0
            ),
            "healthy_datasets": sum(d.is_healthy for d in self.datasets),
            "degraded_datasets": sum(d.is_degraded for d in self.datasets),
            "lost_datasets": sum(d.is_lost for d in self.datasets),
            "average_replica_count": (
                sum(replicas) / len(replicas) if replicas else 0.0
            ),
            "data_availability": (
                sum(min(1.0, d.replicas / d.target_replicas) for d in self.datasets)
                / len(self.datasets)
                if self.datasets
                else 1.0
            ),
            "repair_requests": repair_requests,
            "successful_repairs": successful_repairs,
            "repair_cost": repair_cost,
            "storage_reward": storage_reward,
            "demand_arrivals": self._demand_arrivals,
            "fully_replicated_arrivals": self._demand_arrivals - sum(
                1 for d in self.datasets if d.dataset_id.startswith(f"dataset-demand-{self.time}-")
                and d.replica_deficit > 0
            ),
            "unserved_demand_gb": self._unserved_demand_gb,
            "endowment": self.endowment,
        }
        if environment_state is not None:
            metric["cost_growth"] = float(environment_state["cost_growth"])
        self.metrics.append(metric)
        return metric

    def step(self, forced_exit_rate: float = 0.0) -> dict[str, float | int]:
        """Advance a month; an optional forced exit shock occurs at month start."""
        environment_state = None
        random_failures = 0
        self._demand_arrivals = 0
        self._unserved_demand_gb = 0.0
        if self.environment is not None:
            environment_state = self.environment.advance(len(self.miners))
            self.yield_rate = float(environment_state["yield_rate"])
            self.token_price = float(environment_state["token_price"])
            self.slippage = float(environment_state["slippage"])
            random_failures = self._apply_random_failures(
                environment_state["miner_failures"]
            )
            self._demand_arrivals = len(environment_state["demand_sizes"])
            _, self._unserved_demand_gb = self._admit_demand(
                environment_state["demand_sizes"]
            )

        forced_exits = self.process_forced_miner_exits(forced_exit_rate)
        utilization = self.calculate_utilization()
        storage_price = self.update_storage_price(utilization)
        self.calculate_miner_economics(storage_price)

        # Storage reward is the total monthly operating payout in this minimal model.
        storage_reward = sum(
            miner.reward
            for miner in self.miners
            if miner.status == MinerAgent.ACTIVE
        )

        voluntary_exits = self.process_miner_exits()
        repair_requests, successful_repairs, repair_cost = self.run_repair_game()
        settlement_cost = storage_reward / (1.0 - self.slippage)
        self.update_endowment(settlement_cost, repair_cost)
        result = self.collect_metrics(
            utilization=utilization,
            storage_price=storage_price,
            repair_requests=repair_requests,
            successful_repairs=successful_repairs,
            repair_cost=repair_cost,
            storage_reward=storage_reward,
            forced_exits=len(forced_exits),
            voluntary_exits=len(voluntary_exits),
            random_failures=random_failures,
            environment_state=environment_state,
        )
        # Deterministic decline remains available in the legacy mode. In
        # stochastic mode, the GBM increment replaces it to avoid double count.
        if environment_state is None:
            self.update_miner_costs()
        else:
            growth = float(environment_state["cost_growth"])
            for miner in self.miners:
                miner.unit_cost *= growth
        self.time += 1
        return result

    def run(self, months: int | None = None) -> list[dict[str, float | int]]:
        """Run the configured number of monthly steps and return all metrics."""
        if months is None:
            months = int(self.config.get("simulation", {}).get("months", 12))
        if months < 0:
            raise ValueError("months không được âm")
        for _ in range(months):
            self.step()
        return self.metrics
