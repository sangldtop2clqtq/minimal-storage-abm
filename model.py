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
from metrics import (
    accessible_dataset_fraction,
    mean_replica_sufficiency,
    permanent_loss_count,
    replication_compliance_rate,
)
from random_streams import RandomStreams
from repair import PendingRepair
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
        self.initialization_rng = random.Random(self.seed)
        self.random_streams = RandomStreams(self.seed)
        self.time = 0
        self.miners: list[MinerAgent] = []
        self.datasets: list[Dataset] = []
        self.metrics: list[dict[str, Any]] = []
        self.stochastic_cfg = self.config.get("stochastic", {})
        self.stochastic_enabled = bool(self.stochastic_cfg.get("enabled", False))
        if self.stochastic_enabled and float(self.stochastic_cfg.get("dt", 1.0)) != 1.0:
            raise ValueError(
                "StorageModel uses one month per step; stochastic.dt must equal one month"
            )
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
        self.pending_repairs: list[PendingRepair] = []
        self.repair_history: list[PendingRepair] = []
        self._next_repair_id = 0
        self._repair_cancelled_this_step = 0
        self.repair_budget_rejections_this_step = 0
        self.dataset_degraded_since: dict[str, int] = {}
        self.permanent_loss_ids: set[str] = set()
        self.recovery_durations: list[int] = []
        self._recovery_durations_this_step: list[int] = []
        self.cash_flow: dict[str, float] = {
            "opening_endowment": 0.0,
            "yield_income": 0.0,
            "storage_obligation": 0.0,
            "repair_obligation": 0.0,
            "total_obligations": 0.0,
            "paid_obligations": 0.0,
            "obligation_shortfall": 0.0,
        }

        endowment_cfg = self.config.get("endowment", {})
        self.endowment = float(endowment_cfg.get("initial", 10_000.0))
        self.yield_rate = float(endowment_cfg.get("yield_rate", 0.003))
        self.insolvency_policy = str(
            endowment_cfg.get("insolvency_policy", "allow_negative")
        )
        if self.insolvency_policy not in {"allow_negative", "stop"}:
            raise ValueError(
                "endowment.insolvency_policy must be 'allow_negative' or 'stop'"
            )
        if self.insolvency_policy == "stop" and self.endowment < 0:
            raise ValueError("stop insolvency policy requires nonnegative initial Endowment")
        self.insolvent = False
        self.first_insolvency_month: int | None = None
        self.end_reason: str | None = None
        miner_cfg = self.config.get("miners", {})
        self.monthly_cost_decline = float(
            miner_cfg.get("monthly_cost_decline", 0.03)
        )
        if not 0.0 <= self.monthly_cost_decline < 1.0:
            raise ValueError("monthly_cost_decline phải nằm trong [0, 1)")

        self.create_miners()
        self.create_datasets()
        self.initialize_replication()
        self.validate_invariants()

    def update_dataset_health_tracking(self) -> list[int]:
        """Track degradation onset, permanent loss, and recovery durations."""
        recovered: list[int] = []
        for dataset in self.datasets:
            dataset_id = dataset.dataset_id
            if dataset.is_healthy:
                if dataset_id in self.dataset_degraded_since:
                    duration = self.time - self.dataset_degraded_since.pop(dataset_id)
                    recovered.append(duration)
                    self.recovery_durations.append(duration)
                continue
            if not dataset.ever_stored:
                continue
            self.dataset_degraded_since.setdefault(dataset_id, self.time)
            if dataset.is_permanently_lost:
                self.permanent_loss_ids.add(dataset_id)
        self._recovery_durations_this_step.extend(recovered)
        return recovered

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
                capacity=self.initialization_rng.uniform(capacity_min, capacity_max),
                unit_cost=self.initialization_rng.uniform(cost_min, cost_max),
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
                size=self.initialization_rng.uniform(size_min, size_max),
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
        dataset.ever_stored = True

    def _remove_replica(self, miner: MinerAgent, dataset: Dataset) -> None:
        """Remove a replica from both sides; reject inconsistent state."""
        if dataset not in miner.datasets or miner not in dataset.miners:
            raise ValueError("Quan hệ miner-dataset không đồng bộ")
        miner.datasets.remove(dataset)
        dataset.miners.remove(miner)

    def validate_invariants(self) -> None:
        """Raise when capacity or miner-dataset relationships are inconsistent."""
        for miner in self.miners:
            if (
                miner.used_capacity < 0
                or miner.reserved_capacity < 0
                or miner.used_capacity + miner.reserved_capacity > miner.capacity
            ):
                raise ValueError(f"Miner {miner.miner_id} violates capacity invariant")
            if miner.status != MinerAgent.ACTIVE and (
                miner.datasets or miner.reserved_capacity > 0
            ):
                raise ValueError(
                    f"Offline miner {miner.miner_id} still has storage commitments"
                )
            for dataset in miner.datasets:
                if miner not in dataset.miners:
                    raise ValueError("Miner-dataset relationship is not bidirectional")
        for dataset in self.datasets:
            for miner in dataset.miners:
                if dataset not in miner.datasets:
                    raise ValueError("Dataset-miner relationship is not bidirectional")

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
            candidates = self.random_streams.ordered(
                f"initial_replication:{dataset.dataset_id}",
                0,
                candidates,
                key=lambda miner: miner.miner_id,
            )
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
        self._repair_cancelled_this_step += self._cancel_repairs_for_miners(
            {miner.miner_id for miner, should_fail in zip(self.miners, draws) if should_fail}
        )
        return failed

    def _miner_by_id(self, miner_id: int) -> MinerAgent:
        return next(miner for miner in self.miners if miner.miner_id == miner_id)

    def _dataset_by_id(self, dataset_id: str) -> Dataset:
        return next(dataset for dataset in self.datasets if dataset.dataset_id == dataset_id)

    def _cancel_pending_repair(self, repair: PendingRepair) -> None:
        target = self._miner_by_id(repair.target_miner_id)
        dataset = self._dataset_by_id(repair.dataset_id)
        target.release_capacity(dataset.size)
        repair.status = "CANCELLED"
        self.repair_history.append(repair)

    def _cancel_repairs_for_miners(self, miner_ids: set[int]) -> int:
        cancelled = 0
        remaining: list[PendingRepair] = []
        for repair in self.pending_repairs:
            if (
                repair.source_miner_id in miner_ids
                or repair.target_miner_id in miner_ids
            ):
                self._cancel_pending_repair(repair)
                cancelled += 1
            else:
                remaining.append(repair)
        self.pending_repairs = remaining
        return cancelled

    def complete_pending_repairs(self) -> tuple[int, int, float, list[int]]:
        """Complete due repairs; cancel any whose source or target is offline."""
        completed = 0
        cancelled = 0
        payment = 0.0
        latencies: list[int] = []
        remaining: list[PendingRepair] = []
        for repair in self.pending_repairs:
            if repair.completes_at > self.time:
                remaining.append(repair)
                continue
            source = self._miner_by_id(repair.source_miner_id)
            target = self._miner_by_id(repair.target_miner_id)
            dataset = self._dataset_by_id(repair.dataset_id)
            if (
                source.status != MinerAgent.ACTIVE
                or target.status != MinerAgent.ACTIVE
                or source not in dataset.miners
            ):
                self._cancel_pending_repair(repair)
                cancelled += 1
                continue
            target.release_capacity(dataset.size)
            self._assign_replica(target, dataset)
            repair.status = "COMPLETED"
            self.repair_history.append(repair)
            completed += 1
            payment += repair.promised_reward
            latencies.append(self.time - repair.initiated_at)
        self.pending_repairs = remaining
        return completed, cancelled, payment, latencies

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
            candidates = self.random_streams.ordered(
                f"demand_placement:{dataset.dataset_id}",
                self.time,
                candidates,
                key=lambda miner: miner.miner_id,
            )
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
            and self.random_streams.uniform(
                "voluntary_exit", self.time, miner.miner_id
            ) < miner.leave_probability(beta)
        ]

        for miner in leaving:
            for dataset in list(miner.datasets):
                self._remove_replica(miner, dataset)
            miner.go_offline()
        self._repair_cancelled_this_step += self._cancel_repairs_for_miners(
            {miner.miner_id for miner in leaving}
        )
        self.validate_invariants()
        return leaving

    def process_forced_miner_exits(self, exit_rate: float) -> list[MinerAgent]:
        """Force a fraction of active miners offline and remove their replicas."""
        if not 0.0 <= exit_rate <= 1.0:
            raise ValueError("exit_rate phải nằm trong [0, 1]")

        active = [miner for miner in self.miners if miner.status == MinerAgent.ACTIVE]
        exit_count = round(len(active) * exit_rate)
        ordered = self.random_streams.ordered(
            "forced_exit", self.time, active, key=lambda miner: miner.miner_id
        )
        leaving = ordered[:exit_count]
        for miner in leaving:
            for dataset in list(miner.datasets):
                self._remove_replica(miner, dataset)
            miner.go_offline()
        self._repair_cancelled_this_step += self._cancel_repairs_for_miners(
            {miner.miner_id for miner in leaving}
        )
        self.validate_invariants()
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

        initiated = 0
        promised_reward = 0.0
        monthly_budget_value = cfg.get("monthly_budget")
        monthly_budget = (
            None if monthly_budget_value is None else float(monthly_budget_value)
        )
        if monthly_budget is not None and monthly_budget < 0:
            raise ValueError("repair.monthly_budget must be nonnegative or null")

        pending_counts: dict[str, int] = {}
        pending_targets: dict[str, set[int]] = {}
        for pending in self.pending_repairs:
            pending_counts[pending.dataset_id] = pending_counts.get(pending.dataset_id, 0) + 1
            pending_targets.setdefault(pending.dataset_id, set()).add(
                pending.target_miner_id
            )

        requests = sum(
            max(0, dataset.replica_deficit - pending_counts.get(dataset.dataset_id, 0))
            for dataset in self.datasets
        )

        for dataset in self.datasets:
            # A replica can only be copied from a surviving source. Datasets that
            # reach zero replicas require an external backup mechanism, which is
            # outside the Minimal ABM.
            if dataset.replicas == 0:
                continue
            uncovered = max(
                0,
                dataset.replica_deficit - pending_counts.get(dataset.dataset_id, 0),
            )
            if uncovered == 0:
                continue
            sources = [
                miner for miner in dataset.miners
                if miner.status == MinerAgent.ACTIVE
            ]
            sources = self.random_streams.ordered(
                f"repair_sources:{dataset.dataset_id}",
                self.time,
                sources,
                key=lambda miner: miner.miner_id,
            )
            if not sources:
                continue
            source = sources[0]
            excluded_targets = pending_targets.get(dataset.dataset_id, set())
            candidates = [
                miner
                for miner in self.miners
                if miner.status == MinerAgent.ACTIVE
                and dataset not in miner.datasets
                and miner.miner_id not in excluded_targets
                and miner.free_capacity >= dataset.size
            ]
            candidates = self.random_streams.ordered(
                f"repair_candidates:{dataset.dataset_id}",
                self.time,
                candidates,
                key=lambda miner: miner.miner_id,
            )
            reward = repair_reward(
                deficit=dataset.replica_deficit,
                target_replicas=dataset.target_replicas,
                base_reward=base_reward,
                alpha=alpha,
            )
            latency = int(cfg.get("latency_months", 1))
            if latency < 1:
                raise ValueError("repair.latency_months must be at least 1")
            for target in candidates:
                if initiated >= requests or uncovered <= 0:
                    break
                if (
                    monthly_budget is not None
                    and promised_reward + reward > monthly_budget + 1e-12
                ):
                    self.repair_budget_rejections_this_step += uncovered
                    break
                utility = reward - repair_cost_per_replica
                probability = repair_probability(utility, beta)
                draw = self.random_streams.uniform(
                    "repair_acceptance",
                    self.time,
                    dataset.dataset_id,
                    target.miner_id,
                    dataset.replica_deficit,
                )
                if draw >= probability:
                    continue
                target.reserve_capacity(dataset.size)
                repair = PendingRepair(
                    repair_id=f"repair-{self._next_repair_id}",
                    dataset_id=dataset.dataset_id,
                    source_miner_id=source.miner_id,
                    target_miner_id=target.miner_id,
                    initiated_at=self.time,
                    completes_at=self.time + latency,
                    promised_reward=reward,
                )
                self._next_repair_id += 1
                self.pending_repairs.append(repair)
                initiated += 1
                promised_reward += reward
                uncovered -= 1

        return requests, initiated, promised_reward

    def update_endowment(self, storage_reward: float, repair_cost: float) -> None:
        if self.yield_rate < -1.0:
            raise ValueError("yield_rate không thể nhỏ hơn -1")
        opening = self.endowment
        yield_income = opening * self.yield_rate
        total_obligations = storage_reward + repair_cost
        available = opening + yield_income
        if self.insolvency_policy == "stop":
            paid_obligations = min(max(available, 0.0), total_obligations)
            obligation_shortfall = total_obligations - paid_obligations
            self.endowment = max(0.0, available - paid_obligations)
            if obligation_shortfall > 0:
                self.insolvent = True
                if self.first_insolvency_month is None:
                    self.first_insolvency_month = self.time
                self.end_reason = "INSOLVENT"
        else:
            paid_obligations = total_obligations
            obligation_shortfall = 0.0
            self.endowment = available - paid_obligations
        self.cash_flow = {
            "opening_endowment": opening,
            "yield_income": yield_income,
            "storage_obligation": storage_reward,
            "repair_obligation": repair_cost,
            "total_obligations": total_obligations,
            "paid_obligations": paid_obligations,
            "obligation_shortfall": obligation_shortfall,
        }
        expected = (
            max(0.0, available - paid_obligations)
            if self.insolvency_policy == "stop"
            else opening + yield_income - paid_obligations
        )
        tolerance = 1e-9 * max(1.0, abs(expected))
        if abs(self.endowment - expected) > tolerance:
            raise AssertionError("Endowment cash flow does not reconcile")

    def collect_metrics(
        self,
        pricing_total_capacity: float,
        pricing_used_capacity: float,
        pricing_utilization: float,
        storage_price: float,
        repair_requests: int,
        successful_repairs: int,
        repair_cost: float,
        storage_reward: float,
        forced_exits: int,
        voluntary_exits: int,
        repair_initiated: int = 0,
        repair_cancelled: int = 0,
        repair_completion_latencies: list[int] | None = None,
        promised_repair_reward: float = 0.0,
        random_failures: int = 0,
        environment_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = [m for m in self.miners if m.status == MinerAgent.ACTIVE]
        replicas = [d.replicas for d in self.datasets]
        end_total_capacity = self.calculate_total_capacity()
        end_used_capacity = self.calculate_used_storage()
        end_utilization = self.calculate_utilization()
        metric: dict[str, Any] = {
            "time": self.time,
            "active_miners": len(active),
            "forced_exits": forced_exits,
            "voluntary_exits": voluntary_exits,
            "random_failures": random_failures,
            "pricing_total_capacity": pricing_total_capacity,
            "pricing_used_capacity": pricing_used_capacity,
            "pricing_utilization": pricing_utilization,
            "total_capacity": end_total_capacity,
            "used_capacity": end_used_capacity,
            "utilization": end_utilization,
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
            "accessible_dataset_fraction": accessible_dataset_fraction(self.datasets),
            "replication_compliance_rate": replication_compliance_rate(self.datasets),
            "mean_replica_sufficiency": mean_replica_sufficiency(self.datasets),
            "permanently_lost_datasets": permanent_loss_count(self.datasets),
            "never_stored_datasets": sum(
                not dataset.ever_stored for dataset in self.datasets
            ),
            "repair_requests": repair_requests,
            "repair_initiated": repair_initiated,
            "repair_completed": successful_repairs,
            "repair_cancelled": repair_cancelled,
            "repair_pending": len(self.pending_repairs),
            "mean_repair_latency": (
                sum(repair_completion_latencies) / len(repair_completion_latencies)
                if repair_completion_latencies else None
            ),
            "promised_repair_reward": promised_repair_reward,
            "repair_budget_rejections": self.repair_budget_rejections_this_step,
            "successful_repairs": successful_repairs,
            "repair_cost": repair_cost,
            "storage_reward": storage_reward,
            **self.cash_flow,
            "demand_arrivals": self._demand_arrivals,
            "fully_replicated_arrivals": self._demand_arrivals - sum(
                1 for d in self.datasets if d.dataset_id.startswith(f"dataset-demand-{self.time}-")
                and d.replica_deficit > 0
            ),
            "unserved_demand_gb": self._unserved_demand_gb,
            "endowment": self.endowment,
            "is_insolvent": int(self.insolvent),
            "first_insolvency_month": self.first_insolvency_month,
            "end_reason": self.end_reason,
            "mean_dataset_recovery_time": (
                sum(self._recovery_durations_this_step)
                / len(self._recovery_durations_this_step)
                if self._recovery_durations_this_step else None
            ),
        }
        if environment_state is not None:
            metric["cost_growth"] = float(environment_state["cost_growth"])
        self.metrics.append(metric)
        return metric

    def step(self, forced_exit_rate: float = 0.0) -> dict[str, Any]:
        """Advance a month; an optional forced exit shock occurs at month start."""
        environment_state = None
        random_failures = 0
        self._repair_cancelled_this_step = 0
        self.repair_budget_rejections_this_step = 0
        self._recovery_durations_this_step = []
        self._demand_arrivals = 0
        self._unserved_demand_gb = 0.0
        demand_sizes: list[float] = []
        if self.environment is not None:
            environment_state = self.environment.advance(len(self.miners))
            self.yield_rate = float(environment_state["yield_rate"])
            self.token_price = float(environment_state["token_price"])
            self.slippage = float(environment_state["slippage"])
            random_failures = self._apply_random_failures(
                environment_state["miner_failures"]
            )
            demand_sizes = environment_state["demand_sizes"]

        forced_exits = self.process_forced_miner_exits(forced_exit_rate)
        self.update_dataset_health_tracking()
        completed_repairs, cancelled_at_completion, repair_payment, repair_latencies = (
            self.complete_pending_repairs()
        )
        self.update_dataset_health_tracking()
        self._repair_cancelled_this_step += cancelled_at_completion
        self._demand_arrivals = len(demand_sizes)
        _, self._unserved_demand_gb = self._admit_demand(demand_sizes)
        self.update_dataset_health_tracking()
        pricing_total_capacity = self.calculate_total_capacity()
        pricing_used_capacity = self.calculate_used_storage()
        pricing_utilization = self.calculate_utilization()
        storage_price = self.update_storage_price(pricing_utilization)
        self.calculate_miner_economics(storage_price)

        # Storage reward is the total monthly operating payout in this minimal model.
        storage_reward = sum(
            miner.reward
            for miner in self.miners
            if miner.status == MinerAgent.ACTIVE
        )

        voluntary_exits = self.process_miner_exits()
        self.update_dataset_health_tracking()
        repair_requests, repair_initiated, promised_repair_reward = self.run_repair_game()
        settlement_cost = storage_reward / (1.0 - self.slippage)
        self.update_endowment(settlement_cost, repair_payment)
        result = self.collect_metrics(
            pricing_total_capacity=pricing_total_capacity,
            pricing_used_capacity=pricing_used_capacity,
            pricing_utilization=pricing_utilization,
            storage_price=storage_price,
            repair_requests=repair_requests,
            successful_repairs=completed_repairs,
            repair_cost=repair_payment,
            storage_reward=storage_reward,
            forced_exits=len(forced_exits),
            voluntary_exits=len(voluntary_exits),
            repair_initiated=repair_initiated,
            repair_cancelled=self._repair_cancelled_this_step,
            repair_completion_latencies=repair_latencies,
            promised_repair_reward=promised_repair_reward,
            random_failures=random_failures,
            environment_state=environment_state,
        )
        self.validate_invariants()
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

    def run(self, months: int | None = None) -> list[dict[str, Any]]:
        """Run the configured number of monthly steps and return all metrics."""
        if months is None:
            months = int(self.config.get("simulation", {}).get("months", 12))
        if months < 0:
            raise ValueError("months không được âm")
        for _ in range(months):
            self.step()
            if self.insolvency_policy == "stop" and self.insolvent:
                break
        return self.metrics
