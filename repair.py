"""State for repairs that take one or more model months to complete."""

from dataclasses import dataclass


@dataclass
class PendingRepair:
    repair_id: str
    dataset_id: str
    source_miner_id: int
    target_miner_id: int
    initiated_at: int
    completes_at: int
    promised_reward: float
    status: str = "PENDING"

