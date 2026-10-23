class Dataset:
    def __init__(self, dataset_id: str, size: float, target_replicas: int):
        if not dataset_id:
            raise ValueError("dataset_id không được để trống")
        if size <= 0:
            raise ValueError("size phải lớn hơn 0")
        if target_replicas <= 0:
            raise ValueError("target_replicas phải lớn hơn 0")

        self.dataset_id = dataset_id
        self.size = size
        self.target_replicas = target_replicas
        self.miners = set()
        self.ever_stored = False

    @property
    def replicas(self) -> int:
        return len(self.miners)

    @property
    def replica_deficit(self) -> int:
        return max(0, self.target_replicas - self.replicas)

    @property
    def is_healthy(self) -> bool:
        return self.replicas >= self.target_replicas

    @property
    def is_degraded(self) -> bool:
        return 0 < self.replicas < self.target_replicas

    @property
    def is_lost(self) -> bool:
        return self.replicas == 0

    def add_miner(self, miner) -> bool:
        if miner in self.miners:
            return False
        self.miners.add(miner)
        self.ever_stored = True
        return True

    @property
    def is_permanently_lost(self) -> bool:
        return self.ever_stored and self.replicas == 0

    def remove_miner(self, miner) -> bool:
        if miner not in self.miners:
            return False
        self.miners.remove(miner)
        return True
