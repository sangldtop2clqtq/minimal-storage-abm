"""Pure metric functions for dataset availability and redundancy."""

from collections.abc import Sequence

from dataset import Dataset


def accessible_dataset_fraction(datasets: Sequence[Dataset]) -> float:
    if not datasets:
        return 1.0
    return sum(dataset.replicas > 0 for dataset in datasets) / len(datasets)


def replication_compliance_rate(datasets: Sequence[Dataset]) -> float:
    """Fraction of datasets meeting or exceeding their target replicas."""
    if not datasets:
        return 1.0
    return sum(dataset.is_healthy for dataset in datasets) / len(datasets)


def mean_replica_sufficiency(datasets: Sequence[Dataset]) -> float:
    if not datasets:
        return 1.0
    return sum(
        min(1.0, dataset.replicas / dataset.target_replicas)
        for dataset in datasets
    ) / len(datasets)


def permanent_loss_count(datasets: Sequence[Dataset]) -> int:
    return sum(dataset.is_permanently_lost for dataset in datasets)
