import pytest

from dataset import Dataset
from metrics import (
    accessible_dataset_fraction,
    mean_replica_sufficiency,
    permanent_loss_count,
    replication_compliance_rate,
)


def dataset_with_replicas(dataset_id, replicas, target, ever_stored=None):
    dataset = Dataset(dataset_id, size=1.0, target_replicas=target)
    dataset.miners.update(object() for _ in range(replicas))
    dataset.ever_stored = replicas > 0 if ever_stored is None else ever_stored
    return dataset


def test_availability_redundancy_and_loss_metrics_are_distinct():
    datasets = [
        dataset_with_replicas("healthy", 2, 2),
        dataset_with_replicas("degraded", 1, 2),
        dataset_with_replicas("never-stored", 0, 2, ever_stored=False),
        dataset_with_replicas("permanent-loss", 0, 2, ever_stored=True),
    ]

    assert accessible_dataset_fraction(datasets) == pytest.approx(0.5)
    assert replication_compliance_rate(datasets) == pytest.approx(0.25)
    assert mean_replica_sufficiency(datasets) == pytest.approx(0.375)
    assert permanent_loss_count(datasets) == 1
