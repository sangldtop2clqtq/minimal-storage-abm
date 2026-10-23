import pytest

from model import StorageModel
from tests.test_model import small_config


def test_dataset_with_no_surviving_source_cannot_be_repaired():
    model = StorageModel(small_config())
    dataset = model.datasets[0]
    for miner in list(dataset.miners):
        model._remove_replica(miner, dataset)

    requests, successes, expenditure = model.run_repair_game()

    assert requests == dataset.target_replicas
    assert successes == 0
    assert expenditure == 0.0
    assert dataset.replicas == 0


def test_invariant_validator_rejects_one_sided_dataset_link():
    model = StorageModel(small_config())
    dataset = model.datasets[0]
    miner = next(iter(dataset.miners))
    miner.datasets.remove(dataset)

    with pytest.raises(ValueError, match="bidirectional"):
        model.validate_invariants()


def test_endowment_cash_flow_reconciles_recorded_components():
    config = small_config()
    config["miner_behavior"]["beta_leave"] = 0.0
    model = StorageModel(config)

    row = model.step()

    assert row["endowment"] == pytest.approx(
        row["opening_endowment"]
        + row["yield_income"]
        - row["paid_obligations"]
    )
    assert row["total_obligations"] == pytest.approx(
        row["storage_obligation"] + row["repair_obligation"]
    )
    assert row["obligation_shortfall"] == 0.0
