"""Run the configured no-shock baseline simulation."""

from copy import deepcopy

from config import CONFIG
from experiments.common import save_run
from model import StorageModel


def main() -> None:
    config = deepcopy(CONFIG)
    seed = int(config["simulation"]["seed"])
    months = int(config["simulation"]["months"])

    model = StorageModel(config=config, seed=seed)
    metrics = model.run(months=months)
    run_name = f"baseline_seed_{seed}"
    csv_path, metadata_path = save_run(
        experiment_name="baseline",
        run_name=run_name,
        metrics=metrics,
        config=config,
        metadata={"months": months, "seed": seed, "shock": None},
    )

    print(f"Baseline completed: {months} months, seed={seed}")
    print(f"Metrics: {csv_path}")
    print(f"Metadata: {metadata_path}")
    if metrics:
        final = metrics[-1]
        print(f"Active miners: {final['active_miners']}")
        print(f"Data availability: {final['data_availability']:.1%}")
        print(f"Endowment: {final['endowment']:.2f}")


if __name__ == "__main__":
    main()
