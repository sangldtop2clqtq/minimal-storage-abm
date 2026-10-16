"""Run a baseline with a one-time forced miner exit shock."""

from copy import deepcopy

from config import CONFIG
from experiments.common import save_run
from model import StorageModel


def main() -> None:
    config = deepcopy(CONFIG)
    seed = int(config["simulation"]["seed"])
    months = int(config["simulation"]["months"])
    shock_month = 6
    exit_rate = 0.30

    model = StorageModel(config=config, seed=seed)
    metrics = []
    for _ in range(months):
        rate = exit_rate if model.time == shock_month else 0.0
        metrics.append(model.step(forced_exit_rate=rate))

    run_name = f"miner_exit_{int(exit_rate * 100)}pct_month_{shock_month}_seed_{seed}"
    csv_path, metadata_path = save_run(
        experiment_name="miner_exit",
        run_name=run_name,
        metrics=metrics,
        config=config,
        metadata={
            "months": months,
            "seed": seed,
            "shock": {"month": shock_month, "forced_exit_rate": exit_rate},
        },
    )

    print(
        f"Miner-exit experiment completed: {exit_rate:.0%} forced exit "
        f"at model month {shock_month}, seed={seed}"
    )
    print(f"Metrics: {csv_path}")
    print(f"Metadata: {metadata_path}")
    if metrics:
        final = metrics[-1]
        print(f"Active miners: {final['active_miners']}")
        print(f"Data availability: {final['data_availability']:.1%}")
        print(f"Endowment: {final['endowment']:.2f}")


if __name__ == "__main__":
    main()
