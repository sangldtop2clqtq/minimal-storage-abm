from config import CONFIG
from model import StorageModel


def main() -> None:
    model = StorageModel(config=CONFIG)
    results = model.run()

    print(f"Simulation completed: {len(results)} months, seed={model.seed}")
    if results:
        final = results[-1]
        print(f"Active miners: {final['active_miners']}")
        print(f"Data availability: {final['data_availability']:.1%}")
        print(f"Endowment: {final['endowment']:.2f}")


if __name__ == "__main__":
    main()
