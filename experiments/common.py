"""Shared helpers for saving experiment output."""

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_run(
    experiment_name: str,
    run_name: str,
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a legacy single run into a new immutable run directory."""
    # Local import avoids a module cycle: output.py imports PROJECT_ROOT here.
    from experiments.output import (
        capture_provenance,
        create_run_directory,
        save_json,
        save_rows,
    )

    run_dir = create_run_directory(
        experiment_name, output_root=PROJECT_ROOT / "results"
    )
    csv_path = save_rows(run_dir / "monthly.csv", metrics)
    run_metadata = {
        "experiment": experiment_name,
        "run_name": run_name,
        "config": config,
        "provenance": capture_provenance(),
    }
    if metadata:
        run_metadata.update(metadata)
    json_path = save_json(run_dir / "metadata.json", run_metadata)
    return csv_path, json_path
