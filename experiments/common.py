"""Shared helpers for saving experiment output."""

import csv
import json
from datetime import datetime, timezone
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
    """Write monthly metrics as CSV and run configuration as JSON."""
    raw_dir = PROJECT_ROOT / "results" / "raw" / experiment_name
    metadata_dir = PROJECT_ROOT / "results" / "metadata" / experiment_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / f"{run_name}.csv"
    json_path = metadata_dir / f"{run_name}.json"

    if metrics:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
    else:
        csv_path.write_text("", encoding="utf-8")

    run_metadata = {
        "experiment": experiment_name,
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    if metadata:
        run_metadata.update(metadata)
    json_path.write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return csv_path, json_path
