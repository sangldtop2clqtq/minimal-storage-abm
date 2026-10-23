import csv
import json

import pytest

from experiments import common
from experiments.output import (
    capture_provenance,
    create_run_directory,
    save_json,
    save_rows,
)


def test_existing_experiment_run_directory_is_never_overwritten(tmp_path):
    first = create_run_directory("paired", output_root=tmp_path, run_id="run-1")
    assert first.is_dir()
    with pytest.raises(FileExistsError):
        create_run_directory("paired", output_root=tmp_path, run_id="run-1")


def test_result_files_use_exclusive_creation(tmp_path):
    run_dir = create_run_directory("paired", output_root=tmp_path, run_id="run-1")
    save_rows(run_dir / "monthly.csv", [{"time": 0, "value": 1.5}])
    with pytest.raises(FileExistsError):
        save_rows(run_dir / "monthly.csv", [{"time": 1, "value": 2.0}])

    with (run_dir / "monthly.csv").open(encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == [{"time": "0", "value": "1.5"}]


def test_json_and_provenance_capture_reproducibility_fields(tmp_path):
    run_dir = create_run_directory("paired", output_root=tmp_path, run_id="run-1")
    provenance = capture_provenance()
    save_json(run_dir / "metadata.json", provenance)

    stored = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert stored["python_version"]
    assert stored["platform"]
    assert "git_commit" in stored
    assert "git_dirty" in stored
    assert "package_versions" in stored


def test_legacy_save_run_creates_unique_directories(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "PROJECT_ROOT", tmp_path)
    rows = [{"time": 0, "value": 1.0}]

    first_csv, first_metadata = common.save_run(
        "baseline", "seed_1", rows, {"simulation": {"seed": 1}}
    )
    second_csv, second_metadata = common.save_run(
        "baseline", "seed_1", rows, {"simulation": {"seed": 1}}
    )

    assert first_csv != second_csv
    assert first_metadata != second_metadata
    assert first_csv.exists() and second_csv.exists()
