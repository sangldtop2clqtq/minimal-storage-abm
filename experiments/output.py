"""Immutable experiment outputs and reproducibility metadata."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from experiments.common import PROJECT_ROOT


def _safe_component(value: str, field: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{field} must be one safe path component")
    return value


def create_run_directory(
    experiment: str,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    experiment = _safe_component(experiment, "experiment")
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    run_id = _safe_component(run_id, "run_id")
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "results"
    run_dir = root / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return destination


def save_json(path: str | Path, value: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
    return destination


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def capture_provenance() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "matplotlib", "pytest"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    status = _git_value("status", "--porcelain")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": packages,
    }
