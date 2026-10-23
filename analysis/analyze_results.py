"""Paired run-level analysis for fixed-versus-dynamic repair experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CONTINUOUS_OUTCOMES = [
    "mean_completed_recovery_time",
    "mean_replication_compliance_rate",
    "mean_accessible_dataset_fraction",
    "mean_replica_sufficiency",
    "permanent_loss_count",
    "total_repair_expenditure",
    "total_storage_expenditure",
    "total_obligation_shortfall",
    "months_observed",
    "final_endowment",
]


def paired_differences(
    summary: pd.DataFrame, outcomes: Iterable[str]
) -> pd.DataFrame:
    """Return one dynamic-minus-fixed difference per pair and comparison."""
    outcomes = list(outcomes)
    required = {"pair_id", "policy", *outcomes}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    frame = summary.copy()
    if "comparison" not in frame:
        frame["comparison"] = "unspecified"
    duplicated = frame.duplicated(["comparison", "pair_id", "policy"])
    if duplicated.any():
        raise ValueError("each comparison/pair/policy must have one summary row")
    policies = set(frame["policy"].unique())
    if not policies <= {"fixed", "dynamic"}:
        raise ValueError("policy must be fixed or dynamic")
    pivot = frame.pivot(
        index=["comparison", "pair_id"], columns="policy", values=outcomes
    )
    if not {"fixed", "dynamic"} <= set(pivot.columns.get_level_values(1)):
        raise ValueError("every analysis requires fixed and dynamic policies")
    records: list[dict[str, object]] = []
    for index, values in pivot.iterrows():
        record: dict[str, object] = {
            "comparison": index[0],
            "pair_id": index[1],
        }
        for outcome in outcomes:
            record[outcome] = values[(outcome, "dynamic")] - values[(outcome, "fixed")]
        records.append(record)
    return pd.DataFrame(records)


def summarize_paired_effects(
    differences: pd.DataFrame,
    outcomes: Iterable[str],
    bootstrap_samples: int = 5_000,
    seed: int = 20261023,
) -> pd.DataFrame:
    """Summarize paired differences with a run-level percentile bootstrap CI."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    comparisons = (
        differences["comparison"].drop_duplicates().tolist()
        if "comparison" in differences else ["unspecified"]
    )
    for comparison in comparisons:
        group = (
            differences[differences["comparison"] == comparison]
            if "comparison" in differences else differences
        )
        for outcome in outcomes:
            values = pd.to_numeric(group[outcome], errors="coerce").dropna().to_numpy(float)
            n_total = len(group)
            n = len(values)
            if n:
                sampled = rng.choice(values, size=(bootstrap_samples, n), replace=True)
                bootstrap_means = sampled.mean(axis=1)
                lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
                mean_value = float(values.mean())
                median_value = float(np.median(values))
            else:
                lower = upper = mean_value = median_value = math.nan
            records.append({
                "comparison": comparison,
                "outcome": outcome,
                "contrast": "dynamic - fixed",
                "n_pairs_total": n_total,
                "n_pairs_analyzed": n,
                "n_pairs_missing": n_total - n,
                "mean_difference": mean_value,
                "median_difference": median_value,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "uncertainty": "95% paired bootstrap CI for mean difference",
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": seed,
            })
    return pd.DataFrame(records)


def wilson_interval(events: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial risk."""
    if n == 0:
        return math.nan, math.nan
    if events < 0 or events > n:
        raise ValueError("events must be between zero and n")
    p = events / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    lower = 0.0 if events == 0 else max(0.0, center - half)
    upper = 1.0 if events == n else min(1.0, center + half)
    return lower, upper


def summarize_policy_risks(
    summary: pd.DataFrame, outcomes: Iterable[str]
) -> pd.DataFrame:
    """Estimate per-policy event risks while retaining all trajectory rows."""
    records: list[dict[str, object]] = []
    group_keys = ["policy"]
    if "comparison" in summary:
        group_keys.insert(0, "comparison")
    for keys, group in summary.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = dict(zip(group_keys, keys))
        for outcome in outcomes:
            values = pd.to_numeric(group[outcome], errors="raise")
            if values.isna().any() or not values.isin([0, 1]).all():
                raise ValueError(f"{outcome} must be complete binary data")
            events = int(values.sum())
            n = len(values)
            lower, upper = wilson_interval(events, n)
            records.append({
                **labels,
                "outcome": outcome,
                "events": events,
                "n_runs": n,
                "risk": events / n if n else math.nan,
                "ci_lower": lower,
                "ci_upper": upper,
                "uncertainty": "95% Wilson binomial interval",
            })
    return pd.DataFrame(records)


def _write_markdown_table(path: Path, frame: pd.DataFrame) -> None:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_paired_run(
    run_dir: str | Path,
    bootstrap_samples: int = 5_000,
    seed: int = 20261023,
    output_name: str = "analysis",
) -> Path:
    """Analyze an immutable paired run and create a new analysis directory."""
    run_path = Path(run_dir)
    summary = pd.read_csv(run_path / "summary.csv")
    outcomes = [name for name in DEFAULT_CONTINUOUS_OUTCOMES if name in summary]
    differences = paired_differences(summary, outcomes)
    effects = summarize_paired_effects(
        differences, outcomes, bootstrap_samples=bootstrap_samples, seed=seed
    )
    binary = [name for name in ("is_insolvent", "any_permanent_loss") if name in summary]
    risks = summarize_policy_risks(summary, binary)

    output = run_path / output_name
    output.mkdir(exist_ok=False)
    differences.to_csv(output / "paired_differences.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)
    risks.to_csv(output / "policy_risks.csv", index=False)
    _write_markdown_table(output / "paired_effects.md", effects)
    _write_markdown_table(output / "policy_risks.md", risks)
    (output / "analysis_metadata.json").write_text(
        json.dumps({
            "source_summary": str((run_path / "summary.csv").resolve()),
            "independent_unit": "pair / simulated trajectory",
            "contrast": "dynamic - fixed",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "recovery_missing_policy": (
                "No completed recovery remains missing; it is not replaced with zero. "
                "Unrecovered episode counts are retained separately."
            ),
        }, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output-name", default="analysis")
    args = parser.parse_args()
    print(analyze_paired_run(
        args.run_dir, args.bootstrap_samples, args.seed, args.output_name
    ))


if __name__ == "__main__":
    main()
