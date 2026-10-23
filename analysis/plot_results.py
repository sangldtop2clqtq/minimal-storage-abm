"""Reproducible, run-level scientific figures for experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.analyze_results import wilson_interval


POLICIES = ("fixed", "dynamic")
COLORS = {"fixed": "#0072B2", "dynamic": "#D55E00"}
LINESTYLES = {"fixed": "--", "dynamic": "-"}


def _save_both(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def _band(ax: plt.Axes, frame: pd.DataFrame, field: str, ylabel: str) -> None:
    for policy in POLICIES:
        group = frame[frame.policy == policy]
        aggregate = group.groupby("time")[field].agg(
            median="median",
            lower=lambda x: x.quantile(0.10),
            upper=lambda x: x.quantile(0.90),
            n="count",
        )
        ax.plot(
            aggregate.index, aggregate["median"], label=policy.capitalize(),
            color=COLORS[policy], linestyle=LINESTYLES[policy], linewidth=1.8,
        )
        ax.fill_between(
            aggregate.index, aggregate["lower"], aggregate["upper"],
            color=COLORS[policy], alpha=0.16,
        )
    ax.set(xlabel="Model month", ylabel=ylabel)
    ax.grid(alpha=0.25)


def plot_paired_run(run_dir: str | Path, figure_dir_name: str = "figures") -> Path:
    """Plot paired trajectories and run-level outcomes as PNG and PDF."""
    run_path = Path(run_dir)
    monthly = pd.read_csv(run_path / "monthly.csv")
    summary = pd.read_csv(run_path / "summary.csv")
    output = run_path / figure_dir_name
    output.mkdir(exist_ok=False)

    trajectory = monthly.sort_values(["policy", "pair_id", "time"]).copy()
    trajectory["cumulative_storage_cost"] = trajectory.groupby(
        ["policy", "pair_id"]
    )["storage_obligation"].cumsum()
    trajectory["cumulative_repair_cost"] = trajectory.groupby(
        ["policy", "pair_id"]
    )["repair_cost"].cumsum()

    with plt.rc_context({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        _band(axes[0, 0], trajectory, "endowment", "Endowment (USD)")
        axes[0, 0].set_title("A  Endowment")
        _band(
            axes[0, 1], trajectory, "replication_compliance_rate", "RCR (fraction)"
        )
        axes[0, 1].set_ylim(0, 1.02)
        axes[0, 1].set_title("B  Replication compliance")
        _band(axes[1, 0], trajectory, "repair_pending", "Pending repairs (count)")
        axes[1, 0].set_title("C  Repair backlog")
        _band(
            axes[1, 1], trajectory, "cumulative_storage_cost",
            "Cumulative storage obligation (USD)",
        )
        axes[1, 1].set_title("D  Storage cost")
        axes[0, 0].legend(
            title="Median; shading = 10th–90th percentiles", frameon=False,
            loc="upper right",
        )
        _save_both(fig, output, "paired_trajectories")

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        x = np.arange(2)
        width = 0.34
        for offset, policy in enumerate(POLICIES):
            group = summary[summary.policy == policy]
            risks = []
            lower_errors = []
            upper_errors = []
            for outcome in ("is_insolvent", "any_permanent_loss"):
                events = int(group[outcome].sum())
                risk = events / len(group)
                low, high = wilson_interval(events, len(group))
                risks.append(risk)
                lower_errors.append(risk - low)
                upper_errors.append(high - risk)
            axes[0, 0].bar(
                x + (offset - 0.5) * width, risks, width,
                yerr=np.array([lower_errors, upper_errors]), capsize=3,
                color=COLORS[policy], alpha=0.85, label=policy.capitalize(),
            )
        axes[0, 0].set_xticks(x, ["Insolvency", "Permanent loss"])
        axes[0, 0].set_ylim(0, 1.08)
        axes[0, 0].set_ylabel("Risk (fraction; 95% Wilson interval)")
        axes[0, 0].set_title("A  Run-level risks")
        axes[0, 0].legend(frameon=False)

        for index, (field, title, ylabel) in enumerate([
            ("mean_completed_recovery_time", "B  Completed recovery time", "Months"),
            ("total_repair_expenditure", "C  Paid repair expenditure", "USD/run"),
            ("months_observed", "D  Observed duration", "Months/run"),
        ]):
            ax = axes.flat[index + 1]
            values = [summary.loc[summary.policy == p, field].dropna() for p in POLICIES]
            ax.boxplot(values, tick_labels=["F", "D"], showfliers=True)
            for position, policy_values in enumerate(values, start=1):
                ax.scatter(
                    np.full(len(policy_values), position), policy_values,
                    s=10, alpha=0.35, color=COLORS[POLICIES[position - 1]],
                )
            missing = [int(summary.loc[summary.policy == p, field].isna().sum()) for p in POLICIES]
            ax.set(title=f"{title}\nmissing: fixed={missing[0]}, dynamic={missing[1]}", ylabel=ylabel)
            ax.grid(axis="y", alpha=0.25)
        _save_both(fig, output, "paired_outcomes")

    (output / "README.txt").write_text(
        "Source: ../monthly.csv and ../summary.csv\n"
        "Independent unit: simulated trajectory/pair.\n"
        "Trajectory bands are 10th–90th percentiles, not confidence intervals.\n"
        "Missing recovery times mean no completed recovery and are not zeros.\n",
        encoding="utf-8",
    )
    return output


def plot_sensitivity_run(
    run_dir: str | Path, figure_dir_name: str = "figures"
) -> Path:
    """Plot OAT sensitivity summaries with trajectory-level distributions."""
    run_path = Path(run_dir)
    summary = pd.read_csv(run_path / "summary.csv")
    output = run_path / figure_dir_name
    output.mkdir(exist_ok=False)
    ordered = summary["setting_id"].drop_duplicates().tolist()
    labels = [value.replace("_", " ").replace("-", ": ") for value in ordered]
    grouped = summary.groupby("setting_id", sort=False)
    duration = grouped["months_observed"].agg(
        median="median", lower=lambda x: x.quantile(0.10), upper=lambda x: x.quantile(0.90)
    ).reindex(ordered)
    rcr = grouped["mean_replication_compliance_rate"].agg(
        median="median", lower=lambda x: x.quantile(0.10), upper=lambda x: x.quantile(0.90)
    ).reindex(ordered)
    insolvency = grouped["is_insolvent"].mean().reindex(ordered)
    utilization = grouped["realized_initial_utilization"].median().reindex(ordered)

    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
        y = np.arange(len(ordered))
        for ax, data, title, ylabel in [
            (axes[0, 0], duration, "A  Observed duration", "Months"),
            (axes[0, 1], rcr, "B  Replication compliance", "Mean RCR"),
        ]:
            ax.errorbar(
                data["median"], y,
                xerr=np.vstack([data["median"] - data["lower"], data["upper"] - data["median"]]),
                fmt="o", color="#0072B2", ecolor="#56B4E9", capsize=2,
            )
            ax.set(title=title, xlabel=ylabel)
            ax.grid(axis="x", alpha=0.25)
        axes[1, 0].barh(y, insolvency, color="#D55E00")
        axes[1, 0].set(title="C  Insolvency risk", xlabel="Fraction of runs", xlim=(0, 1.05))
        axes[1, 1].barh(y, utilization, color="#009E73")
        axes[1, 1].set(title="D  Realized initial utilization", xlabel="Fraction")
        for ax in axes.flat:
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
        fig.suptitle(
            "One-factor-at-a-time exploratory screen\n"
            "Points/bands: median and 10th–90th trajectory percentiles"
        )
        _save_both(fig, output, "sensitivity_screen")
    (output / "README.txt").write_text(
        "Source: ../summary.csv. OAT results do not estimate interactions.\n"
        "Intervals are trajectory percentiles, not confidence intervals.\n",
        encoding="utf-8",
    )
    return output


def plot_stress_run(
    run_dir: str | Path, figure_dir_name: str = "figures"
) -> Path:
    """Plot dynamic-policy median paths for every prespecified stress scenario."""
    run_path = Path(run_dir)
    monthly = pd.read_csv(run_path / "monthly.csv")
    monthly = monthly[monthly.policy == "dynamic"]
    output = run_path / figure_dir_name
    output.mkdir(exist_ok=False)
    scenarios = monthly["scenario"].drop_duplicates().tolist()
    colors = plt.colormaps["tab10"](np.linspace(0, 0.9, len(scenarios)))
    styles = ["-", "--", "-.", ":"]
    panels = [
        ("endowment", "A  Endowment", "USD"),
        ("active_miners", "B  Active miners", "Count"),
        ("replication_compliance_rate", "C  Replication compliance", "RCR"),
        ("token_price_usd", "D  Token price", "USD/STRAT"),
    ]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.subplots_adjust(right=0.78, wspace=0.30, hspace=0.32)
        for ax, (field, title, ylabel) in zip(axes.flat, panels):
            for index, scenario in enumerate(scenarios):
                group = monthly[monthly.scenario == scenario].groupby("time")[field].median()
                ax.plot(
                    group.index, group.values, color=colors[index],
                    linestyle=styles[index % len(styles)], linewidth=1.6,
                    label=scenario.replace("_", " "),
                )
            ax.axvline(6, color="#333333", linestyle=":", linewidth=1.2)
            ax.set(title=title, xlabel="Model month", ylabel=ylabel)
            ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.80, 0.5), frameon=False)
        fig.suptitle(
            "Stress scenarios, dynamic policy — median trajectory\n"
            "Vertical dotted line: scheduled shock month 6",
            y=0.98,
        )
        _save_both(fig, output, "stress_overview")
    (output / "README.txt").write_text(
        "Source: ../monthly.csv. Lines are medians across 20 trajectories for the dynamic policy.\n"
        "No uncertainty band is drawn in this crowded overview; raw trajectories remain in monthly.csv.\n"
        "The vertical dotted line marks scheduled shock month 6.\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--kind", choices=("paired", "sensitivity", "stress"), default="paired"
    )
    parser.add_argument("--output-name", default="figures")
    args = parser.parse_args()
    functions = {
        "paired": plot_paired_run,
        "sensitivity": plot_sensitivity_run,
        "stress": plot_stress_run,
    }
    function = functions[args.kind]
    print(function(args.run_dir, args.output_name))


if __name__ == "__main__":
    main()
