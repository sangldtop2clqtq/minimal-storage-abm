import pandas as pd

from analysis.plot_results import plot_paired_run, plot_stress_run


def test_paired_plotter_exports_png_and_pdf(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monthly = []
    summary = []
    for pair_id in (0, 1):
        for policy in ("fixed", "dynamic"):
            summary.append({
                "pair_id": pair_id, "policy": policy,
                "is_insolvent": pair_id, "any_permanent_loss": 0,
                "mean_completed_recovery_time": 1.0 + pair_id,
                "total_repair_expenditure": 2.0 + pair_id,
                "months_observed": 2,
            })
            for time in (0, 1):
                monthly.append({
                    "pair_id": pair_id, "policy": policy, "time": time,
                    "endowment": 100.0 - 10 * time,
                    "accessible_dataset_fraction": 1.0,
                    "replication_compliance_rate": 0.9,
                    "repair_pending": time,
                    "storage_obligation": 5.0,
                    "repair_cost": float(time),
                })
    pd.DataFrame(monthly).to_csv(run_dir / "monthly.csv", index=False)
    pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)

    figure_dir = plot_paired_run(run_dir)

    assert (figure_dir / "paired_trajectories.png").exists()
    assert (figure_dir / "paired_trajectories.pdf").exists()
    assert (figure_dir / "paired_outcomes.png").exists()
    assert (figure_dir / "paired_outcomes.pdf").exists()


def test_stress_plotter_exports_shock_overview(tmp_path):
    run_dir = tmp_path / "stress"
    run_dir.mkdir()
    rows = []
    for scenario in ("normal", "yield_drop"):
        for pair_id in (0, 1):
            for time in range(8):
                rows.append({
                    "scenario": scenario, "pair_id": pair_id,
                    "policy": "dynamic", "time": time,
                    "endowment": 100 - time,
                    "active_miners": 10,
                    "replication_compliance_rate": 1.0,
                    "token_price_usd": 1.0,
                })
    pd.DataFrame(rows).to_csv(run_dir / "monthly.csv", index=False)

    figure_dir = plot_stress_run(run_dir)

    assert (figure_dir / "stress_overview.png").exists()
    assert (figure_dir / "stress_overview.pdf").exists()
