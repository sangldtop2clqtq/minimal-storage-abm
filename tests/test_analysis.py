import math

import pandas as pd
import pytest

from analysis.analyze_results import (
    paired_differences,
    summarize_paired_effects,
    summarize_policy_risks,
    wilson_interval,
)


def fixture_summary():
    return pd.DataFrame([
        {"comparison": "A", "pair_id": 0, "policy": "fixed", "score": 1.0,
         "is_insolvent": 0, "any_permanent_loss": 0},
        {"comparison": "A", "pair_id": 0, "policy": "dynamic", "score": 2.0,
         "is_insolvent": 1, "any_permanent_loss": 0},
        {"comparison": "A", "pair_id": 1, "policy": "fixed", "score": 3.0,
         "is_insolvent": 1, "any_permanent_loss": 0},
        {"comparison": "A", "pair_id": 1, "policy": "dynamic", "score": 6.0,
         "is_insolvent": 1, "any_permanent_loss": 1},
    ])


def test_paired_differences_use_dynamic_minus_fixed_once_per_pair():
    differences = paired_differences(fixture_summary(), ["score"])

    assert differences["pair_id"].tolist() == [0, 1]
    assert differences["score"].tolist() == [1.0, 3.0]


def test_paired_effect_summary_matches_hand_calculated_mean_and_median():
    differences = paired_differences(fixture_summary(), ["score"])

    effects = summarize_paired_effects(
        differences, ["score"], bootstrap_samples=500, seed=5
    )

    row = effects.iloc[0]
    assert row["n_pairs_total"] == 2
    assert row["n_pairs_analyzed"] == 2
    assert row["mean_difference"] == pytest.approx(2.0)
    assert row["median_difference"] == pytest.approx(2.0)
    assert row["ci_lower"] <= 2.0 <= row["ci_upper"]
    assert row["uncertainty"] == "95% paired bootstrap CI for mean difference"


def test_missing_paired_outcome_is_counted_not_replaced_with_zero():
    frame = fixture_summary()
    frame.loc[(frame.pair_id == 1) & (frame.policy == "dynamic"), "score"] = math.nan
    differences = paired_differences(frame, ["score"])

    effects = summarize_paired_effects(differences, ["score"], bootstrap_samples=100)

    assert effects.iloc[0]["n_pairs_total"] == 2
    assert effects.iloc[0]["n_pairs_analyzed"] == 1
    assert effects.iloc[0]["n_pairs_missing"] == 1


def test_policy_risks_retain_all_runs_and_use_wilson_intervals():
    risks = summarize_policy_risks(
        fixture_summary(), ["is_insolvent", "any_permanent_loss"]
    )
    dynamic_loss = risks[
        (risks.policy == "dynamic") & (risks.outcome == "any_permanent_loss")
    ].iloc[0]

    assert dynamic_loss["events"] == 1
    assert dynamic_loss["n_runs"] == 2
    assert dynamic_loss["risk"] == pytest.approx(0.5)
    assert dynamic_loss["ci_lower"] < 0.5 < dynamic_loss["ci_upper"]
    assert wilson_interval(0, 0) == (math.nan, math.nan)
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == 1.0
