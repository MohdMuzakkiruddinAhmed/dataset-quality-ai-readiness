import pandas as pd
import pytest

from dataset_readiness import readiness_challenge as policy
from examples.offline_readiness import synthetic_frame
from scripts.reproduce import summarize


def test_saved_summaries_match_manuscript():
    result = summarize()
    assert {k: round(v["mean_benefit_pp"], 3) for k, v in result["dataset_means"].items()} == {"adult": -.485, "diabetes": .315, "steel": -.108}
    methods = {r["method"]: r for r in result["assessor_summary"]}
    assert [(methods[m]["completed"], methods[m]["correct"]) for m in ("rules", "single_llm", "multi_agent")] == [(6, 4), (5, 4), (3, 3)]


@pytest.mark.parametrize("case", policy.CASES)
def test_original_case_generation_and_repair_protect_values(case):
    reference = synthetic_frame()
    before = reference.copy(deep=True)
    observed, manifest = policy.generate(reference, case)
    repeated, repeated_manifest = policy.generate(reference, case)
    pd.testing.assert_frame_equal(reference, before)
    pd.testing.assert_frame_equal(observed, repeated)
    assert manifest == repeated_manifest
    original_observed = observed.copy(deep=True)
    repaired, _, verification = policy.repair(observed)
    pd.testing.assert_frame_equal(observed, original_observed)
    pd.testing.assert_series_equal(repaired[policy.TASK.target], observed[policy.TASK.target])
    assert verification["unapproved_value_changes"] == 0
    assert verification["protected_target_and_structure"]


def test_label_policy_boundaries_and_abstention():
    reference = dict.fromkeys(policy.METRICS, .8)
    critical = {"unusable_target": False, "numeric_domain_violation": False, "all_predictors_unavailable": False}
    result = policy.label(reference, reference, reference, before_critical=critical, after_critical=critical, safe=True)
    assert result["label"] == "READY"
    domain = {**critical, "numeric_domain_violation": True}
    assert policy.label(reference, reference, reference, before_critical=domain, after_critical=critical, safe=True)["label"] == "REMEDIATE"
    assert policy.label(reference, None, None, before_critical=critical, after_critical=None, safe=None)["label"] == "ABSTAIN"
    assert policy.label(reference, None, None, before_critical={**critical, "unusable_target": True}, after_critical=None, safe=None)["label"] == "NOT_READY"


def test_replay_keeps_failed_predictions_in_denominator():
    truth = ["READY", "READY", "REMEDIATE", "REMEDIATE", "NOT_READY", "NOT_READY"]
    predictions = [None, None, "REMEDIATE", "REMEDIATE", None, "NOT_READY"]
    metrics = policy.readiness_metrics(truth, predictions)
    assert metrics["all_attempt_success_rate_determinate"] == .5
    assert metrics["completed_predictions_on_determinate_truth"] == 3
    assert metrics["macro_f1_completed_fixed_three_labels"] == pytest.approx(2 / 3)
