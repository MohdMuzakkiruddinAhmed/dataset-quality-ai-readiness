import json

import numpy as np
import pandas as pd
import pytest

from dataset_readiness.artifacts import json_bytes
from dataset_readiness.corruption import MissingnessConfig, inject_missing_values
from dataset_readiness.evaluation import (
    CellRef, EvaluationReport, cells_from_manifest, evaluate_detection,
    evaluate_downstream, evaluate_harmful_actions, evaluate_missing_repairs, score_items,
)


@pytest.mark.parametrize("predicted, truth, counts, scores", [
    ([1, 2, 3], [1, 2, 4], (2, 1, 1), (2/3, 2/3, 2/3)),
    ([1, 1, 1], [1], (1, 0, 0), (1, 1, 1)),
    ([], [1], (0, 0, 1), (None, 0, 0)),
    ([1], [], (0, 1, 0), (0, None, 0)),
    ([], [], (0, 0, 0), (None, None, None)),
])
def test_detection_counts_and_undefined_denominators(predicted, truth, counts, scores):
    result = score_items(predicted, truth)
    assert (result.tp, result.fp, result.fn) == counts
    assert (result.precision, result.recall, result.f1) == scores


def test_detection_projects_cell_predictions_without_inflating_cell_accuracy():
    report = evaluate_detection(
        [CellRef(10, "a"), CellRef(10, "b"), CellRef(10, "a")],
        [CellRef(10, "a"), CellRef(20, "b")], row_ids=[10, 20, 30], columns=["a", "b"],
    )
    assert report.cell.f1 == 0.5
    assert report.row.precision == 1
    assert report.row.recall == 0.5
    assert report.column.f1 == 1
    assert report.dataset.f1 == 1
    with pytest.raises(ValueError, match="outside"):
        evaluate_detection([CellRef(999, "a")], [], row_ids=[10], columns=["a"])
    with pytest.raises(ValueError, match="unique"):
        evaluate_detection([], [], row_ids=[10, 10], columns=["a"])


def test_manifest_adapter_and_integrity_checks(adult_frame):
    corrupted = inject_missing_values(
        adult_frame.drop(columns="income"), adult_frame.income,
        MissingnessConfig(0.1, 42, ("age", "occupation")),
    )
    manifest = json.loads(json_bytes(corrupted.manifest.to_dict()))
    coordinates = cells_from_manifest(manifest)
    assert len(coordinates) == 16
    assert coordinates == {CellRef(c.row_id, c.column) for c in corrupted.manifest.affected_cells}
    manifest["injected_cell_count"] += 1
    with pytest.raises(ValueError, match="count"):
        cells_from_manifest(manifest)
    manifest["affected_cells"].append(manifest["affected_cells"][0])
    with pytest.raises(ValueError, match="duplicate"):
        cells_from_manifest(manifest)


@pytest.fixture
def repair_case():
    reference = pd.DataFrame({"a": [10, 20, 30], "b": ["A", "B", "C"], "target": [0, 1, 0]}, index=[10, 20, 30])
    reference["a"] = reference["a"].astype("Int64")
    corrupted = reference.copy()
    corrupted.loc[[10, 30], "a"] = pd.NA
    corrupted.loc[20, "b"] = np.nan
    truth = [CellRef(10, "a"), CellRef(20, "b"), CellRef(30, "a")]
    repaired = corrupted.copy()
    repaired.loc[10, "a"] = 10
    repaired.loc[20, "b"] = "wrong"
    repaired.loc[10, "b"] = "overcleaned"
    return reference, corrupted, repaired, truth


def test_repair_separates_exact_restoration_wrong_fills_and_overcleaning(repair_case):
    reference, corrupted, repaired, truth = repair_case
    originals = [frame.copy(deep=True) for frame in repair_case[:3]]
    result = evaluate_missing_repairs(reference, corrupted, repaired, truth, target="target")
    assert result.known_defect_cells == 3
    assert result.correctly_restored_cells == 1
    assert result.remaining_defect_cells == 2
    assert result.remaining_missing_cells == 1
    assert result.incorrectly_filled_cells == 1
    assert result.originally_clean_cells == 3  # Excludes the target.
    assert result.changed_clean_cells == 1
    assert result.total_changed_cells == 3
    assert result.repair_accuracy == pytest.approx(1/3)
    assert result.remaining_defect_rate == pytest.approx(2/3)
    assert result.remaining_missing_rate == pytest.approx(1/3)
    assert result.overclean_rate == pytest.approx(1/3)
    for frame, original in zip(repair_case[:3], originals):
        pd.testing.assert_frame_equal(frame, original)
    reordered = repaired.iloc[::-1, ::-1]
    assert evaluate_missing_repairs(reference, corrupted, reordered, truth, target="target") == result


def test_repair_noop_and_oracle_are_only_validation_controls(repair_case):
    reference, corrupted, _, truth = repair_case
    noop = evaluate_missing_repairs(reference, corrupted, corrupted, truth, target="target")
    oracle = evaluate_missing_repairs(reference, corrupted, reference, truth, target="target")
    assert (noop.repair_accuracy, noop.remaining_defect_rate, noop.overclean_rate) == (0, 1, 0)
    assert (oracle.repair_accuracy, oracle.remaining_defect_rate, oracle.overclean_rate) == (1, 0, 0)


def test_repair_rejects_missing_truth_structural_changes_and_target_edits(repair_case):
    reference, corrupted, repaired, truth = repair_case
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_missing_repairs(reference, corrupted, repaired, truth[:-1], target="target")
    with pytest.raises(ValueError, match="additions or removals"):
        evaluate_missing_repairs(reference, corrupted, repaired.drop(index=10), truth, target="target")
    with pytest.raises(ValueError, match="additions or removals"):
        evaluate_missing_repairs(reference, corrupted, repaired.drop(columns="a"), truth, target="target")
    with pytest.raises(ValueError, match="protected target"):
        evaluate_missing_repairs(reference, corrupted, repaired.assign(target=1), truth, target="target")
    nonmissing_corruption = reference.copy()
    nonmissing_corruption.loc[10, "a"] = 999
    with pytest.raises(ValueError, match="observed-to-missing"):
        evaluate_missing_repairs(reference, nonmissing_corruption, reference, [CellRef(10, "a")], target="target")


def test_repair_empty_denominators_and_large_integers():
    reference = pd.DataFrame({"a": pd.Series([2**53 + 1], dtype="Int64"), "target": [1]})
    unchanged = evaluate_missing_repairs(reference, reference, reference, [], target="target")
    assert unchanged.repair_accuracy is None and unchanged.overclean_rate == 0
    corrupted = reference.copy()
    corrupted["a"] = pd.Series([pd.NA], dtype="Int64")
    rounded = reference.copy()
    rounded["a"] = float(2**53)
    result = evaluate_missing_repairs(reference, corrupted, rounded, [CellRef(0, "a")], target="target")
    assert result.repair_accuracy == 0 and result.incorrectly_filled_cells == 1
    assert result.overclean_rate is None


def test_repair_compares_missing_sentinels_and_different_category_vocabularies():
    reference = pd.DataFrame({"a": ["A", None, "B"], "target": [0, 1, 0]})
    reference["a"] = reference["a"].astype("category")
    corrupted = reference.copy()
    corrupted.loc[0, "a"] = np.nan
    repaired = pd.DataFrame({"a": pd.Categorical(["A", None, "B"], categories=["A", "B", "C"]), "target": [0, 1, 0]})
    result = evaluate_missing_repairs(reference, corrupted, repaired, [CellRef(0, "a")], target="target")
    assert result.repair_accuracy == 1
    assert result.overclean_rate == 0


@pytest.mark.parametrize("clean, corrupted, repaired, expected_rate", [
    (0.8, 0.6, 0.7, 0.5), (0.8, 0.6, 0.9, 1.5), (0.8, 0.6, 0.5, -0.5),
])
def test_recovery_supports_partial_above_one_and_harmful_results(clean, corrupted, repaired, expected_rate):
    result = evaluate_downstream(clean, corrupted, repaired)
    assert result.performance_loss == pytest.approx(0.2)
    assert result.recovery_rate == pytest.approx(expected_rate)
    assert result.recovery_rate_reason is None
    assert result.repair_harmed_performance == (expected_rate < 0)


@pytest.mark.parametrize("clean, corrupted, repaired, reason", [
    (0.8, 0.8, 0.9, "no_positive_performance_loss"),
    (0.8, 0.9, 0.85, "no_positive_performance_loss"),
    (0.8, 0.8 - 1e-14, 0.9, "no_positive_performance_loss"),
    (0.8, 0.6, None, "repair_not_evaluated"),
    (0.8, 0.9, None, "no_positive_performance_loss"),
])
def test_undefined_recovery_is_explicit(clean, corrupted, repaired, reason):
    result = evaluate_downstream(clean, corrupted, repaired)
    assert result.recovery_rate is None
    assert result.recovery_rate_reason == reason
    if repaired is None:
        assert result.performance_recovery is None
        assert result.repair_harmed_performance is None


def test_lower_is_better_metrics_and_action_harm():
    result = evaluate_downstream(10, 20, 15, metric="rmse", higher_is_better=False)
    assert (result.performance_loss, result.performance_recovery, result.recovery_rate) == (10, 5, 0.5)
    actions = evaluate_harmful_actions([(0.8, 0.7), (0.8, 0.9), (0.8, 0.8)])
    assert actions.harmful_actions == 1 and actions.evaluated_actions == 3
    assert actions.harmful_action_rate == pytest.approx(1/3)
    assert evaluate_harmful_actions([]).harmful_action_rate is None
    assert evaluate_harmful_actions([(10, 20)], higher_is_better=False).harmful_action_rate == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "0.8"])
def test_invalid_scores_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        evaluate_downstream(bad, 0.6, 0.7)
    with pytest.raises(ValueError, match="finite"):
        evaluate_downstream(0.8, 0.6, bad)


def test_evaluation_report_is_strict_json_with_nulls(repair_case):
    reference, corrupted, repaired, truth = repair_case
    report = EvaluationReport(
        downstream=evaluate_downstream(0.8, 0.9),
        detection=evaluate_detection([], truth, row_ids=reference.index, columns=["a", "b"]),
        repair=evaluate_missing_repairs(reference, corrupted, repaired, truth, target="target"),
    )
    serialized = json.loads(json_bytes(report.to_dict()))
    assert serialized["schema_version"] == 1
    assert serialized["detection"]["cell"]["precision"] is None
    assert serialized["downstream"]["recovery_rate"] is None
    assert serialized["harmful_actions"] is None
    assert "NaN" not in json_bytes(report.to_dict()).decode()
