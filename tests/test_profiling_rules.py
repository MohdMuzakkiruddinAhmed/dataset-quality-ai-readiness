import builtins
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset_readiness.artifacts import json_bytes
from dataset_readiness.baselines.rules import assess_rules
from dataset_readiness.profiling import AssessmentSchema, NumericRange, ProfileOptions, build_profile


@pytest.fixture
def observations():
    frame = pd.DataFrame({
        "number": ["1", "2", None, "oops", np.inf, True, "-3", "100", "1", "1"],
        "category": ["Private", " Private ", "private", "Private", None, "B", "Private", "B", "Private", "Private"],
        "constant": [7] * 9 + [None], "empty": [None] * 10,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }, index=pd.Index(range(10, 110, 10), name="row_id"))
    schema = AssessmentSchema("target", ("number", "constant", "empty"), ("category",))
    return frame, schema


def test_numeric_statistics_parseability_and_original_values(observations):
    frame, schema = observations
    original = frame.copy(deep=True)
    profile = build_profile(frame, schema)
    summary = profile["columns"]["number"]
    expected = pd.Series([1, 2, -3, 100, 1, 1], dtype=float)
    assert summary["missing_count"] == 1
    assert summary["missing_rate"] == 0.1
    assert summary["numeric"]["parse_failure_count"] == 2
    assert summary["numeric"]["nonfinite_count"] == 1
    assert summary["numeric"]["finite_count"] == 6
    for key, value in {"mean": expected.mean(), "std": expected.std(ddof=1), "median": expected.median(), "minimum": -3, "maximum": 100}.items():
        assert summary["numeric"][key] == pytest.approx(value)
    assert summary["numeric"]["q1"] == expected.quantile(0.25)
    assert summary["numeric"]["q3"] == expected.quantile(0.75)
    assert summary["numeric"]["outlier_candidate_count"] == 2
    assert profile["feature_missing_count"] == int(frame.drop(columns="target").isna().sum().sum())
    assert profile["target"]["class_counts"] == [{"value": 0, "count": 5}, {"value": 1, "count": 5}]
    assert profile["target"]["imbalance_ratio"] == 1
    assert profile["constant_features"] == ["constant"]
    assert profile["columns"]["empty"]["numeric"]["mean"] is None
    assert profile["columns"]["empty"]["all_missing"] is True
    pd.testing.assert_frame_equal(frame, original)
    assert "NaN" not in json_bytes(profile).decode()
    nonfinite_sample = profile["samples"]["nonfinite"][0]
    assert nonfinite_sample["values"]["number"] == {"nonfinite": "infinity"}


def test_categories_are_measured_without_normalization(observations):
    frame, schema = observations
    stats = build_profile(frame, schema, ProfileOptions(rare_category_threshold=0.2))["columns"]["category"]["categorical"]
    assert stats["top_categories"][0] == {"value": "Private", "count": 5}
    assert stats["whitespace_affected_row_count"] == 1
    assert stats["whitespace_variants"]["group_count"] == 1
    assert stats["case_variants"]["group_count"] == 1
    assert stats["rare_category_count"] == 2
    assert stats["rare_row_count"] == 2
    assert frame.loc[20, "category"] == " Private "


def test_rules_report_known_evidence_and_only_declared_ranges(observations):
    frame, schema = observations
    report = assess_rules(frame, schema)
    numeric = [issue for issue in report.issues if issue.column == "number"]
    types = {issue.issue_type: issue for issue in numeric}
    assert types["numeric_type_violation"].rows == (40, 60)
    assert types["nonfinite_numeric"].rows == (50,)
    assert types["missing_values"].rows == (30,)
    assert "range_violation" not in types
    assert not any("outlier" in issue.issue_type for issue in report.issues)
    assert {issue.column for issue in report.issues if issue.issue_type == "constant_feature"} == {"constant"}
    assert (30, "number") in report.feature_missing_cells("target")
    rule = NumericRange("number", 0, 1000, "Synthetic fixture constraint")
    ranged = assess_rules(frame, replace(schema, numeric_ranges=(rule,)))
    violations = [issue for issue in ranged.issues if issue.issue_type == "range_violation"]
    assert len(violations) == 1 and violations[0].rows == (70,)
    assert violations[0].evidence["source"] == rule.source
    assert build_profile(frame, replace(schema, numeric_ranges=(rule,)))["columns"]["number"]["declared_range"]["violation_count"] == 1
    json.loads(json_bytes(report.to_dict()))


def test_duplicates_include_target_and_exclude_row_ids():
    frame = pd.DataFrame({"number": pd.array([1, 1, 1, None, None], dtype="Int64"), "category": ["A", "A", "A", None, None], "target": [0, 0, 1, 1, 1]}, index=[11, 22, 33, 44, 55])
    schema = AssessmentSchema("target", ("number",), ("category",))
    profile = build_profile(frame, schema)
    assert profile["duplicates"]["group_count"] == 2
    assert profile["duplicates"]["excess_row_count"] == 2
    assert profile["duplicates"]["member_row_count"] == 4
    duplicate_issues = [issue for issue in assess_rules(frame, schema).issues if issue.issue_type == "duplicate_records"]
    assert [issue.rows for issue in duplicate_issues] == [(11, 22), (44, 55)]


@pytest.mark.parametrize("target_values, status", [([None, None], "all_missing"), ([0, None], "contains_missing"), ([1, 1], "single_class"), ([], "empty")])
def test_target_edge_cases(target_values, status):
    frame = pd.DataFrame({"number": range(len(target_values)), "target": target_values})
    schema = AssessmentSchema("target", ("number",), ())
    profile = build_profile(frame, schema)
    assert profile["target"]["status"] == status
    json_bytes(profile)
    report = assess_rules(frame, schema)
    assert not report.feature_missing_cells("target")
    if not target_values:
        assert profile["columns"]["number"]["missing_rate"] is None
        assert profile["columns"]["number"]["numeric"]["std"] is None


def test_absent_target_missing_features_and_unexpected_features():
    frame = pd.DataFrame({"unexpected": [3, 4]})
    schema = AssessmentSchema("target", ("number",), ())
    profile = build_profile(frame, schema)
    assert profile["target"]["status"] == "absent"
    assert profile["schema"] == {"missing_features": ["number"], "unexpected_features": ["unexpected"]}
    assert {issue.issue_type for issue in assess_rules(frame, schema).issues} == {"missing_target", "missing_feature"}


def test_reports_bound_high_cardinality_output_and_reproduce_samples():
    frame = pd.DataFrame({"category": [f"id_{i}" for i in range(100)], "target": [0, 1] * 50})
    schema = AssessmentSchema("target", (), ("category",))
    options = ProfileOptions(sample_size=2, category_limit=3, rare_category_threshold=0.02)
    state = np.random.get_state()
    first = build_profile(frame, schema, options)
    repeated = build_profile(frame, schema, options)
    assert json_bytes(first) == json_bytes(repeated)
    stats = first["columns"]["category"]["categorical"]
    assert len(stats["top_categories"]) == len(stats["rare_category_examples"]) == 3
    assert stats["frequencies_truncated"] is True
    assert stats["rare_category_count"] == stats["rare_row_count"] == 100
    assert all(len(sample) <= 2 for sample in first["samples"].values())
    assert build_profile(frame, schema, replace(options, sample_seed=18))["samples"]["random"] != first["samples"]["random"]
    assert all(not rows for rows in build_profile(frame, schema, replace(options, sample_size=0))["samples"].values())
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


def test_pure_assessors_need_no_files_or_evaluator_data(monkeypatch, observations):
    frame, schema = observations
    def forbidden(*args, **kwargs):
        pytest.fail("Pure assessors must not read files")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    build_profile(frame, schema)
    assess_rules(frame, schema)


@pytest.mark.parametrize("change", [{"sample_size": -1}, {"sample_size": 1000}, {"category_limit": 0}, {"sample_seed": True}, {"rare_category_threshold": float("nan")}, {"rare_category_threshold": 2}])
def test_invalid_profile_options_are_rejected(change):
    with pytest.raises(ValueError):
        ProfileOptions(**change)


def test_schema_and_row_ids_are_validated(observations):
    frame, schema = observations
    with pytest.raises(ValueError, match="distinct"):
        AssessmentSchema("target", ("number",), ("number",))
    with pytest.raises(ValueError, match="source"):
        NumericRange("number", 0, 100, "")
    with pytest.raises(ValueError, match="exceeds"):
        NumericRange("number", 5, 1, "fixture")
    with pytest.raises(ValueError, match="unique"):
        build_profile(pd.concat([frame, frame]), schema)
    with pytest.raises(ValueError, match="index"):
        build_profile(frame.reset_index(), schema)
    with pytest.raises(ValueError, match="integer"):
        assess_rules(frame.set_axis(frame.index.astype(str)), schema)
