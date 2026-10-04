"""Cell-preserving repairs. No reference data, truth, holdout, or model access."""

from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from numbers import Integral

import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

from dataset_readiness.artifacts import csv_bytes, json_bytes, sha256
from dataset_readiness.baselines.rules import assess_rules
from dataset_readiness.profiling import AssessmentSchema, ProfileOptions, build_profile
from dataset_readiness.profiling.inspection import scalar, validate_frame

VERSION = "missing_imputation_v1"
NUMERIC = "IMPUTE_NUMERIC_MEDIAN"
CATEGORICAL = "IMPUTE_CATEGORICAL_MODE"
POLICY = {
    "scope": "exact missing feature cells in accepted source findings only",
    "fit_scope": "all observed values in the original observed training column",
    "numeric": "exact middle value or mean of middle pair; reject fractional medians for integer dtype",
    "categorical": "string mode; ties choose lexicographically smallest Unicode string",
    "all_missing": "reject; no fallback constant or evaluator values",
    "overlap": "merge identical actions with all sources; reject conflicting overlapping proposals",
    "coarse_findings": "reject; do not expand column/row/dataset claims into cell edits",
    "dtype": "preserve every dtype; reject unrepresentable fill values",
}


@dataclass(frozen=True)
class Finding:
    issue_id: str
    issue_type: str
    level: str
    column: str | None
    rows: tuple[int, ...]
    action: str


@dataclass(frozen=True)
class RepairStep:
    step_id: str
    column: str
    action: str
    rows: tuple[int, ...]
    source_issue_ids: tuple[str, ...]
    fill_value: int | float | str
    observed_fit_count: int
    mode_tie_count: int | None


@dataclass(frozen=True)
class PlanDecision:
    source_issue_id: str
    proposed_action: str
    status: str
    reason: str
    step_id: str | None = None


@dataclass(frozen=True)
class RepairPlan:
    input_sha256: str
    findings_sha256: str
    task_schema: dict
    steps: tuple[RepairStep, ...]
    decisions: tuple[PlanDecision, ...]
    version: str = VERSION
    schema_version: int = 1

    def to_dict(self):
        return {**asdict(self), "policy": POLICY.copy()}


def _validate_inputs(frame, schema, findings):
    validate_frame(frame, schema)
    if set(frame.columns) != {schema.target, *schema.numeric_features, *schema.categorical_features}:
        raise ValueError("Repair requires the complete declared schema without extra columns")
    if not frame[schema.target].notna().all():
        raise ValueError("Repair requires a complete protected target")
    ids = set()
    for finding in findings:
        if not isinstance(finding, Finding) or not isinstance(finding.issue_id, str) or not finding.issue_id or finding.issue_id in ids:
            raise ValueError("Source findings require unique nonempty issue IDs")
        ids.add(finding.issue_id)
        if finding.level not in ("cell", "row", "column", "dataset"):
            raise ValueError("Invalid source finding level")
        if any(not isinstance(value, str) or not value for value in (finding.issue_type, finding.action)):
            raise ValueError("Invalid source finding type/action")
        if finding.column is not None and (not isinstance(finding.column, str) or not finding.column):
            raise ValueError("Invalid source column")
        if not isinstance(finding.rows, tuple) or any(isinstance(r, bool) or not isinstance(r, Integral) for r in finding.rows):
            raise ValueError("Source row IDs must be integer tuples")
        if len(set(finding.rows)) != len(finding.rows):
            raise ValueError("Source coordinates must not contain duplicates")


def _fit(series, action):
    observed = series.dropna()
    if observed.empty:
        raise ValueError("all_missing_column")
    if action == NUMERIC:
        if is_bool_dtype(series.dtype) or not (is_integer_dtype(series.dtype) or is_float_dtype(series.dtype)):
            raise ValueError("numeric_dtype_required")
        values = [int(value) if is_integer_dtype(series.dtype) else float(value) for value in observed]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("nonfinite_numeric_observation")
        values.sort()
        n = len(values)
        exact = (Fraction(values[(n - 1) // 2]) + Fraction(values[n // 2])) / 2
        if is_integer_dtype(series.dtype):
            if exact.denominator != 1:
                raise ValueError("fractional_median_for_integer")
            fill = int(exact)
        else:
            fill = float(exact)
        ties = None
    else:
        values = [scalar(value) for value in observed]
        if any(type(value) is not str for value in values):
            raise ValueError("string_categories_required")
        counts = Counter(values)
        winners = sorted(value for value, count in counts.items() if count == max(counts.values()))
        fill, ties = winners[0], len(winners)
    # Probe in isolation; never convert the full column or round observed integers.
    probe = series.loc[series.isna()].iloc[:1].copy()
    try:
        probe.iloc[0] = fill
    except (TypeError, ValueError, OverflowError):
        raise ValueError("fill_not_representable") from None
    if probe.dtype != series.dtype or scalar(probe.iloc[0]) != fill:
        raise ValueError("fill_not_representable")
    return fill, len(observed), ties


def build_plan(frame: pd.DataFrame, schema: AssessmentSchema, findings: tuple[Finding, ...]) -> RepairPlan:
    """Record every proposal, accepting only supported, fully grounded edits."""
    _validate_inputs(frame, schema, findings)
    findings = tuple(sorted(findings, key=lambda item: item.issue_id))
    decisions, accepted = {}, {}
    conflicts = set()
    cell_proposals = [f for f in findings if f.level == "cell" and f.column is not None]
    for i, left in enumerate(cell_proposals):
        for right in cell_proposals[i + 1:]:
            if left.column == right.column and left.action != right.action and set(left.rows) & set(right.rows):
                conflicts.update((left.issue_id, right.issue_id))
    for finding in findings:
        reason = None
        if finding.issue_id in conflicts:
            reason = "conflicting_overlapping_actions"
        elif finding.action == "NO_ACTION":
            decisions[finding.issue_id] = PlanDecision(finding.issue_id, finding.action, "skipped", "no_action_proposed")
            continue
        elif finding.action not in (NUMERIC, CATEGORICAL):
            reason = "operation_not_supported"
        elif finding.issue_type != "missing_values":
            reason = "missing_values_issue_required"
        elif finding.column == schema.target:
            reason = "protected_target"
        elif finding.column not in frame.columns:
            reason = "unknown_feature"
        elif finding.level != "cell" or not finding.rows:
            reason = "exact_cell_coordinates_required"
        elif not set(finding.rows) <= set(frame.index):
            reason = "unknown_row_ids"
        elif (finding.action == NUMERIC) != (finding.column in schema.numeric_features):
            reason = "action_schema_mismatch"
        elif not frame.loc[list(finding.rows), finding.column].isna().all():
            reason = "nonmissing_cell_overwrite"
        if reason is None:
            try:
                fit = _fit(frame[finding.column], finding.action)
                for rule in schema.numeric_ranges:
                    if rule.column == finding.column and ((rule.minimum is not None and fit[0] < rule.minimum)
                                                          or (rule.maximum is not None and fit[0] > rule.maximum)):
                        raise ValueError("fill_outside_declared_range")
            except ValueError as exc:
                reason = str(exc)
        if reason:
            decisions[finding.issue_id] = PlanDecision(finding.issue_id, finding.action, "rejected", reason)
        else:
            accepted.setdefault((finding.column, finding.action), []).append((finding, fit))
    steps = []
    for (column, action), sources in sorted(accepted.items()):
        rows = tuple(sorted({int(row) for finding, _ in sources for row in finding.rows}))
        source_ids = tuple(finding.issue_id for finding, _ in sources)
        step_id = "step_" + sha256(json_bytes([column, action, rows, source_ids]))[:24]
        fill, count, ties = sources[0][1]
        steps.append(RepairStep(step_id, column, action, rows, source_ids, fill, count, ties))
        for finding, _ in sources:
            decisions[finding.issue_id] = PlanDecision(finding.issue_id, finding.action, "accepted", "validated_observed_missing_cells", step_id)
    return RepairPlan(sha256(csv_bytes(frame)), sha256(json_bytes([{**asdict(f), "rows": [int(r) for r in f.rows]} for f in findings])),
                      asdict(schema), tuple(steps), tuple(decisions[f.issue_id] for f in findings))


def validate_plan(plan, frame, schema, findings):
    # Re-derive fitting and provenance at the execution boundary: no caller may
    # change rows, operations, fill values, source references, policy, or schema.
    expected = build_plan(frame, schema, findings)
    try:
        matches = isinstance(plan, RepairPlan) and json_bytes(plan.to_dict()) == json_bytes(expected.to_dict())
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError("Repair plan does not match the observed input and source findings")


def execute_plan(frame, schema, findings, plan):
    """Validate the whole plan before any mutation, then modify a private copy."""
    validate_plan(plan, frame, schema, findings)
    before = csv_bytes(frame)
    repaired = frame.copy(deep=True)
    by_id = {finding.issue_id: set(finding.rows) for finding in findings}
    changes = []
    for step in plan.steps:
        repaired.loc[list(step.rows), step.column] = step.fill_value
        for row in step.rows:
            changes.append({"row_id": row, "column": step.column, "before": None,
                            "after": scalar(repaired.at[row, step.column]), "step_id": step.step_id,
                            "source_issue_ids": [sid for sid in step.source_issue_ids if row in by_id[sid]]})
    if csv_bytes(frame) != before:
        raise ValueError("Repair mutated the source input")
    manifest = {"schema_version": 1, "executor_version": VERSION,
                "plan_sha256": sha256(json_bytes(plan.to_dict())), "input_sha256": sha256(before),
                "repaired_sha256": sha256(csv_bytes(repaired)), "changed_cell_count": len(changes), "changes": changes}
    return repaired, manifest


def _changed_cells(before, after):
    result = set()
    for column in before:
        both = before[column].notna() & after[column].notna()
        equal = before[column].isna() & after[column].isna()
        equal.loc[both] = before.loc[both, column].astype(object).eq(after.loc[both, column].astype(object)).to_numpy()
        result.update((int(row), column) for row in before.index[~equal])
    return result


def verify_repairs(before, after, schema, findings, plan, manifest, options=ProfileOptions()):
    """Check exact edits independently, then re-profile and report remaining/new observations."""
    validate_plan(plan, before, schema, findings)
    if (not before.index.equals(after.index) or before.index.dtype != after.index.dtype
            or before.index.name != after.index.name or not before.columns.equals(after.columns)):
        raise ValueError("Verification failed: row IDs or column order changed")
    if not before.dtypes.equals(after.dtypes) or not before[schema.target].equals(after[schema.target]):
        raise ValueError("Verification failed: dtype or protected target changed")
    expected = {(row, step.column): step for step in plan.steps for row in step.rows}
    actual = _changed_cells(before, after)
    if actual != set(expected):
        raise ValueError("Verification failed: unintended or missing edits")
    if manifest["input_sha256"] != sha256(csv_bytes(before)) or manifest["repaired_sha256"] != sha256(csv_bytes(after)) or manifest["plan_sha256"] != sha256(json_bytes(plan.to_dict())):
        raise ValueError("Verification failed: artifact hash mismatch")
    entries = {(entry["row_id"], entry["column"]): entry for entry in manifest["changes"]}
    if len(entries) != len(manifest["changes"]) or set(entries) != actual or manifest["changed_cell_count"] != len(actual):
        raise ValueError("Verification failed: manifest coordinates/count")
    sources = {finding.issue_id: set(finding.rows) for finding in findings}
    for coordinate, step in expected.items():
        row, column = coordinate
        entry = entries[coordinate]
        expected_entry = dict(row_id=row, column=column, before=None, after=step.fill_value, step_id=step.step_id,
                              source_issue_ids=[sid for sid in step.source_issue_ids if row in sources[sid]])
        if not pd.isna(before.at[row, column]) or scalar(after.at[row, column]) != step.fill_value or entry != expected_entry:
            raise ValueError("Verification failed: wrong fill or manifest provenance")
    before_rules, after_rules = assess_rules(before, schema), assess_rules(after, schema)
    key = lambda issue: (issue.issue_type, issue.column, issue.rows)
    existing = {key(issue) for issue in before_rules.issues}
    new_issues = [asdict(issue) for issue in after_rules.issues if key(issue) not in existing]
    original_status = []
    decision_by_id = {d.source_issue_id: d for d in plan.decisions}
    for finding in findings:
        status = {"issue_id": finding.issue_id, "plan_status": decision_by_id[finding.issue_id].status,
                  "verification_status": "not_rechecked", "remaining_missing_cells": None}
        if finding.issue_type == "missing_values" and finding.column in after and finding.level == "cell" and set(finding.rows) <= set(after.index):
            count = int(after.loc[list(finding.rows), finding.column].isna().sum())
            status.update(verification_status="resolved_missingness" if not count else "remaining_missingness", remaining_missing_cells=count)
        elif finding.issue_type == "missing_values" and finding.column in after and finding.level == "column":
            status.update(verification_status="remaining_missingness" if after[finding.column].isna().any() else "resolved_missingness",
                          remaining_missing_cells=int(after[finding.column].isna().sum()))
        elif finding.issue_type == "duplicate_records" and finding.rows and set(finding.rows) <= set(after.index):
            group = after.loc[list(finding.rows)]
            status["verification_status"] = "still_identical" if len(group.drop_duplicates()) == 1 else "no_longer_identical"
        original_status.append(status)
    profile = build_profile(after, schema, options)
    verification = {"schema_version": 1, "status": "passed", "verification_version": VERSION,
                    "checks": {"row_ids_and_order_preserved": True, "columns_and_order_preserved": True,
                               "target_preserved": True, "dtypes_preserved": True, "only_planned_missing_cells_changed": True,
                               "manifest_exact": True},
                    "changed_cell_count": len(actual), "remaining_feature_missing_cells": profile["feature_missing_count"],
                    "original_issue_status": original_status, "new_rule_findings": new_issues,
                    "new_rule_finding_count": len(new_issues),
                    "new_findings_scope": "New exact rule signatures; observational flags, not certified defects or readiness labels",
                    "dtype_before": {c: str(before[c].dtype) for c in before},
                    "dtype_after": {c: str(after[c].dtype) for c in after}}
    return verification, profile, after_rules.to_dict()
