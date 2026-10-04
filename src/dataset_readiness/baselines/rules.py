"""Deterministic findings with exact evidence; no repair or readiness claims."""

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from dataset_readiness.profiling.inspection import duplicate_groups, inspect_numeric, range_violations, scalar, validate_frame
from dataset_readiness.profiling.schemas import AssessmentSchema

RULE_VERSION = "rules_v1"


@dataclass(frozen=True)
class RuleIssue:
    issue_id: str
    issue_type: str
    level: Literal["cell", "row", "column", "dataset"]
    column: str | None
    rows: tuple[int, ...]
    description: str
    evidence: dict
    severity: Literal["info", "medium", "high"]
    agent: str = "rules"
    rule_version: str = RULE_VERSION


@dataclass(frozen=True)
class RuleReport:
    issues: tuple[RuleIssue, ...]
    schema_version: int = 1
    rule_version: str = RULE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def feature_missing_cells(self, target: str) -> set[tuple[int, str]]:
        return {(row, issue.column) for issue in self.issues if issue.issue_type == "missing_values" and issue.level == "cell" and issue.column is not None and issue.column != target for row in issue.rows}


def assess_rules(frame: pd.DataFrame, schema: AssessmentSchema) -> RuleReport:
    validate_frame(frame, schema)
    issues = []
    def add(kind, level, column, rows, description, evidence, severity="medium"):
        issues.append(RuleIssue(f"rule_{len(issues) + 1:04d}", kind, level, column, tuple(sorted(int(row) for row in rows)), description, evidence, severity))

    if schema.target not in frame:
        add("missing_target", "dataset", schema.target, (), "The declared target column is absent.", {}, "high")
    elif frame[schema.target].nunique(dropna=True) < 2:
        add("insufficient_target_classes", "column", schema.target, (), "Fewer than two observed target classes are available.", {"class_count": int(frame[schema.target].nunique(dropna=True))}, "high")
    for column in (*schema.numeric_features, *schema.categorical_features):
        if column not in frame:
            add("missing_feature", "column", column, (), "A declared feature column is absent.", {}, "high")
    for column in frame:
        series = frame[column]
        missing = series.isna()
        if missing.any():
            add("missing_target_values" if column == schema.target else "missing_values", "cell", column, frame.index[missing], "Observed missing values.", {"count": int(missing.sum()), "rate": float(missing.mean())}, "high" if column == schema.target else "medium")
        if column == schema.target:
            continue
        if series.nunique(dropna=True) == 1:
            add("constant_feature", "column", column, (), "All observed feature values are identical; utility requires task context.", {"observed_value": scalar(series.dropna().iloc[0]), "observed_count": int(series.notna().sum()), "missing_count": int(missing.sum())}, "info")
        if column in schema.numeric_features:
            numeric = inspect_numeric(series)
            for kind, mask, description in (
                ("numeric_type_violation", numeric.parse_failure, "Values cannot be parsed as real numbers; booleans are not accepted as numeric features."),
                ("nonfinite_numeric", numeric.nonfinite, "Nonfinite numeric values are present."),
            ):
                if mask.any():
                    add(kind, "cell", column, frame.index[mask], description, {"count": int(mask.sum())})
            for rule in schema.numeric_ranges:
                if rule.column == column:
                    invalid = range_violations(numeric, rule)
                    if invalid.any():
                        add("range_violation", "cell", column, frame.index[invalid], "Values violate a supplied inclusive range.", {**asdict(rule), "count": int(invalid.sum())})
    for group in duplicate_groups(frame):
        add("duplicate_records", "row", None, group, "Complete records are equal; no original row is inferred.", {"group_size": len(group), "excess_rows": len(group) - 1, "comparison": "all observed columns, including target; row ID excluded"}, "info")
    return RuleReport(tuple(issues))
