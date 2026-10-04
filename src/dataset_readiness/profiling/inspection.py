"""Shared read-only observations used by the profiler and deterministic rules."""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
import pandas as pd

from dataset_readiness.profiling.schemas import AssessmentSchema, NumericRange


def scalar(value: object) -> object:
    """Encode scalar observations without inventing values for nonfinite numbers."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            return {"nonfinite": "infinity" if value > 0 else "negative_infinity"}
    if type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"Unsupported tabular scalar: {type(value).__name__}")


def validate_frame(frame: pd.DataFrame, schema: AssessmentSchema) -> None:
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Assessment requires unique row IDs and column names")
    if any(isinstance(row, (bool, np.bool_)) or not isinstance(row, Integral) for row in frame.index):
        raise ValueError("Assessment requires integer row IDs")
    if any(not isinstance(column, str) or not column for column in frame.columns):
        raise ValueError("Assessment columns must be nonempty strings")
    if schema.row_id_column in frame.columns:
        raise ValueError("Audit row IDs must be the index, not an assessed feature")


@dataclass(frozen=True)
class NumericObservation:
    parsed: pd.Series
    finite: pd.Series
    parse_failure: pd.Series
    nonfinite: pd.Series


def inspect_numeric(series: pd.Series) -> NumericObservation:
    disallowed = series.map(lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating)))
    parsed = pd.to_numeric(series.astype(object).mask(disallowed, np.nan), errors="coerce")
    finite = pd.Series(np.isfinite(parsed.to_numpy(dtype=float, na_value=np.nan)), index=series.index)
    invalid = series.notna() & (parsed.isna() | disallowed)
    nonfinite = series.notna() & ~invalid & ~finite
    return NumericObservation(parsed, finite & ~invalid, invalid, nonfinite)


def range_violations(observation: NumericObservation, rule: NumericRange) -> pd.Series:
    violations = pd.Series(False, index=observation.parsed.index)
    if rule.minimum is not None:
        violations |= observation.parsed.lt(rule.minimum).fillna(False)
    if rule.maximum is not None:
        violations |= observation.parsed.gt(rule.maximum).fillna(False)
    return violations & observation.finite


def duplicate_groups(frame: pd.DataFrame) -> list[tuple[int, ...]]:
    """Complete records, including target if present; row IDs excluded."""
    if not len(frame.columns) or frame.empty:
        return []
    duplicates = frame.loc[frame.duplicated(keep=False)]
    if duplicates.empty:
        return []
    groups = [
        tuple(sorted(int(row) for row in group.index))
        for _, group in duplicates.groupby(list(frame.columns), dropna=False, sort=False, observed=True)
    ]
    return sorted(groups)
