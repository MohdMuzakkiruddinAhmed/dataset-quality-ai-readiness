"""Compact profiling from supplied observations alone; no mutations or file IO."""

from dataclasses import asdict
import json
import math

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from dataset_readiness.profiling.inspection import duplicate_groups, inspect_numeric, range_violations, scalar, validate_frame
from dataset_readiness.profiling.schemas import PROFILER_VERSION, AssessmentSchema, ProfileOptions


def _number(value: object) -> float | None:
    return float(value) if value is not None and pd.notna(value) and math.isfinite(float(value)) else None


def _frequencies(series: pd.Series) -> list[dict]:
    counts = [{"value": scalar(value), "count": int(count)} for value, count in series.value_counts(dropna=True).items() if count]
    return sorted(counts, key=lambda item: (-item["count"], json.dumps(item["value"], sort_keys=True)))


def _variants(series: pd.Series, normalize, limit: int) -> dict:
    groups = {}
    for value in series.dropna().unique():
        if isinstance(value, str):
            groups.setdefault(normalize(value), []).append(value)
    variants = [{"normalized": key, "values": sorted(values)[:limit], "value_count": len(values)} for key, values in sorted(groups.items()) if len(values) > 1]
    return {"group_count": len(variants), "examples": variants[:limit], "truncated": len(variants) > limit}


def build_profile(frame: pd.DataFrame, schema: AssessmentSchema, options: ProfileOptions = ProfileOptions()) -> dict:
    validate_frame(frame, schema)
    rows = len(frame)
    expected = (*schema.numeric_features, *schema.categorical_features)
    features = [column for column in frame if column != schema.target]
    groups = duplicate_groups(frame)
    missing_rows = set(int(row) for row in frame.index[frame.isna().any(axis=1)])
    suspicious = {"missing": missing_rows, "duplicates": {row for group in groups for row in group}, "parse_failures": set(), "nonfinite": set(), "outlier_candidates": set(), "rare_categories": set(), "range_violations": set()}
    columns = {}
    constants = []
    for column in features:
        series = frame[column]
        missing, observed = int(series.isna().sum()), int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        if observed and unique == 1:
            constants.append(column)
        numeric = column in schema.numeric_features or (column not in expected and is_numeric_dtype(series.dtype))
        summary = {
            "observed_dtype": str(series.dtype), "kind": "numeric" if numeric else "categorical",
            "missing_count": missing, "missing_rate": missing / rows if rows else None,
            "observed_count": observed, "unique_count": unique,
            "cardinality_ratio": unique / observed if observed else None,
            "all_missing": bool(rows and missing == rows), "observed_constant": bool(observed and unique == 1),
        }
        if numeric:
            inspection = inspect_numeric(series)
            values = inspection.parsed.loc[inspection.finite]
            quantiles = values.quantile([0.25, 0.5, 0.75], interpolation="linear")
            q1, median, q3 = (_number(quantiles.loc[q]) for q in (0.25, 0.5, 0.75))
            iqr = q3 - q1 if q1 is not None and q3 is not None else None
            lower = _number(q1 - 1.5 * iqr) if iqr is not None else None
            upper = _number(q3 + 1.5 * iqr) if iqr is not None else None
            outliers = inspection.finite & (inspection.parsed.lt(lower) | inspection.parsed.gt(upper)) if lower is not None and upper is not None else pd.Series(False, index=frame.index)
            summary["numeric"] = {
                "finite_count": len(values), "parse_failure_count": int(inspection.parse_failure.sum()),
                "nonfinite_count": int(inspection.nonfinite.sum()), "mean": _number(values.mean()),
                "std": _number(values.std(ddof=1)), "minimum": _number(values.min()), "maximum": _number(values.max()),
                "q1": q1, "median": median, "q3": q3, "iqr": _number(iqr),
                "iqr_lower_fence": lower, "iqr_upper_fence": upper, "outlier_candidate_count": int(outliers.sum()),
            }
            for name, mask in (("parse_failures", inspection.parse_failure), ("nonfinite", inspection.nonfinite), ("outlier_candidates", outliers)):
                suspicious[name].update(int(row) for row in frame.index[mask])
            for rule in schema.numeric_ranges:
                if rule.column == column:
                    invalid = range_violations(inspection, rule)
                    summary["declared_range"] = {**asdict(rule), "violation_count": int(invalid.sum())}
                    suspicious["range_violations"].update(int(row) for row in frame.index[invalid])
        else:
            frequencies = _frequencies(series)
            rare = [item for item in frequencies if item["count"] / observed < options.rare_category_threshold]
            # Use actual pandas keys for masks; JSON representations are only for output.
            counts = series.value_counts(dropna=True)
            rare_values = counts.index[(counts > 0) & (counts / observed < options.rare_category_threshold)] if observed else []
            suspicious["rare_categories"].update(int(row) for row in frame.index[series.isin(rare_values)])
            summary["categorical"] = {
                "top_categories": frequencies[:options.category_limit], "frequencies_truncated": len(frequencies) > options.category_limit,
                "rare_category_count": len(rare), "rare_row_count": sum(item["count"] for item in rare),
                "rare_category_examples": rare[:options.category_limit],
                "whitespace_affected_row_count": int(series.astype(object).map(lambda value: isinstance(value, str) and value != value.strip()).sum()),
                "whitespace_variants": _variants(series, str.strip, options.category_limit),
                "case_variants": _variants(series, str.casefold, options.category_limit),
            }
        columns[column] = summary
    target = {"column": schema.target, "status": "absent"}
    if schema.target in frame:
        series = frame[schema.target]
        counts = _frequencies(series)
        present = int(series.notna().sum())
        target.update({
            "status": "empty" if not rows else "all_missing" if not present else "contains_missing" if present < rows else "single_class" if len(counts) < 2 else "available",
            "missing_count": rows - present, "class_count": len(counts),
            "class_counts": counts[:options.category_limit], "class_counts_truncated": len(counts) > options.category_limit,
            "majority_fraction": counts[0]["count"] / present if present else None,
            "imbalance_ratio": counts[0]["count"] / counts[-1]["count"] if counts else None,
        })
    generator = np.random.Generator(np.random.PCG64(options.sample_seed))
    samples = {}
    for name, candidates in {"random": set(int(row) for row in frame.index), **suspicious}.items():
        ids = np.array(sorted(candidates), dtype=object)
        selected = sorted(int(row) for row in generator.choice(ids, size=min(len(ids), options.sample_size), replace=False))
        samples[name] = [{"row_id": row, "values": {column: scalar(frame.at[row, column]) for column in frame}} for row in selected]
    return {
        "schema_version": 1, "profiler_version": PROFILER_VERSION, "options": asdict(options), "task_schema": asdict(schema),
        "shape": {"rows": rows, "features": len(features), "columns_including_target": len(frame.columns)},
        "column_names": list(frame.columns), "schema": {
            "missing_features": [column for column in expected if column not in frame],
            "unexpected_features": [column for column in features if column not in expected],
        },
        "conventions": {
            "missing_rate_denominator": "all rows; pandas missing sentinels only",
            "std_ddof": 1, "quantile_interpolation": "linear", "numeric_statistics": "finite parseable observations only",
            "rare_categories": "frequency / nonmissing observations < configured threshold",
            "duplicates": "complete records including target when present; row ID excluded; missing sentinels compare equal",
            "outliers": "outside Q1 - 1.5*IQR or Q3 + 1.5*IQR; candidates, not definite errors",
            "constant_features": "one distinct observed value; all-missing features excluded",
        },
        "feature_missing_count": int(frame[features].isna().to_numpy().sum()),
        "columns": columns, "constant_features": constants, "target": target,
        "duplicates": {"group_count": len(groups), "excess_row_count": sum(len(group) - 1 for group in groups), "member_row_count": sum(len(group) for group in groups), "examples": [{"row_ids": list(group[:options.sample_size]), "group_size": len(group)} for group in groups[:options.sample_size]]},
        "samples": samples,
    }
