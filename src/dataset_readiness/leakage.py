"""Training-only exact numeric target proxies; no causal leakage inference."""

from dataset_readiness.added_features import validate_frame
from dataset_readiness.artifacts import csv_bytes, sha256

DETECTOR = "exact_binary_target_proxy_v1"
MIN_CLASS_SUPPORT = 5
CLASSES = ("<=50K", ">50K")


def detector_receipt():
    return {"version": DETECTOR, "minimum_rows_per_class": MIN_CLASS_SUPPORT,
            "scope": "declared numeric predictors only; target excluded; observed training rows only",
            "criterion": "complete nonmissing two-valued feature; exactly one distinct value per target class; class values differ",
            "interpretation": "Exact empirical target proxy requiring provenance review; not proof of temporal or causal leakage",
            "abstention": "missing feature values, absent/rare classes, varying within-class values, constants, or categorical predictors"}


def exact_proxies(frame, schema):
    validate_frame(frame, schema)
    counts = {label: int((frame[schema.target] == label).sum()) for label in CLASSES}
    if min(counts.values()) < MIN_CLASS_SUPPORT: return {}
    result = {}
    for column in schema.numeric_features:
        series = frame[column]
        if series.isna().any() or series.nunique() != 2: continue
        subsets = {label: series[frame[schema.target] == label] for label in CLASSES}
        if any(values.nunique() != 1 for values in subsets.values()): continue
        mapping = {label: int(values.iloc[0]) for label, values in subsets.items()}
        if len(set(mapping.values())) != 2: continue
        result[column] = {"row_count": len(frame), "missing_count": 0, "unique_count": 2,
                          "minimum_rows_per_class": MIN_CLASS_SUPPORT, "class_counts": counts.copy(),
                          "class_value_mapping": mapping, "column_sha256": sha256(csv_bytes(frame[[column]])),
                          "target_sha256": sha256(csv_bytes(frame[[schema.target]]))}
    return result
