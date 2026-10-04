"""Explicit dynamic schemas for the isolated added-feature experiment."""

from dataclasses import asdict, dataclass
from io import BytesIO
import re

import pandas as pd

from dataset_readiness.artifacts import csv_bytes
from dataset_readiness.datasets.adult import ADULT
from dataset_readiness.profiling import AssessmentSchema

ML_VERSION = "adult_dynamic_features_lr_v1"
DETECTOR = "complete_constant_features_v1"
NAMES = ("attribute_01", "attribute_02")


@dataclass(frozen=True)
class FeatureSchema:
    columns: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target: str = ADULT.target
    version: str = "adult_dynamic_features_v1"

    def __post_init__(self):
        if self.target != ADULT.target or self.version != "adult_dynamic_features_v1":
            raise ValueError("Unsupported feature schema")
        if any(not isinstance(x, tuple) for x in (self.columns, self.numeric_features, self.categorical_features)):
            raise ValueError("Feature schema lists must be tuples")
        AssessmentSchema(self.target, self.numeric_features, self.categorical_features)
        if (not self.columns or len(set(self.columns)) != len(self.columns)
                or self.columns[-1] != self.target
                or set(self.columns) != {self.target, *self.numeric_features, *self.categorical_features}
                or len(self.columns) < 2
                or any(not isinstance(c, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in self.columns)):
            raise ValueError("Invalid ordered feature schema")

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, value):
        required = {"columns", "numeric_features", "categorical_features", "target", "version"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("Incomplete feature schema")
        return cls(**{k: tuple(v) if k in required - {"target", "version"} else v for k, v in value.items()})

    def assessment(self): return AssessmentSchema(self.target, self.numeric_features, self.categorical_features)

    def without(self, columns):
        if self.target in columns or not set(columns) <= set(self.columns):
            raise ValueError("Cannot remove target or unknown columns")
        return FeatureSchema(*(tuple(c for c in values if c not in columns)
                               for values in (self.columns, self.numeric_features, self.categorical_features)))


def original_schema(): return FeatureSchema(ADULT.columns, ADULT.numeric, ADULT.categorical)


def validate_frame(frame, schema):
    if (list(frame.columns) != list(schema.columns) or frame.index.name != "row_id"
            or str(frame.index.dtype) != "int64" or not frame.index.is_unique):
        raise ValueError("Observed rows/columns differ from explicit feature schema")
    if any(str(frame[c].dtype) != "Int64" for c in schema.numeric_features):
        raise ValueError("Dynamic numeric observations require nullable signed int64")
    for c in (*schema.categorical_features, schema.target):
        if frame[c].dtype != object or any(not isinstance(x, str) for x in frame[c].dropna()):
            raise ValueError("Categorical observations require faithful strings")
    if frame[schema.target].isna().any() or not set(frame[schema.target]) <= {"<=50K", ">50K"}:
        raise ValueError("Missing or unknown target label")


def read_frame(payload, schema):
    frame = pd.read_csv(BytesIO(payload), index_col="row_id", keep_default_na=False, na_values=[""],
                        dtype={"row_id": "int64", **{c: "Int64" for c in schema.numeric_features},
                               **{c: object for c in (*schema.categorical_features, schema.target)}})
    validate_frame(frame, schema)
    if csv_bytes(frame) != payload: raise ValueError("Dynamic observations did not round-trip exactly")
    return frame


def constant_columns(frame, schema):
    validate_frame(frame, schema)
    return [c for c in schema.columns if c != schema.target and len(frame) >= 2
            and not frame[c].isna().any() and frame[c].nunique(dropna=False) == 1]


def matched_view(full_test, full_schema, retained_schema):
    """Only evaluation uses this explicit projection after repair verification."""
    validate_frame(full_test, full_schema)
    if full_schema.without(set(full_schema.columns) - set(retained_schema.columns)) != retained_schema:
        raise ValueError("Evaluation schema is not a retained-column projection")
    result = full_test[list(retained_schema.columns)].copy()
    validate_frame(result, retained_schema)
    return result
