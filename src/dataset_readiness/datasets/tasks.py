"""Explicit binary task schemas, independent of dataset-specific preparation."""

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinaryTask:
    dataset: str
    version: str
    features: tuple[str, ...]
    numeric: tuple[str, ...]
    target: str
    negative_label: str
    positive_label: str
    forbidden_predictors: tuple[str, ...]

    def __post_init__(self):
        if (not self.features or len(set(self.features)) != len(self.features)
                or len(set(self.numeric)) != len(self.numeric) or not set(self.numeric) <= set(self.features)
                or self.target in self.features or set(self.features) & set(self.forbidden_predictors)
                or self.positive_label == self.negative_label):
            raise ValueError("Invalid binary task schema")

    @property
    def categorical(self): return tuple(c for c in self.features if c not in self.numeric)

    @property
    def columns(self): return (*self.features, self.target)

    def receipt(self): return asdict(self)

    def validate_frame(self, frame, *, training=False):
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != self.columns:
            raise ValueError("Frame must match the exact task schema; IDs/source targets are excluded")
        if (frame.empty or not frame.index.is_unique or frame.index.name != "row_id"
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in frame.index)):
            raise ValueError("Task requires nonempty frames and unique nonnegative integer row IDs")
        labels = frame[self.target]
        if labels.isna().any() or not labels.isin([self.negative_label, self.positive_label]).all():
            raise ValueError("Missing or unknown target label")
        if training and labels.nunique() != 2:
            raise ValueError("Training requires both binary labels")
        for col in self.numeric:
            if not pd.api.types.is_numeric_dtype(frame[col].dtype) or pd.api.types.is_bool_dtype(frame[col].dtype):
                raise ValueError("Numeric predictor dtype required: " + col)
            if not np.isfinite(frame[col].dropna().to_numpy(dtype=float)).all():
                raise ValueError("Nonfinite numeric predictor: " + col)
        for col in self.categorical:
            if any(type(v) is not str or not v for v in frame[col].dropna()):
                raise ValueError("Categorical predictors must be nonempty strings or missing: " + col)
