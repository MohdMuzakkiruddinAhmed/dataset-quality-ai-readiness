"""Explicit multiclass schema; existing binary protocols remain unchanged."""

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MulticlassTask:
    dataset: str
    version: str
    features: tuple[str, ...]
    numeric: tuple[str, ...]
    target: str
    labels: tuple[str, ...]
    forbidden_predictors: tuple[str, ...]

    def __post_init__(self):
        if (not self.features or len(set(self.features)) != len(self.features)
                or len(set(self.numeric)) != len(self.numeric) or not set(self.numeric) <= set(self.features)
                or self.target in self.features or set(self.features) & set(self.forbidden_predictors)
                or len(self.labels) < 3 or len(set(self.labels)) != len(self.labels)
                or any(type(v) is not str or not v for v in (*self.features, *self.labels))):
            raise ValueError('Invalid multiclass task schema')

    @property
    def categorical(self): return tuple(c for c in self.features if c not in self.numeric)

    @property
    def columns(self): return (*self.features, self.target)

    def receipt(self): return asdict(self)

    def validate_frame(self, frame, *, training=False):
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != self.columns:
            raise ValueError('Frame must match the exact task schema; source IDs and current sensors are excluded')
        if (frame.empty or not frame.index.is_unique or frame.index.name != 'row_id'
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in frame.index)):
            raise ValueError('Task requires nonempty frames and unique nonnegative integer row IDs')
        if frame[self.target].isna().any() or not frame[self.target].isin(self.labels).all():
            raise ValueError('Missing or unknown target label')
        if training and set(frame[self.target]) != set(self.labels):
            raise ValueError('Training requires every declared class')
        for col in self.numeric:
            if (not pd.api.types.is_numeric_dtype(frame[col].dtype) or pd.api.types.is_bool_dtype(frame[col].dtype)
                    or not np.isfinite(frame[col].dropna().to_numpy(dtype=float)).all()):
                raise ValueError('Numeric finite predictor required: ' + col)
        for col in self.categorical:
            if any(type(v) is not str or not v for v in frame[col].dropna()):
                raise ValueError('Categorical predictors require nonempty strings or missing: ' + col)
