"""Assessment input checks allow missing labels without making a frame trainable."""
from numbers import Integral
import numpy as np
import pandas as pd
from dataset_readiness.datasets.diabetes import DIABETES


def validate_observation(frame, task):
    if task != DIABETES:
        raise ValueError('Frozen Diabetes assessment task required')
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != task.columns:
        raise ValueError('Exact observed task columns required')
    if (frame.empty or not frame.index.is_unique or frame.index.name != 'row_id'
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in frame.index)):
        raise ValueError('Nonempty observations with unique integer row IDs required')
    labels = frame[task.target].dropna()
    if not labels.isin([task.negative_label, task.positive_label]).all():
        raise ValueError('Unknown observed target label')
    for column in task.numeric:
        series = frame[column]
        if (not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype)
                or not np.isfinite(series.dropna().to_numpy(dtype=float)).all()):
            raise ValueError('Finite numeric predictor dtype required: ' + column)
    for column in task.categorical:
        if any(type(v) is not str or not v for v in frame[column].dropna()):
            raise ValueError('Categorical values must be nonempty strings or missing')
