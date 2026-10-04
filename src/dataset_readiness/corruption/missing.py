"""Seeded missingness without modifying the caller's data or target."""

from decimal import Decimal, ROUND_FLOOR
import math
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype, is_integer_dtype, is_unsigned_integer_dtype

from dataset_readiness.corruption.schemas import (
    ENGINE_VERSION, CellChange, CorruptionManifest, CorruptionResult, JsonScalar, MissingnessConfig,
)


def _json_scalar(value: object) -> JsonScalar:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (str, int, float, bool):
        raise ValueError(f"Unsupported cell value for a corruption manifest: {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite cell values cannot be stored in the manifest")
    return value


def inject_missing_values(
    features: pd.DataFrame, labels: pd.Series, config: MissingnessConfig,
) -> CorruptionResult:
    """Corrupt exactly floor(severity * observed selected cells) in a deep copy.

    Inputs are features and labels separately. Only selected, originally observed
    feature cells can change. Stable integer row IDs, column order, and input row
    order are retained. The seed never changes NumPy's global random state.
    """
    if not features.columns.is_unique or not features.index.is_unique:
        raise ValueError("Feature columns and row IDs must be unique")
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in features.index):
        raise ValueError("Corruption requires stable integer row IDs")
    if not features.index.equals(labels.index):
        raise ValueError("Feature and label row IDs must match in the same order")
    if labels.name is None or labels.name in features.columns:
        raise ValueError("Named target must be supplied separately from feature columns")
    if labels.name in config.columns:
        raise ValueError("Target corruption is not allowed for missing_values")
    unknown = set(config.columns) - set(features.columns)
    if unknown:
        raise ValueError(f"Unknown feature columns: {sorted(unknown)}")

    eligible_mask = features.loc[:, list(config.columns)].notna().to_numpy()
    candidates = np.flatnonzero(eligible_mask.ravel())
    count = int((Decimal(str(config.severity)) * len(candidates)).to_integral_value(rounding=ROUND_FLOOR))
    generator = np.random.Generator(np.random.PCG64(config.seed))
    selected = np.sort(generator.choice(candidates, size=count, replace=False))
    row_positions, column_positions = np.divmod(selected, len(config.columns))
    changes = tuple(
        CellChange(
            row_id=int(features.index[row]), column=config.columns[column],
            original_value=_json_scalar(features.iat[int(row), features.columns.get_loc(config.columns[column])]),
        )
        for row, column in zip(row_positions, column_positions)
    )

    corrupted = features.copy(deep=True)
    for position, column in enumerate(config.columns):
        rows = row_positions[column_positions == position]
        if not len(rows):
            continue
        series = corrupted[column]
        # Nullable integers retain exact values, including integers above 2**53.
        if is_integer_dtype(series.dtype) and not is_extension_array_dtype(series.dtype):
            series = series.astype("UInt64" if is_unsigned_integer_dtype(series.dtype) else "Int64")
        mask = np.zeros(len(series), dtype=bool)
        mask[rows] = True
        corrupted[column] = series.mask(mask)

    manifest = CorruptionManifest(
        schema_version=1, engine_version=ENGINE_VERSION, corruption_type="missing_values",
        severity=config.severity, seed=config.seed, selected_columns=config.columns,
        selected_cell_count=int(eligible_mask.size), eligible_cell_count=len(candidates),
        preexisting_missing_count=int(eligible_mask.size - len(candidates)), affected_cells=changes,
    )
    return CorruptionResult(corrupted, labels.copy(deep=True), manifest)
