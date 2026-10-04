"""Seeded whole-record copies; insertion lineage is evaluator-only evidence."""

from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral

import numpy as np
import pandas as pd

from dataset_readiness.artifacts import csv_bytes
from dataset_readiness.corruption.schemas import DuplicateConfig

VERSION = "exact_duplicates_v1"


@dataclass(frozen=True)
class DuplicateResult:
    dataframe: pd.DataFrame
    manifest: dict


def inject_duplicates(frame: pd.DataFrame, config: DuplicateConfig) -> DuplicateResult:
    if not isinstance(config, DuplicateConfig):
        raise ValueError("Expected DuplicateConfig")
    if not frame.columns.is_unique or not len(frame.columns) or not frame.index.is_unique:
        raise ValueError("Duplicate insertion requires unique row IDs and columns")
    if any(isinstance(r, (bool, np.bool_)) or not isinstance(r, Integral) or not -(2**63) <= int(r) < 2**63 for r in frame.index):
        raise ValueError("Duplicate insertion requires signed int64 row IDs")
    if frame.index.dtype != np.dtype("int64"):
        raise ValueError("Duplicate insertion requires signed int64 row IDs")
    before = csv_bytes(frame)
    count = int(Decimal(str(config.severity)) * len(frame))
    rng = np.random.Generator(np.random.PCG64(config.seed))
    positions = sorted(int(p) for p in rng.choice(len(frame), size=count, replace=False))
    used, inserted = set(frame.index), []
    for _ in positions:
        while True:
            row = -int(rng.integers(1, 2**63, dtype=np.int64))
            if row not in used:
                used.add(row); inserted.append(row); break
    copies = frame.iloc[positions].copy(deep=True)
    copies.index = pd.Index(inserted, dtype="int64", name=frame.index.name)
    observed = pd.concat([frame, copies]) if count else frame.copy(deep=True)
    if not observed.dtypes.equals(frame.dtypes) or not observed.iloc[:len(frame)].equals(frame) or csv_bytes(frame) != before:
        raise ValueError("Duplicate insertion modified source rows or dtypes")
    manifest = {"schema_version": 1, "engine_version": VERSION, "corruption_type": "exact_duplicates",
                "severity": config.severity, "seed": config.seed, "original_row_count": len(frame),
                "injected_row_count": count, "observed_row_count": len(observed),
                "realized_severity": count / len(frame) if len(frame) else 0.0,
                "preexisting_excess_rows": int(frame.duplicated().sum()),
                "comparison_columns": list(frame.columns),
                "insertions": [{"row_id": row, "source_row_id": int(frame.index[pos])} for row, pos in zip(inserted, positions)],
                "selection_policy": "floor(decimal severity * original rows); PCG64(seed) uniform source positions without replacement, sorted; copy all columns including labels",
                "id_policy": "Unique random negative int64 IDs from the same local RNG after source selection; collision checked; IDs encode no source relationship",
                "order_policy": "Preserve original rows and order; append copies in selected source-position order"}
    return DuplicateResult(observed, manifest)
