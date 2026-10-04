"""Seeded categorical representation variants, with truth separate from observations."""

from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral

import numpy as np
import pandas as pd

from dataset_readiness.categorical import DOMAINS, resolve_category, validate_domains, schema_receipt
from dataset_readiness.corruption.schemas import CategoricalConfig

VERSION = "categorical_variants_v1"
OPERATIONS = ("pad_spaces", "swapcase")


def variants(value, vocabulary):
    """Exclude unchanged, valid-category, ambiguous, or non-restorable collisions."""
    if resolve_category(value, vocabulary)[0] != "valid":
        return ()
    proposed = (("pad_spaces", " " + value + " "), ("swapcase", value.swapcase()))
    return tuple((operation, token) for operation, token in proposed
                 if token != value and resolve_category(token, vocabulary) == ("variant", value))


@dataclass(frozen=True)
class CategoricalResult:
    dataframe: pd.DataFrame
    labels: pd.Series
    manifest: dict


def inject_categories(features, labels, config: CategoricalConfig, *, domains=None):
    vocabulary = DOMAINS if domains is None else domains
    validate_domains(vocabulary, features.columns)
    if not features.columns.is_unique or not features.index.is_unique:
        raise ValueError("Categorical corruption requires unique features and row IDs")
    if any(isinstance(r, (bool, np.bool_)) or not isinstance(r, Integral) for r in features.index):
        raise ValueError("Categorical corruption requires integer row IDs")
    if not features.index.equals(labels.index) or labels.name is None or labels.name in features:
        raise ValueError("A protected target must be supplied separately with matching row IDs")
    if labels.name in config.columns or not set(config.columns) <= set(features.columns) & set(vocabulary):
        raise ValueError("Corruption requires selected declared categorical features")
    candidates, choices, missing = [], {}, 0
    # Cache by value; selection remains row-major and independent of frequencies.
    for c in config.columns:
        if features[c].dtype != object:
            raise ValueError("Categorical observations require object dtype")
        for value in features[c].dropna().unique():
            if resolve_category(value, vocabulary[c])[0] != "valid":
                raise ValueError("Injection requires originally declared valid categorical values")
            choices[c, value] = variants(value, vocabulary[c])
    for row, values in enumerate(features[list(config.columns)].itertuples(index=False, name=None)):
        for column, value in enumerate(values):
            if pd.isna(value):
                missing += 1
            elif choices[config.columns[column], value]:
                candidates.append(row * len(config.columns) + column)
    count = int(Decimal(str(config.severity)) * len(candidates))
    rng = np.random.Generator(np.random.PCG64(config.seed))
    positions = np.sort(rng.choice(candidates, size=count, replace=False))
    result, entries = features.copy(deep=True), []
    for position in positions:
        r, c = divmod(int(position), len(config.columns))
        row, column = int(features.index[r]), config.columns[c]
        original = features.at[row, column]
        options = choices[column, original]
        operation, token = options[int(rng.integers(0, len(options)))]
        entries.append({"row_id": row, "column": column, "original_value": original,
                        "corrupted_value": token, "operation": operation})
        result.at[row, column] = token
    selected = len(features) * len(config.columns)
    manifest = {"schema_version": 1, "engine_version": VERSION, "corruption_type": "categorical_variants",
                "severity": config.severity, "seed": config.seed, "selected_columns": list(config.columns),
                "selected_cell_count": selected, "eligible_cell_count": len(candidates), "preexisting_missing_count": missing,
                "ineligible_observed_count": selected-missing-len(candidates), "injected_cell_count": count,
                "affected_cells": entries, "affected_rows": sorted({e["row_id"] for e in entries}),
                "counts_by_column": {c: sum(e["column"] == c for e in entries) for c in config.columns},
                "counts_by_operation": {op: sum(e["operation"] == op for e in entries) for op in OPERATIONS},
                "realized_severity": count/len(candidates) if candidates else 0.0, "categorical_schema": schema_receipt(domains),
                "selection_policy": "row-major eligible observed cells; decimal floor; PCG64 seed choice without replacement, sorted positions, then uniform choice among each cell's valid pad_spaces/swapcase operations"}
    return CategoricalResult(result, labels.copy(deep=True), manifest)
