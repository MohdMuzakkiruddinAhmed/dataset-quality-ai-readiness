"""Typed corruption settings and exact cell-level ground truth."""

from dataclasses import asdict, dataclass
import math
from typing import Any

import pandas as pd

JsonScalar = str | int | float | bool | None
ENGINE_VERSION = "missing_values_v1"


@dataclass(frozen=True)
class MissingnessConfig:
    severity: float
    seed: int
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.severity) not in (int, float) or not math.isfinite(self.severity):
            raise ValueError("Missingness severity must be a finite number")
        if not 0 <= self.severity <= 1:
            raise ValueError("Missingness severity must be between 0 and 1")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Corruption seed must be an integer between 0 and 2**32 - 1")
        if not isinstance(self.columns, tuple) or not self.columns:
            raise ValueError("Selected columns must be a nonempty tuple")
        if any(not isinstance(c, str) or not c for c in self.columns):
            raise ValueError("Selected column names must be nonempty strings")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Selected columns must be unique")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MissingnessConfig":
        if not isinstance(value, dict) or set(value) != {"type", "severity", "seed", "columns"}:
            raise ValueError("Corruption requires type, severity, seed, and columns")
        if value["type"] != "missing_values":
            raise ValueError("Only missing_values corruption is implemented")
        if not isinstance(value["columns"], list):
            raise ValueError("Corruption columns must be a list")
        return cls(value["severity"], value["seed"], tuple(value["columns"]))


@dataclass(frozen=True)
class DuplicateConfig:
    severity: float
    seed: int

    def __post_init__(self):
        if type(self.severity) not in (int, float) or not math.isfinite(self.severity) or not 0 <= self.severity <= 1:
            raise ValueError("Duplicate severity must be finite and between 0 and 1")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Duplicate seed must be an integer between 0 and 2**32 - 1")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"type", "severity", "seed"} or value["type"] != "exact_duplicates":
            raise ValueError("Exact duplicates require type, severity, seed; all columns including target are copied")
        return cls(value["severity"], value["seed"])


@dataclass(frozen=True)
class LabelFlipConfig:
    severity: float
    seed: int

    def __post_init__(self):
        if type(self.severity) not in (int, float) or not math.isfinite(self.severity) or not 0 <= self.severity <= 1:
            raise ValueError("Label severity must be finite and between 0 and 1")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Label seed must be an integer between 0 and 2**32 - 1")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"type", "severity", "seed"} or value["type"] != "binary_label_flips":
            raise ValueError("Binary label flips require only type, severity, seed")
        return cls(value["severity"], value["seed"])


@dataclass(frozen=True)
class NumericTypeConfig(MissingnessConfig):
    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"type", "severity", "seed", "columns"} or value["type"] != "numeric_type_tokens":
            raise ValueError("Numeric type corruption requires type, severity, seed, columns")
        if not isinstance(value["columns"], list):
            raise ValueError("Numeric type columns must be a list")
        return cls(value["severity"], value["seed"], tuple(value["columns"]))


@dataclass(frozen=True)
class CategoricalConfig(MissingnessConfig):
    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"type", "severity", "seed", "columns"} or value["type"] != "categorical_variants":
            raise ValueError("Categorical corruption requires type, severity, seed, columns")
        if not isinstance(value["columns"], list):
            raise ValueError("Categorical columns must be a list")
        return cls(value["severity"], value["seed"], tuple(value["columns"]))


@dataclass(frozen=True)
class NumericDomainConfig(MissingnessConfig):
    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"type", "severity", "seed", "columns"} or value["type"] != "numeric_domain_negatives":
            raise ValueError("Numeric domain corruption requires type, severity, seed, columns")
        if not isinstance(value["columns"], list):
            raise ValueError("Numeric domain columns must be a list")
        return cls(value["severity"], value["seed"], tuple(value["columns"]))


def corruption_config(value):
    if isinstance(value, dict) and value.get("type") == "missing_and_duplicates":
        from dataset_readiness.corruption.missing_duplicates import MissingDuplicatesConfig
        return MissingDuplicatesConfig.from_mapping(value)
    if isinstance(value, dict) and value.get("type") == "combined_feature_defects":
        from dataset_readiness.corruption.combined import CombinedConfig
        return CombinedConfig.from_mapping(value)
    if isinstance(value, dict) and value.get("type") == "exact_target_encoding":
        from dataset_readiness.corruption.leakage import LeakageConfig
        return LeakageConfig.from_mapping(value)
    if isinstance(value, dict) and value.get("type") == "added_irrelevant_features":
        from dataset_readiness.corruption.irrelevant import AddedFeatureConfig
        return AddedFeatureConfig.from_mapping(value)
    if isinstance(value, dict) and value.get("type") == "binary_label_flips":
        return LabelFlipConfig.from_mapping(value)
    if isinstance(value, dict) and value.get("type") == "numeric_domain_negatives":
        return NumericDomainConfig.from_mapping(value)
    if isinstance(value, dict) and value.get("type") == "categorical_variants":
        return CategoricalConfig.from_mapping(value)
    if isinstance(value, dict) and value.get("type") == "numeric_type_tokens":
        return NumericTypeConfig.from_mapping(value)
    if isinstance(value, dict) and value.get("type") == "exact_duplicates":
        return DuplicateConfig.from_mapping(value)
    return MissingnessConfig.from_mapping(value)


@dataclass(frozen=True)
class CellChange:
    row_id: int
    column: str
    original_value: JsonScalar
    corrupted_value: None = None


@dataclass(frozen=True)
class CorruptionManifest:
    schema_version: int
    engine_version: str
    corruption_type: str
    severity: float
    seed: int
    selected_columns: tuple[str, ...]
    selected_cell_count: int
    eligible_cell_count: int
    preexisting_missing_count: int
    affected_cells: tuple[CellChange, ...]

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["injected_cell_count"] = len(self.affected_cells)
        record["affected_rows"] = sorted({cell.row_id for cell in self.affected_cells})
        counts = dict.fromkeys(self.selected_columns, 0)
        for cell in self.affected_cells:
            counts[cell.column] += 1
        record["counts_by_column"] = counts
        record["affected_columns"] = [c for c, count in counts.items() if count]
        record["realized_severity"] = (
            len(self.affected_cells) / self.eligible_cell_count if self.eligible_cell_count else 0.0
        )
        record["selection_policy"] = (
            "Uniform sampling without replacement across observed cells in selected columns; "
            "floor(severity * eligible_cell_count), using decimal severity; "
            "NumPy Generator(PCG64(seed)); row-major candidate order."
        )
        return record


@dataclass(frozen=True)
class CorruptionResult:
    dataframe: pd.DataFrame
    labels: pd.Series
    manifest: CorruptionManifest
