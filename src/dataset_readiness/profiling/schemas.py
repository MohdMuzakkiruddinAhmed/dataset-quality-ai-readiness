"""Allowed task/schema metadata and bounded profiling settings."""

from dataclasses import dataclass
import math

PROFILER_VERSION = "profile_v1"


@dataclass(frozen=True)
class NumericRange:
    column: str
    minimum: float | None
    maximum: float | None
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column or not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Numeric ranges require a column and a documented constraint source")
        if self.minimum is None and self.maximum is None:
            raise ValueError("Numeric ranges require at least one bound")
        for bound in (self.minimum, self.maximum):
            if bound is not None and (type(bound) not in (float, int) or not math.isfinite(bound)):
                raise ValueError("Range bounds must be finite numbers")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Range minimum exceeds maximum")


@dataclass(frozen=True)
class AssessmentSchema:
    target: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_ranges: tuple[NumericRange, ...] = ()
    row_id_column: str = "row_id"

    def __post_init__(self) -> None:
        if not all(isinstance(value, tuple) for value in (self.numeric_features, self.categorical_features, self.numeric_ranges)):
            raise ValueError("Schema feature lists and ranges must be tuples")
        names = (self.target, self.row_id_column, *self.numeric_features, *self.categorical_features)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Target, row ID, and feature names must be nonempty and distinct")
        if any(not isinstance(rule, NumericRange) or rule.column not in self.numeric_features for rule in self.numeric_ranges):
            raise ValueError("Ranges must refer to declared numeric features")
        if len({rule.column for rule in self.numeric_ranges}) != len(self.numeric_ranges):
            raise ValueError("Only one declared range per feature is allowed")


@dataclass(frozen=True)
class ProfileOptions:
    sample_seed: int = 17
    sample_size: int = 5
    category_limit: int = 10
    rare_category_threshold: float = 0.01

    def __post_init__(self) -> None:
        if type(self.sample_seed) is not int or not 0 <= self.sample_seed < 2**32:
            raise ValueError("Sample seed must be an integer in [0, 2**32)")
        if type(self.sample_size) is not int or not 0 <= self.sample_size <= 50:
            raise ValueError("Sample size must be an integer in [0, 50]")
        if type(self.category_limit) is not int or not 1 <= self.category_limit <= 100:
            raise ValueError("Category limit must be an integer in [1, 100]")
        if type(self.rare_category_threshold) not in (int, float) or not math.isfinite(self.rare_category_threshold) or not 0 <= self.rare_category_threshold <= 1:
            raise ValueError("Rare-category threshold must be finite and in [0, 1]")


