"""Typed, JSON-safe metric records. Undefined quantities use None, never NaN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from dataset_readiness.evaluation.duplicates import DuplicateDetectionReport, DuplicateRepairMetrics

EVALUATOR_VERSION = "evaluation_v1"


@dataclass(frozen=True, order=True)
class CellRef:
    row_id: int
    column: str

    def __post_init__(self) -> None:
        if isinstance(self.row_id, bool) or not isinstance(self.row_id, Integral):
            raise ValueError("Cell row_id must be an integer")
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("Cell column must be a nonempty string")
        object.__setattr__(self, "row_id", int(self.row_id))


def cells_from_manifest(manifest: dict) -> frozenset[CellRef]:
    """Read coordinates from the current missingness manifest (evaluator only)."""
    if manifest.get("schema_version") != 1 or manifest.get("corruption_type") != "missing_values":
        raise ValueError("Expected a version 1 missing_values manifest")
    entries = manifest.get("affected_cells")
    if not isinstance(entries, (list, tuple)):
        raise ValueError("Manifest affected_cells must be a sequence")
    try:
        cells = [CellRef(cell["row_id"], cell["column"]) for cell in entries]
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed affected cell in manifest") from exc
    if len(set(cells)) != len(cells):
        raise ValueError("Manifest contains duplicate coordinates")
    if type(manifest.get("injected_cell_count")) is not int or manifest["injected_cell_count"] != len(cells):
        raise ValueError("Manifest count does not match its affected cells")
    return frozenset(cells)


@dataclass(frozen=True)
class DetectionMetrics:
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class DetectionReport:
    defect_type: str
    cell: DetectionMetrics
    row: DetectionMetrics
    column: DetectionMetrics
    dataset: DetectionMetrics
    evaluated_rows: int
    evaluated_columns: int


@dataclass(frozen=True)
class RepairMetrics:
    known_defect_cells: int
    correctly_restored_cells: int
    remaining_defect_cells: int
    remaining_missing_cells: int
    incorrectly_filled_cells: int
    originally_clean_cells: int
    changed_clean_cells: int
    total_changed_cells: int
    repair_accuracy: float | None
    remaining_defect_rate: float | None
    remaining_missing_rate: float | None
    overclean_rate: float | None


@dataclass(frozen=True)
class DownstreamMetrics:
    metric: str
    higher_is_better: bool
    zero_tolerance: float
    clean_score: float
    corrupted_score: float
    repaired_score: float | None
    performance_loss: float
    performance_recovery: float | None
    recovery_rate: float | None
    recovery_rate_reason: Literal["no_positive_performance_loss", "repair_not_evaluated"] | None
    repair_harmed_performance: bool | None


@dataclass(frozen=True)
class HarmfulActionMetrics:
    evaluated_actions: int
    harmful_actions: int
    harmful_action_rate: float | None


@dataclass(frozen=True)
class EvaluationReport:
    downstream: DownstreamMetrics | None = None
    detection: DetectionReport | DuplicateDetectionReport | None = None
    repair: RepairMetrics | DuplicateRepairMetrics | None = None
    harmful_actions: HarmfulActionMetrics | None = None
    schema_version: int = 1
    evaluator_version: str = EVALUATOR_VERSION

    def to_dict(self) -> dict:
        return asdict(self)
