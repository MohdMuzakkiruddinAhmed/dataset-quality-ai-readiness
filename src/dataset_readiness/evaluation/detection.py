"""Set-based detection metrics, with explicit undefined-denominator handling."""

from collections.abc import Hashable, Iterable
from numbers import Integral

from dataset_readiness.evaluation.schemas import CellRef, DetectionMetrics, DetectionReport


def score_items(predicted: Iterable[Hashable], truth: Iterable[Hashable]) -> DetectionMetrics:
    """Duplicate reports count once; empty denominators return None."""
    predicted, truth = set(predicted), set(truth)
    tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
    return DetectionMetrics(
        tp=tp, fp=fp, fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    )


def validate_universe(row_ids: Iterable[int], columns: Iterable[str]) -> tuple[set[int], set[str]]:
    rows, columns = list(row_ids), list(columns)
    if any(isinstance(row, bool) or not isinstance(row, Integral) for row in rows):
        raise ValueError("Evaluation row IDs must be integers")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("Evaluation columns must be nonempty strings")
    if len(set(rows)) != len(rows) or len(set(columns)) != len(columns):
        raise ValueError("Evaluation row IDs and columns must be unique")
    return set(rows), set(columns)


def validate_cells(cells: Iterable[CellRef], rows: set[int], columns: set[str]) -> set[CellRef]:
    result = set(cells)
    if any(not isinstance(cell, CellRef) for cell in result):
        raise ValueError("Expected CellRef coordinates")
    if any(cell.row_id not in rows or cell.column not in columns for cell in result):
        raise ValueError("Cell coordinate is outside the evaluated data")
    return result


def evaluate_detection(
    predicted: Iterable[CellRef], truth: Iterable[CellRef], *,
    row_ids: Iterable[int], columns: Iterable[str], defect_type: str = "missing_values",
) -> DetectionReport:
    """Evaluate one defect family; coarse metrics project the reported cells.

    A column-only suspicion cannot be expanded into cell predictions. Such outputs
    can be measured directly using score_items on column names. Invalid coordinates
    raise instead of being silently discarded or treated as valid predictions.
    """
    if not isinstance(defect_type, str) or not defect_type:
        raise ValueError("A defect type is required")
    rows, columns = validate_universe(row_ids, columns)
    predicted, truth = validate_cells(predicted, rows, columns), validate_cells(truth, rows, columns)
    return DetectionReport(
        defect_type=defect_type,
        cell=score_items(predicted, truth),
        row=score_items({cell.row_id for cell in predicted}, {cell.row_id for cell in truth}),
        column=score_items({cell.column for cell in predicted}, {cell.column for cell in truth}),
        dataset=score_items({defect_type} if predicted else set(), {defect_type} if truth else set()),
        evaluated_rows=len(rows), evaluated_columns=len(columns),
    )
