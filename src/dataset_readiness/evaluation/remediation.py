"""Reference-based evaluation of cell-preserving missing-value repairs."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from dataset_readiness.evaluation.detection import validate_cells, validate_universe
from dataset_readiness.evaluation.schemas import CellRef, RepairMetrics


def _equal_values(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    # Object comparison preserves large integers and permits category vocabularies
    # to differ. Equivalent missing sentinels compare equal; dtypes are not scored.
    result = {}
    for column in left:
        observed = left[column].notna() & right[column].notna()
        same = left[column].isna() & right[column].isna()
        # Compare only observed operands: pandas.NA has no boolean truth value.
        same.loc[observed] = (
            left.loc[observed, column].astype(object)
            .eq(right.loc[observed, column].astype(object)).to_numpy()
        )
        result[column] = same
    return pd.DataFrame(result, index=left.index)


def evaluate_missing_repairs(
    reference: pd.DataFrame, corrupted: pd.DataFrame, repaired: pd.DataFrame,
    truth: Iterable[CellRef], *, target: str,
) -> RepairMetrics:
    """Score exact restoration separately from simply filling a missing cell.

    Frames include the target, whose values are protected and excluded from cell
    denominators. This version rejects dropped/added rows or columns. Reordering
    is allowed and aligned by IDs. Known defects must exactly match the observed
    reference-to-corrupted differences and must be observed-to-missing changes.
    """
    rows, columns = validate_universe(reference.index, reference.columns)
    if target not in columns:
        raise ValueError("Target column is required for repair evaluation")
    for name, frame in (("corrupted", corrupted), ("repaired", repaired)):
        frame_rows, frame_columns = validate_universe(frame.index, frame.columns)
        if frame_rows != rows or frame_columns != columns:
            raise ValueError(f"{name}: row/column additions or removals are not supported by cell repair metrics")
    corrupted = corrupted.loc[reference.index, reference.columns]
    repaired = repaired.loc[reference.index, reference.columns]
    if not _equal_values(reference[[target]], corrupted[[target]])[target].all():
        raise ValueError("Corruption changed the protected target")
    if not _equal_values(reference[[target]], repaired[[target]])[target].all():
        raise ValueError("Repair changed the protected target")
    feature_columns = [column for column in reference if column != target]
    truth = validate_cells(truth, rows, set(feature_columns))
    reference = reference[feature_columns]
    corrupted = corrupted[feature_columns]
    repaired = repaired[feature_columns]
    known = pd.DataFrame(False, index=reference.index, columns=reference.columns)
    for cell in truth:
        known.at[cell.row_id, cell.column] = True
    actual_changes = ~_equal_values(reference, corrupted)
    if not np.array_equal(actual_changes.to_numpy(), known.to_numpy()):
        raise ValueError("Manifest coordinates do not exactly match the injected changes")
    if (known & (reference.isna() | corrupted.notna())).to_numpy().any():
        raise ValueError("Expected observed-to-missing corruption at every known coordinate")
    restored = _equal_values(reference, repaired)
    changed = ~_equal_values(corrupted, repaired)
    n_defects = len(truth)
    n_restored = int((known & restored).to_numpy().sum())
    n_missing = int((known & repaired.isna()).to_numpy().sum())
    n_clean = int(reference.size) - n_defects
    overcleaned = int((~known & changed).to_numpy().sum())
    return RepairMetrics(
        known_defect_cells=n_defects, correctly_restored_cells=n_restored,
        remaining_defect_cells=n_defects - n_restored, remaining_missing_cells=n_missing,
        incorrectly_filled_cells=n_defects - n_restored - n_missing,
        originally_clean_cells=n_clean, changed_clean_cells=overcleaned,
        total_changed_cells=int(changed.to_numpy().sum()),
        repair_accuracy=n_restored / n_defects if n_defects else None,
        remaining_defect_rate=(n_defects - n_restored) / n_defects if n_defects else None,
        remaining_missing_rate=n_missing / n_defects if n_defects else None,
        overclean_rate=overcleaned / n_clean if n_clean else None,
    )
