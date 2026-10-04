"""Group detection and deletion utility; row origin is a separate measurement."""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral

from dataset_readiness.artifacts import json_bytes
from dataset_readiness.corruption.duplicates import VERSION
from dataset_readiness.corruption.schemas import DuplicateConfig
from dataset_readiness.evaluation.detection import score_items, validate_universe
from dataset_readiness.evaluation.schemas import DetectionMetrics
from dataset_readiness.profiling.inspection import duplicate_groups, scalar


def row_keys(frame):
    return {int(row): json_bytes([scalar(v) for v in values]) for row, values in zip(frame.index, frame.itertuples(index=False, name=None))}


def validate_manifest(original, observed, manifest):
    if manifest.get("schema_version") != 1 or manifest.get("engine_version") != VERSION or manifest.get("corruption_type") != "exact_duplicates":
        raise ValueError("Expected exact_duplicates_v1 manifest")
    config = DuplicateConfig(manifest["severity"], manifest["seed"])
    for frame in (original, observed):
        validate_universe(frame.index, frame.columns)
    entries = manifest.get("insertions")
    if not isinstance(entries, list) or any(not isinstance(e, dict) or set(e) != {"row_id", "source_row_id"} for e in entries):
        raise ValueError("Invalid duplicate insertion entries")
    if any(type(e[k]) is not int for e in entries for k in ("row_id", "source_row_id")):
        raise ValueError("Insertion IDs must be integers")
    added, sources = [e["row_id"] for e in entries], [e["source_row_id"] for e in entries]
    n = int(Decimal(str(config.severity)) * len(original))
    if (len(added) != n or len(set(added)) != n or len(set(sources)) != n
            or set(added) & set(original.index) or not set(sources) <= set(original.index)):
        raise ValueError("Duplicate insertion count, IDs or source lineage is invalid")
    if any(type(manifest.get(k)) is not int or manifest[k] != v for k, v in (
        ("original_row_count", len(original)), ("observed_row_count", len(observed)),
        ("injected_row_count", n), ("preexisting_excess_rows", int(original.duplicated().sum())))):
        raise ValueError("Duplicate manifest counts differ from data")
    if (not observed.columns.equals(original.columns) or not observed.dtypes.equals(original.dtypes)
            or observed.index.dtype != original.index.dtype or observed.index.name != original.index.name
            or manifest["comparison_columns"] != list(original.columns)
            or list(observed.index) != [*original.index, *added] or not observed.iloc[:len(original)].equals(original)):
        raise ValueError("Duplicate insertion changed original data or schema")
    if not observed.loc[added].reset_index(drop=True).equals(original.loc[sources].reset_index(drop=True)):
        raise ValueError("Inserted rows do not equal their declared sources")
    return set(added)


@dataclass(frozen=True)
class DuplicateDetectionReport:
    group: DetectionMetrics
    participating_row: DetectionMetrics
    dataset: DetectionMetrics
    evaluated_rows: int
    observed_duplicate_groups: int
    injected_affected_groups: int
    injected_group_recall: float | None
    preexisting_duplicate_groups: int
    defect_type: str = "exact_duplicates"
    injected_row_localization: None = None
    interpretation: str = "Group membership includes retained originals; no prediction of insertion origin. Exact complete groups are the scoring unit."


def evaluate_duplicate_detection(predicted, original, observed, manifest):
    inserted = validate_manifest(original, observed, manifest)
    universe = set(observed.index)
    predicted_groups = set()
    for rows in predicted:
        rows = tuple(rows)
        if (len(rows) < 2 or len(set(rows)) != len(rows)
                or any(isinstance(r, bool) or not isinstance(r, Integral) for r in rows) or not set(rows) <= universe):
            raise ValueError("Duplicate predictions require distinct known integer row IDs")
        predicted_groups.add(tuple(sorted(int(r) for r in rows)))
    actual = set(duplicate_groups(observed))
    affected = {g for g in actual if inserted & set(g)}
    return DuplicateDetectionReport(
        score_items(predicted_groups, actual),
        score_items({r for g in predicted_groups for r in g}, {r for g in actual for r in g}),
        score_items({"exact_duplicates"} if predicted_groups else set(), {"exact_duplicates"} if actual else set()),
        len(observed), len(actual), len(affected), len(predicted_groups & affected)/len(affected) if affected else None,
        len(duplicate_groups(original)))


@dataclass(frozen=True)
class DuplicateRepairMetrics:
    original_rows: int
    injected_rows: int
    total_deleted_rows: int
    deleted_injected_rows: int
    retained_injected_rows: int
    deleted_original_rows: int
    original_row_identity_loss_rate: float | None
    unique_original_rows_removed: int
    unique_original_row_removal_rate: float | None
    original_rows_with_equivalent_retained: int
    original_value_groups_lost: int
    retained_data_equivalent: bool
    original_row_multiset_restored: bool
    injected_affected_groups: int
    resolved_injected_groups: int
    lost_injected_groups: int
    injected_group_resolution_rate: float | None
    remaining_injected_excess_rows: int
    remaining_duplicate_groups: int
    remaining_excess_rows: int
    preexisting_excess_rows: int
    defect_type: str = "exact_duplicates"


def evaluate_duplicate_repairs(original, observed, repaired, manifest):
    inserted = validate_manifest(original, observed, manifest)
    validate_universe(repaired.index, repaired.columns)
    if (not set(repaired.index) <= set(observed.index) or not repaired.columns.equals(observed.columns)
            or repaired.index.dtype != observed.index.dtype or repaired.index.name != observed.index.name
            or not repaired.dtypes.equals(observed.dtypes) or not repaired.equals(observed.loc[repaired.index])):
        raise ValueError("Duplicate repair evaluation permits only deletion of observed rows; values/labels are protected")
    original_keys, observed_keys, after_keys = (row_keys(f) for f in (original, observed, repaired))
    original_counts, observed_counts, after_counts = (Counter(k.values()) for k in (original_keys, observed_keys, after_keys))
    removed = set(observed.index) - set(repaired.index)
    original_removed = removed & set(original.index)
    unique = {r for r, key in original_keys.items() if observed_counts[key] == 1}
    affected = {observed_keys[r] for r in inserted}
    resolved = sum(after_counts[k] == 1 for k in affected)
    return DuplicateRepairMetrics(
        len(original), len(inserted), len(removed), len(removed & inserted), len(inserted - removed), len(original_removed),
        len(original_removed)/len(original) if len(original) else None,
        len(removed & unique), len(removed & unique)/len(unique) if unique else None,
        sum(original_keys[r] in after_counts for r in original_removed), len(original_counts.keys() - after_counts.keys()),
        original_counts.keys() == after_counts.keys(), original_counts == after_counts,
        len(affected), resolved, sum(after_counts[k] == 0 for k in affected), resolved/len(affected) if affected else None,
        sum(max(after_counts[k]-1, 0) for k in affected), sum(n > 1 for n in after_counts.values()),
        sum(n-1 for n in after_counts.values()), int(original.duplicated().sum()))
