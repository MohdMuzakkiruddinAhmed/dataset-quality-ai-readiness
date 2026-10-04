"""Direction-aware performance loss, recovery, and action-level harm."""

from collections.abc import Iterable
import math
from numbers import Real

from dataset_readiness.evaluation.schemas import DownstreamMetrics, HarmfulActionMetrics


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _settings(higher_is_better: bool, zero_tolerance: float) -> tuple[int, float]:
    if type(higher_is_better) is not bool:
        raise ValueError("higher_is_better must be a boolean")
    tolerance = _finite(zero_tolerance, "zero_tolerance")
    if tolerance < 0:
        raise ValueError("zero_tolerance must be nonnegative")
    return (1 if higher_is_better else -1), tolerance


def evaluate_downstream(
    clean_score: float, corrupted_score: float, repaired_score: float | None = None,
    *, metric: str = "macro_f1", higher_is_better: bool = True, zero_tolerance: float = 1e-12,
) -> DownstreamMetrics:
    """Positive loss means degradation; positive recovery means improvement.

    Rate is undefined when loss <= tolerance or no repair has been evaluated.
    Valid ratios are not clipped: negative and above-one values are informative.
    The tolerance is numerical, not an empirical readiness threshold.
    """
    direction, tolerance = _settings(higher_is_better, zero_tolerance)
    if not isinstance(metric, str) or not metric:
        raise ValueError("A metric name is required")
    clean, corrupted = _finite(clean_score, "clean_score"), _finite(corrupted_score, "corrupted_score")
    repaired = _finite(repaired_score, "repaired_score") if repaired_score is not None else None
    loss = _finite(direction * (clean - corrupted), "performance_loss")
    recovery = _finite(direction * (repaired - corrupted), "performance_recovery") if repaired is not None else None
    reason = None
    if loss <= tolerance:
        reason = "no_positive_performance_loss"
    elif repaired is None:
        reason = "repair_not_evaluated"
    rate = _finite(recovery / loss, "recovery_rate") if reason is None else None
    return DownstreamMetrics(
        metric=metric, higher_is_better=higher_is_better, zero_tolerance=tolerance,
        clean_score=clean, corrupted_score=corrupted, repaired_score=repaired,
        performance_loss=loss, performance_recovery=recovery, recovery_rate=rate,
        recovery_rate_reason=reason,
        repair_harmed_performance=recovery < -tolerance if recovery is not None else None,
    )


def evaluate_harmful_actions(
    action_scores: Iterable[tuple[float, float]], *,
    higher_is_better: bool = True, zero_tolerance: float = 1e-12,
) -> HarmfulActionMetrics:
    """Each pair is an actually evaluated (before, after) action comparison.

    Whole-plan scores cannot identify which individual action caused harm.
    """
    direction, tolerance = _settings(higher_is_better, zero_tolerance)
    total = harmful = 0
    for before, after in action_scores:
        change = _finite(direction * (_finite(after, "after") - _finite(before, "before")), "change")
        total += 1
        harmful += int(change < -tolerance)
    return HarmfulActionMetrics(total, harmful, harmful / total if total else None)
