"""Evaluator-only measurements; never expose reference truth to agents."""

from dataset_readiness.evaluation.detection import evaluate_detection, score_items
from dataset_readiness.evaluation.downstream import evaluate_downstream, evaluate_harmful_actions
from dataset_readiness.evaluation.remediation import evaluate_missing_repairs
from dataset_readiness.evaluation.schemas import CellRef, EvaluationReport, cells_from_manifest

__all__ = [
    "CellRef", "EvaluationReport", "cells_from_manifest", "evaluate_detection",
    "evaluate_downstream", "evaluate_harmful_actions", "evaluate_missing_repairs", "score_items",
]
