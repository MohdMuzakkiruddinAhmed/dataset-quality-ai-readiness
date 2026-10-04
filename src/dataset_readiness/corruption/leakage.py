"""Intentional synthetic outcome encoding; originals never modified."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset_readiness.added_features import FeatureSchema, original_schema, validate_frame
from dataset_readiness.artifacts import csv_bytes, json_bytes, sha256, write_once
from dataset_readiness.leakage import CLASSES

VERSION = "exact_target_encoding_v1"
NAME = "attribute_01"
EVALUATION_POLICY = "intentional_outcome_leakage_v1"


@dataclass(frozen=True)
class LeakageConfig:
    count: int
    seed: int

    def __post_init__(self):
        if type(self.count) is not int or self.count != 1: raise ValueError("Exact target encoding adds exactly one column")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32: raise ValueError("Invalid leakage seed")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"type", "count", "seed"} or value["type"] != "exact_target_encoding":
            raise ValueError("Target encoding requires type, count, seed")
        return cls(value["count"], value["seed"])


def encoding(seed):
    LeakageConfig(1, seed)
    stream = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1801])))
    values = stream.choice(2001, size=2, replace=False) - 1000
    return {label: int(value) for label, value in zip(CLASSES, values)}


def protocol_receipt():
    return {"version": VERSION, "count": 1, "name": NAME, "classes": list(CLASSES),
            "mapping": "Generator(PCG64(SeedSequence([seed, 1801]))).choice(2001, size=2, replace=False) - 1000; assign in class order",
            "representation": "distinct signed integers [-1000,1000]; nullable Int64; exact decimal CSV",
            "evaluation_policy": EVALUATION_POLICY,
            "test_policy": "same mapping applied to held-out labels in a separate intentionally leaky diagnostic view; not deployable or valid generalization evidence",
            "seed_interpretation": "encoding variation of the same target information, not independent label/noise draws"}


def inject_leakage(frame, config):
    if not isinstance(config, LeakageConfig): raise ValueError("Leakage injection requires its isolated configuration")
    validate_frame(frame, original_schema())
    if NAME in frame: raise ValueError("Leaked feature name collision")
    base = original_schema(); mapping = encoding(config.seed)
    schema = FeatureSchema((*base.columns[:-1], NAME, base.target), (*base.numeric_features, NAME), base.categorical_features)
    after = frame.copy(deep=True)
    after[NAME] = pd.array([mapping[label] for label in frame[base.target]], dtype="Int64")
    after = after[list(schema.columns)]; validate_frame(after, schema)
    return after, schema


def manifest_for(train, test, observed, view, config, schema):
    return {"schema_version": 1, "engine_version": VERSION, "corruption_type": "exact_target_encoding",
            "seed": config.seed, "added_feature_count": config.count, "protocol": protocol_receipt(),
            "class_value_mapping": encoding(config.seed), "original_schema": original_schema().to_dict(), "observed_schema": schema.to_dict(),
            "train_sha256": sha256(csv_bytes(train)), "test_sha256": sha256(csv_bytes(test)),
            "observed_sha256": sha256(csv_bytes(observed)), "evaluation_view_sha256": sha256(csv_bytes(view)),
            "added_columns": [{"column": NAME, "kind": "exact_outcome_encoding"}],
            "affected_rows": [int(r) for r in train.index], "evaluation_rows": [int(r) for r in test.index]}


def prepare_leakage(root, train, test, config):
    observed, schema = inject_leakage(train, config); view, view_schema = inject_leakage(test, config)
    assert schema == view_schema
    manifest = manifest_for(train, test, observed, view, config, schema)
    case = "case_" + sha256(json_bytes(manifest))[:24]
    directory = root / "data/corrupted/adult" / case
    evaluation = root / "data/evaluation_views/adult" / case / "intentionally_leaky_test.csv"
    notice = evaluation.with_suffix(".json")
    truth = root / "artifacts/corruption_manifests" / (case + ".json")
    bundle = {"dataset": "adult", "task": "binary_classification", "schema": schema.to_dict(),
              "observations": "train.csv", "row_id_role": "audit metadata only; never a predictor"}
    note = {"policy": EVALUATION_POLICY, "contains_outcome_derived_predictor": True,
            "valid_deployment_estimate": False, "purpose": "controlled measurement of apparent score inflation from known injected leakage",
            "original_test_sha256": manifest["test_sha256"], "view_sha256": manifest["evaluation_view_sha256"]}
    outputs = {directory / "train.csv": csv_bytes(observed), directory / "agent_input.json": json_bytes(bundle),
               evaluation: csv_bytes(view), notice: json_bytes(note), truth: json_bytes(manifest)}
    for path, payload in outputs.items(): write_once(path, payload)
    receipt = {"case_id": case, "engine_version": VERSION, "feature_schema": schema.to_dict(), "added_feature_count": 1,
               "evaluation_policy": EVALUATION_POLICY, "leaky_score_is_valid_deployment_estimate": False}
    for key, path in (("corrupted_train", directory / "train.csv"), ("agent_input", directory / "agent_input.json"),
                      ("evaluation_view", evaluation), ("evaluation_notice", notice), ("evaluator_manifest", truth)):
        receipt[key] = path.relative_to(root).as_posix()
        receipt["manifest_sha256" if key == "evaluator_manifest" else key + "_sha256"] = sha256(outputs[path])
    return observed, view, schema, receipt
