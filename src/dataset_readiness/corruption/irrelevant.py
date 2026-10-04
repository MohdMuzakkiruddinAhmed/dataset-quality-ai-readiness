"""Two declared added columns; independent partition streams, no label access."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset_readiness.added_features import FeatureSchema, NAMES, original_schema, validate_frame
from dataset_readiness.artifacts import csv_bytes, json_bytes, sha256, write_once

VERSION = "added_irrelevant_features_v1"


@dataclass(frozen=True)
class AddedFeatureConfig:
    count: int
    seed: int

    def __post_init__(self):
        if type(self.count) is not int or self.count != 2:
            raise ValueError("The first added-feature protocol requires exactly two columns")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid added-feature seed")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"type", "count", "seed"} or value["type"] != "added_irrelevant_features":
            raise ValueError("Added features require type, count, seed; severity is a column count")
        return cls(value["count"], value["seed"])


def protocol_receipt():
    return {"version": VERSION, "count": 2, "names": list(NAMES), "constant": 7,
            "noise": "uniform signed integers [-1000000, 1000000], inclusive; nullable Int64; exact decimal CSV",
            "rng": "Generator(PCG64(SeedSequence([seed, partition, 1701])))",
            "partitions": {"train": 0, "test": 1}, "sampling": "one noise draw per row in saved partition order; no labels/features/outcomes used",
            "test_policy": "separate derived view; original holdout bytes, labels, row IDs and features preserved"}


def inject_features(frame, config, partition):
    validate_frame(frame, original_schema())
    if partition not in ("train", "test"): raise ValueError("Unknown feature partition")
    if set(NAMES) & set(frame.columns): raise ValueError("Added feature name collision")
    stream = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.seed, {"train": 0, "test": 1}[partition], 1701])))
    after = frame.copy(deep=True)
    after[NAMES[0]] = pd.array([7] * len(frame), dtype="Int64")
    after[NAMES[1]] = pd.array(stream.integers(-1000000, 1000001, size=len(frame), dtype=np.int64), dtype="Int64")
    base = original_schema()
    schema = FeatureSchema((*base.columns[:-1], *NAMES, base.target), (*base.numeric_features, *NAMES), base.categorical_features)
    after = after[list(schema.columns)]
    validate_frame(after, schema)
    return after, schema


def manifest_for(train, test, observed, evaluation_view, config, schema):
    return {"schema_version": 1, "engine_version": VERSION, "corruption_type": "added_irrelevant_features",
            "seed": config.seed, "added_feature_count": config.count, "protocol": protocol_receipt(),
            "original_schema": original_schema().to_dict(), "observed_schema": schema.to_dict(),
            "train_sha256": sha256(csv_bytes(train)), "test_sha256": sha256(csv_bytes(test)),
            "observed_sha256": sha256(csv_bytes(observed)), "evaluation_view_sha256": sha256(csv_bytes(evaluation_view)),
            "added_columns": [{"column": NAMES[0], "kind": "constant", "value": 7}, {"column": NAMES[1], "kind": "independent_noise"}]}


def prepare_features(root, train, test, config):
    observed, schema = inject_features(train, config, "train")
    view, view_schema = inject_features(test, config, "test")
    assert view_schema == schema
    manifest = manifest_for(train, test, observed, view, config, schema)
    case = "case_" + sha256(json_bytes(manifest))[:24]
    directory = root / "data/corrupted/adult" / case
    evaluation = root / "data/evaluation_views/adult" / case / "test.csv"
    truth = root / "artifacts/corruption_manifests" / (case + ".json")
    bundle = {"dataset": "adult", "task": "binary_classification", "schema": schema.to_dict(),
              "observations": "train.csv", "row_id_role": "audit metadata only; never a predictor"}
    outputs = {directory / "train.csv": csv_bytes(observed), directory / "agent_input.json": json_bytes(bundle),
               evaluation: csv_bytes(view), truth: json_bytes(manifest)}
    for path, data in outputs.items(): write_once(path, data)
    receipt = {"case_id": case, "engine_version": VERSION, "feature_schema": schema.to_dict(), "added_feature_count": config.count}
    for key, path in (("corrupted_train", directory / "train.csv"), ("agent_input", directory / "agent_input.json"),
                      ("evaluation_view", evaluation), ("evaluator_manifest", truth)):
        receipt[key] = path.relative_to(root).as_posix()
        receipt["manifest_sha256" if key == "evaluator_manifest" else key + "_sha256"] = sha256(outputs[path])
    return observed, view, schema, receipt
