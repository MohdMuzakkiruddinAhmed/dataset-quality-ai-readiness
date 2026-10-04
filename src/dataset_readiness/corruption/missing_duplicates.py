"""Numeric missingness followed by exact observed-row copies; lineage stays hidden."""

from dataclasses import dataclass, asdict

import numpy as np
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from dataset_readiness.artifacts import csv_bytes, json_bytes, sha256, write_once
from dataset_readiness.categorical import read_categorical_adult
from dataset_readiness.corruption.schemas import MissingnessConfig, DuplicateConfig
from dataset_readiness.corruption.missing import inject_missing_values
from dataset_readiness.corruption.duplicates import inject_duplicates
from dataset_readiness.datasets.adult import ADULT

VERSION = "missing_then_duplicates_v1"
COMPONENTS = ("numeric_missing", "exact_duplicates")


def protocol_receipt():
    return {"version": VERSION, "component_order": list(COMPONENTS),
            "seed_derivation": "uint32 SeedSequence([parent_seed, 2001, component_index]).generate_state(1)[0]; fixed indices 0/1 in every arm",
            "missing": "missing_values_v1; decimal floor of eligible original numeric cells; row-major PCG64 sampling without replacement",
            "duplicates": "exact_duplicates_v1; decimal floor of pre-copy rows; independent PCG64 source positions then random unique negative int64 IDs",
            "lineage": "primary missing cells on original rows; inherited injected missing cells on copies; preexisting missingness separate",
            "controls": "enable none, missing only, duplicates only, or both; component streams fixed across controls",
            "protected": "original files, heldout, original row order/values except declared primary missing cells; copied target remains exact"}


@dataclass(frozen=True)
class MissingDuplicatesConfig:
    severity: float
    seed: int
    components: tuple[str, ...]
    numeric_columns: tuple[str, ...]

    def __post_init__(self):
        MissingnessConfig(self.severity, self.seed, self.numeric_columns)
        if not isinstance(self.components, tuple) or self.components != tuple(c for c in COMPONENTS if c in self.components):
            raise ValueError("Missing/duplicate components require unique supported names in frozen order")

    @classmethod
    def from_mapping(cls, value):
        if (not isinstance(value, dict) or set(value) != {"type", "severity", "seed", "components", "numeric_columns"}
                or value["type"] != "missing_and_duplicates" or not isinstance(value["components"], list)
                or not isinstance(value["numeric_columns"], list)):
            raise ValueError("Explicit missing-and-duplicates declaration required")
        return cls(value["severity"], value["seed"], tuple(value["components"]), tuple(value["numeric_columns"]))


def component_seed(seed, index):
    return int(np.random.SeedSequence([seed, 2001, index]).generate_state(1)[0])


def inject_missing_duplicates(frame, config, schema):
    if not isinstance(config, MissingDuplicatesConfig):
        raise ValueError("MissingDuplicatesConfig required")
    if set(frame.columns) != {schema.target, *schema.numeric_features, *schema.categorical_features} or not set(config.numeric_columns) <= set(schema.numeric_features):
        raise ValueError("Missing/duplicate injection requires complete declared schema and numeric columns")
    if frame[schema.target].isna().any():
        raise ValueError("Protected target must be complete")
    if any(not is_numeric_dtype(frame[c].dtype) or is_bool_dtype(frame[c].dtype) for c in config.numeric_columns):
        raise ValueError("Missing component requires numeric dtype")
    before = csv_bytes(frame)
    missing = inject_missing_values(frame.drop(columns=schema.target), frame[schema.target],
        MissingnessConfig(config.severity if COMPONENTS[0] in config.components else 0, component_seed(config.seed, 0), config.numeric_columns))
    primary = missing.dataframe.assign(**{schema.target: missing.labels}).loc[:, frame.columns]
    duplicated = inject_duplicates(primary, DuplicateConfig(config.severity if COMPONENTS[1] in config.components else 0, component_seed(config.seed, 1)))
    missing_manifest = missing.manifest.to_dict()
    by_source = {}
    for e in missing_manifest["affected_cells"]:
        by_source.setdefault(e["row_id"], []).append(e)
    inherited = [{**cell, "row_id": entry["row_id"], "source_row_id": entry["source_row_id"]}
                 for entry in duplicated.manifest["insertions"] for cell in by_source.get(entry["source_row_id"], [])]
    if csv_bytes(frame) != before:
        raise ValueError("Composition changed original input")
    return duplicated.dataframe, primary, {"schema_version": 1, "protocol": protocol_receipt(),
        "configuration": {"severity": config.severity, "seed": config.seed, "components": list(config.components), "numeric_columns": list(config.numeric_columns)},
        "missing": missing_manifest, "duplicates": duplicated.manifest, "inherited_missing_cells": inherited,
        "primary_missing_count": missing_manifest["injected_cell_count"], "inherited_missing_count": len(inherited),
        "observed_injected_missing_count": missing_manifest["injected_cell_count"] + len(inherited),
        "pre_duplicate_sha256": sha256(csv_bytes(primary))}


def prepare_missing_duplicates(root, train, split, config, schema):
    before = csv_bytes(train)
    if sha256(before) != split["train_sha256"]:
        raise ValueError("Frozen training hash mismatch")
    # Use one faithful nullable representation in all four control arms.
    normalized = read_categorical_adult(before)
    observed, _, injection = inject_missing_duplicates(normalized, config, schema)
    payload = csv_bytes(observed)
    if not read_categorical_adult(payload).equals(observed):
        raise ValueError("Missing/duplicate observations failed exact CSV round-trip")
    case = "case_" + sha256(json_bytes({"protocol": protocol_receipt(), "configuration": asdict(config), "train_sha256": sha256(before)}))[:24]
    directory = root / "data/corrupted/adult" / case
    truth = root / "artifacts/corruption_manifests" / (case + ".json")
    manifest = {"injection": injection, "input_train_sha256": sha256(before), "corrupted_train_sha256": sha256(payload), "clean_test_sha256": split["test_sha256"]}
    bundle = {"dataset": "adult", "task": "binary_classification", "data_file": "train.csv", "target": ADULT.target,
              "row_id_column": "row_id", "numeric_features": list(ADULT.numeric), "categorical_features": list(ADULT.categorical),
              "observed_schema_version": "adult_categorical_observed_v1"}
    write_once(directory / "train.csv", payload); write_once(directory / "agent_input.json", json_bytes(bundle)); write_once(truth, json_bytes(manifest))
    return observed, {"case_id": case, "engine_version": VERSION,
                     "corrupted_train": (directory / "train.csv").relative_to(root).as_posix(), "corrupted_train_sha256": sha256(payload),
                     "agent_input": (directory / "agent_input.json").relative_to(root).as_posix(),
                     "evaluator_manifest": truth.relative_to(root).as_posix(), "manifest_sha256": sha256(json_bytes(manifest))}
