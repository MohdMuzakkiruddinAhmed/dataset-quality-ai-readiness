"""Explicit disjoint numeric-missing/category-variant composition and controls."""

from dataclasses import asdict, dataclass

import numpy as np
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from dataset_readiness.artifacts import csv_bytes, json_bytes, sha256, write_once
from dataset_readiness.categorical import schema_receipt, read_categorical_adult
from dataset_readiness.corruption.missing import inject_missing_values
from dataset_readiness.corruption.categorical import inject_categories
from dataset_readiness.corruption.schemas import MissingnessConfig, CategoricalConfig
from dataset_readiness.datasets.adult import ADULT

VERSION = "disjoint_feature_defects_v1"
COMPONENTS = ("numeric_missing", "categorical_variants")


def protocol_receipt():
    return {"version": VERSION, "component_order": list(COMPONENTS),
            "seed_derivation": "uint32 SeedSequence([parent_seed, 1901, component_index]).generate_state(1)[0]; indices 0/1 fixed even in single-family controls",
            "numeric": "missing_values_v1; observed numeric cells; row-major; decimal floor; PCG64 choice without replacement",
            "categorical": "categorical_variants_v1; valid reversible declared tokens; row-major; decimal floor; sorted selection then uniform supported variant",
            "overlap": "reject intersecting column sets; no chains or overlapping cells",
            "controls": "same severity and streams; enable none, numeric only, category only, or both",
            "protected": "target, row IDs/order, feature schema, original files and entire held-out partition"}


@dataclass(frozen=True)
class CombinedConfig:
    severity: float
    seed: int
    components: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    def __post_init__(self):
        MissingnessConfig(self.severity, self.seed, self.numeric_columns)
        CategoricalConfig(self.severity, self.seed, self.categorical_columns)
        if not isinstance(self.components, tuple) or self.components != tuple(c for c in COMPONENTS if c in self.components):
            raise ValueError("Combined components require unique declared names in frozen order")
        if set(self.numeric_columns) & set(self.categorical_columns):
            raise ValueError("Overlapping component columns are not supported")

    @classmethod
    def from_mapping(cls, value):
        required = {"type", "severity", "seed", "components", "numeric_columns", "categorical_columns"}
        if not isinstance(value, dict) or set(value) != required or value["type"] != "combined_feature_defects":
            raise ValueError("Combined corruption requires the explicit disjoint component declaration")
        if any(not isinstance(value[k], list) for k in ("components", "numeric_columns", "categorical_columns")):
            raise ValueError("Combined columns and components must be lists")
        return cls(value["severity"], value["seed"], *(tuple(value[k]) for k in ("components", "numeric_columns", "categorical_columns")))


def component_seed(seed, index):
    return int(np.random.SeedSequence([seed, 1901, index]).generate_state(1)[0])


def inject_combined(features, labels, config, *, domains=None):
    if not isinstance(config, CombinedConfig):
        raise ValueError("Explicit combined configuration required")
    if not set(config.numeric_columns + config.categorical_columns) <= set(features):
        raise ValueError("Components must name observed feature columns")
    if any(not is_numeric_dtype(features[c].dtype) or is_bool_dtype(features[c].dtype) for c in config.numeric_columns):
        raise ValueError("Missing component requires numeric columns")
    observed = features.copy(deep=True)
    manifests = {}
    # Execute both validators even for disabled components (zero severity).
    for index, component in enumerate(COMPONENTS):
        severity = config.severity if component in config.components else 0
        seed = component_seed(config.seed, index)
        if component == COMPONENTS[0]:
            result = inject_missing_values(observed, labels, MissingnessConfig(severity, seed, config.numeric_columns))
            manifest = result.manifest.to_dict()
        else:
            result = inject_categories(observed, labels, CategoricalConfig(severity, seed, config.categorical_columns), domains=domains)
            manifest = result.manifest
        observed = result.dataframe
        manifests[component] = manifest
    cells = [{"component": name, **cell} for name in COMPONENTS for cell in manifests[name]["affected_cells"]]
    if len({(c["row_id"], c["column"]) for c in cells}) != len(cells):
        raise ValueError("Component cells overlap")
    return observed.assign(**{labels.name: labels.copy(deep=True)}), {
        "schema_version": 1, "protocol": protocol_receipt(),
        "configuration": {"severity": config.severity, "seed": config.seed, "components": list(config.components),
                          "numeric_columns": list(config.numeric_columns), "categorical_columns": list(config.categorical_columns)},
        "components": manifests, "affected_cells": cells, "injected_cell_count": len(cells),
        "categorical_schema": schema_receipt(domains)}


def prepare_combined(root, train, split, config):
    before = csv_bytes(train)
    if sha256(before) != split["train_sha256"]:
        raise ValueError("Combined training input differs from frozen split")
    observed, injection = inject_combined(train.drop(columns=ADULT.target), train[ADULT.target], config)
    observed = read_categorical_adult(csv_bytes(observed.loc[:, train.columns]))
    payload = csv_bytes(observed)
    identity = {"protocol": protocol_receipt(), "configuration": asdict(config),
                "train_sha256": sha256(before), "categorical_schema": schema_receipt()}
    case = "case_" + sha256(json_bytes(identity))[:24]
    directory = root / "data/corrupted/adult" / case
    manifest = {"injection": injection, "input_train_sha256": sha256(before),
                "corrupted_train_sha256": sha256(payload), "clean_test_sha256": split["test_sha256"]}
    truth = root / "artifacts/corruption_manifests" / (case + ".json")
    bundle = {"dataset": "adult", "task": "binary_classification", "data_file": "train.csv",
              "target": ADULT.target, "row_id_column": "row_id", "numeric_features": list(ADULT.numeric),
              "categorical_features": list(ADULT.categorical), "categorical_schema": schema_receipt(),
              "observed_schema_version": "adult_categorical_observed_v1"}
    write_once(directory / "train.csv", payload)
    write_once(directory / "agent_input.json", json_bytes(bundle))
    write_once(truth, json_bytes(manifest))
    if csv_bytes(train) != before or not train[ADULT.target].equals(observed[ADULT.target]):
        raise ValueError("Combined injection changed source or protected target")
    return observed, {"case_id": case, "engine_version": VERSION,
                      "corrupted_train": (directory / "train.csv").relative_to(root).as_posix(),
                      "corrupted_train_sha256": sha256(payload),
                      "agent_input": (directory / "agent_input.json").relative_to(root).as_posix(),
                      "evaluator_manifest": truth.relative_to(root).as_posix(),
                      "manifest_sha256": sha256(json_bytes(manifest))}
