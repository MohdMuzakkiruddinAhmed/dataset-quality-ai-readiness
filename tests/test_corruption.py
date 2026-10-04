from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from dataset_readiness.artifacts import json_bytes
from dataset_readiness.corruption import MissingnessConfig, inject_missing_values


@pytest.fixture
def mixed_data():
    frame = pd.DataFrame({
        "integer": pd.Series([2**53 + i for i in range(10)], dtype="int64"),
        "numeric": [np.nan, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0, 11.25],
        "category": pd.Categorical(["A", "B", "A", "B", None, "A", "B", "A", "B", "A"]),
        "text": pd.Series(["hello", "world"] * 5, dtype="string"),
        "untouched": np.arange(10),
    })
    frame.index = pd.Index(range(10, 110, 10), name="row_id")
    labels = pd.Series(["no", "yes"] * 5, index=frame.index, name="outcome")
    return frame, labels


def test_exact_changes_match_manifest_and_preserve_inputs(mixed_data):
    features, labels = mixed_data
    before, before_labels = features.copy(deep=True), labels.copy(deep=True)
    config = MissingnessConfig(0.3, 42, ("integer", "numeric", "category", "text"))
    result = inject_missing_values(features, labels, config)
    pd.testing.assert_frame_equal(features, before)
    pd.testing.assert_series_equal(labels, before_labels)
    pd.testing.assert_series_equal(result.labels, labels)
    pd.testing.assert_series_equal(result.dataframe.untouched, features.untouched)
    assert result.dataframe.index.equals(features.index)
    assert result.dataframe.columns.equals(features.columns)

    newly_missing = result.dataframe.isna() & features.notna()
    actual = {(int(features.index[r]), features.columns[c]) for r, c in zip(*np.where(newly_missing))}
    manifest = result.manifest.to_dict()
    assert manifest["eligible_cell_count"] == 38
    assert manifest["preexisting_missing_count"] == 2
    assert manifest["injected_cell_count"] == 11  # floor(0.3 * 38)
    declared = {(cell.row_id, cell.column) for cell in result.manifest.affected_cells}
    assert actual == declared
    assert len(declared) == len(result.manifest.affected_cells)
    for cell in result.manifest.affected_cells:
        assert cell.original_value == features.at[cell.row_id, cell.column]
        assert cell.corrupted_value is None
    for column in features:
        pd.testing.assert_series_equal(
            result.dataframe.loc[~newly_missing[column], column],
            features.loc[~newly_missing[column], column], check_dtype=False,
        )
    # Nullable integers must preserve values too large to represent as float64.
    assert str(result.dataframe.integer.dtype) == "Int64"
    serialized = json.loads(json_bytes(manifest))
    assert len(serialized["affected_cells"]) == 11
    assert sum(serialized["counts_by_column"].values()) == 11


def test_reproducibility_and_local_random_generator(mixed_data):
    features, labels = mixed_data
    config = MissingnessConfig(0.5, 42, ("integer", "category"))
    global_state = np.random.get_state()
    first = inject_missing_values(features, labels, config)
    repeated = inject_missing_values(features, labels, config)
    other_seed = inject_missing_values(features, labels, replace(config, seed=43))
    pd.testing.assert_frame_equal(first.dataframe, repeated.dataframe)
    assert json_bytes(first.manifest.to_dict()) == json_bytes(repeated.manifest.to_dict())
    assert first.manifest.affected_cells != other_seed.manifest.affected_cells
    after = np.random.get_state()
    assert global_state[0] == after[0]
    np.testing.assert_array_equal(global_state[1], after[1])
    assert global_state[2:] == after[2:]


@pytest.mark.parametrize("severity, expected", [(0.0, 0), (0.001, 0), (0.01, 1), (0.29, 29), (1.0, 100)])
def test_severity_rounding_and_endpoints(severity, expected):
    features = pd.DataFrame({"number": range(100)})
    labels = pd.Series([0, 1] * 50, name="target")
    result = inject_missing_values(features, labels, MissingnessConfig(severity, 42, ("number",)))
    assert len(result.manifest.affected_cells) == expected
    assert int(result.dataframe.isna().sum().sum()) == expected
    if not expected:
        pd.testing.assert_frame_equal(features, result.dataframe)


@pytest.mark.parametrize("row_count", [0, 10])
def test_no_eligible_cells_is_a_recorded_noop(row_count):
    features = pd.DataFrame({"number": [np.nan] * row_count})
    labels = pd.Series([0] * row_count, name="target", dtype=int)
    result = inject_missing_values(features, labels, MissingnessConfig(1.0, 42, ("number",)))
    pd.testing.assert_frame_equal(features, result.dataframe)
    assert result.manifest.eligible_cell_count == 0
    assert result.manifest.to_dict()["injected_cell_count"] == 0
    assert result.manifest.to_dict()["realized_severity"] == 0.0


@pytest.mark.parametrize("severity", [-0.1, 1.1, float("nan"), float("inf"), "0.1", True])
def test_invalid_severity_is_rejected(severity):
    with pytest.raises(ValueError, match="severity"):
        MissingnessConfig(severity, 42, ("age",))


@pytest.mark.parametrize("seed", [-1, 2**32, 0.5, "42", True])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        MissingnessConfig(0.1, seed, ("age",))


@pytest.mark.parametrize("columns", [(), ("age", "age"), ("",), (1,), "age"])
def test_invalid_column_selection_is_rejected(columns):
    with pytest.raises(ValueError, match="column"):
        MissingnessConfig(0.1, 42, columns)


def test_target_unknown_columns_and_row_alignment_are_protected(mixed_data):
    features, labels = mixed_data
    with pytest.raises(ValueError, match="Target corruption"):
        inject_missing_values(features, labels, MissingnessConfig(0.1, 42, ("outcome",)))
    with pytest.raises(ValueError, match="Unknown feature"):
        inject_missing_values(features, labels, MissingnessConfig(0.1, 42, ("unknown",)))
    config = MissingnessConfig(0.1, 42, ("integer",))
    with pytest.raises(ValueError, match="separately"):
        inject_missing_values(features.assign(outcome=labels), labels, config)
    with pytest.raises(ValueError, match="same order"):
        inject_missing_values(features, labels.iloc[::-1], config)
    with pytest.raises(ValueError, match="unique"):
        inject_missing_values(pd.concat([features, features]), pd.concat([labels, labels]), config)
    with pytest.raises(ValueError, match="stable integer"):
        inject_missing_values(features.set_axis(features.index.astype(str)), labels, config)
