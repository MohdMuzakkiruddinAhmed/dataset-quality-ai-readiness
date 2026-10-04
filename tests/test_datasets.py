import pandas as pd
import pytest

from dataset_readiness.artifacts import write_once
from dataset_readiness.datasets.adult import ADULT, build_reference, download_adult, parse_adult
from dataset_readiness.datasets.split import fixed_split


def test_parser_handles_uci_format(tmp_path):
    source = tmp_path / "adult.test"
    source.write_text(
        "|1x3 Cross validator\n"
        "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K.\n"
        "50, ?, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, >50K.\n",
        encoding="utf-8",
    )
    frame = parse_adult(source)
    assert len(frame) == 2
    assert frame["income"].tolist() == ["<=50K", ">50K"]
    assert frame.loc[0, "workclass"] == "State-gov"
    assert pd.isna(frame.loc[1, "workclass"])
    assert str(frame["age"].dtype) == "Int64"


def test_reference_audits_exclusions_and_preserves_ids(tmp_path, adult_frame):
    frame = adult_frame.iloc[:4].copy()
    frame.loc[1, "occupation"] = "?"
    frame.loc[2] = frame.loc[0]
    frame.iloc[:3].to_csv(tmp_path / "adult.data", header=False, index=False)
    frame.iloc[3:].to_csv(tmp_path / "adult.test", header=False, index=False)
    reference, audit = build_reference(tmp_path)
    assert reference.index.tolist() == [0, 3]
    assert audit["excluded_missing_row_ids"] == [1]
    assert audit["excluded_duplicate_row_ids"] == [2]
    assert audit["raw_rows"] == 4


def test_fixed_split_is_reproducible_and_groups_identical_features(adult_frame):
    frame = adult_frame.copy()
    frame.loc[80] = frame.loc[0]
    frame.loc[80, "income"] = "<=50K"  # Same features, conflicting label.
    before = frame.copy(deep=True)
    train, test = fixed_split(frame, "income", seed=42)
    train_again, test_again = fixed_split(frame, "income", seed=42)
    pd.testing.assert_frame_equal(train, train_again)
    pd.testing.assert_frame_equal(test, test_again)
    pd.testing.assert_frame_equal(frame, before)
    assert not set(train.index) & set(test.index)
    assert set(train.index) | set(test.index) == set(frame.index)
    assert (0 in train.index) == (80 in train.index)
    assert set(train.income) == set(test.income) == {"<=50K", ">50K"}


def test_artifacts_are_write_once(tmp_path):
    path = tmp_path / "reference.csv"
    write_once(path, b"original")
    modified_at = path.stat().st_mtime_ns
    write_once(path, b"original")
    assert path.stat().st_mtime_ns == modified_at
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_once(path, b"changed")
    assert path.read_bytes() == b"original"


def test_source_tampering_is_rejected_before_network(tmp_path):
    (tmp_path / "adult.data").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_adult(tmp_path)
