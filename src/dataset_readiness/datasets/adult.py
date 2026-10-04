"""UCI Adult ingestion. All normalization is confined to reference preparation."""

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from dataset_readiness.artifacts import sha256, write_once

SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
SOURCE_FILES = {
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
    "adult.names": "c248284c0b5de30c9e1958d6cdd168a34a654758b620e68f46aefa83fc0a576a",
}


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    columns: tuple[str, ...]
    numeric: tuple[str, ...]
    target: str
    positive_label: str

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(c for c in self.columns if c != self.target)

    @property
    def categorical(self) -> tuple[str, ...]:
        return tuple(c for c in self.features if c not in self.numeric)


ADULT = DatasetSchema(
    name="adult",
    columns=(
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ),
    numeric=("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"),
    target="income",
    positive_label=">50K",
)


def download_adult(directory: Path) -> dict[str, str]:
    """Download once; verify pinned source bytes on every invocation."""
    for name, expected in SOURCE_FILES.items():
        path = directory / name
        if path.exists():
            payload = path.read_bytes()
        else:
            with urlopen(f"{SOURCE_URL}/{name}", timeout=45) as response:
                payload = response.read()
        if sha256(payload) != expected:
            raise ValueError(f"Source checksum mismatch: {path}")
        write_once(path, payload)
    return dict(SOURCE_FILES)


def parse_adult(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, header=None, names=ADULT.columns, skipinitialspace=True,
        na_values=["?"], keep_default_na=False, comment="|",
    )
    for column in (*ADULT.categorical, ADULT.target):
        frame[column] = frame[column].str.strip().replace("?", float("nan"))
    frame[ADULT.target] = frame[ADULT.target].str.removesuffix(".")
    labels = frame[ADULT.target].dropna()
    if not labels.isin(["<=50K", ">50K"]).all():
        raise ValueError("Unexpected Adult target label")
    for column in ADULT.numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        present = frame[column].dropna()
        if (present % 1 != 0).any():
            raise ValueError(f"Non-integer value in {column}")
        frame[column] = frame[column].astype("Int64")
    return frame


def build_reference(raw_directory: Path) -> tuple[pd.DataFrame, dict]:
    parts = [parse_adult(raw_directory / name) for name in ("adult.data", "adult.test")]
    raw = pd.concat(parts, ignore_index=True)
    raw.index.name = "row_id"
    complete = raw.dropna()
    reference = complete.drop_duplicates().copy()
    audit = {
        "policy": "complete_cases_v1",
        "raw_rows": len(raw),
        "source_rows": {name: len(frame) for name, frame in zip(("adult.data", "adult.test"), parts)},
        "missing_by_column": {c: int(n) for c, n in raw.isna().sum().items()},
        "excluded_missing_row_ids": raw.index[raw.isna().any(axis=1)].tolist(),
        "excluded_duplicate_row_ids": complete.index[complete.duplicated()].tolist(),
        "reference_rows": len(reference),
        "feature_duplicate_rows_remaining": int(reference.duplicated(subset=list(ADULT.features)).sum()),
        "row_id_policy": "Zero-based record offset in adult.data followed by adult.test, before filtering.",
        "limitations": "Complete-case selection changes the population; source labels are not verified ground truth.",
    }
    return reference, audit
