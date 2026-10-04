"""Explicit categorical vocabulary; preserve valid values and reject ambiguity."""

from functools import lru_cache
from io import BytesIO

import pandas as pd

from dataset_readiness.artifacts import csv_bytes
from dataset_readiness.datasets.adult import ADULT
from dataset_readiness.datasets.adult_categories import DOMAINS, VERSION, SOURCE_FILE, SOURCE_SHA256

OBSERVED_VERSION = "adult_categorical_observed_v1"
NORMALIZATION = "Unicode strip then casefold for lookup only; exact declared values protected; unique matches only"


@lru_cache(maxsize=128)
def _index(vocabulary):
    if (not isinstance(vocabulary, tuple) or not vocabulary or
            any(not isinstance(v, str) or not v or v != v.strip() for v in vocabulary) or
            len(set(vocabulary)) != len(vocabulary)):
        raise ValueError("Vocabulary requires unique nonempty canonical strings without surrounding whitespace")
    index = {}
    for value in vocabulary:
        index.setdefault(value.casefold(), []).append(value)
    return frozenset(vocabulary), {k: tuple(v) for k, v in index.items()}


def resolve_category(value, vocabulary):
    allowed, lookup = _index(vocabulary)
    if not isinstance(value, str):
        return ("missing", None) if pd.isna(value) else ("unknown", None)
    if value in allowed:
        return "valid", value
    candidates = lookup.get(value.strip().casefold(), ())
    if len(candidates) == 1:
        return "variant", candidates[0]
    return ("ambiguous", None) if candidates else ("unknown", None)


def validate_domains(domains, categorical_features):
    if not isinstance(domains, dict) or not domains or not set(domains) <= set(categorical_features):
        raise ValueError("Domains must name declared categorical features only")
    for vocabulary in domains.values():
        if not isinstance(vocabulary, tuple):
            raise ValueError("Domain vocabulary must be a tuple")
        _index(vocabulary)


def schema_receipt(domains=None):
    return {"version": VERSION if domains is None else "explicit_supplied_domains",
            **({"source_file": SOURCE_FILE, "source_sha256": SOURCE_SHA256} if domains is None else {}),
            "normalization": NORMALIZATION, "domains": {k: list(v) for k, v in (DOMAINS if domains is None else domains).items()}}


def read_categorical_adult(payload):
    """Empty alone is missing; preserve exact categorical whitespace/case/NA text."""
    frame = pd.read_csv(BytesIO(payload), index_col="row_id", keep_default_na=False, na_values=[""],
                        dtype={"row_id": "int64", **{c: "Int64" for c in ADULT.numeric},
                               **{c: object for c in (*ADULT.categorical, ADULT.target)}})
    if not frame.index.is_unique or list(frame.columns) != list(ADULT.columns) or csv_bytes(frame) != payload:
        raise ValueError("Categorical Adult observations did not round-trip with unique rows and exact schema")
    return frame
