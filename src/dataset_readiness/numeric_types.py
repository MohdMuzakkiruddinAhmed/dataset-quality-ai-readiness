"""Versioned integer observations: exact parsing, faithful CSV loading, ML copies."""

from decimal import Decimal, InvalidOperation
from io import BytesIO
import math
from numbers import Integral, Real
import re

import pandas as pd

from dataset_readiness.artifacts import csv_bytes
from dataset_readiness.datasets.adult import ADULT

VERSION = "adult_numeric_observed_v1"
ML_VERSION = "adult_lr_numeric_tokens_v1"
MINIMUM, MAXIMUM = -(2**63), 2**63 - 1


def integer_observation(value):
    """Return (missing/valid/parseable/invalid, exact int or None).

    Strings use finite decimal arithmetic, never a float intermediate. Numeric
    syntax accepts surrounding whitespace, signs, decimals and exponents only
    when exactly integral and within signed int64. Booleans are invalid.
    """
    if value is None or value is pd.NA or (isinstance(value, Real) and not isinstance(value, Integral) and math.isnan(value)):
        return "missing", None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return ("valid", int(value)) if MINIMUM <= value <= MAXIMUM else ("invalid", None)
    if isinstance(value, Real) and not isinstance(value, bool):
        if math.isfinite(value) and value == int(value) and MINIMUM <= int(value) <= MAXIMUM:
            return "valid", int(value)
        return "invalid", None
    if isinstance(value, str) and len(value) <= 128:
        token = value.strip()
        if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token):
            try:
                number = Decimal(token)
                if number.is_finite() and MINIMUM <= number <= MAXIMUM and number == number.to_integral_value():
                    return "parseable", int(number)
            except InvalidOperation:
                pass
    return "invalid", None


def observed_numeric_copy(frame, columns=ADULT.numeric):
    result = frame.copy(deep=True)
    for column in columns:
        result[column] = result[column].astype(object)
    return result


def read_observed_adult(payload):
    """CSV has lexical types: canonical int64 tokens are integers; others stay text.

    Empty fields alone mean missing. Strings such as NA, NaN, 001, 1.0, or a
    whitespace-padded value survive unchanged. Quoting cannot encode a numeric
    string's original runtime type; that distinction is outside this protocol.
    """
    frame = pd.read_csv(BytesIO(payload), index_col="row_id", keep_default_na=False,
                        dtype={"row_id": "int64", **{c: object for c in ADULT.columns}})
    if list(frame.columns) != list(ADULT.columns) or not frame.index.is_unique:
        raise ValueError("Observed Adult data requires the exact schema and unique row IDs")
    for column in ADULT.columns:
        def decode(value):
            if value == "":
                return pd.NA
            if column in ADULT.numeric:
                status, number = integer_observation(value)
                if status == "parseable" and str(number) == value:
                    return number
            return value
        frame[column] = pd.Series([decode(v) for v in frame[column]], index=frame.index, dtype=object)
    if csv_bytes(frame) != payload:
        raise ValueError("Observed Adult CSV did not round-trip exactly")
    return frame


def numeric_ml_copy(frame, columns=ADULT.numeric):
    """Declared ML-only fallback: lossless integers, invalid tokens become missing."""
    result = frame.copy(deep=True)
    for column in columns:
        values = [integer_observation(value)[1] for value in frame[column]]
        result[column] = pd.Series(pd.array(values, dtype="Int64"), index=frame.index)
    return result
