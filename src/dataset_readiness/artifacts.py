"""Content fingerprints and write-once experiment artifacts."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Stable CSV representation, retaining audit row IDs and empty missing cells."""
    return frame.to_csv(index=True, index_label="row_id", lineterminator="\n").encode()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_once(path: Path, payload: bytes) -> None:
    """Reuse identical artifacts, but never replace existing different bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Artifact mismatch; refusing to overwrite {path}") from None
