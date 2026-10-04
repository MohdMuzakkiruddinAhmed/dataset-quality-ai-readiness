"""Verify bundled evidence and source bytes without importing scientific packages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for record in manifest["files"]:
        path = (ROOT / record["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Manifest path escapes package")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(record["path"])
    if failures:
        raise ValueError("Missing or changed package files: " + ", ".join(failures))
    sources = json.loads((ROOT / "evidence/source_manifest.json").read_text(encoding="utf-8"))
    for record in sources:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["published_sha256"], record["path"]
    print(json.dumps({"verified_files": len(manifest["files"]), "source_provenance_records": len(sources)}))


if __name__ == "__main__":
    verify()
