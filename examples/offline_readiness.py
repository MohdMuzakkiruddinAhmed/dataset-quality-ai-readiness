"""Illustrative synthetic data only: exercise original deterministic research code."""
import json

import pandas as pd

from dataset_readiness.datasets.diabetes import DIABETES
from dataset_readiness import readiness_challenge as policy


def synthetic_frame():
    task = DIABETES
    frame = pd.DataFrame({c: pd.Series([i % 7 + 1 for i in range(20)], dtype="Int64")
                          if c in task.numeric else ["category_a" if i % 2 else "category_b" for i in range(20)]
                          for c in task.features})
    frame[task.target] = [task.positive_label if i % 2 else task.negative_label for i in range(20)]
    frame = frame.loc[:, list(task.columns)]
    frame.index.name = "row_id"
    return frame


def main():
    original = synthetic_frame()
    observed, injection = policy.generate(original, policy.CASES[2])
    decision = policy.rules_readiness(observed)
    repaired, manifest, verification = policy.repair(observed)
    assert verification["protected_target_and_structure"]
    assert verification["unapproved_value_changes"] == 0
    assert not policy.critical(repaired)["numeric_domain_violation"]
    print(json.dumps({"illustrative_synthetic_example": True, "rows": len(original),
                      "injected_cells": injection["changed_cells"], "rules_decision": decision,
                      "repaired_cells": manifest["changed_cells"], "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
