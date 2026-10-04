import pandas as pd
import pytest

from dataset_readiness.datasets.adult import ADULT


@pytest.fixture
def adult_frame():
    rows = []
    for i in range(80):
        rows.append([
            20 + i, "Private" if i % 2 else "State-gov", 10000 + i * 120,
            "Bachelors" if i % 2 else "HS-grad", 13 if i % 2 else 9,
            "Never-married", "Sales", "Not-in-family", "White", "Female",
            i * 30, 0, 30 + i % 20, "United-States", ">50K" if i % 3 == 0 else "<=50K",
        ])
    frame = pd.DataFrame(rows, columns=ADULT.columns)
    frame.index.name = "row_id"
    return frame


@pytest.fixture
def config():
    return {
        "dataset": "adult", "reference_version": "complete_cases_v1",
        "split": {"seed": 42, "folds": 5, "test_fold": 0},
        "ml": {"model": "logistic_regression", "seed": 42, "C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
    }
