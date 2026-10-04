"""Deterministic holdout with identical feature vectors kept in one partition."""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def fixed_split(
    frame: pd.DataFrame, target: str, *, seed: int, folds: int = 5, test_fold: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not frame.index.is_unique:
        raise ValueError("Reference row IDs must be unique")
    if folds < 2 or not 0 <= test_fold < folds:
        raise ValueError("Require folds >= 2 and 0 <= test_fold < folds")
    if frame[target].isna().any() or frame[target].nunique() < 2:
        raise ValueError("Splitting requires non-missing classification labels")
    if frame[target].value_counts().min() < folds:
        raise ValueError("Each class must have at least folds rows")
    features = frame.drop(columns=target)
    groups = pd.util.hash_pandas_object(features, index=False)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    train_positions, test_positions = list(splitter.split(features, frame[target], groups))[test_fold]
    train = frame.iloc[train_positions].sort_index().copy()
    test = frame.iloc[test_positions].sort_index().copy()
    if set(groups.iloc[train_positions]) & set(groups.iloc[test_positions]):
        raise ValueError("Identical features crossed the train/test boundary")
    if set(train[target]) != set(frame[target]) or set(test[target]) != set(frame[target]):
        raise ValueError("Both partitions must contain every class")
    return train, test
