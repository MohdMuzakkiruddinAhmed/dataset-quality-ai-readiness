"""Versioned deterministic patient-disjoint binary-classification holdout."""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

VERSION = 'patient_stratified_group_v1'


def validate_split(config):
    if not isinstance(config, dict) or set(config) != {'protocol','seed','folds','test_fold'} or config['protocol'] != VERSION:
        raise ValueError('Patient split requires protocol, seed, folds and test_fold')
    if any(type(config[k]) is not int for k in ('seed','folds','test_fold')):
        raise ValueError('Patient split settings must be integers')
    if not 0 <= config['seed'] < 2**32 or config['folds'] < 2 or not 0 <= config['test_fold'] < config['folds']:
        raise ValueError('Invalid patient split settings')


def patient_split(frame, task, patients, config):
    validate_split(config); task.validate_frame(frame, training=True)
    if (not isinstance(patients, pd.Series) or not patients.index.equals(frame.index)
            or patients.isna().any() or not pd.api.types.is_integer_dtype(patients.dtype)
            or pd.api.types.is_bool_dtype(patients.dtype) or (patients <= 0).any()):
        raise ValueError('Patient groups require aligned, complete positive integer IDs')
    if patients.nunique() < config['folds'] or frame[task.target].value_counts().min() < config['folds']:
        raise ValueError('Insufficient groups or class rows for requested folds')
    splitter = StratifiedGroupKFold(n_splits=config['folds'], shuffle=True, random_state=config['seed'])
    train_pos, test_pos = list(splitter.split(frame[list(task.features)], frame[task.target], patients))[config['test_fold']]
    train, test = frame.iloc[train_pos].sort_index().copy(), frame.iloc[test_pos].sort_index().copy()
    if set(patients.loc[train.index]) & set(patients.loc[test.index]): raise ValueError('Patient overlap across partitions')
    if train[task.target].nunique() != 2 or test[task.target].nunique() != 2:
        raise ValueError('Both partitions must contain both classes')
    return train, test
