"""Isolated fixed multiclass models, train-only preprocessing and OvR metrics."""

import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from dataset_readiness.datasets.steel import STEEL
from dataset_readiness.model_config import validate_ml_config


def validate_config(config):
    if not isinstance(config, dict): raise ValueError('Invalid Steel model configuration')
    model = config.get('model')
    expected = 'steel_lr_dense_v1' if model == 'logistic_regression' else 'steel_trees_dense_v1'
    if config.get('preprocessing') != expected: raise ValueError('Steel preprocessing protocol required')
    copied = dict(config)
    copied['preprocessing'] = 'diabetes_lr_dense32_v1' if model == 'logistic_regression' else 'diabetes_trees_dense32_v1'
    if model == 'xgboost':
        if (config.get('objective') != 'multi:softprob' or config.get('eval_metric') != 'mlogloss'
                or type(config.get('num_class')) is not int or config['num_class'] != len(STEEL.labels)):
            raise ValueError('Steel XGBoost requires three-class soft probabilities')
        copied.pop('num_class'); copied.update(objective='binary:logistic', eval_metric='logloss')
    validate_ml_config(copied, dataset='diabetes')


def preprocessing(config):
    validate_config(config)
    return {'version': config['preprocessing'], 'task_schema': STEEL.receipt(),
        'fit_scope': 'Training partition only; no validation data or early stopping',
        'numeric': 'Median with empty-column zero fallback; ' + ('StandardScaler' if config['model'] == 'logistic_regression' else 'unscaled'),
        'categorical': 'Constant __MISSING__ fallback; full dense float64 one-hot vocabulary; unknown categories all zeros',
        'fallback_scope': 'Private model matrix only; stored reference unchanged', 'labels': list(STEEL.labels),
        'probability': 'Convert predict_proba to float64 and normalize each row by its sum; declared label order',
        'decision': 'Probability argmax; ties choose the first declared label',
        'metrics': 'Macro F1 includes all three labels; macro ROC-AUC/AP is unweighted mean of three one-vs-rest scores; log loss uses declared labels',
        'undefined': 'Any missing test class makes balanced accuracy and both macro OvR metrics null; per-class undefined values are explicit',
        'holdout_tuning': False, 'early_stopping': False}


def build_pipeline(config):
    validate_config(config)
    numeric = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if config['model'] == 'logistic_regression': numeric.append(('scaler', StandardScaler()))
    prep = ColumnTransformer([
        ('numeric', Pipeline(numeric), list(STEEL.numeric)),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__', keep_empty_features=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), list(STEEL.categorical))])
    params = {k: v for k, v in config.items() if k not in ('model', 'preprocessing', 'seed')}
    params['random_state'] = config['seed']
    if config['model'] == 'logistic_regression': model = LogisticRegression(**params)
    elif config['model'] == 'random_forest': model = RandomForestClassifier(**params, criterion='gini', bootstrap=True, class_weight=None)
    else: model = XGBClassifier(**params, enable_categorical=False, verbosity=0)
    return Pipeline([('preprocessing', prep), ('classifier', model)])


def normalized_probabilities(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Invalid three-class probabilities')
    return values / values.sum(axis=1, keepdims=True)


def score(truth, labels, probabilities):
    truth = np.asarray(truth); labels = np.asarray(labels); probabilities = np.asarray(probabilities)
    if (truth.ndim != 1 or labels.shape != truth.shape or not len(truth)
            or probabilities.shape != (len(truth), 3) or not np.isin(truth, [0, 1, 2]).all()
            or not np.isin(labels, [0, 1, 2]).all()): raise ValueError('Invalid multiclass scoring input')
    normalized_probabilities(probabilities)
    all_classes = len(np.unique(truth)) == 3
    per_class = {}
    for i, name in enumerate(STEEL.labels):
        positive = truth == i; present = bool(positive.any()); both = present and not bool(positive.all())
        per_class[name] = {'support': int(positive.sum()),
            'roc_auc_ovr': float(roc_auc_score(positive, probabilities[:, i])) if both else None,
            'average_precision': float(average_precision_score(positive, probabilities[:, i])) if present else None,
            'f1': float(f1_score(truth, labels, labels=[i], average='macro', zero_division=0)),
            'undefined_metric_reasons': {**({} if both else {'roc_auc_ovr': 'one_binary_class'}),
                **({} if present else {'average_precision': 'no_positive_test_labels'})}}
    return {'accuracy': float(accuracy_score(truth, labels)),
        'macro_f1': float(f1_score(truth, labels, labels=[0, 1, 2], average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(truth, labels, labels=[0, 1, 2], average='weighted', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, labels)) if all_classes else None,
        'roc_auc_ovr_macro': float(np.mean([v['roc_auc_ovr'] for v in per_class.values()])) if all_classes else None,
        'average_precision_macro': float(np.mean([v['average_precision'] for v in per_class.values()])) if all_classes else None,
        'log_loss': float(log_loss(truth, probabilities, labels=[0, 1, 2])),
        'undefined_metric_reasons': {} if all_classes else {k: 'missing_test_class' for k in ('balanced_accuracy', 'roc_auc_ovr_macro', 'average_precision_macro')},
        'per_class': per_class, 'confusion_matrix_labels': list(STEEL.labels),
        'confusion_matrix': confusion_matrix(truth, labels, labels=[0, 1, 2]).tolist()}


def evaluate(train, test, config):
    STEEL.validate_frame(train, training=True); STEEL.validate_frame(test)
    if set(train.index) & set(test.index): raise ValueError('Training and test row IDs overlap')
    mapping = dict(zip(STEEL.labels, range(3)))
    y_train = train.Load_Type.map(mapping).to_numpy(); y_test = test.Load_Type.map(mapping).to_numpy()
    pipeline = build_pipeline(config); started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipeline.fit(train[list(STEEL.features)], y_train)
    fitted_at = time.perf_counter(); model = pipeline.named_steps['classifier']
    if model.classes_.tolist() != [0, 1, 2]: raise ValueError('Unexpected fitted class order')
    probabilities = normalized_probabilities(pipeline.predict_proba(test[list(STEEL.features)]))
    labels = probabilities.argmax(axis=1); predicted_at = time.perf_counter()
    metrics = score(y_test, labels, probabilities)
    metrics['hyperparameters'] = {k: 'NaN' if isinstance(v, float) and math.isnan(v) else v for k, v in model.get_params().items()}
    metrics['solver_iterations'] = model.n_iter_.tolist() if config['model'] == 'logistic_regression' else None
    metrics['training_diagnostics'] = {'model': config['model'], 'classes': model.classes_.tolist(), 'training_rows': len(train),
        'transformed_feature_count': int(model.n_features_in_),
        'fitted_rounds_or_trees': None if config['model'] == 'logistic_regression' else len(model.estimators_) if config['model'] == 'random_forest' else model.get_booster().num_boosted_rounds(),
        'holdout_used_in_fit': False, 'early_stopping': False, 'convergence_warnings_are_errors': True}
    prep = pipeline.named_steps['preprocessing']; numeric = prep.named_transformers_['numeric']
    encoder = prep.named_transformers_['categorical'].named_steps['encoder']
    fitted = {'numeric_medians': numeric.named_steps['imputer'].statistics_.tolist(),
        'numeric_scaler_mean': numeric.named_steps['scaler'].mean_.tolist() if 'scaler' in numeric.named_steps else None,
        'numeric_scaler_scale': numeric.named_steps['scaler'].scale_.tolist() if 'scaler' in numeric.named_steps else None,
        'categorical_vocabulary': [v.tolist() for v in encoder.categories_],
        'output_feature_names': prep.get_feature_names_out().tolist()}
    predictions = pd.DataFrame({'predicted_label': [STEEL.labels[i] for i in labels]}, index=test.index)
    for i, name in enumerate(STEEL.labels): predictions['probability_' + name] = probabilities[:, i]
    prior = np.bincount(y_train, minlength=3) / len(y_train)
    control = {'policy': 'Training class frequencies as constant probabilities; training majority argmax with declared-order ties',
        'training_class_probabilities': prior.tolist(), 'predicted_label': STEEL.labels[int(prior.argmax())],
        'metrics': score(y_test, np.full(len(test), prior.argmax()), np.tile(prior, (len(test), 1)))}
    return metrics, predictions, fitted, control, {'fit_seconds': fitted_at-started, 'predict_seconds': predicted_at-fitted_at}
