"""Schema-driven binary-model evaluation; isolated Diabetes preprocessing v1."""

import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_readiness.ml import build_classifier
from dataset_readiness.model_config import validate_ml_config


def preprocessing_for_task(config, task):
    validate_ml_config(config, dataset=task.dataset)
    return {'version': config['preprocessing'], 'task_schema': task.receipt(), 'fit_scope':'training partition only',
            'numeric':'training median; empty-column zero fallback; '+('StandardScaler' if config['model']=='logistic_regression' else 'no scaling'),
            'categorical':'constant __MISSING__ fallback; dense float64 one-hot; max_categories=32 per column; unknown categories all zeros',
            'category_grouping':'training frequencies only; includes infrequent bucket in maximum 32; raw categories preserved in saved data',
            'fallback_scope':'private model matrices only; no changes to stored reference, training or test data',
            'labels':[task.negative_label,task.positive_label], 'decision':'classifier.predict; binary library threshold 0.5',
            'probability':'class 1 is the declared positive label', 'holdout_tuning':False, 'early_stopping':False}


def build_task_pipeline(config, task):
    validate_ml_config(config, dataset=task.dataset)
    if task.dataset != 'diabetes': raise ValueError('This preprocessing adapter currently supports Diabetes only')
    numeric=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if config['model']=='logistic_regression': numeric.append(('scaler',StandardScaler()))
    prep=ColumnTransformer([
        ('numeric',Pipeline(numeric),list(task.numeric)),
        ('categorical',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='__MISSING__',keep_empty_features=True)),
                                 ('encoder',OneHotEncoder(handle_unknown='ignore',max_categories=32,sparse_output=False))]),list(task.categorical))])
    return Pipeline([('preprocessing',prep),('classifier',build_classifier(config,dataset=task.dataset))])


def score_binary(y_true, labels, probabilities, task):
    both=len(np.unique(y_true))==2; positive=bool(np.any(y_true==1))
    return {'accuracy':float(accuracy_score(y_true,labels)),
            'macro_f1':float(f1_score(y_true,labels,labels=[0,1],average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y_true,labels,labels=[0,1],average='weighted',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y_true,labels)) if both else None,
            'roc_auc':float(roc_auc_score(y_true,probabilities)) if both else None,
            'average_precision':float(average_precision_score(y_true,probabilities)) if positive else None,
            'undefined_metric_reasons':{**({} if both else {'roc_auc':'test_contains_one_class','balanced_accuracy':'test_contains_one_class'}),
                                        **({} if positive else {'average_precision':'no_positive_test_labels'})},
            'positive_label':task.positive_label,'confusion_matrix_labels':[task.negative_label,task.positive_label],
            'confusion_matrix':confusion_matrix(y_true,labels,labels=[0,1]).tolist()}


def evaluate_task(train, test, config, task):
    task.validate_frame(train,training=True);task.validate_frame(test)
    if set(train.index)&set(test.index): raise ValueError('Training and test row IDs overlap')
    pipeline=build_task_pipeline(config,task);features=list(task.features)
    y_train=train[task.target].eq(task.positive_label).astype(int);y_test=test[task.target].eq(task.positive_label).astype(int)
    started=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipeline.fit(train[features],y_train)
    fitted=time.perf_counter()
    model=pipeline.named_steps['classifier']
    if model.classes_.tolist()!=[0,1]: raise ValueError('Unexpected fitted class/probability order')
    labels=pipeline.predict(test[features]);probabilities=pipeline.predict_proba(test[features])[:,1]
    if not np.isfinite(probabilities).all() or np.any(probabilities<0) or np.any(probabilities>1): raise ValueError('Invalid model probabilities')
    predicted=time.perf_counter()
    metrics=score_binary(y_test.to_numpy(),labels,probabilities,task)
    metrics['solver_iterations']=model.n_iter_.tolist() if config['model']=='logistic_regression' else None
    metrics['hyperparameters']={k:'NaN' if isinstance(v,float) and math.isnan(v) else v for k,v in model.get_params().items()}
    metrics['training_diagnostics']={'model':config['model'],'classes':model.classes_.tolist(),
        'training_rows':len(train),'transformed_feature_count':int(model.n_features_in_),
        'fitted_trees':None if config['model']=='logistic_regression' else len(model.estimators_) if config['model']=='random_forest' else model.get_booster().num_boosted_rounds(),
        'holdout_used_in_fit':False,'early_stopping':False,'convergence_warnings_are_errors':True}
    prep=pipeline.named_steps['preprocessing'];numeric=prep.named_transformers_['numeric'];encoder=prep.named_transformers_['categorical'].named_steps['encoder']
    fitted_receipt={'numeric_medians':[float(v) for v in numeric.named_steps['imputer'].statistics_],
                    'numeric_scaler_mean':numeric.named_steps['scaler'].mean_.tolist() if 'scaler' in numeric.named_steps else None,
                    'numeric_scaler_scale':numeric.named_steps['scaler'].scale_.tolist() if 'scaler' in numeric.named_steps else None,
                    'categorical_vocabulary':[values.tolist() for values in encoder.categories_],
                    'infrequent_categories':[values.tolist() if values is not None else None for values in encoder.infrequent_categories_],
                    'output_feature_names':prep.get_feature_names_out().tolist()}
    predictions=pd.DataFrame({'predicted_label':np.where(labels==1,task.positive_label,task.negative_label),
                              'positive_probability':probabilities.astype(float)},index=test.index)
    return metrics, predictions, fitted_receipt, {'fit_seconds':fitted-started,'predict_seconds':predicted-fitted}
