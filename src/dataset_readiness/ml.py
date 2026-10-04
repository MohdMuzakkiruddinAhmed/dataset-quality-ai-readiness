"""Training-only preprocessing for pinned classification baselines."""

import warnings
import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_readiness.datasets.adult import ADULT
from dataset_readiness.numeric_types import ML_VERSION, numeric_ml_copy
from dataset_readiness.added_features import ML_VERSION as FEATURE_ML_VERSION, validate_frame as validate_feature_frame
from dataset_readiness.model_config import TREE_PREPROCESSING, validate_ml_config

PREPROCESSING = {
    "version": "adult_lr_v1",
    "numeric": "training median fallback, then StandardScaler",
    "categorical": "constant __MISSING__ fallback, then one-hot encoding; ignore unknown categories",
    "fit_scope": "training partition only",
    "fallback_imputation": "Explicit ML fallback only; never applied to saved assessment data.",
    "feature_policy": "All 14 source features, including fnlwgt as an ordinary feature; row_id excluded.",
    "threshold": 0.5,
}


def preprocessing_for(config):
    if config.get("preprocessing") == TREE_PREPROCESSING:
        return {**PREPROCESSING, "version": TREE_PREPROCESSING,
                "numeric": "training median fallback; empty-column fallback zero; no scaling",
                "categorical": "constant __MISSING__ fallback, then dense one-hot encoding; ignore unknown categories",
                "matrix": "dense float64; encoded zeros remain observed zeros for both tree models",
                "prediction": "classifier.predict; >50K probability from class 1; fixed library binary threshold; no holdout tuning"}
    if config.get("preprocessing") == FEATURE_ML_VERSION:
        return {**PREPROCESSING, "version": FEATURE_ML_VERSION,
                "feature_policy": "All predictors in an explicit ordered schema; dynamic numeric/categorical subsets; row_id excluded; matched evaluation schema required"}
    if config.get("preprocessing", "adult_lr_v1") == ML_VERSION:
        return {**PREPROCESSING, "version": ML_VERSION,
                "numeric_observations": "Exact signed-int64 parsing; invalid tokens become missing in a private ML copy before training-median fallback; empty-column fallback zero. Same conversion for train and test.",
                "fit_scope": "conversion is stateless; imputation/scaling/encoding fit on training only"}
    return PREPROCESSING.copy()


def build_pipeline(config: dict, *, feature_schema=None) -> Pipeline:
    validate_ml_config(config)
    if (config.get("preprocessing") == FEATURE_ML_VERSION) != (feature_schema is not None):
        raise ValueError("Dynamic ML requires an explicit feature schema and its isolated protocol")
    trees = config["model"] != "logistic_regression"
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ] + ([] if trees else [("scaler", StandardScaler())]))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not trees)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, list(feature_schema.numeric_features if feature_schema else ADULT.numeric)),
        ("categorical", categorical, list(feature_schema.categorical_features if feature_schema else ADULT.categorical)),
    ])
    return Pipeline([("preprocessing", preprocessing), ("classifier", build_classifier(config))])


def build_classifier(config, *, dataset="adult"):
    """Shared model factory; dataset-specific preprocessing is validated separately."""
    validate_ml_config(config, dataset=dataset)
    if config["model"] == "logistic_regression":
        classifier = LogisticRegression(
            C=config["C"], solver=config["solver"], max_iter=config["max_iter"], random_state=config["seed"],
        )
    else:
        parameters = {k: v for k, v in config.items() if k not in ("model", "seed", "preprocessing")}
        parameters["random_state"] = config["seed"]
        if config["model"] == "random_forest":
            classifier = RandomForestClassifier(**parameters, criterion="gini", bootstrap=True, class_weight=None)
        else:
            from xgboost import XGBClassifier
            classifier = XGBClassifier(**parameters, enable_categorical=False, verbosity=0)
    return classifier


def evaluate_baseline(train: pd.DataFrame, test: pd.DataFrame, config: dict, *, feature_schema=None) -> dict:
    if feature_schema is not None:
        validate_feature_frame(train, feature_schema)
        validate_feature_frame(test, feature_schema)
    elif config.get("preprocessing") == FEATURE_ML_VERSION:
        raise ValueError("Dynamic ML requires an explicit feature schema")
    if config.get("preprocessing") == ML_VERSION:
        train, test = numeric_ml_copy(train), numeric_ml_copy(test)
    pipeline = build_pipeline(config, feature_schema=feature_schema) if feature_schema else build_pipeline(config)
    features = [c for c in feature_schema.columns if c != feature_schema.target] if feature_schema else list(ADULT.features)
    y_train = (train[ADULT.target] == ADULT.positive_label).astype(int)
    y_test = (test[ADULT.target] == ADULT.positive_label).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(train[features], y_train)
    predictions = pipeline.predict(test[features])
    probabilities = pipeline.predict_proba(test[features])[:, 1]
    classifier = pipeline.named_steps["classifier"]
    # XGBoost exposes NaN as its missing sentinel. Encode that setting explicitly
    # without allowing nonstandard JSON NaN or altering any computed score.
    parameters = {k: "NaN" if isinstance(v, float) and math.isnan(v) else v for k, v in classifier.get_params().items()}
    diagnostics = {} if config["model"] == "logistic_regression" else {
        "training_diagnostics": {"model": config["model"], "classes": classifier.classes_.tolist(),
                                 "transformed_feature_count": int(classifier.n_features_in_),
                                 "fitted_trees": len(classifier.estimators_) if config["model"] == "random_forest" else classifier.get_booster().num_boosted_rounds(),
                                 "early_stopping": False, "holdout_used_in_fit": False}}
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "positive_label": ADULT.positive_label,
        "confusion_matrix_labels": ["<=50K", ">50K"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "solver_iterations": classifier.n_iter_.tolist() if config["model"] == "logistic_regression" else None,
        "hyperparameters": parameters,
        **diagnostics,
    }
