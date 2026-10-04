"""Strict downstream model settings; tree experiments have an isolated protocol."""

import math

TREE_PREPROCESSING = "adult_trees_dense_v1"
DIABETES_LR_PREPROCESSING = "diabetes_lr_dense32_v1"
DIABETES_TREE_PREPROCESSING = "diabetes_trees_dense32_v1"
MODELS = ("logistic_regression", "random_forest", "xgboost")


def validate_ml_config(config, *, dataset="adult"):
    if dataset not in ("adult", "diabetes"):
        raise ValueError("Unsupported ML dataset adapter")
    if not isinstance(config, dict) or config.get("model") not in MODELS:
        raise ValueError("Unsupported ML model configuration")
    model = config["model"]
    if dataset == "diabetes" and config.get("preprocessing") != (DIABETES_LR_PREPROCESSING if model == "logistic_regression" else DIABETES_TREE_PREPROCESSING):
        raise ValueError("Diabetes models require their explicit dataset preprocessing protocol")
    required = {"model", "seed"}
    if model == "logistic_regression":
        required |= {"C", "max_iter", "solver"}
    elif model == "random_forest":
        required |= {"preprocessing", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_jobs"}
    else:
        required |= {"preprocessing", "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                     "min_child_weight", "reg_alpha", "reg_lambda", "gamma", "max_bin", "tree_method", "device",
                     "n_jobs", "objective", "eval_metric", "booster"}
    if not required <= set(config) or set(config) - required - {"preprocessing"}:
        raise ValueError("ML configuration fields do not match model " + model)
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Seeds must be integers between 0 and 2**32 - 1")
    if model == "logistic_regression":
        if config["solver"] != "lbfgs":
            raise ValueError("Logistic regression requires lbfgs")
        if (type(config["C"]) not in (int, float) or not math.isfinite(config["C"]) or config["C"] <= 0
                or type(config["max_iter"]) is not int or config["max_iter"] <= 0):
            raise ValueError("C and max_iter must be positive and finite; max_iter must be integer")
        protocols = ("adult_lr_v1", "adult_lr_numeric_tokens_v1", "adult_dynamic_features_lr_v1") if dataset == "adult" else (DIABETES_LR_PREPROCESSING,)
        if config.get("preprocessing", "adult_lr_v1") not in protocols:
            raise ValueError("Unsupported ML preprocessing protocol")
        return
    if config["preprocessing"] != (TREE_PREPROCESSING if dataset == "adult" else DIABETES_TREE_PREPROCESSING):
        raise ValueError("Tree models require the isolated dense preprocessing protocol")
    for key in ("n_estimators", "max_depth", "min_samples_leaf" if model == "random_forest" else "max_bin"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if type(config["n_jobs"]) is not int or config["n_jobs"] != 1:
        raise ValueError("Tree model experiments require n_jobs=1")
    if model == "random_forest":
        if config["max_features"] != "sqrt":
            raise ValueError("Random forest requires max_features=sqrt")
        return
    for key in ("learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda", "gamma"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative")
        if key in ("learning_rate", "subsample", "colsample_bytree") and not 0 < value <= 1:
            raise ValueError(key + " must be in (0, 1]")
    for key, expected in {"tree_method": "hist", "device": "cpu", "objective": "binary:logistic",
                          "eval_metric": "logloss", "booster": "gbtree"}.items():
        if config[key] != expected:
            raise ValueError(key + " must be " + expected)
    if config["max_bin"] < 2:
        raise ValueError("max_bin must be at least 2")
