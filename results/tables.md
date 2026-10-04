# Reproduced manuscript summaries

Calculated only from the bundled saved measurements. No new observations or model fits.

## Missingness (Table II)

| Dataset | Rate | Mean benefit (pp) | Sample SD (pp) | Positive / negative |
|---|---:|---:|---:|---:|
| adult | 5% | -0.425 | 0.059 | 0 / 3 |
| adult | 10% | -0.398 | 0.202 | 0 / 3 |
| adult | 20% | -0.633 | 0.313 | 0 / 3 |
| diabetes | 5% | +0.197 | 0.143 | 3 / 0 |
| diabetes | 10% | +0.370 | 0.288 | 3 / 0 |
| diabetes | 20% | +0.379 | 0.166 | 3 / 0 |
| steel | 5% | -0.168 | 0.166 | 1 / 2 |
| steel | 10% | -0.109 | 0.022 | 0 / 3 |
| steel | 20% | -0.047 | 0.174 | 1 / 2 |

## Declared Adult examples (Table III)

| Defect | Unit | Detection F1 | Benefit (pp) | Recovery |
|---|---|---:|---:|---:|
| missing_values | cell | 1.000 | -0.418 | Undefined |
| exact_duplicates | groups | 1.000 | +0.140 | 1.000 |
| numeric_type_tokens | cell | 1.000 | +0.000 | Undefined |
| numeric_domain_negatives | cell | 1.000 | +0.294 | 1.638 |
| categorical_variants | cell | 1.000 | -0.037 | Undefined |
| binary_label_flips | cell | 0.066 | +0.504 | 0.729 |
| added_irrelevant_features | column | 0.667 | +0.000 | 0.000 |
| exact_target_encoding | column | 1.000 | Invalid | Undefined |

## Native Diabetes (Table IV)

| Learner | Metric | Benefit (pp) |
|---|---|---:|
| logistic_regression | macro_f1 | +0.247 |
| logistic_regression | roc_auc | -0.200 |
| logistic_regression | average_precision | -0.282 |
| random_forest | macro_f1 | +0.000 |
| random_forest | roc_auc | -0.117 |
| random_forest | average_precision | -0.373 |
| xgboost | macro_f1 | -0.261 |
| xgboost | roc_auc | -0.307 |
| xgboost | average_precision | -0.790 |

## Readiness and completion (Table V)

| Assessor | Completed | Correct / assigned | F1 completed | Requests | Tools | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|
| always_ready | 6/6 | 2/6 | 0.167 | — | — | — |
| rules | 6/6 | 4/6 | 0.556 | — | — | — |
| single_llm | 5/6 | 4/6 | 0.600 | 14 | 48 | 70.3 |
| multi_agent | 3/6 | 3/6 | 0.667 | 37 | 84 | 209.9 |
