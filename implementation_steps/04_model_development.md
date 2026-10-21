# Ticket 04 — Model Development

| Field | Value |
|-------|-------|
| **Step** | 4 of 7 |
| **Depends on** | [03 — Feature Engineering](03_feature_engineering.md) |
| **Blocks** | [05 — Depreciation Analysis](05_depreciation_analysis.md), [06 — Visualization](06_visualization.md) |
| **Report sections** | Model Development with discussion (5%), Evaluation Results with discussion (5%) |
| **Deliverable weight** | Code Quality (20%) |

## Summary

Train, tune, and compare at least three regression algorithms that predict `value_retention_ratio`, using a sound validation strategy and at least two metrics.

## Tasks

- [ ] **Baseline:** a naive predictor (e.g. the training-set mean, or mean ratio per make/model) to give the metrics a reference point.
- [ ] **Validation strategy:**
  - Hold out a test set (e.g. 80/20, fixed `random_state`). Consider stratifying by make or binned target.
  - Use k-fold CV (e.g. 5-fold) on the training set for model selection and tuning.
  - Optional: a cross-source check (train on source A, test on source B) to measure how well models generalize across sources.
- [ ] **Train ≥ 3 algorithms**, for example:
  - Linear model: Linear / Ridge / Lasso (interpretable coefficients for [Ticket 05](05_depreciation_analysis.md)).
  - Tree ensemble: Random Forest.
  - Gradient boosting: XGBoost / LightGBM / sklearn `HistGradientBoostingRegressor`.
  - Optional: KNN, SVR, or a small MLP.
- [ ] **Hyperparameter tuning** with `GridSearchCV` / `RandomizedSearchCV` (or Optuna) inside the CV loop.
- [ ] **Evaluate** on the held-out test set with **≥ 2 metrics**: MAE, RMSE, R².
- [ ] **Residual analysis:** where do the models fail (e.g. by make, by age, by source)?
- [ ] Save the results table (`reports/model_results.csv`) and the fitted best model(s).

## Acceptance Criteria

- [ ] **≥ 3 different regression algorithms** trained and compared.
- [ ] A suitable train/test split **and/or** cross-validation strategy is used and justified.
- [ ] **≥ 2 regression metrics** reported for every model (CV mean ± std, plus test score).
- [ ] No data leakage: encoders and scalers are fit only on training folds.
- [ ] Results are reproducible (fixed seeds, documented environment).

## Artifacts

- `src/train.py` (or `notebooks/04_modeling.ipynb`)
- `reports/model_results.csv`
- `models/` (serialized best model(s), optional)

## Notes for the Report

- Why each algorithm was chosen and its key hyperparameters.
- Results table and interpretation: which model wins and why.
- Overfitting / underfitting observations and residual patterns.
