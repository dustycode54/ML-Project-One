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

- [x] **≥ 3 different regression algorithms** trained and compared.
- [x] A suitable train/test split **and/or** cross-validation strategy is used and justified.
- [x] **≥ 2 regression metrics** reported for every model (CV mean ± std, plus test score).
- [x] No data leakage: encoders and scalers are fit only on training folds.
- [x] Results are reproducible (fixed seeds, documented environment).

## Artifacts

- `src/train.py` (or `notebooks/04_modeling.ipynb`)
- `reports/model_results.csv`
- `models/` (serialized best model(s), optional)

## Implementation Notes

**Command:** `python -m src.train` ([src/train.py](../src/train.py)) reads `data/processed/features.csv` and runs in about a minute:

| Output | Contents |
|--------|----------|
| `reports/model_results.csv` | One row per model: selected hyperparameters, CV MAE / RMSE / R² (mean and std), train MAE, test MAE / RMSE / R² |
| `reports/model_results.md` | The same results as Markdown tables, plus the cross-source check, residuals of the best model, and the environment |
| `reports/cross_source.csv` | Each tuned pipeline refit on one source and scored on each source's test rows |
| `reports/residuals_by_group.csv` | Test MAE and mean residual by make, model, source, seller type, age bin and mileage bin, for every model |
| `reports/test_predictions.csv` | Test-set target and every model's prediction (for Ticket 06 plots) |
| `models/<model>.joblib` | Fitted pipelines (preprocessor + model). Git-ignored because the random forest is 11 MB; re-run to recreate them |

**Validation strategy.**
- *Hold-out test set.* 433 rows (20%) via `src.features.split()`, stratified by make with seed 42, so all four makes have the same share in train and test. These rows are used once, to score the final refit models.
- *5-fold CV on the 1,728 training rows* (shuffled `KFold`, seed 42) for all tuning and model selection. With ~1.7k rows a single validation split would be noisy, and 5 folds give a standard deviation for each score.
- *No leakage.* Each candidate is a `Pipeline(build_preprocessor(), model)`, so imputation, scaling, one-hot vocabularies, frequency tables and target encodings are fit on the 4 training folds and applied to the fifth. `TargetEncoder` additionally cross-fits within those folds (Ticket 03). `check_leakage()` runs before training.
- *Reproducibility.* Every split, search and model uses `RANDOM_STATE = 42`. Three back-to-back runs gave identical results. Package versions are pinned in `requirements.txt` and printed in `model_results.md` (Python 3.14.2, scikit-learn 1.9.1).

**Algorithms and tuning.** The high-cardinality encoding (`onehot` / `frequency` / `target`) is part of every search, because Ticket 03 found them within one CV standard deviation of each other. Models are selected on CV MAE.

| Model | Why | Search | Selected |
|-------|-----|--------|----------|
| Mean (baseline) | Floor: predicts the training mean | — | — |
| Make/model mean (baseline) | What you get by knowing only the car's nameplate | — | — |
| Ridge | Linear and interpretable: its coefficients feed Ticket 05. The L2 penalty handles ~300 one-hot columns and the correlated age / mileage / msrp | Grid: `alpha` 0.01–1000 (11 values, log scale) × 3 encodings = 33 | one-hot, `alpha=1` |
| Random Forest | Bagged deep trees capture interactions (age × model) with little tuning | Random, 20 of 144: `max_features`, `min_samples_leaf`, `max_depth`, encoding. 300 trees | one-hot, `max_features=0.33`, `min_samples_leaf=1`, unlimited depth |
| HistGradientBoosting | Boosted shallow trees, usually the strongest on tabular data | Random, 30 of 729: `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, encoding | frequency, `learning_rate=0.05`, `max_iter=800`, `max_leaf_nodes=7`, `min_samples_leaf=20`, `l2=0.1` |

**Results** (`reports/model_results.csv`)

| Model | CV MAE | CV RMSE | CV R² | Train MAE | Test MAE | Test RMSE | Test R² |
|-------|--------|---------|-------|-----------|----------|-----------|---------|
| Mean baseline | 0.1929 ± 0.0034 | 0.2318 ± 0.0019 | −0.001 | 0.1928 | 0.1861 | 0.2245 | 0.000 |
| Make/model mean | 0.1608 ± 0.0058 | 0.1960 ± 0.0065 | 0.284 ± 0.040 | 0.1592 | 0.1610 | 0.1942 | 0.252 |
| Ridge | 0.0687 ± 0.0005 | 0.0915 ± 0.0032 | 0.844 ± 0.009 | 0.0574 | 0.0671 | 0.0953 | 0.820 |
| Random Forest | 0.0725 ± 0.0036 | 0.0962 ± 0.0053 | 0.827 ± 0.017 | 0.0260 | 0.0693 | 0.0978 | 0.810 |
| **HistGradientBoosting** | **0.0630 ± 0.0024** | **0.0853 ± 0.0035** | **0.864 ± 0.010** | 0.0365 | **0.0635** | **0.0906** | **0.837** |

- **HistGradientBoosting wins** on every CV and test metric. Its test MAE of 0.064 means a typical prediction is off by about 6 percentage points of MSRP (roughly $1,600 on a $25k car). Its CV MAE is 0.006 better than Ridge, about 2.5 CV standard deviations.
- **Every model beats both baselines by far.** Knowing only the make and model cuts the error of the mean predictor by 17%. The full feature set cuts it by another 60%. Ticket 05 breaks down which features carry that.
- **Ridge is a close second**, so most of the signal is roughly linear in the standardized features. Boosting gains on the non-linear effects and interactions that Ridge can only fit additively.
- **Overfitting.** The Random Forest fits the training data almost perfectly (train MAE 0.026 vs. CV 0.073). Its search picked fully grown trees, and every constrained setting scored worse in CV (`min_samples_leaf=8` was worst, about 0.080), so the forest is limited by variance, not by its settings; with ~1.7k rows and ~300 columns the bagged trees stay noisy. Boosting's shallow 7-leaf trees and leaf-size limit keep the gap smaller (0.037 vs. 0.063). Ridge has the smallest gap (0.057 vs. 0.069): it underfits slightly rather than overfits.
- **Test vs. CV.** Test MAE is within 0.0035 of CV MAE for every tuned model, and the ranking is the same, so the hold-out set confirms the CV estimate. Test R² is about 0.02–0.03 lower than CV R² because the test target has slightly lower variance (the mean baseline's test RMSE is 0.225 vs. 0.232 in CV), not because of larger errors.
- **Encoding.** Ridge and the Random Forest picked one-hot, boosting picked frequency, matching Ticket 03.

**Cross-source check** (`reports/cross_source.csv`, test MAE)

| Model | Autolist → Autolist | Autolist → Craigslist | Craigslist → Craigslist | Craigslist → Autolist |
|-------|---------------------|-----------------------|-------------------------|-----------------------|
| Ridge | 0.053 | 0.081 | 0.081 | 0.107 |
| Random Forest | 0.065 | 0.094 | 0.084 | 0.091 |
| HistGB | 0.053 | 0.083 | 0.075 | 0.078 |

- Dealer listings (Autolist) are easier to predict than Craigslist: in-source MAE 0.053 vs. 0.075. Craigslist prices are noisier: private sellers, negotiable asking prices, and less structured data.
- A model trained only on Autolist does about as well on Craigslist as one trained on Craigslist itself (0.083 vs. 0.075 for boosting). The reverse direction costs more: Ridge trained on Craigslist reaches 0.107 MAE on Autolist, about twice its in-source error. Linear coefficients learned from noisy private-seller prices transfer worse than the trees' piecewise fits.
- Pooling both sources is slightly better than training on either alone. Boosting's test MAE is 0.073 on Craigslist when trained on both sources (vs. 0.075 on Craigslist only) and 0.052 on Autolist (vs. 0.053 on Autolist only). The noisier Craigslist rows don't hurt the dealer predictions.

**Residual analysis** (best model, `reports/model_results.md`)
- *By model.* Tacoma has the largest error (MAE 0.117, almost twice the average). Its retention is far above every other model's (mean 0.79 vs. 0.36–0.69), so the shared age and mileage effects fit it worst. Camry, Civic and Corolla follow at 0.076–0.083. The low-retention Explorer, Escape, Malibu and Equinox are the easiest (0.041–0.048). Error roughly tracks a model's retention level. Mean residuals stay within ±0.022, so no model is systematically over- or under-predicted.
- *By source and seller.* Craigslist MAE is 0.073 vs. 0.052 for Autolist, and private sellers 0.084 vs. 0.060 for dealers. This is the noisier private-seller pricing seen in the cross-source check.
- *By age and mileage.* The error is largest for the newest and least-driven cars (age 0–2: 0.087; under 30k miles: 0.093) and smallest for the oldest and highest-mileage ones (150k+: 0.050). Newer cars have the highest ratios (mean 0.81 at age 0–2) but the fewest rows (134 overall, 30 in test), so the model has the least data where the values are largest. The model slightly over-predicts these young, low-mileage cars (mean residual −0.012 to −0.014).

## Notes for the Report

- Why each algorithm was chosen and its key hyperparameters.
- Results table and interpretation: which model wins and why.
- Overfitting / underfitting observations and residual patterns.
