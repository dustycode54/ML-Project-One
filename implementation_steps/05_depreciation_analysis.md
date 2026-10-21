# Ticket 05 — Depreciation Analysis and Feature Ranking

| Field | Value |
|-------|-------|
| **Step** | 5 of 7 |
| **Depends on** | [04 — Model Development](04_model_development.md) |
| **Blocks** | [06 — Visualization](06_visualization.md), [07 — Discussion & Conclusion](07_discussion_and_conclusion.md) |
| **Report section** | Depreciation Analysis (5%) |

## Summary

Quantify how vehicle age, mileage, and other features drive depreciation, and rank features by importance using one or more interpretability methods.

## Tasks

- [ ] **Linear-model coefficients:** from the Ridge/Lasso model (on standardized features), report the size and sign of each coefficient. Interpret these, e.g. "each additional year of age reduces retention by ~X percentage points."
- [ ] **Tree-model importance:** impurity-based importance from Random Forest / gradient boosting.
- [ ] **Permutation importance** on the test set for the best model; this is more reliable than impurity-based importance when features are correlated or high-cardinality.
- [ ] Optional: **SHAP** values (summary and dependence plots) for the best tree model.
- [ ] **Depreciation curves:** mean `value_retention_ratio` vs. age and vs. mileage, overall and per make (or per model). Identify the makes/models that hold value best and worst.
- [ ] Check whether the methods agree on the top features, and explain any disagreements.
- [ ] Assess the text-derived features from [Ticket 03](03_feature_engineering.md): did `one_owner`, `clean_title`, etc. matter?
- [ ] Save the feature-ranking tables (`reports/feature_importance.csv`).

## Acceptance Criteria

- [x] Contributions of **vehicle age** and **mileage** to depreciation are explicitly quantified and discussed.
- [x] Features are ranked using **≥ 1 method** (ideally 2+ for comparison).
- [x] Other important features (make, model, condition indicators, etc.) are identified and discussed.
- [x] Ranking outputs are saved for use in [Ticket 06](06_visualization.md).

## Artifacts

- `src/depreciation.py` (or `notebooks/05_depreciation_analysis.ipynb`)
- `reports/feature_importance.csv`

## Implementation Notes

**Command:** `python -m src.depreciation` ([src/depreciation.py](../src/depreciation.py)) runs in about 30 seconds. It loads the three pipelines saved by `python -m src.train` (run that first), so every number comes from the Ticket 04 models and the same train/test split. Model-based numbers are on the 433 held-out test rows; the curves use all 2,161 listings.

| Output | Contents |
|--------|----------|
| `reports/feature_importance.csv` | One row per input feature (33): Ridge contribution std, RF impurity importance, permutation importance (mean and std) for each model, the rank under each method, and the mean rank |
| `reports/grouped_importance.csv` | Permutation importance of correlated feature groups, shuffled together |
| `reports/ridge_coefficients.csv` | Every encoded Ridge column: standardized coefficient, scale, and effect per raw unit |
| `reports/age_mileage_effects.csv` | Average change in predicted retention for +1 year, +10k miles and +1 typical year of use, per model |
| `reports/partial_dependence.csv` | Mean prediction over a grid of ages (2–14) and mileages (0–200k), per model |
| `reports/depreciation_curves.csv` | Mean / median retention by age and by mileage bin, overall, per make and per model (for Ticket 06) |
| `reports/model_retention.csv` | Per model and per make: linear fit of retention on age and mileage, and the fitted retention of a 5-year-old, 60k-mile car |
| `reports/depreciation_analysis.md` | All of the above as tables, plus the CV ablation and the method-agreement tables |

SHAP (optional) was not used, because `shap` is not installed. Partial dependence and grouped permutation cover the same questions with scikit-learn only.

**Methods.**
- *Ridge coefficients.* The inputs are standardized, so each coefficient is the effect of one standard deviation. Dividing by the scaler's SD converts it to raw units (per year, per 10k miles, per flag). To rank Ridge features, the script sums each feature's one-hot columns and takes the std of the result over the training rows, so a 16-level `model` and a single numeric column are measured on the same scale.
- *Impurity importance* from the Random Forest, with one-hot columns summed back to their feature. sklearn's `HistGradientBoostingRegressor` has no impurity importance.
- *Permutation importance* (increase in test MAE, 10 repeats) for all three models. It permutes the raw input columns, so a categorical feature is permuted as one unit.
- *Grouped permutation* shuffles correlated sets together (age + mileage + miles/year; make + model + trim + MSRP; …). Permuting one feature of a correlated set understates the set, because the model can recover the information from the other features.
- *Ablation.* 5-fold CV of the best pipeline (HistGB) refit without a feature group, on the training split.
- *Consistent counterfactuals.* For "+1 year" and the partial-dependence grids, `miles_per_year` is recomputed from the changed age and mileage. Otherwise the probe would describe a car that cannot exist (older, with the same mileage and the same miles per year).

**Age and mileage** (`age_mileage_effects.csv`, test-set average, pp = percentage points of MSRP)

| Scenario | Ridge | Random Forest | HistGB |
|----------|-------|---------------|--------|
| +1 year, same mileage | −2.0 | −0.9 | −1.4 |
| +10k miles, same age | −2.3 | −2.3 | −2.3 |
| +1 year and +12.1k miles (median annual use) | −4.8 | −3.6 | −4.3 |

- **A typical year of ownership costs about 4–5 pp of MSRP** (−4.3 pp for the best model, about $1,100 on a $25k car). About two thirds of that comes from the miles driven and one third from age itself.
- **Mileage is the strongest single driver.** All three models agree on −2.3 pp per 10k miles. Ridge gives a nearly identical −2.1 pp per 10k miles, holding everything else fixed. Partial dependence for HistGB falls from 0.79 at 0 miles to 0.48 at 120k and 0.33 at 200k. The slope is steepest from 40k to 160k miles and flattens after that.
- **Age at fixed mileage matters less than its raw curve suggests.** Raw mean retention drops from 0.81 at age 2 to 0.38 at age 14 (about 3.6 pp/year), but most of that drop comes with mileage. Holding each car's mileage fixed, HistGB's partial dependence falls only from 0.63 to 0.45 over the same span (−1.4 pp/year). It is nearly flat for ages 2–5 and steeper for ages 6–12. Ridge (−2.0 to −2.3 pp/year) is linear by construction. The forest's lower estimate (−0.9) is likely because it splits on mileage first and leaves little to age.
- **The two are hard to separate** (Pearson r = 0.54). Dropping age alone from HistGB raises CV MAE by just 0.0024, and dropping mileage alone by 0.0014, because each stands in for the other. Dropping age, mileage and miles/year together raises it by 0.056 (0.063 → 0.119), almost back to the make/model baseline (0.161). Grouped permutation tells the same story: shuffling the age/mileage group adds about 0.12 MAE in every model, more than any other group.

**Feature ranking** (`feature_importance.csv`, sorted by mean rank over the 5 methods)

| Rank | Feature | Notes |
|------|---------|-------|
| 1 | `mileage` | #1 under all 5 methods |
| 2 | `vehicle_age` | #2–3 everywhere |
| 3 | `make` | #2 for both tree-model permutations |
| 4 | `model` | #3–5 |
| 5 | `msrp` | Ridge: −0.44 pp per $1k. Pricier trims/models lose a larger share of MSRP |
| 6 | `log_description_len` | Mostly marks whether the listing has a description, i.e. the source (see below) |
| 7 | `trim_key` | High for Ridge (one-hot), low for the trees |
| 8 | `body_type` | Trucks vs. SUVs vs. sedans. #4–6 for the trees, low for Ridge |

- **Nameplate is the second major factor.** Make, model, trim and MSRP shuffled together add 0.067–0.097 MAE. Ridge's largest indicators are model Escape (−15.0 pp) and Tacoma (+14.9 pp), condition "fair" (−12.7 pp), F-150 (+12.2 pp) and make Toyota (+9.3 pp). Condition "excellent" (+5.8 pp) and hybrid fuel (+5.3 pp) are the largest non-nameplate indicators.
- **Everything else is small.** Body/drivetrain/engine adds 0.003–0.016 MAE when shuffled together, location 0.002–0.006, and source + seller type about 0.003.

**Do the methods agree?** On the top features, yes: mileage, age, make and model are the top 4 under every method, in slightly different orders. Further down they disagree, and the Spearman correlations reflect that: 0.85–0.95 within a model family (Ridge contribution vs. Ridge permutation, RF impurity vs. tree permutations) but only 0.53–0.71 between families. The disagreements have clear causes:
- *Make vs. model vs. trim.* These are nested (the trim key includes make and model, and the model determines the make), so a model can put the same information in any of them. Ridge spreads it across the one-hot `model` and `trim_key` columns. The trees, with frequency-encoded trims, split on `make` first. Grouped permutation, which shuffles all three together, gives consistent results across models.
- *Impurity importance inflates continuous features.* `miles_per_year` is #5 by RF impurity but #8–19 by permutation, and `lat`/`lon` are #11–12 by impurity but near zero by permutation for the forest. Impurity rewards features with many split points even when the splits fit noise. Permutation on held-out rows doesn't.
- *Location.* `state`, `region` and `city_key` rank 8–10 for Ridge but near 0 for the trees. Ridge fits a regional offset per one-hot level, while the trees use `lon` (#7 for HistGB permutation) and seldom split on the sparse categories.
- *Linear vs. non-linear.* `body_type` and `cylinders` rank high for the trees, which use them in interactions (trucks depreciate differently with mileage), and low for Ridge, which only fits them additively.

**Depreciation by make and model** (`model_retention.csv`). Raw means are confounded: the models differ in average age and mileage. So each model gets its own linear fit `ratio ~ age + mileage`, and the models are compared at the same reference car (5 years, 60k miles).

| Holds value best | Retention @ 5 yr / 60k | Holds value worst | Retention @ 5 yr / 60k |
|------------------|------------------------|-------------------|------------------------|
| Toyota Tacoma | 0.89 | Ford Escape | 0.51 |
| Toyota RAV4 | 0.83 | Ford Explorer | 0.53 |
| Honda Civic | 0.83 | Chevrolet Malibu | 0.55 |
| Toyota Corolla | 0.83 | Chevrolet Equinox | 0.57 |

- By make: Toyota 0.83, Honda 0.75, Ford 0.62, Chevrolet 0.60. The gap is visible at every age in the raw curves too: at age 10, Toyota's mean is 0.61 vs. 0.34 for Chevrolet.
- The makes differ mostly in their **level**, not their rate: per-make slopes are −1.5 to −2.0 pp/year and −2.4 to −2.7 pp per 10k miles. The Tacoma's fitted age slope is about 0 (it loses value only with mileage). The Mustang is the most mileage-sensitive (−3.9 pp per 10k), likely because its listings have low mileage (mean 77k) and are often weekend cars. The Tahoe, Civic and Equinox lose the most per year (−2.6 to −2.9 pp).

**Text-derived features** (Ticket 03). **They did not matter for prediction.** Permutation of all 8 flags together adds only 0.0006–0.0011 test MAE. Dropping them from HistGB *improves* CV MAE slightly (0.0630 → 0.0623, within one CV std), and no flag ranks above 17th for the best model. Reasons:
- *Most raw associations are confounded.* `txt_clean_title` has a raw median difference of −18 pp, and Ridge still gives it −2.7 pp after controls. That is not because a clean title lowers the price: 56% of private-seller listings say "clean title" vs. 13% of dealer listings, and the flagged cars are older (9.7 vs. 8.0 years) and higher-mileage (124k vs. 96k). The flag marks a "private-sale older car", which age, mileage and seller type already describe. Likewise `txt_one_owner` is +5 pp raw but −0.8 pp in Ridge.
- *The flags are sparse.* `txt_needs_work` is the only flag with a large controlled effect (Ridge −10.3 pp), but it fires on 9 listings (0.4%), too few for the trees to split on.
- *Description presence is a source marker.* `log_description_len` (#6) and `has_description` are among the more important features, but together they mostly encode the source (only Autolist listings lack descriptions). Ridge's +1.6 pp per log-character and −8.7 pp for `has_description` largely cancel out.

**Caveats for the report.**
- *Correlation, not causation.* All effects are associations in asking prices. "−2.3 pp per 10k miles" is the price difference between otherwise similar listings, and it includes whatever correlates with mileage and isn't measured (wear, maintenance, accident history). Asking prices are not transaction prices.
- *Correlated features.* Single-feature importances for age, mileage and miles/year, and for make, model and trim, understate their groups and split the credit between them unpredictably. The grouped and ablation results are the reliable numbers for those groups.
- *Extrapolation.* There are almost no cars under 2 years old (1 listing) or past 14 years, and few under 30k miles (148). The partial-dependence ends and the "+1 year" effect for the newest cars rest on few data points, where Ticket 04 also found the largest errors.
- *Linear per-model fits* are a summary for ranking. The Ridge and partial-dependence curves show the age effect is non-linear (flat early, steeper after year 5).

## Notes for the Report

- Ranked list of top features, with interpretation.
- Per-make or per-model value retention comparison.
- Caveats: correlated features (age vs. mileage), and the difference between correlation and causation.
