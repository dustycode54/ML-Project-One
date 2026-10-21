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

- [ ] Contributions of **vehicle age** and **mileage** to depreciation are explicitly quantified and discussed.
- [ ] Features are ranked using **≥ 1 method** (ideally 2+ for comparison).
- [ ] Other important features (make, model, condition indicators, etc.) are identified and discussed.
- [ ] Ranking outputs are saved for use in [Ticket 06](06_visualization.md).

## Artifacts

- `notebooks/05_depreciation_analysis.ipynb`
- `reports/feature_importance.csv`

## Notes for the Report

- Ranked list of top features, with interpretation.
- Per-make or per-model value retention comparison.
- Caveats: correlated features (age vs. mileage), and the difference between correlation and causation.
