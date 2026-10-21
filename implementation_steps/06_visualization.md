# Ticket 06 — Visualization

| Field | Value |
|-------|-------|
| **Step** | 6 of 7 |
| **Depends on** | [02 — Preprocessing](02_preprocessing_and_integration.md), [04 — Modeling](04_model_development.md), [05 — Depreciation Analysis](05_depreciation_analysis.md) |
| **Blocks** | [07 — Discussion & Conclusion](07_discussion_and_conclusion.md) |
| **Report section** | Visualizations with discussion (5%) |

## Summary

Produce clear, report-ready figures covering data characteristics, feature ranking, and model performance comparison.

## Tasks

- [ ] **Data characteristics** (from the integrated dataset):
  - Distributions of price and `value_retention_ratio` (histogram / KDE).
  - Mileage distribution.
  - Listing counts by make and by model (bar chart), optionally split by source.
  - Retention vs. age and retention vs. mileage (scatter with trend line), optionally colored by make.
  - Optional: correlation heatmap of numeric features; per-source comparison box plots.
- [ ] **Feature ranking:**
  - Horizontal bar chart of the top-N features (permutation importance and/or model importance).
  - Optional: SHAP summary plot; coefficient plot for the linear model.
- [ ] **Model performance comparison:**
  - Grouped bar chart of MAE / RMSE / R² by model (with CV error bars).
  - Predicted vs. actual scatter for the best model.
  - Optional: residual plot.
- [ ] Apply a consistent style: labeled axes with units, titles, readable fonts, colorblind-safe palette.
- [ ] Export every figure to `reports/figures/` as PNG (and SVG/PDF if needed for the report).

## Acceptance Criteria

- [ ] **≥ 1 visualization** of data characteristics (price distribution, make/model frequencies, mileage distribution, etc.).
- [ ] **≥ 1 visualization** of feature ranking.
- [ ] **≥ 1 visualization** comparing performance across the trained models.
- [ ] Every figure can be regenerated from code and is saved to `reports/figures/`.
- [ ] Each figure has a short written interpretation ready for the report.

## Artifacts

- `src/plots.py` (or `notebooks/06_visualization.ipynb`)
- `reports/figures/*.png`

## Notes for the Report

- Each figure should come with 1–3 sentences on what it shows and why it matters.
