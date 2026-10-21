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

- [x] **≥ 1 visualization** of data characteristics (price distribution, make/model frequencies, mileage distribution, etc.).
- [x] **≥ 1 visualization** of feature ranking.
- [x] **≥ 1 visualization** comparing performance across the trained models.
- [x] Every figure can be regenerated from code and is saved to `reports/figures/`.
- [x] Each figure has a short written interpretation ready for the report.

## Artifacts

- `src/plots.py` (or `notebooks/06_visualization.ipynb`)
- `reports/figures/*.png`

## Implementation Notes

**Command:** `python -m src.plots` ([src/plots.py](../src/plots.py)) runs in a few seconds. It reads only the tables written by Tickets 02–05 (`data/processed/*.csv`, `reports/*.csv`) and refits nothing. It checks that `test_predictions.csv` matches the current split. Each figure is saved to `reports/figures/` as PNG (200 dpi) and SVG; the PDF report embeds the SVGs.

| Requirement | Figures |
|-------------|---------|
| Data characteristics | 01 price + retention histograms · 02 mileage and age by source · 03 listings per model by source · 04 retention vs. age / mileage (scatter + binned mean) · 05 mean retention by age and mileage per make · 06 per-model retention at a 5-yr / 60k reference car · 07 correlation heatmap |
| Feature ranking | 08 HistGB permutation importance (top 15, ± SD) · 09 rank of the top 12 features under all 5 methods · 10 grouped permutation per model · 11 largest Ridge indicator coefficients · 12 partial dependence on age and mileage |
| Model performance | 13 CV (mean ± SD) and test MAE / RMSE / R² for all 5 models · 14 predicted vs. actual for HistGB, by source · 15 residuals vs. prediction, and test MAE per vehicle model |

**Style.**
- *Colors.* Categorical colors are assigned in a fixed order and keyed to the entity: each make (and each source, and each model) has the same color in every figure. The four-slot palette was checked with the dataviz skill's validator on a white surface: worst adjacent-pair CVD ΔE 9.1. Scatters use at most three colors, the limit that also passes all-pairs.
- *Contrast.* Aqua and yellow fall below 3:1 contrast, so every chart that uses them also has a legend.
- *Scales.* One-series charts use a single blue. Ridge coefficients and the correlation heatmap use a red–gray–blue diverging scale, and the method-rank heatmap uses a one-hue blue ramp.
- *Marks and text.* Hairline value-axis gridlines only, units on every axis, no dual axes.

**Interpretations** are the figure captions in Section 8 of [reports/final_report.md](../reports/final_report.md).

## Notes for the Report

- Each figure should come with 1–3 sentences on what it shows and why it matters.
