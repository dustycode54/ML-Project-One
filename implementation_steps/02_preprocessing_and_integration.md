# Ticket 02 — Data Preprocessing and Integration

| Field | Value |
|-------|-------|
| **Step** | 2 of 7 |
| **Depends on** | [01 — Data Collection](01_data_collection.md) |
| **Blocks** | [03 — Feature Engineering](03_feature_engineering.md) |
| **Report section** | Preprocessing and Integration with discussion (5%) |
| **Deliverable weight** | Data (20%), Code Quality (20%) |

## Summary

Merge the per-source raw files and the MSRP data into one clean, unified dataset. Compute the target variable and explore summary statistics.

## Tasks

- [ ] **Schema mapping.** Define a single target schema and map each source's column names onto it (e.g. `odometer` / `miles` → `mileage`).
- [ ] **Normalize formats and units:**
  - Price: strip `$` and `,`, convert to numeric, and remove placeholder prices (e.g. `$1`, `$123456`).
  - Mileage: parse values such as `45k` and `45,000 mi`; convert km → mi if any source uses km.
  - Categorical values: standardize spelling and case for make, model, trim, transmission, fuel type, color, etc.
  - Year: integer, within a valid range.
- [ ] **Add a `source` column** so the origin of each row can be traced.
- [ ] **Deduplicate** within and across sources (e.g. by VIN, or by a fuzzy match on year/make/model/mileage/price/location). Document the rule used.
- [ ] **Handle missing values.** Drop, impute, or flag each one, and justify the choice per attribute.
- [ ] **Remove outliers and invalid rows** (e.g. salvage titles if out of scope, mileage of 0 on old cars, prices far outside a plausible range).
- [ ] **Merge MSRP** on make/model/year (and trim if available). Report the match rate and how unmatched rows were handled.
- [ ] **Compute the target:** `value_retention_ratio = price / msrp`. Sanity-check its range.
- [ ] **Exploratory statistics:** `describe()` for numeric columns; value counts for categorical columns; missing-value summary; per-source comparison.
- [ ] Save the integrated dataset to `data/processed/integrated.csv`.

## Acceptance Criteria

- [ ] A single integrated dataset with consistent column names, formats, and units.
- [ ] **≥ 1,000 listings** after preprocessing and duplicate removal.
- [ ] Every row has a valid MSRP and `value_retention_ratio`.
- [ ] Summary statistics of the collected attributes are produced and saved (notebook output or `reports/eda_summary.*`).
- [ ] The pipeline can be re-run from raw files with one command or notebook.

## Artifacts

- `src/preprocess.py` (or `notebooks/02_preprocessing.ipynb`)
- `data/processed/integrated.csv`
- EDA outputs (tables or figures), reusable in [Ticket 06](06_visualization.md)

## Notes for the Report

- Table of schema mappings across sources.
- Row counts at each stage (raw → deduped → MSRP-matched → final).
- Main inconsistencies found and how each was resolved.
