# Ticket 03 — Feature Engineering

| Field | Value |
|-------|-------|
| **Step** | 3 of 7 |
| **Depends on** | [02 — Preprocessing & Integration](02_preprocessing_and_integration.md) |
| **Blocks** | [04 — Model Development](04_model_development.md) |
| **Report section** | Feature Engineering with discussion (5%) |
| **Deliverable weight** | Code Quality (20%) |

## Summary

Derive features that may improve prediction of `value_retention_ratio`, including features extracted from free-text listing descriptions, and encode categorical variables. At least one encoding must be suited to high-cardinality features.

## Tasks

- [ ] **Structured features:**
  - `vehicle_age = listing_year - model_year` (use the scrape or post date).
  - `miles_per_year = mileage / max(vehicle_age, 1)`.
  - Optional extras: luxury-brand flag, regional grouping from location, segment / body type.
- [ ] **Text-derived features** from listing descriptions (regex or keyword matching, with optional NLP):
  - `one_owner` (e.g. "one owner", "1 owner", "single owner").
  - `clean_title` / `accident_free` / `no_accidents`.
  - Other candidates: `service_records`, `new_tires`, `garage_kept`, `certified_pre_owned`, `needs_work`.
  - Validate each extractor by spot-checking a sample of matches and non-matches.
- [ ] **Encoding:**
  - Low-cardinality categoricals (e.g. fuel type, transmission): one-hot.
  - **High-cardinality** categoricals (e.g. model, trim, location, color): apply at least one of **target encoding** (fit inside CV folds to avoid leakage), **frequency/count encoding**, or **hashing**.
  - Compare at least two approaches if time allows.
- [ ] **Leakage check:** confirm that no feature encodes `price` or `msrp` directly. If `msrp` is used as a feature, justify it and note the effect.
- [ ] Wrap the transformations in a reusable sklearn `Pipeline` / `ColumnTransformer` so the same steps apply to training and test data.
- [ ] Save the feature matrix to `data/processed/features.csv` (or produce it on the fly from the pipeline).

## Acceptance Criteria

- [ ] **≥ 2 meaningful features** extracted from listing descriptions, with extraction logic documented.
- [ ] **≥ 1 encoding technique for high-cardinality features** applied **and discussed** (why it was chosen, trade-offs, leakage handling).
- [ ] Vehicle age (plus any other derived numeric features) created.
- [ ] Feature transformations are reproducible and fit on training data only.

## Artifacts

- `src/features.py` (or `notebooks/03_features.ipynb`)
- `data/processed/features.csv` (optional)

## Notes for the Report

- Prevalence of each text feature (% of listings flagged).
- Why each feature should plausibly affect value retention.
- Encoding comparison and leakage considerations.
