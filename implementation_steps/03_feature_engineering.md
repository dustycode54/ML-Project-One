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

- [x] **≥ 2 meaningful features** extracted from listing descriptions, with extraction logic documented.
- [x] **≥ 1 encoding technique for high-cardinality features** applied **and discussed** (why it was chosen, trade-offs, leakage handling).
- [x] Vehicle age (plus any other derived numeric features) created.
- [x] Feature transformations are reproducible and fit on training data only.

## Artifacts

- `src/features.py` (or `notebooks/03_features.ipynb`)
- `data/processed/features.csv` (optional)

## Implementation Notes

**Command:** `python -m src.features` ([src/features.py](../src/features.py)) reads `data/processed/integrated.csv` and runs in about 15 seconds:

| Output | Contents |
|--------|----------|
| `data/processed/features.csv` | 2,161 rows: `listing_id`, the target and 33 features, before encoding |
| `reports/text_features.md` | Regex patterns, prevalence by source, validation against Autolist history fields, spot-check samples |
| `reports/encoding_comparison.csv` | 5-fold CV (MAE / RMSE / R², mean ± std) for each high-cardinality encoding × Ridge / HistGradientBoosting |

`features.csv` is saved **before encoding**. Encoders learn from the target or from category counts, so they must be fit per training split. Ticket 04 should load `features.csv` and wrap `build_preprocessor()` and a model in one `Pipeline`. `split()` gives the shared 80/20 train/test split (seed 42, stratified by make).

**Structured features**

| Feature | Definition |
|---------|------------|
| `vehicle_age` | Listing year (from `posted_at`) − model year, clipped at 0. Range 0–14 |
| `miles_per_year` | `mileage / max(vehicle_age, 1)` |
| `region` | US Census region from `state` (northeast / midwest / south / west / unknown) |
| `trim_key` | `make model trim` (178 levels). Trim names repeat across models (`LX`, `SE`, `LT`); missing trim → `unknown` |
| `city_key` | `city, state` (880 levels, 613 seen only once) |
| `has_description`, `log_description_len` | Whether a description exists, and log(1 + characters) |

No luxury flag: all four makes in scope (Honda, Toyota, Ford, Chevrolet) are mainstream brands.

**Text features.** Each extractor is a regex over the lowercased `listing_title + " . " + description`. Some phrases are positive only when not negated: "not one owner", "no check engine light on", "no engine issues". For those, a match is dropped if a negation word (`no`, `not`, `never`, `without`, `nothing`, `-n't`) appears in the 15 characters before it. Extractors whose pattern already contains the negation ("no accidents", "never smoked") skip this check.

| Feature | Flagged | Autolist | Craigslist | Why it should affect retention |
|---------|---------|----------|------------|--------------------------------|
| `txt_one_owner` | 15.4% | 14.7% | 16.0% | Fewer owners suggests consistent care and a known history |
| `txt_clean_title` | 19.9% | 2.6% | 35.7% | Private sellers state it to reassure buyers. Branded titles were already removed, so it mostly marks private-seller listings |
| `txt_clean_history` | 10.7% | 13.2% | 8.5% | Clean Carfax / AutoCheck report |
| `txt_accident_free` | 7.5% | 5.4% | 9.3% | Accident history lowers resale value |
| `txt_service_records` | 4.3% | 3.3% | 5.2% | Documented maintenance lowers buyer risk |
| `txt_new_tires` | 5.3% | 2.1% | 8.2% | Recent spend the buyer won't have to make |
| `txt_non_smoker` | 4.6% | 3.3% | 5.8% | Interior condition |
| `txt_needs_work` | 0.4% | 0.4% | 0.4% | Known mechanical problems. Rare, but a strong signal (median ratio 0.28 vs 0.53) |

**Validation.**
- *Against labels.* Autolist's structured history fields serve as labels for two extractors. `txt_one_owner` has **0.89 precision** against `owner_count == 1` (recall 0.34). `txt_accident_free` has **0.95 precision** against `accident_count == 0` (recall 0.08). Recall is low because most dealer descriptions don't restate the history report. Precision is what matters for a feature flag.
- *Spot checks.* Two rounds of spot checks (matches, plus listings with the keyword that weren't flagged) led to several fixes:
  - added "clean texas title", "never any accidents", "1 previous owner" / "original owner", "no smoking" and "serviced regularly";
  - allowed run-together text ("wirelessnon-smoker");
  - dropped "as is", which is mostly legal disclaimers.
- *Candidates rejected.* Certified pre-owned, warranty and garage-kept were tried and dropped. CPO and warranty mostly appear in SEO keyword lists and dealer boilerplate ("certified pre-owned preowned pre owned…", "warranty available"). Garage-kept matched only 0.3% of listings.
- *Remaining false positives.* These are dealer boilerplate ("all vehicles have a clean title", "vehicles that need work after you buy it").

**Encoding** (`build_preprocessor(high_card=...)`)

| Group | Columns | Encoding |
|-------|---------|----------|
| Numeric | vehicle_age, mileage, miles_per_year, msrp, cylinders, lat, lon, has_description, log_description_len, 8 text flags | Median impute → standardize |
| Low cardinality | source, make, model (16), body_type, drivetrain, transmission, fuel_type, exterior_color, condition, title_status, seller_type, msrp_match_level, region | One-hot, unseen levels ignored |
| High cardinality | trim_key (178), state (50), city_key (880) | `frequency` (default), `target` or `onehot` → standardize |

- **Frequency encoding** (custom `FrequencyEncoder`) replaces a level with its share of training rows. It uses no target, so it can't leak it, and unseen levels map to 0. The downside is that distinct levels with equal counts become indistinguishable.
- **Target encoding** (sklearn `TargetEncoder`) replaces a level with a smoothed mean of the target. To avoid leakage it **cross-fits**: in `fit`, each training row is encoded from the other 4/5 of the training rows, so a row's own price never feeds its encoding. On new rows, `transform` uses the encoding learned from all training rows. Smoothing shrinks rare levels (most cities) toward the global mean.
- **One-hot** (levels with < 5 rows grouped as infrequent) is the baseline: 297 columns versus 95 for the other two.

**Encoding comparison** (5-fold CV on the 1,728-row training split; `reports/encoding_comparison.csv`)

| High-card encoding | Ridge MAE | Ridge R² | HistGB MAE | HistGB R² |
|--------------------|-----------|----------|------------|-----------|
| one-hot | 0.0687 | 0.844 | 0.0661 | 0.853 |
| frequency | 0.0702 | 0.835 | 0.0664 | 0.853 |
| target | 0.0695 | 0.840 | 0.0676 | 0.847 |
| target, no `msrp` | 0.0707 | 0.835 | 0.0691 | 0.840 |

The encodings differ by about 0.001–0.0015 MAE, which is about the CV standard deviation (0.002–0.004), so none clearly wins. These columns add little beyond `msrp`, `model` and `vehicle_age`. The trim signal is mostly already in `msrp`, and 613 of 880 cities appear only once, so a city's target mean is mostly smoothing. Frequency is the default: it ties the boosted model's best score and needs neither the target nor 200 extra columns. Ticket 04 should still treat the encoding as a hyperparameter, since Ridge does best with one-hot.

**Leakage check.** `check_leakage()` fails the run if `price`, `value_retention_ratio` or an ID column (`listing_id`, `vin`, `listing_url`, `source_id`) is among the feature columns. Text flags are binary, so the dollar amounts in descriptions never reach the model. Imputation, scaling and every encoder are fit inside the pipeline, so CV folds and the test set never feed their statistics.

`msrp` **is used as a feature.** It is known before the car is listed and doesn't depend on the listing price. It also places the car within its model's price range (trim and equipment level). Removing it raises CV MAE by 0.0012 (Ridge) and 0.0015 (HistGB). One caveat: the target is `price / msrp`, so an MSRP that is too high (for example, the median-of-styles fallback for a truck) mechanically lowers the ratio. The model can partly learn this through `msrp` and `msrp_match_level`.

**Not used as features:** Autolist-only `owner_count`, `accident_count` and equipment flags. They are missing for every Craigslist row, so they would mostly act as a proxy for `source`. The text flags are the cross-source equivalent.

## Notes for the Report

- Prevalence of each text feature (% of listings flagged).
- Why each feature should plausibly affect value retention.
- Encoding comparison and leakage considerations.
