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

- [x] A single integrated dataset with consistent column names, formats, and units.
- [x] **≥ 1,000 listings** after preprocessing and duplicate removal.
- [x] Every row has a valid MSRP and `value_retention_ratio`.
- [x] Summary statistics of the collected attributes are produced and saved (notebook output or `reports/eda_summary.*`).
- [x] The pipeline can be re-run from raw files with one command or notebook.

## Artifacts

- `src/preprocess.py` (or `notebooks/02_preprocessing.ipynb`)
- `data/processed/integrated.csv`
- EDA outputs (tables or figures), reusable in [Ticket 06](06_visualization.md)

## Implementation Notes

**Command:** `python -m src.preprocess` ([src/preprocess.py](../src/preprocess.py)) rebuilds everything from `data/raw/` in a few seconds:

| Output | Contents |
|--------|----------|
| `data/processed/integrated.csv` | 2,161 listings × 44 columns |
| `reports/eda_summary.md` | Stage counts, schema mapping, `describe()`, per-source / per-model comparison, missing values, categorical value counts |
| `reports/eda/*.csv` | The same tables as CSV, for Ticket 06 |
| `reports/preprocessing_log.json` | Stage counts and MSRP match rates |

**Row counts by stage** (each row is counted under the first rule it fails)

| Stage | Rows |
|-------|------|
| Raw (Autolist 1,248 + Craigslist 1,440) | 2,688 |
| − EV / other-nameplate variants (Mach-E, F-150 Lightning, Equinox EV, Shelby GT350/500, Tundra, Crosstour) | −32 |
| − price missing or < $1,000 (Autolist "call for price" = 0, Craigslist `$1` placeholders) | −238 |
| − price < $5,000 with "$X down" / "down payment" text (listed price is the down payment) | −33 |
| − mileage missing | −16 |
| − odometer flagged broken / rolled over | −8 |
| − mileage < 100 on a car ≥ 2 model years old, or > 400k | −2 |
| − branded title (salvage, rebuilt, missing; Autolist `is_salvage`) | −154 |
| **After invalid / out-of-scope removal** | **2,205** |
| − duplicate VIN | −4 |
| − duplicate year/make/model/exact mileage, price within 10% | −17 |
| **After deduplication** | **2,184** |
| − no MSRP for make/model/year | 0 |
| − `value_retention_ratio` outside [0.10, 1.50] | −23 |
| **Final** (Autolist 1,031 / Craigslist 1,130) | **2,161** |

**Normalization**
- Price and mileage: `parse_number` strips `$`, `,`, `mi`/`miles` and expands `45k`. Neither source reports km (Autolist is US-only; Craigslist's odometer field is miles).
- Make/model come from the config-driven search keys, so spelling and case are already canonical. Craigslist's free-text `make_model` (e.g. `HONDA CR-V EX L AWDSUV`) is kept as `trim_raw` and parsed for trim.
- Categories are mapped to shared labels; anything unmapped becomes `other`. Craigslist has no AWD or flex-fuel options, so `awd`/`4wd`/`4x4` become `awd_4wd`, and flex fuel becomes `gas`.
- Autolist `is_salvage` becomes `title_status`, and Autolist rows get `seller_type = dealer`. Craigslist `"6 cylinders"` becomes `6`. Post dates become `YYYY-MM-DD`.

**Deduplication rule.** Rows are sorted by source priority (Autolist first, for its richer structured fields), then by how many fields are filled. Then (1) drop repeat VINs, and (2) drop rows with the same year, make, model and exact odometer reading whose price is within 10% of a kept row. Exact mileage within a model year is close to unique. The 10% tolerance allows for price changes between reposts (Craigslist dealers post the same car in several metros) and across sites.

**MSRP matching.** CarsDirect lists one MSRP per trim *style* (cab, bed and drive variants). For each listing, the pipeline:
1. Finds the longest CarsDirect trim name that appears as whole tokens in the listing's trim text. Matching ignores case and hyphens (`EX-L` = `EXL`, `Off-Road` = `Off Road`). For Craigslist it tries `make_model`, then the post title.
2. If none matches, tries aliases: engine labels stripped (`2.4 LX` → `LX`, `TRD Off Road V6` → `TRD Off Road`), `Base` dropped (`Base V6` → `V6`), and Chevrolet equipment groups collapsed (`1LT`/`2LT` → `LT`, `2LZ` → `LTZ`).
3. Takes the **median MSRP across that trim's styles** (`msrp_match_level = trim`). If no trim is recognized, takes the median across all styles of that make/model/year (`model_year`).

Match rate: **100%** of rows get an MSRP. 98.2% of Autolist rows match at trim level; the misses are fleet/police trims and package names. 78.3% of Craigslist rows match at trim level; the misses are mostly posts that never name a trim.

**Missing-value handling**

| Attribute | Handling | Why |
|-----------|----------|-----|
| price, mileage | Drop row | Price is the target; mileage is a primary predictor and only 16 rows lacked it |
| trim | Fall back to model-year median MSRP; `trim` left blank and `msrp_match_level` records it | Keeps 22% of Craigslist rows that never name a trim |
| body_type, drivetrain, transmission, fuel_type, exterior_color, condition, title_status | Fill `unknown` | Keeps missingness as a category level for encoding in Ticket 03 |
| cylinders, lat/lon, city/state/zip | Leave NaN | Impute inside the model pipeline (Ticket 04) so test rows don't leak |
| owner_count, accident_count, feature flags (leather, sunroof, …) | Leave NaN for Craigslist | Structurally missing (Autolist-only); Ticket 03 extracts equivalents from description text |
| vin | Leave NaN | Only used for deduplication |
| condition (Autolist) | NaN → `unknown` | Autolist reports every row as "used", which says nothing |

**Sanity check:** `value_retention_ratio` ranges 0.10–1.31 (median 0.53). By model year, the median is 0.38 for 2012–14, 0.44 for 2015–17, 0.56 for 2018–20 and 0.73 for 2021–24. Tacoma (0.80) and Corolla (0.71) retain the most value; Escape and Explorer (0.33) the least.

**Caveats for later tickets**
- CarsDirect has no Camry, Accord, CR-V or RAV4 hybrid trims, so hybrids get the gas trim's MSRP. This slightly overstates their retention ratio. `fuel_type = hybrid` lets the model learn the offset.
- Median MSRP across styles hides cab and bed differences on trucks. A Silverado listing that names only "LT" gets the median LT price.
- Craigslist prices run lower than Autolist (median ratio 0.45 vs 0.60), partly from higher mileage (median 112k vs 87k) and private sellers. `source` should be a model feature or a stratification variable.

## Notes for the Report

- Table of schema mappings across sources.
- Row counts at each stage (raw → deduped → MSRP-matched → final).
- Main inconsistencies found and how each was resolved.
