# ML-Project-One

Predicting used-car **value retention ratio** (`listing price / original MSRP`) from multi-source listing data. See [project_overview.md](project_overview.md) for the assignment and [implementation_steps/](implementation_steps/) for the per-step tickets.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Collection

Vehicle scope (makes, models, model years) and scraper settings live in [config/vehicles.yaml](config/vehicles.yaml).

| Source | Role | Command | Output |
|--------|------|---------|--------|
| Autolist | Listings (dealer inventory, vehicle-history fields) | `python -m src.scrapers.autolist` | `data/raw/autolist_listings.csv` |
| Craigslist | Listings (owner + dealer posts, free-text descriptions) | `python -m src.scrapers.craigslist` | `data/raw/craigslist_listings.csv` |
| CarsDirect | Original MSRP per make / model / year / trim | `python -m src.scrapers.carsdirect_msrp` | `data/raw/msrp.csv` |

Each scraper rate-limits its requests and caches every response under `data/cache/` (git-ignored), so re-running after a parser change does not re-download pages. Delete `data/cache/<source>/` to force a fresh scrape. Row counts, scrape timestamps and HTTP stats are written to `data/raw/collection_log.json`.

## Preprocessing and Integration

```bash
python -m src.preprocess
```

This merges the raw listings and MSRP into `data/processed/integrated.csv`, which has one row per vehicle and a `value_retention_ratio` column. It also writes EDA tables to `reports/eda_summary.md` and `reports/eda/*.csv`, and stage counts to `reports/preprocessing_log.json`. The cleaning, deduplication and MSRP-matching rules are documented in [Ticket 02](implementation_steps/02_preprocessing_and_integration.md#implementation-notes).

## Feature Engineering

```bash
python -m src.features
```

This adds derived features (vehicle age, miles per year, region, and text flags from listing descriptions such as one owner, clean title and accident-free) and writes them to `data/processed/features.csv`. The encoding is not applied there: `src.features.build_preprocessor()` returns a `ColumnTransformer` that is fit on training rows inside each model pipeline. The run also writes extractor prevalence, validation and spot checks to `reports/text_features.md`, and a CV comparison of high-cardinality encodings to `reports/encoding_comparison.csv`. See [Ticket 03](implementation_steps/03_feature_engineering.md#implementation-notes).

## Model Development

```bash
python -m src.train
```

This tunes Ridge, Random Forest and HistGradientBoosting with 5-fold CV on the training split (the high-cardinality encoding is tuned too), compares them with two baselines, and scores each on the held-out test set with MAE, RMSE and R². Results go to `reports/model_results.csv` and `reports/model_results.md`, together with a cross-source check (`reports/cross_source.csv`), residuals by group (`reports/residuals_by_group.csv`) and test-set predictions (`reports/test_predictions.csv`). The fitted pipelines are saved to `models/` (git-ignored). See [Ticket 04](implementation_steps/04_model_development.md#implementation-notes).

## Depreciation Analysis

```bash
python -m src.depreciation   # after python -m src.train
```

This loads the fitted pipelines from `models/` and quantifies the age and mileage effects (counterfactual changes and partial dependence). It ranks the features with Ridge contributions, Random Forest impurity, test-set permutation importance (single and grouped) and a CV ablation, and builds depreciation curves and a value-retention ranking per make and model. The feature ranking goes to `reports/feature_importance.csv` and the full write-up to `reports/depreciation_analysis.md`, together with CSVs for Ticket 06 plots (`depreciation_curves.csv`, `partial_dependence.csv`, `model_retention.csv`, …). See [Ticket 05](implementation_steps/05_depreciation_analysis.md#implementation-notes).

## Visualization

```bash
python -m src.plots   # after python -m src.depreciation
```

This draws 15 report figures from the saved tables, covering data characteristics, feature ranking and model performance. Each figure is written to `reports/figures/` as PNG and SVG. Nothing is refit, so it runs in seconds. See [Ticket 06](implementation_steps/06_visualization.md#implementation-notes).

## Final Report

```bash
python -m src.report   # after python -m src.plots; needs Chrome, Chromium or Edge
```

This renders [reports/final_report.md](reports/final_report.md) to `reports/final_report.pdf` with headless Chrome. Set `CHROME=/path/to/browser` if the browser isn't in a standard location. The report covers all ten graded sections. See [Ticket 07](implementation_steps/07_discussion_and_conclusion.md#implementation-notes).

## Reproducing Everything

The raw scraped files are committed, so the full pipeline runs offline from `data/raw/`:

```bash
python -m src.preprocess && python -m src.features && python -m src.train \
  && python -m src.depreciation && python -m src.plots && python -m src.report
```

Re-running the scrapers first fetches the current listings, so the counts and results will differ from the report.

## Repository Layout

```
config/            vehicle scope and scraper settings
data/raw/          parsed output from each source (one file per source) + MSRP
data/processed/    integrated, cleaned dataset
data/cache/        cached HTTP responses (not committed)
models/            fitted model pipelines (not committed; created by src.train)
reports/           EDA summary, preprocessing log, text-feature report, encoding comparison, model results,
                   feature importance, depreciation analysis, and the final report (.md source + .pdf)
reports/figures/   report figures (PNG + SVG, created by src.plots)
src/scrapers/      one module per source, plus shared HTTP/IO helpers
src/preprocess.py  raw -> integrated pipeline
src/features.py    derived features + encoding ColumnTransformer
src/train.py       model tuning, evaluation and residual analysis
src/depreciation.py  age/mileage effects, feature ranking, depreciation curves
src/plots.py       report figures
src/report.py      final report Markdown -> PDF
implementation_steps/  per-step tickets
```
