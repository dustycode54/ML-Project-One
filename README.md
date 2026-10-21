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

## Repository Layout

```
config/            vehicle scope and scraper settings
data/raw/          parsed output from each source (one file per source) + MSRP
data/processed/    integrated, cleaned dataset
data/cache/        cached HTTP responses (not committed)
models/            fitted model pipelines (not committed; created by src.train)
reports/           EDA summary, preprocessing log, text-feature report, encoding comparison, model results
src/scrapers/      one module per source, plus shared HTTP/IO helpers
src/preprocess.py  raw -> integrated pipeline
src/features.py    derived features + encoding ColumnTransformer
src/train.py       model tuning, evaluation and residual analysis
implementation_steps/  per-step tickets
```
