# Ticket 01 — Data Collection

| Field | Value |
|-------|-------|
| **Step** | 1 of 7 |
| **Depends on** | — |
| **Blocks** | [02 — Preprocessing & Integration](02_preprocessing_and_integration.md) |
| **Report section** | Data Collection with discussion (5%) |
| **Deliverable weight** | Data (20%), Code Quality (20%) |

## Summary

Scrape used-car listings from at least two independent online sources and collect MSRP data for every make / model / model year in scope. Save the raw parsed output from each source as its own file.

## Background

The target variable is `value_retention_ratio = listing_price / original_MSRP`, so every listing needs a matching MSRP. Choose the vehicle scope (makes/models) **before** scraping so both sources and the MSRP lookup cover the same vehicles.

## Tasks

- [ ] **Define scope.** Pick 3–5 makes, and 3–5 models per make. Prefer popular models so each source has enough listings. Record the choice in a config file (e.g. `config/vehicles.yaml`).
- [ ] **Select sources.** Choose ≥ 2 independent listing sites (e.g. Craigslist, Cars.com, CarGurus, Autotrader). Check each site's `robots.txt` and Terms of Service. Avoid sources that require a login (e.g. Facebook Marketplace) unless you have a workable, permitted approach.
- [ ] **Build a scraper per source** (e.g. `src/scrapers/<source>.py`):
  - Search by make/model, paginate, and parse listing detail pages.
  - Capture as many raw attributes as possible: price, year, make, model, trim, mileage, location, title status, transmission, drivetrain, fuel type, exterior color, body type, number of owners, accident history, listing description, URL, listing ID, post date.
  - Rate-limit requests, retry on failure, and cache raw HTML so a re-parse doesn't require re-scraping.
- [ ] **Collect MSRP data** for every make/model/year (and trim where possible) from a reliable source (manufacturer archives, Edmunds, KBB, etc.). Write down the source and method used. Save to `data/raw/msrp.csv`.
- [ ] **Save raw parsed data** separately for each source, e.g. `data/raw/<source>_listings.csv`.
- [ ] **Log collection metadata:** scrape dates, row counts per source, and any failures or blocked requests.

## Acceptance Criteria

- [ ] Data collected from **≥ 2 sources**.
- [ ] Each source contributes **≥ 500 listings** before integration.
- [ ] **3–5 makes**, each with **3–5 models**.
- [ ] **≥ 10 original attributes** collected (excluding engineered features).
- [ ] MSRP collected for every make, model, and model year present in the listings.
- [ ] One raw data file per source committed (or linked, if too large) in the repo.
- [ ] Scrapers can be re-run from a documented command.

> Scrape more than the minimum (≥ 700 per source is a safe target). Duplicate removal and dropping rows without an MSRP match in Step 2 must still leave **≥ 1,000** listings.

## Artifacts

- `src/scrapers/` — one module per source
- `data/raw/<source>_listings.csv` (×2+)
- `data/raw/msrp.csv`
- `config/vehicles.yaml`

## Notes for the Report

- Why these sources and vehicles were chosen.
- Obstacles: anti-bot measures, pagination limits, missing fields, inconsistent listing formats.
- MSRP source reliability and how trims were handled (base MSRP vs. trim-level MSRP).
