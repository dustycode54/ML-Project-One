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

- [x] Data collected from **≥ 2 sources**.
- [x] Each source contributes **≥ 500 listings** before integration.
- [x] **3–5 makes**, each with **3–5 models**.
- [x] **≥ 10 original attributes** collected (excluding engineered features).
- [x] MSRP collected for every make, model, and model year present in the listings.
- [x] One raw data file per source committed (or linked, if too large) in the repo.
- [x] Scrapers can be re-run from a documented command.

> Scrape more than the minimum (≥ 700 per source is a safe target). Duplicate removal and dropping rows without an MSRP match in Step 2 must still leave **≥ 1,000** listings.

## Artifacts

- `src/scrapers/` — one module per source
- `data/raw/<source>_listings.csv` (×2+)
- `data/raw/msrp.csv`
- `config/vehicles.yaml`

## Implementation Notes

**Scope** ([config/vehicles.yaml](../config/vehicles.yaml)): 4 makes × 4 models, model years 2012–2024.

| Make | Models |
|------|--------|
| Toyota | Camry, Corolla, RAV4, Tacoma |
| Honda | Civic, Accord, CR-V, Pilot |
| Ford | F-150, Escape, Explorer, Mustang |
| Chevrolet | Silverado 1500, Equinox, Malibu, Tahoe |

**Sources chosen**

| Source | Access method | Why |
|--------|---------------|-----|
| Autolist | Public JSON search API (`/api/v3/search`) + per-vehicle detail endpoint for description text | robots.txt allows all; structured fields incl. owner count, accident count, salvage flag. Queried per model *per model year* so the sample isn't dominated by late-model inventory. |
| Craigslist | Static search results across 14 metro regions → posting pages | Mix of private-party and dealer sellers; rich free-text descriptions for Ticket 03 text features. |
| CarsDirect | Model-year research pages (embedded Nuxt JSON) | Trim-level original MSRP for every model year in scope; robots.txt permits these pages. |

**Sources evaluated and rejected** (as of 2026-10-21): Cars.com, CarGurus, CarMax, TrueCar, Edmunds, Carvana, eBay Motors and Carfax all return 403/406 to scripted requests; Autotrader is behind an Akamai block page; Facebook Marketplace requires login. For MSRP, KBB rejects non-browser HTTP clients, MotorTrend serves a bot challenge after a few requests, and Car and Driver lacks pages for many older model years. We did not attempt to bypass any bot protection.

**Results (scraped 2026-10-21)**

| File | Rows | Notes |
|------|------|-------|
| `data/raw/autolist_listings.csv` | 1,248 | 312 per make, 96 per model year; 0 duplicate VINs; 55 columns; 84% have a description |
| `data/raw/craigslist_listings.csv` | 1,440 | 360 per make; 14 regions; 1,011 dealer / 429 owner posts; 0 duplicate post IDs; ~1,349 with price ≥ $1,000 + odometer + year |
| `data/raw/msrp.csv` | 3,022 | Trim-style MSRPs for all 16 models × 13 model years (no gaps) |

**Known caveats to carry into Ticket 02**
- 156 Autolist rows (12.5%) have `price = 0` ("call for price"); the API ignored the `price_min` filter. Drop them in preprocessing (1,092 priced rows remain).
- Craigslist prices are raw strings (`$7,000`, `$7999.00`) and a few are placeholders (`$1`); odometer is a comma-formatted string. Title status includes 76 rebuilt and 28 salvage vehicles.
- Craigslist posts often omit fields (odometer, title status), and dealer posts can be spammy / duplicated across regions. Within-run duplicates are dropped by post ID; cross-source / repost dedup happens in Ticket 02.
- Autolist's own `msrp` field is unreliable (sometimes equals the listing price) and is not collected; MSRP comes only from CarsDirect.
- CarsDirect lists one MSRP per trim *style* (e.g. cab/bed/drive variants for trucks); Ticket 02 needs a rule for matching listings that only state a base trim (e.g. median or minimum MSRP across styles of that base trim).

## Notes for the Report

- Why these sources and vehicles were chosen.
- Obstacles: anti-bot measures, pagination limits, missing fields, inconsistent listing formats.
- MSRP source reliability and how trims were handled (base MSRP vs. trim-level MSRP).
