"""Collect original MSRP per trim for every make / model / model year in scope from CarsDirect.

CarsDirect's model-year research pages embed trim data in a Nuxt payload (`__NUXT_DATA__`):
a flat JSON array where objects reference other entries by index. We resolve the
"trims-<make>-<model>-<year>-*" entry into plain dicts.

Run: python -m src.scrapers.carsdirect_msrp
"""

import json
import re
from datetime import datetime, timezone

from .common import PoliteClient, iter_models, load_config, log, save_csv, setup_logging, write_log

PAGE_URL = "https://www.carsdirect.com/{make}/{model}/{year}"
NUXT_RE = re.compile(r'<script type="application/json"[^>]*id="__NUXT_DATA__">(.*?)</script>', re.S)
WRAPPERS = {"ShallowReactive", "Reactive", "Ref", "ShallowRef"}

FIELDS = {
    "year": "year",
    "make": "make",
    "model": "model",
    "trim": "name",
    "base_trim": "realTrimName",
    "msrp": "msrpInt",
    "invoice": "invoiceInt",
    "segment": "segment",
    "drive_type": "driveType",
    "engine": "engine",
    "transmission": "transmission",
    "fuel_type": "fuelType",
    "seating_capacity": "seatingCapacity",
    "acode": "acode",
}


def resolve(arr, idx):
    """Resolve a Nuxt (devalue) payload entry into plain Python values."""
    value = arr[idx]
    if isinstance(value, dict):
        return {k: resolve(arr, i) for k, i in value.items()}
    if isinstance(value, list):
        if value and value[0] in WRAPPERS:
            return resolve(arr, value[1])
        return [resolve(arr, i) for i in value]
    return value


def unwrap(arr, idx):
    """Follow reactive wrappers to the underlying raw entry without resolving its children."""
    value = arr[idx]
    while isinstance(value, list) and value and value[0] in WRAPPERS:
        value = arr[value[1]]
    return value


def parse_trims(html):
    match = NUXT_RE.search(html)
    if not match:
        return []
    arr = json.loads(match.group(1))
    # Only resolve the trims entry; other parts of the payload use devalue types we don't need.
    data = unwrap(arr, unwrap(arr, 0)["data"])
    key = next((k for k in data if k.startswith("trims-")), None)
    return resolve(arr, data[key]).get("trims", []) if key else []


def scrape(cfg):
    years = range(cfg["model_years"]["min"], cfg["model_years"]["max"] + 1)
    # CarsDirect pages are large and slow; use a longer delay and timeout than the listing scrapers.
    client = PoliteClient("carsdirect", delay=3.0, timeout=90)
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows, missing = [], []
    for make_cfg, model_cfg in iter_models(cfg):
        for year in years:
            url = PAGE_URL.format(make=make_cfg["carsdirect_make"], model=model_cfg["carsdirect"], year=year)
            html = client.get_text(url)
            trims = parse_trims(html) if html else []
            if not trims:
                missing.append(f"{make_cfg['make']} {model_cfg['name']} {year}")
                log.warning("No trims for %s %s %s", make_cfg["make"], model_cfg["name"], year)
                continue
            for t in trims:
                row = {out: t.get(src) for out, src in FIELDS.items()}
                row.update(search_make=make_cfg["make"], search_model=model_cfg["name"],
                           source_url=url, scraped_at=scraped_at)
                rows.append(row)
        log.info("%s %s: done (%d trim rows so far)", make_cfg["make"], model_cfg["name"], len(rows))

    df = save_csv(rows, "msrp.csv")
    write_log(
        "carsdirect_msrp",
        scraped_at=scraped_at,
        rows=len(df),
        model_years_missing=missing,
        http=client.stats,
    )
    return df


def main():
    setup_logging()
    scrape(load_config())


if __name__ == "__main__":
    main()
