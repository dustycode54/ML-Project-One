"""Scrape used-car listings from Autolist's public search API.

For every model in scope, each model year is queried separately so the sample covers
the full age range rather than being dominated by late-model inventory.

Run: python -m src.scrapers.autolist
"""

from datetime import datetime, timezone

from bs4 import BeautifulSoup

from .common import PoliteClient, iter_models, load_config, log, save_csv, setup_logging, write_log

SEARCH_URL = "https://www.autolist.com/api/v3/search"
DETAIL_URL = "https://www.autolist.com/api/vehicles/{vin}"

# Fields kept from search results (raw names, renamed only where noted).
SEARCH_FIELDS = [
    "id", "vin", "year", "make_name", "model_name", "trim", "condition", "body_type",
    "price", "base_price", "mileage", "transmission", "driveline", "fuel_type",
    "engine_cylinders", "door_count", "city_mpg", "hwy_mpg", "combined_mpg",
    "normalized_color_exterior", "normalized_color_interior",
    "dealer_name", "city", "state", "lat", "lon",
    "accident_count", "owner_count", "one_owner", "is_salvage", "is_damaged",
    "has_frame_damage", "has_theft_history", "damage_score",
    "leather", "navigation", "sunroof", "heated_seats", "backup_camera", "third_row_seats",
    "towing", "warranty", "imv_expected_price", "imv_deal_rating",
    "created_at", "updated_at", "vdp_url",
]

DETAIL_FIELDS = ["description", "engine_type", "exterior_color", "interior_color", "zip", "features"]


def search(client, make, model, year, limit):
    params = {
        "make": make,
        "model": model,
        "condition[]": "used",
        "year_min": year,
        "year_max": year,
        "price_min": 1000,  # skip "call for price" listings, which come back as price 0
        "limit": limit,
        "page": 1,
    }
    data = client.get_json(SEARCH_URL, params=params)
    return data["records"] if data else []


def fetch_detail(client, vin):
    data = client.get_json(DETAIL_URL.format(vin=vin), params={"ads": "web"})
    if not data:
        return {}
    detail = {k: data.get(k) for k in DETAIL_FIELDS}
    if detail["description"]:
        text = detail["description"].replace("[!@@ NL @@!]", "\n")
        detail["description"] = BeautifulSoup(text, "html.parser").get_text("\n", strip=True)
    if isinstance(detail["features"], list):
        detail["features"] = "; ".join(detail["features"])
    return detail


def scrape(cfg):
    settings = cfg["scrape"]["autolist"]
    years = range(cfg["model_years"]["min"], cfg["model_years"]["max"] + 1)
    client = PoliteClient("autolist", delay=cfg["scrape"]["request_delay"])
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows, seen = [], set()
    for make_cfg, model_cfg in iter_models(cfg):
        before = len(rows)
        for year in years:
            for rec in search(client, make_cfg["make"], model_cfg["autolist"], year, settings["per_model_year"]):
                if rec.get("vin") in seen:
                    continue
                seen.add(rec.get("vin"))
                row = {k: rec.get(k) for k in SEARCH_FIELDS}
                row["listing_url"] = "https://www.autolist.com" + (rec.get("vdp_url") or "")
                row["scraped_at"] = scraped_at
                if settings.get("fetch_details"):
                    row.update(fetch_detail(client, rec["vin"]))
                rows.append(row)
        log.info("%s %s: %d listings", make_cfg["make"], model_cfg["name"], len(rows) - before)

    df = save_csv(rows, "autolist_listings.csv")
    write_log(
        "autolist",
        scraped_at=scraped_at,
        rows=len(df),
        rows_per_make=df["make_name"].value_counts().to_dict(),
        http=client.stats,
    )
    return df


def main():
    setup_logging()
    scrape(load_config())


if __name__ == "__main__":
    main()
