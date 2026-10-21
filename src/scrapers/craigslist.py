"""Scrape used-car listings from Craigslist "cars+trucks" searches across several metro regions.

Two stages per model:
  1. Search each region (static HTML results) to collect candidate posting URLs.
  2. Fetch up to `max_posts_per_model` postings, round-robin across regions, and parse
     the attribute table, price, location and free-text description.

Run: python -m src.scrapers.craigslist
"""

import json
import re
from datetime import datetime, timezone
from itertools import chain, zip_longest

from bs4 import BeautifulSoup

from .common import PoliteClient, iter_models, load_config, log, save_csv, setup_logging, write_log

SEARCH_URL = "https://{region}.craigslist.org/search/cta"


def search(client, region, make_model, years):
    """Return [{url, region}] for one region's search results."""
    params = {
        "auto_make_model": make_model,
        "min_auto_year": years["min"],
        "max_auto_year": years["max"],
    }
    html = client.get_text(SEARCH_URL.format(region=region), params=params)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    return [
        {"url": a["href"], "region": region}
        for a in soup.select("li.cl-static-search-result a[href]")
    ]


def _text(el):
    return el.get_text(" ", strip=True) if el else None


def parse_posting(html):
    """Parse one posting page into a flat dict of raw attributes."""
    soup = BeautifulSoup(html, "html.parser")
    row = {}

    important = soup.select_one(".attrgroup .attr.important")
    if important:
        row["year"] = _text(important.select_one(".year"))
        row["make_model"] = _text(important.select_one(".makemodel"))

    # Remaining attributes are "label: value" pairs; key on the CSS class (e.g. auto_miles -> miles).
    for attr in soup.select(".attrgroup .attr:not(.important)"):
        cls = next((c for c in attr.get("class", []) if c != "attr"), None)
        value = attr.select_one(".valu")
        if cls and value:
            row[cls.removeprefix("auto_")] = _text(value)

    price = soup.select_one(".postingtitletext .price")
    row["price"] = _text(price)
    row["title"] = _text(soup.select_one("#titletextonly"))

    ld = soup.select_one("script#ld_posting_data")
    if ld:
        try:
            data = json.loads(ld.string)
            place = data.get("offers", {}).get("availableAtOrFrom", {})
            addr = place.get("address", {})
            row["city"] = addr.get("addressLocality")
            row["state"] = addr.get("addressRegion")
            row["zip"] = addr.get("postalCode")
            row["lat"] = place.get("geo", {}).get("latitude")
            row["lon"] = place.get("geo", {}).get("longitude")
            row["price"] = row["price"] or data.get("offers", {}).get("price")
        except (json.JSONDecodeError, AttributeError):
            pass

    body = soup.select_one("#postingbody")
    if body:
        for qr in body.select(".print-information"):
            qr.decompose()
        row["description"] = body.get_text("\n", strip=True)

    breadcrumbs = _text(soup.select_one(".breadcrumbs")) or ""
    row["seller_type"] = (
        "dealer" if "by dealer" in breadcrumbs else "owner" if "by owner" in breadcrumbs else None
    )

    info = " ".join(_text(p) or "" for p in soup.select(".postinginfo"))
    post_id = re.search(r"post id:\s*(\d+)", info)
    row["post_id"] = post_id.group(1) if post_id else None
    posted = soup.select_one(".postinginfos time.date")
    row["posted_at"] = posted.get("datetime") if posted else None
    return row


def round_robin(results):
    """Interleave search results by region so a capped sample stays geographically mixed."""
    by_region = {}
    for r in results:
        by_region.setdefault(r["region"], []).append(r)
    return [r for r in chain.from_iterable(zip_longest(*by_region.values())) if r]


def scrape(cfg):
    settings = cfg["scrape"]["craigslist"]
    client = PoliteClient("craigslist", delay=cfg["scrape"]["request_delay"])
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows, seen_urls, seen_ids = [], set(), set()
    for make_cfg, model_cfg in iter_models(cfg):
        candidates = []
        for region in settings["regions"]:
            for r in search(client, region, model_cfg["craigslist"], cfg["model_years"]):
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    candidates.append(r)

        kept = 0
        for cand in round_robin(candidates):
            if kept >= settings["max_posts_per_model"]:
                break
            html = client.get_text(cand["url"])
            if not html:
                continue
            row = parse_posting(html)
            if row.get("post_id") in seen_ids:  # same post surfaced in two regions
                continue
            seen_ids.add(row.get("post_id"))
            row.update(
                search_make=make_cfg["make"],
                search_model=model_cfg["name"],
                region=cand["region"],
                listing_url=cand["url"],
                scraped_at=scraped_at,
            )
            rows.append(row)
            kept += 1
        log.info("%s %s: %d candidates, %d postings kept",
                 make_cfg["make"], model_cfg["name"], len(candidates), kept)

    df = save_csv(rows, "craigslist_listings.csv")
    write_log(
        "craigslist",
        scraped_at=scraped_at,
        rows=len(df),
        rows_per_make=df["search_make"].value_counts().to_dict(),
        regions=settings["regions"],
        http=client.stats,
    )
    return df


def main():
    setup_logging()
    scrape(load_config())


if __name__ == "__main__":
    main()
