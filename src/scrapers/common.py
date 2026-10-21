"""Shared helpers for the scrapers: config loading, a polite cached HTTP client, and CSV output."""

import hashlib
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "vehicles.yaml"
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

log = logging.getLogger("scrapers")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def iter_models(cfg):
    """Yield (make_cfg, model_cfg) for every model in scope."""
    for make in cfg["makes"]:
        for model in make["models"]:
            yield make, model


class PoliteClient:
    """HTTP client that rate-limits per host, retries transient failures and caches responses on disk.

    Cached responses mean a parser change can be re-run without re-scraping.
    """

    def __init__(self, source, delay=1.5, max_retries=3, use_cache=True, timeout=30):
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR / source
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._last_request = {}
        self.stats = {"requests": 0, "cache_hits": 0, "failures": 0}

    def _cache_path(self, url, params):
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".cache")

    def _wait(self, host):
        elapsed = time.monotonic() - self._last_request.get(host, 0)
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request[host] = time.monotonic()

    def get_text(self, url, params=None, accept="text/html"):
        """Return the response body, or None if the request ultimately fails."""
        cache_path = self._cache_path(url, params)
        if self.use_cache and cache_path.exists():
            self.stats["cache_hits"] += 1
            return cache_path.read_text(encoding="utf-8")

        host = urlparse(url).netloc
        for attempt in range(1, self.max_retries + 1):
            self._wait(host)
            self.stats["requests"] += 1
            try:
                resp = self.session.get(url, params=params, headers={"Accept": accept}, timeout=self.timeout)
            except requests.RequestException as e:
                log.warning("Request error (%s/%s) %s: %s", attempt, self.max_retries, url, e)
            else:
                if resp.status_code == 200:
                    cache_path.write_text(resp.text, encoding="utf-8")
                    return resp.text
                if resp.status_code in (404, 410):
                    break
                log.warning("HTTP %s (%s/%s) %s", resp.status_code, attempt, self.max_retries, url)
            time.sleep(self.delay * 2 ** attempt)

        self.stats["failures"] += 1
        return None

    def get_json(self, url, params=None):
        text = self.get_text(url, params=params, accept="application/json")
        return json.loads(text) if text else None


def save_csv(rows, filename):
    """Write rows to data/raw/<filename> and return the resulting DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / filename, index=False)
    log.info("Wrote %d rows to %s", len(df), RAW_DIR / filename)
    return df


def write_log(source, **info):
    """Merge collection metadata for one source into data/raw/collection_log.json."""
    path = RAW_DIR / "collection_log.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[source] = info
    path.write_text(json.dumps(data, indent=2, default=str))


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
