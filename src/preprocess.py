"""Merge the raw per-source listings with CarsDirect MSRP into one clean dataset (Ticket 02).

Stages:
  1. Map each source onto a shared schema and normalize formats, units and category labels.
  2. Drop out-of-scope and invalid rows (missing price/mileage, placeholders, branded titles, EVs).
  3. Deduplicate within and across sources (VIN, then year/make/model/mileage/price).
  4. Match MSRP on year/make/model + trim, falling back to the model-year median.
  5. Compute value_retention_ratio = price / msrp and drop implausible ratios.

Outputs data/processed/integrated.csv, reports/eda_summary.md, reports/eda/*.csv and
reports/preprocessing_log.json.

Run: python -m src.preprocess
"""

import json
import logging
import re

import numpy as np
import pandas as pd

from .scrapers.common import RAW_DIR, ROOT, iter_models, load_config, setup_logging

PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"

log = logging.getLogger("preprocess")

# Source priority when two rows describe the same vehicle: Autolist has the richer structured fields.
SOURCE_PRIORITY = {"autolist": 0, "craigslist": 1}

# Target column -> (Autolist column, Craigslist column). None = not collected by that source.
SCHEMA_MAP = {
    "source_id": ("id", "post_id"),
    "vin": ("vin", "vin"),
    "year": ("year", "year"),
    "make": ("make_name", "search_make"),
    "model": ("model_name", "search_model"),
    "trim_raw": ("trim", "make_model"),
    "price": ("price", "price"),
    "mileage": ("mileage", "miles"),
    "body_type": ("body_type", "bodytype"),
    "drivetrain": ("driveline", "drivetrain"),
    "transmission": ("transmission", "transmission"),
    "fuel_type": ("fuel_type", "fuel_type"),
    "cylinders": ("engine_cylinders", "cylinders"),
    "exterior_color": ("normalized_color_exterior", "paint"),
    "condition": (None, "condition"),
    "title_status": ("is_salvage", "title_status"),
    "seller_type": (None, "seller_type"),
    "city": ("city", "city"),
    "state": ("state", "state"),
    "zip": ("zip", "zip"),
    "lat": ("lat", "lat"),
    "lon": ("lon", "lon"),
    "posted_at": ("created_at", "posted_at"),
    "owner_count": ("owner_count", None),
    "accident_count": ("accident_count", None),
    "one_owner": ("one_owner", None),
    "has_frame_damage": ("has_frame_damage", None),
    "has_theft_history": ("has_theft_history", None),
    "leather": ("leather", None),
    "navigation": ("navigation", None),
    "sunroof": ("sunroof", None),
    "heated_seats": ("heated_seats", None),
    "backup_camera": ("backup_camera", None),
    "third_row_seats": ("third_row_seats", None),
    "listing_title": (None, "title"),
    "description": ("description", "description"),
    "listing_url": ("listing_url", "listing_url"),
}

# Raw label -> canonical label. Labels not listed map to "other"; missing stays missing.
CATEGORY_MAPS = {
    "body_type": {
        "suv": "suv", "sedan": "sedan", "pickup": "pickup", "truck": "pickup", "coupe": "coupe",
        "convertible": "convertible", "hatchback": "hatchback", "wagon": "wagon",
    },
    # Craigslist has no AWD option (sellers pick "4wd"), so AWD and 4WD are merged.
    "drivetrain": {"fwd": "fwd", "rwd": "rwd", "awd": "awd_4wd", "4x4": "awd_4wd", "4wd": "awd_4wd"},
    "transmission": {"automatic": "automatic", "manual": "manual"},
    # Craigslist has no flex-fuel option (sellers pick "gas"), so flex fuel is gas.
    "fuel_type": {"gasoline": "gas", "gas": "gas", "flex fuel": "gas", "hybrid": "hybrid",
                  "diesel": "diesel", "electric": "electric"},
    "exterior_color": {c: c for c in ["black", "white", "gray", "silver", "blue", "red", "brown",
                                      "green", "gold", "yellow", "orange"]} | {"grey": "gray"},
}

US_STATES = set(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)

# Variants that share a Craigslist make/model filter value but are different vehicles with no MSRP
# in scope (battery-electric models, Shelby Mustangs, other nameplates).
OUT_OF_SCOPE_PATTERN = re.compile(
    r"\bmach\s?-?\s?e\b|\bmache\b|\blightning\b|\bequinox ev\b|\btundra\b|\bcrosstour\b"
    r"|\bshelby\b|\bgt\s?350\b|\bgt\s?500\b",
    re.I,
)
# "$1,500 down", "2K DOWN", "low down payment": the listed price is a down payment, not the car's price.
DOWN_PAYMENT_PATTERN = re.compile(r"\d\s*k?\s*down\b|\bdown\s*payment|\blow down\b", re.I)
DOWN_PAYMENT_MAX_PRICE = 5_000

BRANDED_TITLES = {"salvage", "rebuilt", "missing", "parts only"}
MIN_PRICE = 1_000
MAX_MILEAGE = 400_000
MIN_MILEAGE_USED = 100  # below this on a car 2+ model years old the odometer is a placeholder
RATIO_RANGE = (0.10, 1.50)
FUZZY_PRICE_TOLERANCE = 0.10


# --------------------------------------------------------------------------- 1. schema mapping

def parse_number(series):
    """'$7,999.00' / '111,447' / '45k' -> float; anything unparseable -> NaN."""
    s = series.astype("string").str.lower().str.replace(r"[$,\s]|mi(les)?", "", regex=True)
    thousands = s.str.endswith("k", na=False)
    num = pd.to_numeric(s.str.removesuffix("k"), errors="coerce")
    return num.where(~thousands, num * 1_000)


def map_category(series, mapping):
    s = series.astype("string").str.strip().str.lower()
    return s.map(mapping).where(s.isna(), s.map(mapping).fillna("other"))


def to_schema(df, source):
    """Rename/derive a source's columns into the shared schema."""
    idx = 0 if source == "autolist" else 1
    out = pd.DataFrame(index=df.index)
    for target, cols in SCHEMA_MAP.items():
        col = cols[idx]
        out[target] = df[col] if col else pd.NA
    out.insert(0, "source", source)
    out.insert(0, "listing_id", source + "_" + out["source_id"].astype(str))

    if source == "autolist":
        out["title_status"] = np.where(df["is_salvage"], "salvage", "clean")
        out["seller_type"] = "dealer"  # Autolist only lists dealer inventory
        out["zip"] = df["zip"].astype("Int64").astype("string").str.zfill(5)
    else:
        out["cylinders"] = parse_number(df["cylinders"].str.extract(r"(\d+)")[0])
        out["condition"] = df["condition"].str.lower()
        broken = df["odometer_broken"].notna() | df["odometer_rolled_over"].notna()
        out["odometer_unreliable"] = broken
    out["vin"] = out["vin"].astype("string").str.upper().str.strip()
    out.loc[out["vin"].str.len() != 17, "vin"] = pd.NA
    return out


def normalize(df):
    df["price"] = parse_number(df["price"])
    df["mileage"] = parse_number(df["mileage"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["cylinders"] = pd.to_numeric(df["cylinders"], errors="coerce")
    for col, mapping in CATEGORY_MAPS.items():
        df[col] = map_category(df[col], mapping)
    df["title_status"] = df["title_status"].astype("string").str.lower()
    df["state"] = df["state"].astype("string").str.upper().str.strip()
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True, format="ISO8601").dt.strftime("%Y-%m-%d")
    # Neither source reports kilometres: Autolist is US-only and Craigslist's odometer field is miles.
    return df


# --------------------------------------------------------------------------- 2. invalid rows

def drop_invalid(df, cfg, counts):
    """Apply each removal rule in order, recording how many rows it removed."""
    years = cfg["model_years"]
    scope = {(mk["make"], md["name"]) for mk, md in iter_models(cfg)}
    text = df["trim_raw"].fillna("") + " " + df["listing_title"].fillna("")
    age = pd.to_datetime(df["posted_at"]).dt.year - df["year"]

    rules = [
        ("make/model/year out of scope",
         ~pd.Series(list(zip(df["make"], df["model"])), index=df.index).isin(scope)
         | ~df["year"].between(years["min"], years["max"])),
        ("electric or other-model variant", text.str.contains(OUT_OF_SCOPE_PATTERN) | (df["fuel_type"] == "electric")),
        ("price missing or < $1,000 (call-for-price / placeholder)", df["price"].isna() | (df["price"] < MIN_PRICE)),
        ("price < $5,000 is a down payment ('$X down' ad)",
         (df["price"] < DOWN_PAYMENT_MAX_PRICE)
         & (text + " " + df["description"].fillna("")).str.contains(DOWN_PAYMENT_PATTERN)),
        ("mileage missing", df["mileage"].isna()),
        ("odometer flagged broken / rolled over", df["odometer_unreliable"].eq(True)),
        ("mileage placeholder or > 400k",
         ((df["mileage"] < MIN_MILEAGE_USED) & (age >= 2)) | (df["mileage"] > MAX_MILEAGE)),
        ("branded title (salvage / rebuilt / missing)",
         df["title_status"].isin(BRANDED_TITLES) | (df["condition"] == "salvage")),
        ("non-US location", df["state"].notna() & ~df["state"].isin(US_STATES)),
    ]
    # Each row is attributed to the first rule it fails.
    removed = pd.Series(False, index=df.index)
    for name, mask in rules:
        mask = mask.fillna(False).astype(bool) & ~removed
        counts[f"removed: {name}"] = int(mask.sum())
        removed |= mask
    return df[~removed].drop(columns="odometer_unreliable")


# --------------------------------------------------------------------------- 3. deduplication

def deduplicate(df, counts):
    """Drop duplicate vehicles, keeping the highest-priority / most complete row.

    Rule 1: identical VIN.
    Rule 2: same year, make, model and exact odometer reading, with prices within 10%.
            Exact mileage on the same model-year is near-unique, and the price tolerance allows
            for small price changes between reposts and across sites.
    """
    df = df.assign(
        _priority=df["source"].map(SOURCE_PRIORITY),
        _filled=df.notna().sum(axis=1),
    ).sort_values(["_priority", "_filled"], ascending=[True, False])

    has_vin = df["vin"].notna()
    vin_dup = has_vin & df.duplicated("vin")
    counts["removed: duplicate VIN"] = int(vin_dup.sum())
    df = df[~vin_dup]

    keep = []
    for _, group in df.groupby(["year", "make", "model", "mileage"], sort=False):
        kept_prices = []
        for idx, price in group["price"].items():
            if any(abs(price - p) <= FUZZY_PRICE_TOLERANCE * p for p in kept_prices):
                continue
            kept_prices.append(price)
            keep.append(idx)
    counts["removed: duplicate year/make/model/mileage/price"] = len(df) - len(keep)
    return df.loc[keep].drop(columns=["_priority", "_filled"]).sort_index()


# --------------------------------------------------------------------------- 4. MSRP matching

def _norm(text):
    """Lowercase, drop hyphens ('EX-L' == 'EXL', 'V-6' == 'V6') and split into tokens."""
    text = str(text).lower().replace("-", "").replace("off road", "offroad").replace("ltd", "limited")
    return tuple(re.findall(r"[a-z0-9.]+", text))


def _aliases(base_trim):
    """Other names sellers use for a CarsDirect trim.

    Strips engine labels ('2.4 LX', 'EX-L 1.5T', 'TRD Off Road V6' -> 'TRD Off Road'), the 'Base'
    prefix ('Base V6' -> 'V6') and Chevrolet's numbered equipment groups ('1LT' / '2LT' -> 'LT',
    '2LZ' -> 'LTZ', '1LS' -> 'LS').
    """
    aliases = {
        re.sub(r"^\d\.\d\s+|\s+(\d\.\dt|v-?6)$", "", base_trim, flags=re.I),
        re.sub(r"^base\s+", "", base_trim, flags=re.I),
        re.sub(r"^[123](LT|LS)$", r"\1", base_trim),
        re.sub(r"^[123]LZ$", "LTZ", base_trim),
    }
    return {_norm(a) for a in aliases} - {_norm(base_trim), ()}


def _contains(tokens, key):
    n = len(key)
    return any(tokens[i:i + n] == key for i in range(len(tokens) - n + 1))


def trim_keys(msrp):
    """One row per (MSRP style, match key): tier 0 = the CarsDirect trim name, tier 1 = an alias."""
    exact = msrp.assign(key=msrp["base_trim"].map(_norm), tier=0)
    alias = msrp.assign(key=msrp["base_trim"].map(lambda t: list(_aliases(t))), tier=1).explode("key")
    keys = pd.concat([exact, alias.dropna(subset="key")])
    return keys.assign(key_len=keys["key"].map(lambda k: len(" ".join(k))))


def find_trim(texts, keys):
    """Return the MSRP styles of the best trim named in `texts`, or None.

    Tries each text in order. Within a text, a CarsDirect trim name beats an alias, and the longest
    match wins ('SE V6' over 'SE'). Single-letter trims ('L', 'S') are only trusted in the first
    (structured) text, not in free-text titles.
    """
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = _norm(text)
        for tier in (0, 1):
            tier_keys = keys[keys["tier"] == tier]
            hits = tier_keys[[
                _contains(tokens, key) and (i == 0 or n > 1)
                for key, n in zip(tier_keys["key"], tier_keys["key_len"])
            ]]
            if len(hits):
                return hits[hits["key_len"] == hits["key_len"].max()]
    return None


def match_msrp(df, msrp, counts):
    """Attach msrp, the matched trim and the match level to every listing.

    Each CarsDirect trim has several styles (cab/bed/drive variants); a matched trim takes the
    median MSRP across its styles. Listings with no recognizable trim get the median MSRP across
    all styles of that make/model/year.
    """
    styles_by_model = {k: g for k, g in msrp.groupby(["year", "make", "model"])}
    keys_by_model = {k: g for k, g in trim_keys(msrp).groupby(["year", "make", "model"])}

    rows = []
    for idx, r in df.iterrows():
        styles = styles_by_model.get((r["year"], r["make"], r["model"]))
        if styles is None:
            rows.append((idx, np.nan, pd.NA, "unmatched", 0))
            continue
        model_prefix = re.compile(rf"^\W*{re.escape(r['make'])}\s+{re.escape(r['model'])}", re.I)
        texts = [model_prefix.sub("", str(r["trim_raw"]))]
        if r["source"] == "craigslist":
            title = re.sub(rf"\b{r['year']}\b", "", str(r["listing_title"]))
            texts.append(re.sub(rf"\b{re.escape(r['make'])}\b|\b{re.escape(r['model'])}\b", "", title, flags=re.I))
        hit = find_trim(texts, keys_by_model[(r["year"], r["make"], r["model"])])
        if hit is not None:
            rows.append((idx, hit["msrp"].median(), hit["base_trim"].iloc[0], "trim", len(hit)))
        else:
            rows.append((idx, styles["msrp"].median(), pd.NA, "model_year", len(styles)))

    m = pd.DataFrame(rows, columns=["idx", "msrp", "trim", "msrp_match_level", "msrp_styles"]).set_index("idx")
    df = df.join(m)
    unmatched = df["msrp_match_level"] == "unmatched"
    counts["removed: no MSRP for make/model/year"] = int(unmatched.sum())
    return df[~unmatched]


# --------------------------------------------------------------------------- 5. target

def add_target(df, counts):
    df["value_retention_ratio"] = (df["price"] / df["msrp"]).round(4)
    lo, hi = RATIO_RANGE
    bad = ~df["value_retention_ratio"].between(lo, hi)
    counts[f"removed: value_retention_ratio outside [{lo}, {hi}]"] = int(bad.sum())
    return df[~bad]


# --------------------------------------------------------------------------- EDA

def _fmt(v):
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def md_table(df, index=True):
    """Render a DataFrame as a GitHub Markdown table (avoids a tabulate dependency)."""
    if index:
        df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


NUMERIC_COLS = ["price", "msrp", "value_retention_ratio", "year", "mileage", "cylinders",
                "owner_count", "accident_count"]
CATEGORICAL_COLS = ["source", "make", "model", "trim", "body_type", "drivetrain", "transmission",
                    "fuel_type", "exterior_color", "condition", "title_status", "seller_type",
                    "state", "msrp_match_level"]


def eda(df, counts):
    """Write summary tables to reports/eda/*.csv and a readable reports/eda_summary.md."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    describe = df[NUMERIC_COLS].describe().T
    missing = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(1),
    })
    missing_by_source = df.groupby("source").agg(lambda s: round(s.isna().mean() * 100, 1)).T
    per_source = df.groupby("source").agg(
        listings=("listing_id", "size"),
        median_price=("price", "median"),
        median_mileage=("mileage", "median"),
        median_year=("year", "median"),
        median_msrp=("msrp", "median"),
        median_ratio=("value_retention_ratio", "median"),
        mean_ratio=("value_retention_ratio", "mean"),
        trim_match_pct=("msrp_match_level", lambda s: round((s == "trim").mean() * 100, 1)),
        with_description_pct=("description", lambda s: round(s.notna().mean() * 100, 1)),
    )
    per_model = df.groupby(["make", "model"]).agg(
        listings=("listing_id", "size"),
        autolist=("source", lambda s: int((s == "autolist").sum())),
        craigslist=("source", lambda s: int((s == "craigslist").sum())),
        median_price=("price", "median"),
        median_ratio=("value_retention_ratio", "median"),
    )
    value_counts = pd.concat([
        df[c].value_counts(dropna=False).rename_axis("value").reset_index().assign(column=c)
        for c in CATEGORICAL_COLS
    ])[["column", "value", "count"]]

    describe.to_csv(EDA_DIR / "numeric_describe.csv")
    missing.to_csv(EDA_DIR / "missing_values.csv")
    missing_by_source.to_csv(EDA_DIR / "missing_by_source_pct.csv")
    per_source.to_csv(EDA_DIR / "per_source.csv")
    per_model.to_csv(EDA_DIR / "per_model.csv")
    value_counts.to_csv(EDA_DIR / "categorical_value_counts.csv", index=False)

    stage_table = pd.DataFrame(list(counts.items()), columns=["stage", "rows"])
    schema_table = pd.DataFrame(
        [(t, a or "—", c or "—") for t, (a, c) in SCHEMA_MAP.items()],
        columns=["integrated column", "Autolist", "Craigslist"],
    )
    small_cats = [c for c in CATEGORICAL_COLS if df[c].nunique() <= 15]
    cat_sections = "\n\n".join(
        f"**{c}**\n\n" + md_table(df[c].value_counts(dropna=False).rename_axis(c).to_frame())
        for c in small_cats
    )
    report = f"""# EDA Summary — Integrated Dataset

Generated by `python -m src.preprocess`. Tables are also saved as CSV under `reports/eda/`.

## Row counts by stage

{md_table(stage_table, index=False)}

## Schema mapping

Autolist `is_salvage` is turned into `title_status`; Autolist rows get `seller_type = dealer`.
`trim_raw` for Craigslist is the free-text make/model field (e.g. `honda cr-v ex-l awd`), parsed for
trim during MSRP matching. `trim` is the CarsDirect trim the listing was matched to.

{md_table(schema_table, index=False)}

## Numeric attributes

{md_table(describe)}

## Per-source comparison

{md_table(per_source)}

## Per-model counts

{md_table(per_model)}

## Missing values (% of rows, by source)

Columns that one source never collects show 100% for that source.

{md_table(missing_by_source[missing_by_source.max(axis=1) > 0])}

## Categorical attributes

`model` and `trim` value counts are in `reports/eda/categorical_value_counts.csv`.

{cat_sections}
"""
    (REPORTS_DIR / "eda_summary.md").write_text(report)
    log.info("Wrote EDA summary to %s", REPORTS_DIR / "eda_summary.md")


# --------------------------------------------------------------------------- pipeline

OUTPUT_COLUMNS = [
    "listing_id", "source", "source_id", "vin", "year", "make", "model", "trim", "trim_raw",
    "price", "mileage", "msrp", "msrp_match_level", "msrp_styles", "value_retention_ratio",
    "body_type", "drivetrain", "transmission", "fuel_type", "cylinders", "exterior_color",
    "condition", "title_status", "seller_type", "city", "state", "zip", "lat", "lon", "posted_at",
    "owner_count", "accident_count", "one_owner", "has_frame_damage", "has_theft_history",
    "leather", "navigation", "sunroof", "heated_seats", "backup_camera", "third_row_seats",
    "listing_title", "description", "listing_url",
]


def fill_missing(df):
    """Fill categorical gaps with 'unknown' so they remain a usable level for encoding.

    Numeric gaps (cylinders, owner/accident counts, location) are left as NaN: they are either
    structurally missing for one source or best imputed inside the model pipeline (Ticket 04)
    so test rows don't leak into the imputation.
    """
    cats = ["body_type", "drivetrain", "transmission", "fuel_type", "exterior_color", "condition",
            "title_status"]
    df[cats] = df[cats].fillna("unknown")
    return df


def run():
    cfg = load_config()
    raw = {
        "autolist": pd.read_csv(RAW_DIR / "autolist_listings.csv"),
        "craigslist": pd.read_csv(RAW_DIR / "craigslist_listings.csv", dtype={"zip": str}),
    }
    msrp = pd.read_csv(RAW_DIR / "msrp.csv")

    counts = {f"raw: {s}": len(d) for s, d in raw.items()}
    df = pd.concat([to_schema(d, s) for s, d in raw.items()], ignore_index=True)
    counts["raw: combined"] = len(df)

    df = normalize(df)
    df = drop_invalid(df, cfg, counts)
    counts["after invalid / out-of-scope removal"] = len(df)
    df = deduplicate(df, counts)
    counts["after deduplication"] = len(df)
    df = match_msrp(df, msrp, counts)
    counts["after MSRP match"] = len(df)
    df = add_target(df, counts)
    df = fill_missing(df)[OUTPUT_COLUMNS]
    counts["final"] = len(df)
    for source, n in df["source"].value_counts().items():
        counts[f"final: {source}"] = int(n)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "integrated.csv", index=False)
    log.info("Wrote %d rows to %s", len(df), PROCESSED_DIR / "integrated.csv")

    match = df.groupby("source")["msrp_match_level"].value_counts(normalize=True).round(3)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "preprocessing_log.json").write_text(json.dumps({
        "stage_counts": counts,
        "msrp_match_rate": {f"{s}/{lvl}": v for (s, lvl), v in match.items()},
    }, indent=2))
    eda(df, counts)
    return df


if __name__ == "__main__":
    setup_logging()
    run()
