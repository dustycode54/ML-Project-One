"""Derive model features from the integrated dataset and define the encoding pipeline (Ticket 03).

Stages:
  1. Structured features: vehicle age, miles per year, census region, description length.
  2. Text features: regex flags from the listing title + description (one owner, clean title, ...),
     with a negation check so "no check engine light" doesn't count as needing work.
  3. Encoding: a ColumnTransformer that one-hot encodes low-cardinality categoricals and
     target- or frequency-encodes high-cardinality ones (trim, city, state). It is fit on
     training rows only; TargetEncoder also cross-fits inside the training rows.
  4. Encoding comparison: 5-fold CV of Ridge and HistGradientBoosting on the training split for
     each high-cardinality encoding, plus a run without msrp.

Outputs data/processed/features.csv (engineered features before encoding; encoding is fit per
split by build_preprocessor), reports/text_features.md and reports/encoding_comparison.csv.

Run: python -m src.features
"""

import logging
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from .preprocess import PROCESSED_DIR, REPORTS_DIR, md_table
from .scrapers.common import setup_logging

log = logging.getLogger("features")

TARGET = "value_retention_ratio"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns that must never be model inputs: the target, its numerator, and row identifiers.
LEAKY_COLUMNS = {"price", TARGET}
ID_COLUMNS = {"listing_id", "source_id", "vin", "listing_url"}

# US Census regions (https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
CENSUS_REGIONS = {
    "northeast": "CT ME MA NH RI VT NJ NY PA",
    "midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "south": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "west": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_REGION = {st: region for region, states in CENSUS_REGIONS.items() for st in states.split()}


# --------------------------------------------------------------------------- 1. structured features

def add_structured_features(df):
    """vehicle_age, miles_per_year, region, trim_key and description length."""
    listing_year = pd.to_datetime(df["posted_at"]).dt.year
    df["vehicle_age"] = (listing_year - df["year"]).clip(lower=0)
    df["miles_per_year"] = (df["mileage"] / df["vehicle_age"].clip(lower=1)).round(0)
    df["region"] = df["state"].map(STATE_REGION).fillna("unknown")
    # Trim names repeat across models ("LX", "SE", "LT"), so the encoded key includes make and model.
    df["trim_key"] = df["make"] + " " + df["model"] + " " + df["trim"].fillna("unknown")
    df["city_key"] = (df["city"].str.lower().str.strip() + ", " + df["state"]).fillna("unknown")
    df["state"] = df["state"].fillna("unknown")
    df["has_description"] = df["description"].notna().astype(int)
    df["log_description_len"] = np.log1p(df["description"].fillna("").str.len()).round(3)
    return df


# --------------------------------------------------------------------------- 2. text features

# Feature -> (pattern, apply negation check). Patterns run on lowercased, whitespace-collapsed
# "title . description". Patterns that already encode the negation themselves ("no accidents")
# skip the negation check.
TEXT_PATTERNS = {
    # "one owner", "1-owner", "single owner", "1 previous owner", "original owner", typos ("1 woner").
    "txt_one_owner": (
        r"\b(?:one|1|single)[\s-]*(?:previous\s*)?(?:owner|wner|woner)\b|\boriginal\s*owner\b", True),
    # "clean title", "clear title", "clean texas title", "title is clean", "clean autocheck & title".
    "txt_clean_title": (
        r"\bclea[nr]\s*(?:\w+\s*(?:(?:&|and)\s*)?)?title\b|\btitle\s*(?:is\s*)?(?:clean|clear)\b", True),
    # A clean vehicle-history report.
    "txt_clean_history": (r"\bclean\s*(?:car-?fax|autocheck|history|vehicle history)\b", True),
    # An explicit statement that the car has had no accidents. Safety-feature copy ("helps prevent
    # accidents") has no negation word and doesn't match.
    "txt_accident_free": (
        r"\b(?:no|zero|0)\s*(?:reported\s*|prior\s*)?accidents?\b|\baccident[\s-]*free\b"
        r"|\bnever\b[\w\s]{0,20}?\baccidents?\b|\bclean\s*accident\s*history\b",
        False),
    # Service / maintenance records or history, or a statement of regular servicing.
    "txt_service_records": (
        r"\b(?:service|maintenance)\s*(?:records?|history)\b|\bdealer[\s-]*(?:maintained|serviced)\b"
        r"|\bregularly\s*(?:maintained|serviced)\b|\bserviced\s*regularly\b",
        True),
    # New tires: "new tires", "4 new michelin tires", "brand new set of tires", "tires are new".
    "txt_new_tires": (
        r"\bnew\s*(?:set\s*of\s*)?(?:[a-z]+\s*)?tires\b|\btires\s*(?:are\s*)?(?:brand\s*)?new\b", True),
    # No leading \b on "non": scraped feature lists run words together ("bluetooth wirelessnon-smoker").
    "txt_non_smoker": (
        r"non[\s-]*smok(?:er|ing)\b|\bsmoke[\s-]*free\b|\bno\s*smok(?:ing|ers?)\b"
        r"|\bnever\s*(?:been\s*)?smoked\b",
        False),
    # Known mechanical problems. "As is" is left out: dealer legal disclaimers use it on every car.
    "txt_needs_work": (
        r"\bneeds?\s*(?:some\s*)?(?:work|repairs?|tlc|(?:a\s*)?(?:new\s*)?(?:engine|transmission))\b"
        r"|\bmechanic(?:al|s)?\s*special\b|\bcheck\s*engine\s*light\s*(?:is\s*)?on\b"
        r"|\b(?:transmission|engine)\s*(?:slips|slipping|problems?|issues?|noise)\b"
        r"|\b(?:not\s*running|does\s*n.?t\s*run|won.?t\s*start)\b",
        True),
}
TEXT_FEATURES = list(TEXT_PATTERNS)

# A match is discarded if one of these appears in the NEGATION_WINDOW characters before it
# ("not one owner", "no check engine light on", "no engine issues", "doesn't need work").
NEGATION = re.compile(r"\b(?:no|not|never|without|nothing|(?:does|do|did|is|was|has|have|wo|ca)n['’]?t)\b")
NEGATION_WINDOW = 15


def listing_text(df):
    text = df["listing_title"].fillna("") + " . " + df["description"].fillna("")
    return text.str.lower().str.replace(r"\s+", " ", regex=True)


def find_match(text, pattern, check_negation):
    """Return the first non-negated match of `pattern` in `text`, or None."""
    for m in re.finditer(pattern, text):
        if check_negation and NEGATION.search(text[max(0, m.start() - NEGATION_WINDOW):m.start()]):
            continue
        return m
    return None


def add_text_features(df):
    text = listing_text(df)
    for name, (pattern, negate) in TEXT_PATTERNS.items():
        df[name] = text.map(lambda t: find_match(t, pattern, negate) is not None).astype(int)
    return df


def build_features(df):
    return add_text_features(add_structured_features(df.copy()))


# --------------------------------------------------------------------------- 3. encoding

NUMERIC_FEATURES = ["vehicle_age", "mileage", "miles_per_year", "msrp", "cylinders", "lat", "lon",
                    "has_description", "log_description_len"] + TEXT_FEATURES
LOW_CARD_FEATURES = ["source", "make", "model", "body_type", "drivetrain", "transmission",
                     "fuel_type", "exterior_color", "condition", "title_status", "seller_type",
                     "msrp_match_level", "region"]
HIGH_CARD_FEATURES = ["trim_key", "state", "city_key"]
FEATURE_COLUMNS = NUMERIC_FEATURES + LOW_CARD_FEATURES + HIGH_CARD_FEATURES


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its share of the training rows; unseen categories get 0.

    Needs no target, so it cannot leak it, but categories with equal counts collide.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.freqs_ = {c: X[c].value_counts(normalize=True) for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return np.column_stack([X[c].map(self.freqs_[c]).fillna(0).to_numpy(float) for c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.feature_names_in_], dtype=object)


def high_card_encoder(kind):
    """Encoder for HIGH_CARD_FEATURES: 'target', 'frequency' or 'onehot' (the comparison baseline)."""
    if kind == "target":
        # Cross-fitted: during fit, each training row is encoded from the other 4/5 of the training
        # rows, so the model never sees a row's own target in its encoding. transform() on new rows
        # uses the encoding learned from all training rows. Smoothing shrinks rare levels to the mean.
        return TargetEncoder(target_type="continuous", smooth="auto",
                             cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
    if kind == "frequency":
        return FrequencyEncoder()
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False)
    raise ValueError(f"unknown encoding {kind!r}")


def build_preprocessor(high_card="frequency", numeric=NUMERIC_FEATURES):
    """ColumnTransformer mapping the build_features() columns to a numeric matrix.

    Numeric: median imputation (cylinders, lat/lon are sometimes missing) + standardization.
    Low-cardinality categoricals: one-hot, ignoring unseen levels.
    High-cardinality categoricals: high_card_encoder(high_card), then standardization. Frequency is
    the default: it tied for best CV score in compare_encodings and uses no target information.
    All statistics are learned in fit(), so inside a Pipeline they come from training rows only.
    """
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(numeric)),
        ("low_card", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARD_FEATURES),
        ("high_card", make_pipeline(high_card_encoder(high_card), StandardScaler()), HIGH_CARD_FEATURES),
    ], verbose_feature_names_out=False)


def check_leakage(columns):
    """Fail if a model input is the target, its numerator, or a row identifier."""
    bad = (LEAKY_COLUMNS | ID_COLUMNS) & set(columns)
    if bad:
        raise ValueError(f"leaky or identifier columns used as features: {sorted(bad)}")


def split(features):
    """The train/test split shared with Ticket 04: 80/20, stratified by make."""
    return train_test_split(features, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                            stratify=features["make"])


# --------------------------------------------------------------------------- 4. encoding comparison

def compare_encodings(train):
    """5-fold CV on the training split for each high-cardinality encoding and two model families."""
    X, y = train[FEATURE_COLUMNS], train[TARGET]
    models = {
        "ridge": lambda: Ridge(alpha=1.0),
        "hist_gb": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    variants = [(enc, NUMERIC_FEATURES) for enc in ("onehot", "frequency", "target")]
    variants.append(("target, no msrp", [c for c in NUMERIC_FEATURES if c != "msrp"]))
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}

    rows = []
    for encoding, numeric in variants:
        for model_name, make_model in models.items():
            pipe = Pipeline([
                ("prep", build_preprocessor(encoding.split(",")[0], numeric)),
                ("model", make_model()),
            ])
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
            row = {"encoding": encoding, "model": model_name}
            for metric in scoring:
                vals = scores[f"test_{metric}"] * (1 if metric == "r2" else -1)
                row[f"{metric}_mean"] = round(vals.mean(), 4)
                row[f"{metric}_std"] = round(vals.std(), 4)
            rows.append(row)
            log.info("%-16s %-8s MAE %.4f  RMSE %.4f  R2 %.3f", encoding, model_name,
                     row["mae_mean"], row["rmse_mean"], row["r2_mean"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- text feature report

SPOT_CHECK_N = 5
SNIPPET_WIDTH = 60


def _snippet(text, m):
    s = text[max(0, m.start() - SNIPPET_WIDTH):m.end() + SNIPPET_WIDTH]
    return "…" + s.replace("|", "/") + "…"


def spot_check(features, text):
    """Sample matches and near misses (a keyword is present but the extractor said no) per feature."""
    keywords = {
        "txt_one_owner": r"owner", "txt_clean_title": r"title", "txt_clean_history": r"carfax|autocheck",
        "txt_accident_free": r"accident", "txt_service_records": r"service|maintenance",
        "txt_new_tires": r"tires", "txt_non_smoker": r"smok", "txt_needs_work": r"needs?|check engine|issue",
    }
    sections = []
    for name, (pattern, negate) in TEXT_PATTERNS.items():
        hits = features.index[features[name] == 1]
        near = features.index[(features[name] == 0) & text.str.contains(keywords[name])]
        lines = [f"### `{name}`", "", "Matches:", ""]
        for i in pd.Series(hits).sample(min(SPOT_CHECK_N, len(hits)), random_state=RANDOM_STATE):
            lines.append(f"- {_snippet(text[i], find_match(text[i], pattern, negate))}")
        lines += ["", f"Not flagged, but contains `{keywords[name]}`:", ""]
        for i in pd.Series(near).sample(min(SPOT_CHECK_N, len(near)), random_state=RANDOM_STATE):
            lines.append(f"- {_snippet(text[i], re.search(keywords[name], text[i]))}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def text_feature_report(features):
    text = listing_text(features)
    prevalence = pd.DataFrame({
        "all_pct": features[TEXT_FEATURES].mean() * 100,
        **{f"{s}_pct": g[TEXT_FEATURES].mean() * 100 for s, g in features.groupby("source")},
        "median_ratio_flagged": [features.loc[features[c] == 1, TARGET].median() for c in TEXT_FEATURES],
        "median_ratio_not_flagged": [features.loc[features[c] == 0, TARGET].median() for c in TEXT_FEATURES],
    }).round(2)

    # Autolist carries structured vehicle-history fields, which act as labels for two extractors.
    auto = features[features["source"] == "autolist"]
    checks = []
    for feat, field, value in [("txt_one_owner", "owner_count", 1), ("txt_accident_free", "accident_count", 0)]:
        known = auto[auto[field].notna()]
        pred, truth = known[feat] == 1, known[field] == value
        checks.append({
            "extractor": feat, "Autolist field": f"{field} == {value}", "rows": len(known),
            "flagged": int(pred.sum()),
            "precision": round((pred & truth).sum() / max(pred.sum(), 1), 3),
            "recall": round((pred & truth).sum() / max(truth.sum(), 1), 3),
        })

    patterns = pd.DataFrame(
        # "\|" keeps regex alternations from splitting the Markdown table cell.
        [(n, "`" + p.replace("|", r"\|") + "`", "yes" if neg else "no") for n, (p, neg) in TEXT_PATTERNS.items()],
        columns=["feature", "pattern", "negation check"],
    )
    report = f"""# Text Features — Prevalence and Validation

Generated by `python -m src.features`. Each extractor is a regex over the lowercased
`listing_title + description`. Where marked, a match is discarded if `no`, `not`, `never`,
`without`, `nothing` or a `-n't` contraction appears in the {NEGATION_WINDOW} characters before it.
Listings without a description ({(1 - features["has_description"].mean()) * 100:.1f}%, all Autolist)
get 0 for every flag; `has_description` lets the model tell those apart.

## Patterns

{md_table(patterns, index=False)}

## Prevalence (% of listings flagged) and median retention ratio

{md_table(prevalence.rename_axis("feature"))}

## Validation against Autolist's structured history fields

Autolist rows have `owner_count` and `accident_count` from the vehicle-history report, so they
can be used as labels. Precision is the share of flagged listings where the field agrees;
recall is the share of qualifying listings that the text flags. Low recall is expected: most
dealer descriptions don't restate the history report.

{md_table(pd.DataFrame(checks), index=False)}

## Spot checks

{SPOT_CHECK_N} random matches and {SPOT_CHECK_N} random listings that contain the keyword but were not
flagged, for each extractor (seed {RANDOM_STATE}).

{spot_check(features, text)}
"""
    (REPORTS_DIR / "text_features.md").write_text(report)
    log.info("Wrote text feature report to %s", REPORTS_DIR / "text_features.md")


# --------------------------------------------------------------------------- pipeline

OUTPUT_COLUMNS = ["listing_id", TARGET] + FEATURE_COLUMNS


def run():
    df = pd.read_csv(PROCESSED_DIR / "integrated.csv", dtype={"zip": str})
    features = build_features(df)
    check_leakage(FEATURE_COLUMNS)

    features[OUTPUT_COLUMNS].to_csv(PROCESSED_DIR / "features.csv", index=False)
    log.info("Wrote %d rows x %d features to %s", len(features), len(FEATURE_COLUMNS),
             PROCESSED_DIR / "features.csv")

    text_feature_report(features)

    train, _ = split(features)
    results = compare_encodings(train)
    results.to_csv(REPORTS_DIR / "encoding_comparison.csv", index=False)
    log.info("Wrote encoding comparison to %s", REPORTS_DIR / "encoding_comparison.csv")
    return features


if __name__ == "__main__":
    setup_logging()
    run()
