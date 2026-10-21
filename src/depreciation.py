"""Quantify what drives depreciation and rank the features (Ticket 05).

Stages:
  1. Ridge coefficients: the standardized coefficient of every encoded column, converted back to
     units (percentage points of MSRP per year, per 10k miles, per text flag).
  2. Age and mileage effects: for every tuned model, the average change in predicted retention on
     the test set when a car is one year older, has 10k more miles, or both. miles_per_year is
     recomputed from the changed age / mileage, so the counterfactual car stays consistent. The
     same recomputation gives partial-dependence curves over age and mileage.
  3. Feature ranking, aggregated to the input columns (one-hot columns summed back to their feature):
     - Ridge: standard deviation of each feature's contribution (coefficient x value) on the training rows.
     - Random Forest: impurity-based importance.
     - Permutation importance on the test set (increase in MAE) for every tuned model, plus grouped
       permutation of correlated feature sets (age + mileage, make + model + trim, location, text).
     - Ablation: 5-fold CV of the best pipeline refit without a feature group.
  4. Depreciation curves: mean retention by age and by mileage, overall, per make and per model, and a
     per-model linear fit of retention on age and mileage that ranks models at a reference car.
  5. Method agreement (Spearman rank correlation, top-feature lists) and the text-feature assessment.

Reads data/processed/features.csv and the pipelines saved by src.train (models/*.joblib).
Outputs reports/feature_importance.csv, reports/grouped_importance.csv, reports/ridge_coefficients.csv,
reports/age_mileage_effects.csv, reports/partial_dependence.csv, reports/depreciation_curves.csv,
reports/model_retention.csv and reports/depreciation_analysis.md.

Run: python -m src.depreciation   (after python -m src.train)
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_validate

from .features import (FEATURE_COLUMNS, NUMERIC_FEATURES, RANDOM_STATE, TARGET, TEXT_FEATURES,
                       build_preprocessor, split)
from .preprocess import PROCESSED_DIR, REPORTS_DIR, md_table
from .scrapers.common import setup_logging
from .train import (AGE_BINS, AGE_LABELS, CV_FOLDS, MILEAGE_BINS, MILEAGE_LABELS, MODELS_DIR,
                    encoding_name, fmt_floats)

log = logging.getLogger("depreciation")

MODEL_NAMES = ("ridge", "random_forest", "hist_gb")
PERMUTATION_REPEATS = 10
MILEAGE_STEP = 10_000
AGE_GRID = np.arange(2, 15)  # the observed range; one listing has age 0
MILEAGE_GRID = np.arange(0, 200_001, 10_000)
REFERENCE_AGE, REFERENCE_MILEAGE = 5, 60_000
MIN_MODEL_ROWS = 30
MIN_CURVE_ROWS = 5  # report tables skip age / mileage cells with fewer listings
TOP_N = 10

# Correlated features permuted together: permuting one alone understates the set when the model
# can recover the information from its partners (age from mileage, model from trim_key).
FEATURE_GROUPS = {
    "age + mileage + miles_per_year": ["vehicle_age", "mileage", "miles_per_year"],
    "make + model + trim_key + msrp": ["make", "model", "trim_key", "msrp"],
    "body / drivetrain / engine": ["body_type", "drivetrain", "transmission", "fuel_type", "cylinders"],
    "location (lat, lon, state, city, region)": ["lat", "lon", "state", "city_key", "region"],
    "source + seller_type": ["source", "seller_type"],
    "text flags + description": TEXT_FEATURES + ["has_description", "log_description_len"],
    "text flags only": TEXT_FEATURES,
}

# Numeric features dropped for the CV ablation of the best model.
ABLATIONS = {
    "none (full model)": [],
    "text flags": TEXT_FEATURES,
    "vehicle_age": ["vehicle_age"],
    "mileage": ["mileage"],
    "miles_per_year": ["miles_per_year"],
    "vehicle_age + mileage + miles_per_year": ["vehicle_age", "mileage", "miles_per_year"],
}


def load_models():
    missing = [n for n in MODEL_NAMES if not (MODELS_DIR / f"{n}.joblib").exists()]
    if missing:
        raise FileNotFoundError(f"missing fitted pipelines {missing}; run python -m src.train first")
    return {n: joblib.load(MODELS_DIR / f"{n}.joblib") for n in MODEL_NAMES}


def best_model_name():
    results = pd.read_csv(REPORTS_DIR / "model_results.csv")
    return results.loc[results["selected"], "model"].iloc[0]


def source_feature(column):
    """Map an encoded column ('make_Toyota', 'trim_key_freq') back to its input feature.

    The longest matching name wins, so 'msrp_match_level_trim' maps to msrp_match_level, not msrp.
    """
    matches = [f for f in FEATURE_COLUMNS if column == f or column.startswith(f + "_")]
    return max(matches, key=len)


# --------------------------------------------------------------------------- 1. ridge coefficients

def column_scales(prep):
    """Standard deviation each encoded column was divided by (1 for the unscaled one-hot block)."""
    scales = np.ones(len(prep.get_feature_names_out()))
    for name in ("num", "high_card"):
        scales[prep.output_indices_[name]] = prep.named_transformers_[name][-1].scale_
    return scales


def ridge_coefficients(ridge):
    prep, model = ridge["prep"], ridge["model"]
    columns = prep.get_feature_names_out()
    scales = column_scales(prep)
    out = pd.DataFrame({
        "column": columns,
        "feature": [source_feature(c) for c in columns],
        "coef": model.coef_,
        "scale": scales,
        # Change in predicted ratio per unit of the raw column (per indicator for one-hot columns).
        "effect_per_unit": model.coef_ / scales,
    })
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")


def ridge_contribution_std(ridge, X):
    """Std over rows of each feature's summed contribution (coef x encoded value).

    Comparable across numeric and one-hot features, unlike raw coefficients.
    """
    prep, model = ridge["prep"], ridge["model"]
    contrib = pd.DataFrame(prep.transform(X) * model.coef_, columns=prep.get_feature_names_out())
    by_feature = contrib.T.groupby([source_feature(c) for c in contrib.columns]).sum().T
    return by_feature.std()


# --------------------------------------------------------------------------- 2. age and mileage effects

def counterfactual(X, age=None, mileage=None):
    """Copy of X with age and/or mileage changed and miles_per_year recomputed as in src.features."""
    X = X.copy()
    if age is not None:
        X["vehicle_age"] = age
    if mileage is not None:
        X["mileage"] = mileage
    X["miles_per_year"] = (X["mileage"] / X["vehicle_age"].clip(lower=1)).round(0)
    return X


def age_mileage_effects(models, X, typical_miles):
    """Average change in predicted retention (test rows) for an older / more-driven car."""
    scenarios = {
        "+1 year, same mileage": counterfactual(X, age=X["vehicle_age"] + 1),
        f"+{MILEAGE_STEP // 1000}k miles, same age": counterfactual(X, mileage=X["mileage"] + MILEAGE_STEP),
        f"+1 year and +{typical_miles / 1000:.1f}k miles (median annual)":
            counterfactual(X, age=X["vehicle_age"] + 1, mileage=X["mileage"] + typical_miles),
    }
    rows = []
    for name, model in models.items():
        base = model.predict(X)
        for scenario, Xc in scenarios.items():
            delta = model.predict(Xc) - base
            rows.append({"model": name, "scenario": scenario, "mean_change": delta.mean(),
                         "median_change": np.median(delta), "share_negative": (delta < 0).mean()})
    return pd.DataFrame(rows)


def partial_dependence(models, X):
    """Mean prediction over test rows with age (or mileage) set to each grid value."""
    rows = []
    for name, model in models.items():
        for feature, grid in (("vehicle_age", AGE_GRID), ("mileage", MILEAGE_GRID)):
            for value in grid:
                Xc = counterfactual(X, **{"age" if feature == "vehicle_age" else "mileage": value})
                rows.append({"model": name, "feature": feature, "value": int(value),
                             "mean_prediction": model.predict(Xc).mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3. feature ranking

def permutation_ranking(models, X, y):
    """Increase in test MAE when a single input column is shuffled (mean and std over repeats)."""
    out = {}
    for name, model in models.items():
        result = permutation_importance(model, X, y, scoring="neg_mean_absolute_error",
                                        n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_STATE,
                                        n_jobs=-1)
        out[f"perm_{name}"] = pd.Series(result.importances_mean, index=X.columns)
        out[f"perm_{name}_std"] = pd.Series(result.importances_std, index=X.columns)
        log.info("Permutation importance for %s done", name)
    return pd.DataFrame(out)


def grouped_permutation(models, X, y):
    """Increase in test MAE when a set of columns is shuffled together (same row order for all)."""
    rng = np.random.default_rng(RANDOM_STATE)
    perms = [rng.permutation(len(X)) for _ in range(PERMUTATION_REPEATS)]
    rows = []
    for name, model in models.items():
        base = mean_absolute_error(y, model.predict(X))
        for group, cols in FEATURE_GROUPS.items():
            increases = []
            for perm in perms:
                Xp = X.copy()
                Xp[cols] = X[cols].iloc[perm].to_numpy()
                increases.append(mean_absolute_error(y, model.predict(Xp)) - base)
            rows.append({"model": name, "group": group, "n_features": len(cols),
                         "mae_increase": np.mean(increases), "mae_increase_std": np.std(increases)})
    return pd.DataFrame(rows)


def ablation(best, X, y):
    """5-fold CV MAE of the best pipeline refit without each numeric feature group."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    encoding = encoding_name(best["prep"])
    rows = []
    for label, dropped in ABLATIONS.items():
        numeric = [c for c in NUMERIC_FEATURES if c not in dropped]
        pipe = clone(best).set_params(prep=build_preprocessor(encoding, numeric))
        scores = -cross_validate(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error",
                                 n_jobs=-1)["test_score"]
        rows.append({"dropped": label, "cv_mae_mean": scores.mean(), "cv_mae_std": scores.std()})
        log.info("Ablation %-40s CV MAE %.4f", label, scores.mean())
    out = pd.DataFrame(rows)
    out["mae_change"] = out["cv_mae_mean"] - out["cv_mae_mean"].iloc[0]
    return out


def feature_ranking(models, train, test):
    X_train, X_test, y_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS], test[TARGET]
    rf = models["random_forest"]
    impurity = pd.Series(rf["model"].feature_importances_, index=rf["prep"].get_feature_names_out())
    table = pd.DataFrame({
        "ridge_contribution_std": ridge_contribution_std(models["ridge"], X_train),
        "rf_impurity": impurity.groupby([source_feature(c) for c in impurity.index]).sum(),
    }).join(permutation_ranking(models, X_test, y_test)).reindex(FEATURE_COLUMNS)
    methods = ["ridge_contribution_std", "rf_impurity"] + [f"perm_{n}" for n in models]
    for m in methods:
        table[f"rank_{m}"] = table[m].rank(ascending=False, method="min").astype(int)
    table["mean_rank"] = table[[f"rank_{m}" for m in methods]].mean(axis=1)
    return table.sort_values("mean_rank").rename_axis("feature"), methods


# --------------------------------------------------------------------------- 4. depreciation curves

def depreciation_curves(df):
    """Mean / median retention by integer age and by mileage bin, overall, per make and per model."""
    df = df.assign(overall="all",
                   mileage_bin=pd.cut(df["mileage"], MILEAGE_BINS, labels=MILEAGE_LABELS))
    frames = []
    for level_col in ("overall", "make", "model"):
        for axis, x_col in (("age", "vehicle_age"), ("mileage", "mileage_bin")):
            g = df.groupby([level_col, x_col], observed=True)[TARGET]
            stats = g.agg(n="size", mean="mean", median="median").reset_index()
            frames.append(stats.rename(columns={level_col: "level", x_col: "x"})
                          .assign(grouping=level_col, axis=axis))
    out = pd.concat(frames, ignore_index=True)
    out["x"] = out["x"].astype(str)
    return out[["grouping", "level", "axis", "x", "n", "mean", "median"]]


def retention_fit(df, group_col):
    """Per group, least squares of retention on age and mileage (per 10k), evaluated at a reference car.

    Comparing groups at the same age and mileage removes the main confounder of raw group means
    (e.g. a model whose listings are mostly older looks worse than it is).
    """
    rows = []
    for level, g in df.groupby(group_col):
        if len(g) < MIN_MODEL_ROWS:
            continue
        A = np.column_stack([np.ones(len(g)), g["vehicle_age"], g["mileage"] / MILEAGE_STEP])
        (intercept, per_year, per_10k), *_ = np.linalg.lstsq(A, g[TARGET], rcond=None)
        resid = g[TARGET] - A @ np.array([intercept, per_year, per_10k])
        rows.append({
            group_col: level, "n": len(g), "mean_ratio": g[TARGET].mean(),
            "mean_age": g["vehicle_age"].mean(), "mean_mileage": g["mileage"].mean(),
            "intercept": intercept, "per_year": per_year, "per_10k_miles": per_10k,
            "fit_r2": 1 - resid.var() / g[TARGET].var(),
            "ratio_at_reference": intercept + per_year * REFERENCE_AGE
                                  + per_10k * REFERENCE_MILEAGE / MILEAGE_STEP,
        })
    out = pd.DataFrame(rows).sort_values("ratio_at_reference", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------- 5. agreement + text

def method_agreement(table, methods):
    corr = table[methods].corr(method="spearman").round(2)
    top = pd.DataFrame({m: table[m].sort_values(ascending=False).index[:TOP_N].tolist()
                        for m in methods}, index=pd.RangeIndex(1, TOP_N + 1, name="rank"))
    return corr, top


def text_assessment(df, coefs, table, best):
    ridge_effect = coefs.set_index("column")["effect_per_unit"]
    rows = []
    for f in TEXT_FEATURES:
        flagged = df[f] == 1
        rows.append({
            "feature": f, "flagged_pct": f"{flagged.mean() * 100:.1f}",
            "raw_median_diff": df.loc[flagged, TARGET].median() - df.loc[~flagged, TARGET].median(),
            "ridge_effect": ridge_effect[f],
            f"perm_{best}": table.loc[f, f"perm_{best}"],
            f"perm_{best}_std": table.loc[f, f"perm_{best}_std"],
            f"rank_perm_{best}": table.loc[f, f"rank_perm_{best}"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report

def pp(v):
    """A ratio change as signed percentage points of MSRP."""
    return f"{v * 100:+.2f} pp"


def write_report(*, best, coefs, effects, pdp, table, methods, groups, ablate, corr, top,
                 curves, model_fit, make_fit, text, typical_miles, age_mileage_corr, n_features):
    numeric_coefs = coefs[coefs["feature"].isin(NUMERIC_FEATURES)].copy()
    numeric_coefs["per unit"] = [
        f"{pp(e * {'mileage': MILEAGE_STEP, 'miles_per_year': 1000, 'msrp': 1000}.get(c, 1))} per "
        + {"vehicle_age": "year", "mileage": "10k miles", "miles_per_year": "1k mi/yr",
           "msrp": "$1k", "cylinders": "cylinder", "lat": "degree", "lon": "degree",
           "log_description_len": "log char"}.get(c, "flag")
        for c, e in zip(numeric_coefs["column"], numeric_coefs["effect_per_unit"])
    ]
    numeric_coefs["coef (per SD)"] = numeric_coefs["coef"].map(pp)
    cat_coefs = coefs[~coefs["feature"].isin(NUMERIC_FEATURES)].head(15).copy()
    cat_coefs["coef"] = cat_coefs["coef"].map(pp)

    eff = effects.pivot(index="scenario", columns="model", values="mean_change")[list(MODEL_NAMES)]
    pd_age = pdp[pdp["feature"] == "vehicle_age"].pivot(index="value", columns="model",
                                                        values="mean_prediction")[list(MODEL_NAMES)]
    pd_mil = pdp[(pdp["feature"] == "mileage") & (pdp["value"] % 40_000 == 0)].pivot(
        index="value", columns="model", values="mean_prediction")[list(MODEL_NAMES)]

    rank_cols = ["ridge_contribution_std", "rf_impurity"] + [f"perm_{n}" for n in MODEL_NAMES] \
        + [f"rank_{m}" for m in methods] + ["mean_rank"]
    grouped = groups.pivot(index="group", columns="model", values="mae_increase")[list(MODEL_NAMES)]
    grouped = grouped.reindex(list(FEATURE_GROUPS))

    shown = curves[curves["n"] >= MIN_CURVE_ROWS]
    overall = shown[shown["grouping"] == "overall"]
    by_age = overall[overall["axis"] == "age"][["x", "n", "mean", "median"]]
    by_mil = overall[overall["axis"] == "mileage"][["x", "n", "mean", "median"]]
    df_make = shown[(shown["grouping"] == "make") & (shown["axis"] == "age")]
    make_age = df_make.assign(age=df_make["x"].astype(int)).pivot(index="age", columns="level",
                                                                   values="mean")

    report = f"""# Depreciation Analysis and Feature Ranking

Generated by `python -m src.depreciation` from the pipelines fitted by `python -m src.train`.
Retention is `price / msrp`; changes are in **percentage points (pp) of MSRP**. Model-based numbers
use the held-out test set unless stated. Best model: `{best}`. {n_features} input features.

## 1. Age and mileage effects

Average change in predicted retention on the test rows when each car is made older or more driven.
`miles_per_year` is recomputed from the changed age and mileage. The median annual mileage in the
training split is {typical_miles:,.0f}. Age and mileage are strongly correlated across all listings
(Pearson r = {age_mileage_corr:.2f}), so neither effect can be read off a single-feature view.

{md_table(fmt_floats(eff.map(lambda v: v * 100), 2).rename_axis("scenario (pp)"))}

### Partial dependence on age (mean predicted retention, mileage held at each car's own value)

{md_table(fmt_floats(pd_age, 3).rename_axis("age"))}

### Partial dependence on mileage (age held at each car's own value)

{md_table(fmt_floats(pd_mil, 3).rename_axis("mileage"))}

## 2. Ridge coefficients (standardized inputs, alpha = {coefs.attrs['alpha']})

Numeric features: the coefficient per standard deviation, and converted back to raw units. Each is
the effect with **all other columns held fixed**, including the correlated ones (age with
mileage and miles_per_year).

{md_table(fmt_floats(numeric_coefs[['column', 'coef (per SD)', 'per unit', 'scale']]), index=False)}

Largest categorical coefficients (one-hot indicators; relative to the other levels of the same feature):

{md_table(cat_coefs[['column', 'coef']], index=False)}

## 3. Feature ranking

`ridge_contribution_std`: std of the feature's Ridge contribution on the training rows.
`rf_impurity`: Random Forest impurity importance (one-hot columns summed). `perm_*`: increase in
test MAE when the column is shuffled ({PERMUTATION_REPEATS} repeats). Sorted by mean rank.

{md_table(fmt_floats(table[rank_cols]))}

### Grouped permutation (increase in test MAE when the whole group is shuffled together)

{md_table(fmt_floats(grouped))}

### Ablation: {CV_FOLDS}-fold CV MAE of `{best}` refit without a feature group (training split)

{md_table(fmt_floats(ablate), index=False)}

### Method agreement

Spearman rank correlation between methods over all {n_features} features:

{md_table(corr.rename_axis('method'))}

Top {TOP_N} features per method:

{md_table(top)}

## 4. Depreciation curves (all listings; cells with fewer than {MIN_CURVE_ROWS} listings omitted)

### By age

{md_table(fmt_floats(by_age.rename(columns={'x': 'age'}), 3), index=False)}

### By mileage

{md_table(fmt_floats(by_mil.rename(columns={'x': 'mileage'}), 3), index=False)}

### Mean retention by age and make

{md_table(fmt_floats(make_age, 3))}

### Value retention by model

Per model, least-squares fit `ratio ~ intercept + per_year * age + per_10k_miles * mileage / 10k`
on all listings, then the fitted ratio for a reference car ({REFERENCE_AGE} years, {REFERENCE_MILEAGE:,}
miles). Ranked by that value. Models with fewer than {MIN_MODEL_ROWS} listings are skipped.

{md_table(fmt_floats(model_fit, 3), index=False)}

### By make (same fit)

{md_table(fmt_floats(make_fit, 3), index=False)}

## 5. Text features

`raw_median_diff`: median retention of flagged minus unflagged listings (no controls).
`ridge_effect`: Ridge coefficient for the flag in ratio units (other features held fixed).
`perm_{best}`: increase in test MAE when the flag is shuffled.

{md_table(fmt_floats(text), index=False)}
"""
    (REPORTS_DIR / "depreciation_analysis.md").write_text(report)


# --------------------------------------------------------------------------- pipeline

def run():
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    train, test = split(features)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test = test[FEATURE_COLUMNS]
    models = load_models()
    best = best_model_name()
    typical_miles = float(train["miles_per_year"].median())
    log.info("Loaded %s; best model %s", ", ".join(models), best)

    coefs = ridge_coefficients(models["ridge"])
    coefs.attrs["alpha"] = models["ridge"]["model"].alpha
    coefs.round(6).to_csv(REPORTS_DIR / "ridge_coefficients.csv", index=False)

    effects = age_mileage_effects(models, X_test, typical_miles)
    effects.round(5).to_csv(REPORTS_DIR / "age_mileage_effects.csv", index=False)
    pdp = partial_dependence(models, X_test)
    pdp.round(4).to_csv(REPORTS_DIR / "partial_dependence.csv", index=False)

    table, methods = feature_ranking(models, train, test)
    table.round(5).to_csv(REPORTS_DIR / "feature_importance.csv")
    groups = grouped_permutation(models, X_test, test[TARGET])
    groups.round(5).to_csv(REPORTS_DIR / "grouped_importance.csv", index=False)
    ablate = ablation(models[best], X_train, y_train)
    corr, top = method_agreement(table, methods)

    curves = depreciation_curves(features)
    curves.round(4).to_csv(REPORTS_DIR / "depreciation_curves.csv", index=False)
    model_fit, make_fit = retention_fit(features, "model"), retention_fit(features, "make")
    pd.concat([model_fit.assign(grouping="model").rename(columns={"model": "level"}),
               make_fit.assign(grouping="make").rename(columns={"make": "level"})]
              ).round(4).to_csv(REPORTS_DIR / "model_retention.csv", index=False)

    text = text_assessment(features, coefs, table, best)
    write_report(best=best, coefs=coefs, effects=effects, pdp=pdp, table=table, methods=methods,
                 groups=groups, ablate=ablate, corr=corr, top=top, curves=curves,
                 model_fit=model_fit, make_fit=make_fit, text=text, typical_miles=typical_miles,
                 age_mileage_corr=features["vehicle_age"].corr(features["mileage"]),
                 n_features=len(FEATURE_COLUMNS))
    log.info("Wrote %s", REPORTS_DIR / "depreciation_analysis.md")
    return table


if __name__ == "__main__":
    setup_logging()
    run()
