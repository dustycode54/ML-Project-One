"""Train, tune and compare regression models for value_retention_ratio (Ticket 04).

Stages:
  1. Split: the shared 80/20 train/test split from src.features (seed 42, stratified by make).
  2. Baselines: the training-set mean, and the mean ratio per make/model.
  3. Tuning: Ridge, Random Forest and HistGradientBoosting, each in a Pipeline with the Ticket 03
     preprocessor. Hyperparameters and the high-cardinality encoding are searched with 5-fold CV
     on the training split only, so encoders, imputers and scalers never see validation rows.
  4. Test evaluation: each tuned model is refit on the full training split and scored once on the
     held-out test set (MAE, RMSE, R²).
  5. Cross-source check: train on one source's training rows, test on the other source's test rows.
  6. Residual analysis: test-set error and bias by make, model, source, seller type, age and mileage.

Outputs reports/model_results.csv, reports/model_results.md, reports/residuals_by_group.csv,
reports/cross_source.csv, reports/test_predictions.csv and models/<model>.joblib.

Run: python -m src.train
"""

import logging
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from .features import (FEATURE_COLUMNS, RANDOM_STATE, TARGET, FrequencyEncoder, build_preprocessor,
                       check_leakage, split)
from .preprocess import PROCESSED_DIR, REPORTS_DIR, md_table
from .scrapers.common import ROOT, setup_logging

log = logging.getLogger("train")

MODELS_DIR = ROOT / "models"
CV_FOLDS = 5
SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
REFIT_METRIC = "mae"
ENCODINGS = ("onehot", "frequency", "target")
ENCODER_NAMES = {OneHotEncoder: "onehot", FrequencyEncoder: "frequency", TargetEncoder: "target"}


# --------------------------------------------------------------------------- 2. baselines

class GroupMeanRegressor(RegressorMixin, BaseEstimator):
    """Predict the training mean of the target for each group (e.g. make + model).

    Groups unseen in training fall back to the overall training mean.
    """

    def __init__(self, group_cols=("make", "model")):
        self.group_cols = group_cols

    def fit(self, X, y):
        keys = X[list(self.group_cols)].astype(str).agg(" ".join, axis=1)
        self.global_mean_ = float(np.mean(y))
        self.group_means_ = pd.Series(np.asarray(y), index=keys.values).groupby(level=0).mean()
        return self

    def predict(self, X):
        keys = X[list(self.group_cols)].astype(str).agg(" ".join, axis=1)
        return keys.map(self.group_means_).fillna(self.global_mean_).to_numpy(float)


BASELINES = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_make_model_mean": GroupMeanRegressor(("make", "model")),
}


# --------------------------------------------------------------------------- 3. tuned models

def pipeline(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def search_space():
    """Model name -> (search class, pipeline, parameter grid / distributions, extra kwargs).

    The preprocessor itself is a hyperparameter: Ticket 03 found the high-cardinality encodings
    within one CV standard deviation of each other, with Ridge preferring one-hot.
    """
    preps = [build_preprocessor(enc) for enc in ENCODINGS]
    return {
        # Linear: interpretable coefficients for Ticket 05. alpha controls L2 shrinkage, which
        # matters with ~300 one-hot columns and correlated age / mileage / msrp.
        "ridge": (GridSearchCV, pipeline(Ridge()), {
            "prep": preps,
            "model__alpha": np.logspace(-2, 3, 11),
        }, {}),
        # Bagged deep trees: captures interactions (age x model) without much tuning.
        # max_features < 1 decorrelates the trees; min_samples_leaf smooths noisy leaves.
        "random_forest": (RandomizedSearchCV, pipeline(RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE)), {
            "prep": preps,
            "model__max_features": [0.2, 0.33, 0.5, 1.0],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_depth": [None, 12, 20],
        }, {"n_iter": 20}),
        # Boosted shallow trees: usually the strongest on tabular data. Learning rate and number of
        # iterations trade off; leaf size and L2 regularization control overfitting.
        "hist_gb": (RandomizedSearchCV, pipeline(HistGradientBoostingRegressor(
            random_state=RANDOM_STATE)), {
            "prep": preps,
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [200, 400, 800],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }, {"n_iter": 30}),
    }


def encoding_name(prep):
    encoder = {name: est for name, est, _ in prep.transformers}["high_card"].steps[0][1]
    return ENCODER_NAMES[type(encoder)]


def describe_params(params):
    """Best-parameter dict with the preprocessor replaced by its encoding name."""
    out = {"encoding": encoding_name(params["prep"])} if "prep" in params else {}
    for k, v in params.items():
        if k != "prep":
            out[k.removeprefix("model__")] = round(float(v), 4) if isinstance(v, float) else v
    return out


def cv_summary(cv_results, index):
    """CV mean ± std of every metric for one row of a search's cv_results_."""
    row = {}
    for metric in SCORING:
        sign = 1 if metric == "r2" else -1
        row[f"cv_{metric}_mean"] = sign * cv_results[f"mean_test_{metric}"][index]
        row[f"cv_{metric}_std"] = cv_results[f"std_test_{metric}"][index]
    return row


def test_scores(y_true, y_pred, prefix="test"):
    return {
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_rmse": root_mean_squared_error(y_true, y_pred),
        f"{prefix}_r2": r2_score(y_true, y_pred),
    }


def evaluate_baselines(X_train, y_train, X_test, y_test, cv):
    rows, preds = [], {}
    for name, est in BASELINES.items():
        scores = cross_validate(est, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name, "params": "-"}
        for metric in SCORING:
            vals = scores[f"test_{metric}"] * (1 if metric == "r2" else -1)
            row[f"cv_{metric}_mean"], row[f"cv_{metric}_std"] = vals.mean(), vals.std()
        fitted = clone(est).fit(X_train, y_train)
        preds[name] = fitted.predict(X_test)
        row["train_mae"] = mean_absolute_error(y_train, fitted.predict(X_train))
        row.update(test_scores(y_test, preds[name]))
        rows.append(row)
        log.info("%-26s CV MAE %.4f  test MAE %.4f  R2 %.3f", name, row["cv_mae_mean"],
                 row["test_mae"], row["test_r2"])
    return rows, preds


def tune_models(X_train, y_train, X_test, y_test, cv):
    rows, preds, fitted = [], {}, {}
    for name, (search_cls, pipe, space, kwargs) in search_space().items():
        if search_cls is RandomizedSearchCV:
            kwargs = {"param_distributions": space, "random_state": RANDOM_STATE, **kwargs}
        else:
            kwargs = {"param_grid": space, **kwargs}
        search = search_cls(pipe, cv=cv, scoring=SCORING, refit=REFIT_METRIC, n_jobs=-1, **kwargs)
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        best = search.best_estimator_
        preds[name] = best.predict(X_test)
        fitted[name] = best
        row = {"model": name, "params": describe_params(search.best_params_),
               **cv_summary(search.cv_results_, search.best_index_),
               "train_mae": mean_absolute_error(y_train, best.predict(X_train)),
               **test_scores(y_test, preds[name]),
               "search_candidates": len(search.cv_results_["params"]),
               "search_seconds": round(elapsed, 1)}
        rows.append(row)
        log.info("%-26s CV MAE %.4f ± %.4f  test MAE %.4f  R2 %.3f  (%s, %.0fs)", name,
                 row["cv_mae_mean"], row["cv_mae_std"], row["test_mae"], row["test_r2"],
                 row["params"], elapsed)
    return rows, preds, fitted


# --------------------------------------------------------------------------- 5. cross-source check

def cross_source(fitted, train, test):
    """Refit each tuned pipeline on one source's training rows; score on each source's test rows.

    Comparing the in-source and cross-source scores shows how much a model relies on
    source-specific patterns (dealer vs. private-seller pricing, description style).
    """
    rows = []
    sources = sorted(train["source"].unique())
    for name, est in fitted.items():
        for train_src in sources:
            tr = train[train["source"] == train_src]
            model = clone(est).fit(tr[FEATURE_COLUMNS], tr[TARGET])
            for test_src in sources:
                te = test[test["source"] == test_src]
                rows.append({"model": name, "train_source": train_src, "test_source": test_src,
                             "n_train": len(tr), "n_test": len(te),
                             **test_scores(te[TARGET], model.predict(te[FEATURE_COLUMNS]), "")})
    out = pd.DataFrame(rows)
    out.columns = [c.lstrip("_") for c in out.columns]
    return out


# --------------------------------------------------------------------------- 6. residual analysis

AGE_BINS = [-1, 2, 5, 8, 11, 20]
AGE_LABELS = ["0-2", "3-5", "6-8", "9-11", "12+"]
MILEAGE_BINS = [-1, 30_000, 60_000, 100_000, 150_000, np.inf]
MILEAGE_LABELS = ["<30k", "30-60k", "60-100k", "100-150k", "150k+"]
RESIDUAL_GROUPS = ["make", "model", "source", "seller_type", "age_bin", "mileage_bin"]


def residuals_by_group(test, preds):
    """MAE and mean residual (actual - predicted; > 0 means under-prediction) per group and model."""
    df = test.assign(
        age_bin=pd.cut(test["vehicle_age"], AGE_BINS, labels=AGE_LABELS),
        mileage_bin=pd.cut(test["mileage"], MILEAGE_BINS, labels=MILEAGE_LABELS),
    )
    frames = []
    for name, pred in preds.items():
        resid = df[TARGET] - pred
        for group in RESIDUAL_GROUPS:
            g = resid.groupby(df[group], observed=True)
            frames.append(pd.DataFrame({
                "model": name, "group": group, "level": g.size().index.astype(str),
                "n": g.size().values, "mae": g.apply(lambda r: r.abs().mean()).values,
                "mean_residual": g.mean().values,
            }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- report

def environment():
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}


def fmt_floats(df, digits=4):
    """Format floats as fixed-precision strings; md_table would round them to 2 decimals."""
    return df.map(lambda v: f"{v:.{digits}f}" if isinstance(v, float) else v)


def write_report(results, residuals, xsource, best, n_train, n_test):
    fmt = results.copy()
    fmt["params"] = fmt["params"].astype(str).str.replace("|", "/")
    for metric in SCORING:
        digits = 3 if metric == "r2" else 4
        fmt[f"CV {metric.upper()}"] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in
                                       zip(fmt[f"cv_{metric}_mean"], fmt[f"cv_{metric}_std"])]
    table = fmt[["model", "CV MAE", "CV RMSE", "CV R2", "train_mae", "test_mae", "test_rmse",
                 "test_r2"]]

    resid = residuals[residuals["model"] == best]
    resid_tables = "\n\n".join(
        f"### By {group}\n\n" + md_table(fmt_floats(resid[resid["group"] == group]
                                                     .drop(columns=["model", "group"])), index=False)
        for group in RESIDUAL_GROUPS
    )
    params = "\n".join(f"- **{r.model}**: `{r.params}`" for r in results.itertuples()
                       if r.params != "-")
    env = ", ".join(f"{k} {v}" for k, v in environment().items())

    report = f"""# Model Results

Generated by `python -m src.train`. Target: `{TARGET}`. Train / test: {n_train} / {n_test} rows
(80/20, stratified by make, seed {RANDOM_STATE}). CV: {CV_FOLDS}-fold shuffled KFold on the training
split, seed {RANDOM_STATE}; models are selected on CV {REFIT_METRIC.upper()}. Environment: {env}.

## Results (CV mean ± std on the training split; train and test scores of the refit model)

{md_table(fmt_floats(table), index=False)}

## Selected hyperparameters

{params}

## Cross-source check

Each tuned pipeline refit on one source's training rows and scored on each source's test rows.

{md_table(fmt_floats(xsource), index=False)}

## Residuals of the best model (`{best}`) on the test set

`mean_residual` is actual − predicted: positive means the model under-predicts retention.

{resid_tables}
"""
    (REPORTS_DIR / "model_results.md").write_text(report)


# --------------------------------------------------------------------------- pipeline

def run():
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    check_leakage(FEATURE_COLUMNS)
    train, test = split(features)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    log.info("Train %d rows, test %d rows, %d features", len(train), len(test), len(FEATURE_COLUMNS))

    base_rows, base_preds = evaluate_baselines(X_train, y_train, X_test, y_test, cv)
    model_rows, model_preds, fitted = tune_models(X_train, y_train, X_test, y_test, cv)
    results = pd.DataFrame(base_rows + model_rows)
    best = results.loc[results["cv_mae_mean"].idxmin(), "model"]
    results["selected"] = results["model"] == best
    results.round(4).to_csv(REPORTS_DIR / "model_results.csv", index=False)
    log.info("Best model by CV MAE: %s", best)

    preds = base_preds | model_preds
    pd.DataFrame({"listing_id": test["listing_id"], TARGET: y_test,
                  **{f"pred_{k}": np.round(v, 4) for k, v in preds.items()}}
                 ).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    residuals = residuals_by_group(test, model_preds)
    residuals.round(4).to_csv(REPORTS_DIR / "residuals_by_group.csv", index=False)

    xsource = cross_source(fitted, train, test)
    xsource.round(4).to_csv(REPORTS_DIR / "cross_source.csv", index=False)

    MODELS_DIR.mkdir(exist_ok=True)
    for name, est in fitted.items():
        joblib.dump(est, MODELS_DIR / f"{name}.joblib", compress=3)
    log.info("Saved %d fitted pipelines to %s", len(fitted), MODELS_DIR)

    write_report(results, residuals, xsource, best, len(train), len(test))
    log.info("Wrote results to %s", REPORTS_DIR / "model_results.md")
    return results


if __name__ == "__main__":
    setup_logging()
    run()
