"""Report figures for data characteristics, feature ranking and model performance (Ticket 06).

Figures:
  Data characteristics
    01 price and value_retention_ratio distributions
    02 mileage and vehicle-age distributions, by source
    03 listing counts per model, split by source
    04 retention vs. age and vs. mileage (listings + binned mean)
    05 mean retention by age and by mileage, per make
    06 per-model retention of a reference car (5 years, 60k miles)
    07 correlation heatmap of the numeric attributes
  Feature ranking
    08 permutation importance of the best model (top features)
    09 rank of the top features under each of the five ranking methods
    10 grouped permutation importance, per model
    11 largest Ridge indicator coefficients
    12 partial dependence on age and mileage, per model
  Model performance
    13 CV and test MAE / RMSE / R2 per model, with baselines
    14 predicted vs. actual retention for the best model
    15 residuals of the best model: vs. prediction, and MAE per vehicle model

Reads only the tables written by earlier steps (data/processed/*.csv and reports/*.csv), so it
refits nothing and runs in a few seconds. Writes reports/figures/figNN_<name>.png and .svg.

Run: python -m src.plots   (after python -m src.depreciation)
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .features import TARGET, split
from .preprocess import PROCESSED_DIR, REPORTS_DIR
from .scrapers.common import setup_logging

log = logging.getLogger("plots")

FIGURES_DIR = REPORTS_DIR / "figures"
DPI = 200
TOP_N = 15

# Colorblind-safe categorical slots, assigned in fixed order (validated adjacent-pair CVD
# separation >= 8 on white; slots 1-3 also all-pairs, so scatters use at most three colors).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED = "#e34948"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
SURFACE = "#ffffff"

MAKE_COLORS = {"Toyota": BLUE, "Honda": ORANGE, "Ford": AQUA, "Chevrolet": YELLOW}
SOURCE_COLORS = {"autolist": BLUE, "craigslist": ORANGE}
SOURCE_LABELS = {"autolist": "Autolist", "craigslist": "Craigslist"}
MODEL_COLORS = {"ridge": BLUE, "random_forest": ORANGE, "hist_gb": AQUA}
MODEL_LABELS = {"baseline_mean": "Mean baseline", "baseline_make_model_mean": "Make/model mean",
                "ridge": "Ridge", "random_forest": "Random Forest", "hist_gb": "HistGB"}
METHOD_LABELS = {"ridge_contribution_std": "Ridge\ncontribution", "rf_impurity": "RF\nimpurity",
                 "perm_ridge": "Permutation\nRidge", "perm_random_forest": "Permutation\nRF",
                 "perm_hist_gb": "Permutation\nHistGB"}

SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("red_blue", [RED, NEUTRAL, BLUE])

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "path",
}

FEATURE_LABELS = {
    "vehicle_age": "Vehicle age", "mileage": "Mileage", "miles_per_year": "Miles per year",
    "msrp": "MSRP", "make": "Make", "model": "Model", "trim_key": "Trim", "body_type": "Body type",
    "log_description_len": "Description length (log)", "has_description": "Has description",
    "drivetrain": "Drivetrain", "state": "State", "condition": "Condition (Craigslist)",
    "source": "Source", "cylinders": "Cylinders", "seller_type": "Seller type", "lon": "Longitude",
    "lat": "Latitude", "city_key": "City", "region": "Census region", "transmission": "Transmission",
    "exterior_color": "Color", "fuel_type": "Fuel",
    "txt_clean_title": "Text: clean title", "txt_one_owner": "Text: one owner",
    "txt_needs_work": "Text: needs work", "value_retention_ratio": "Retention ratio", "price": "Price",
}

thousands = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")
dollars_k = FuncFormatter(lambda v, _: f"${v / 1000:.0f}k")


def label(feature):
    return FEATURE_LABELS.get(feature, feature)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", name)


def bar_axis(ax, category_axis="x"):
    """Keep only the value-axis gridlines on a bar chart."""
    ax.grid(axis=category_axis, visible=False)


def load():
    integrated = pd.read_csv(PROCESSED_DIR / "integrated.csv")
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    tables = {name: pd.read_csv(REPORTS_DIR / f"{name}.csv") for name in (
        "model_results", "feature_importance", "grouped_importance", "ridge_coefficients",
        "partial_dependence", "depreciation_curves", "model_retention", "test_predictions",
        "residuals_by_group")}
    return integrated, features, tables


# ---------------------------------------------------------------- data characteristics

def fig_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), layout="constrained")
    for ax, col, bins, fmt, title in [
        (axes[0], "price", np.arange(0, 65_001, 2_500), dollars_k, "Listing price"),
        (axes[1], TARGET, np.arange(0.1, 1.35, 0.05), None, "Value retention ratio (price / MSRP)"),
    ]:
        ax.hist(df[col], bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
        median = df[col].median()
        ax.axvline(median, color=INK, linewidth=1)
        text = f"median ${median:,.0f}" if col == "price" else f"median {median:.2f}"
        ax.annotate(text, (median, 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
                    textcoords="offset points", va="top", fontsize=8, color=INK)
        if fmt:
            ax.xaxis.set_major_formatter(fmt)
        ax.set(title=title, ylabel="Listings")
        bar_axis(ax)
    axes[0].set_xlabel("Price (USD)")
    axes[1].set_xlabel("Retention ratio")
    save(fig, "fig01_price_retention_distribution")


def fig_mileage_age(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), layout="constrained")
    mileage_bins = np.arange(0, 300_001, 10_000)
    age = (pd.to_datetime(df["posted_at"]).dt.year - df["year"]).clip(lower=0)
    width = 0.4
    for i, (source, group) in enumerate(df.groupby("source")):
        color, name = SOURCE_COLORS[source], SOURCE_LABELS[source]
        counts, edges = np.histogram(group["mileage"].clip(upper=mileage_bins[-1] - 1), mileage_bins)
        axes[0].stairs(counts, edges, color=color, linewidth=2, label=f"{name} (median {group['mileage'].median() / 1000:.0f}k)")
        axes[0].stairs(counts, edges, color=color, alpha=0.1, fill=True)
        by_age = age[group.index].value_counts().sort_index()
        axes[1].bar(by_age.index + (i - 0.5) * width, by_age.values, width=width, color=color,
                    edgecolor=SURFACE, linewidth=0.8, label=name)
    axes[0].xaxis.set_major_formatter(thousands)
    axes[0].set(title="Mileage", xlabel="Odometer (miles; last bin is 290k+)", ylabel="Listings")
    axes[1].set(title="Vehicle age", xlabel="Age at listing (years)", ylabel="Listings")
    axes[1].set_xticks(range(0, 15, 2))
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper left")
    for ax in axes:
        bar_axis(ax)
    save(fig, "fig02_mileage_age_distribution")


def fig_listings_by_model(df):
    counts = df.groupby(["make", "model", "source"]).size().unstack("source", fill_value=0)
    counts = counts.loc[list(MAKE_COLORS)[::-1]]  # Toyota on top
    names = [f"{make} {model}" for make, model in counts.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    left = np.zeros(len(counts))
    for source in SOURCE_COLORS:
        ax.barh(names, counts[source], left=left, height=0.62, color=SOURCE_COLORS[source],
                edgecolor=SURFACE, linewidth=1.5, label=SOURCE_LABELS[source])
        left += counts[source].to_numpy()
    for y, total in enumerate(left):
        ax.annotate(f"{total:.0f}", (total, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_2)
    ax.set(title=f"Listings per model after preprocessing (n = {len(df):,})", xlabel="Listings")
    ax.set_xlim(0, left.max() * 1.1)
    ax.legend(loc="lower right", ncols=2, bbox_to_anchor=(1, 1.0), frameon=False)
    bar_axis(ax, "y")
    save(fig, "fig03_listings_by_model")


def binned_mean(x, y, bins):
    groups = pd.cut(x, bins)
    stats = y.groupby(groups, observed=True).agg(["mean", "size"])
    centers = [interval.mid for interval in stats.index]
    return np.array(centers), stats["mean"].to_numpy(), stats["size"].to_numpy()


def fig_retention_scatter(features):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), layout="constrained", sharey=True)
    y = features[TARGET]
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.3, 0.3, len(features))
    for ax, x, bins, title, xlabel in [
        (axes[0], features["vehicle_age"], np.arange(-0.5, 15), "Retention vs. age", "Vehicle age (years)"),
        (axes[1], features["mileage"], np.arange(0, 260_001, 20_000), "Retention vs. mileage", "Mileage"),
    ]:
        shown_x = x + jitter if x.name == "vehicle_age" else x
        ax.scatter(shown_x, y, s=6, color=MUTED, alpha=0.25, linewidths=0, label="Listing")
        cx, cy, n = binned_mean(x, y, bins)
        keep = n >= 5
        ax.plot(cx[keep], cy[keep], color=BLUE, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label="Binned mean")
        ax.set(title=title, xlabel=xlabel)
    axes[1].xaxis.set_major_formatter(thousands)
    axes[1].set_xlim(0, 260_000)
    axes[0].set_ylabel("Retention ratio")
    axes[0].legend(loc="upper right", markerscale=1.2)
    save(fig, "fig04_retention_vs_age_mileage")


def fig_retention_by_make(curves):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), layout="constrained", sharey=True)
    make = curves[(curves["grouping"] == "make") & (curves["n"] >= 5)]
    mileage_order = ["<30k", "30-60k", "60-100k", "100-150k", "150k+"]
    for name, color in MAKE_COLORS.items():
        by_age = make[(make["level"] == name) & (make["axis"] == "age")]
        by_age = by_age.assign(x=by_age["x"].astype(float)).sort_values("x")
        axes[0].plot(by_age["x"], by_age["mean"], color=color, marker="o", markersize=4,
                     markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        by_miles = make[(make["level"] == name) & (make["axis"] == "mileage")].set_index("x")
        axes[1].plot(range(len(mileage_order)), by_miles.reindex(mileage_order)["mean"], color=color,
                     marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1, label=name)
    axes[0].set(title="Mean retention by age", xlabel="Vehicle age (years)", ylabel="Mean retention ratio")
    axes[0].set_xticks(range(2, 15, 2))
    axes[1].set(title="Mean retention by mileage", xlabel="Mileage bin")
    axes[1].set_xticks(range(len(mileage_order)), mileage_order)
    axes[1].legend(loc="lower left")
    save(fig, "fig05_retention_by_make")


def fig_model_ranking(model_fit, features):
    makes = features.drop_duplicates("model").set_index("model")["make"]
    fit = model_fit[model_fit["grouping"] == "model"].sort_values("ratio_at_reference")
    names = [f"{makes[m]} {m}" for m in fit["level"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
    colors = [MAKE_COLORS[makes[m]] for m in fit["level"]]
    ax.barh(names, fit["ratio_at_reference"], height=0.62, color=colors)
    for y, value in enumerate(fit["ratio_at_reference"]):
        ax.annotate(f"{value:.2f}", (value, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_2)
    ax.set(title="Fitted retention of a 5-year-old, 60k-mile car, by model",
           xlabel="Retention ratio (per-model fit: ratio ~ age + mileage)", xlim=(0, 1))
    ax.legend(handles=[Patch(color=c, label=m) for m, c in MAKE_COLORS.items()], loc="lower right")
    bar_axis(ax, "y")
    save(fig, "fig06_model_retention_ranking")


def fig_correlation(integrated, features):
    cols = ["value_retention_ratio", "price", "msrp", "vehicle_age", "mileage", "miles_per_year",
            "cylinders", "log_description_len"]
    data = features.assign(price=integrated.set_index("listing_id").loc[features["listing_id"], "price"].to_numpy())
    corr = data[cols].corr()
    fig, ax = plt.subplots(figsize=(5.6, 4.6), layout="constrained")
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    names = [label(c) for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), names)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if abs(v) > 0.6 else INK)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation of numeric attributes")
    save(fig, "fig07_correlation_heatmap")


# ---------------------------------------------------------------- feature ranking

def fig_permutation_importance(importance, best):
    top = importance.nlargest(TOP_N, f"perm_{best}").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
    ax.barh([label(f) for f in top["feature"]], top[f"perm_{best}"], xerr=top[f"perm_{best}_std"],
            height=0.62, color=BLUE, error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 2})
    ax.set(title=f"Permutation importance, {MODEL_LABELS[best]} (test set, top {TOP_N})",
           xlabel="Increase in test MAE when the feature is shuffled (± SD over 10 repeats)")
    bar_axis(ax, "y")
    save(fig, "fig08_permutation_importance")


def fig_method_ranks(importance):
    top = importance.sort_values("mean_rank").head(12)
    ranks = top[[f"rank_{m}" for m in METHOD_LABELS]].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 4.4), layout="constrained")
    ax.imshow(ranks, cmap=SEQUENTIAL.reversed(), vmin=1, vmax=25, aspect="auto")
    ax.set_xticks(range(len(METHOD_LABELS)), METHOD_LABELS.values())
    ax.set_yticks(range(len(top)), [label(f) for f in top["feature"]])
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            ax.text(j, i, f"{ranks[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if ranks[i, j] <= 8 else INK)
    ax.set_title("Feature rank under each method (1 = most important; 33 features)", pad=28)
    save(fig, "fig09_ranking_methods")


def fig_grouped_importance(grouped):
    groups = grouped[grouped["model"] == "hist_gb"].sort_values("mae_increase")["group"].tolist()
    y = np.arange(len(groups))
    height = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    for i, (model, color) in enumerate(MODEL_COLORS.items()):
        values = grouped[grouped["model"] == model].set_index("group").loc[groups, "mae_increase"]
        ax.barh(y + (1 - i) * height, values, height=height, color=color, edgecolor=SURFACE,
                linewidth=0.8, label=MODEL_LABELS[model])
    ax.set_yticks(y, groups)
    ax.set(title="Grouped permutation importance (correlated features shuffled together)",
           xlabel="Increase in test MAE")
    ax.legend(loc="lower right")
    bar_axis(ax, "y")
    save(fig, "fig10_grouped_importance")


def fig_ridge_coefficients(coefs, n=18):
    # Indicator columns only (one-hot levels and 0/1 flags), so every bar is in pp per indicator.
    indicators = coefs[(coefs["scale"] == 1.0) | coefs["feature"].str.startswith(("txt_", "has_"))]
    indicators = indicators.assign(effect=indicators["effect_per_unit"] * 100)
    top = indicators.reindex(indicators["effect"].abs().sort_values(ascending=False).index).head(n)
    top = top.iloc[::-1]
    names = [label(f) if c == f else f"{label(f)}: {c[len(f) + 1:]}" for c, f in zip(top["column"], top["feature"])]
    colors = [BLUE if v > 0 else RED for v in top["effect"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    ax.barh(names, top["effect"], height=0.62, color=colors)
    ax.axvline(0, color=BASELINE, linewidth=0.8)
    ax.set(title=f"Largest Ridge indicator coefficients (top {n} by size)",
           xlabel="Effect on retention (pp of MSRP), all other features held fixed")
    ax.legend(handles=[Patch(color=BLUE, label="Raises retention"), Patch(color=RED, label="Lowers retention")],
              loc="lower right")
    bar_axis(ax, "y")
    save(fig, "fig11_ridge_coefficients")


def fig_partial_dependence(pdp):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), layout="constrained", sharey=True)
    for model, color in MODEL_COLORS.items():
        for ax, feature in zip(axes, ("vehicle_age", "mileage")):
            curve = pdp[(pdp["model"] == model) & (pdp["feature"] == feature)]
            ax.plot(curve["value"], curve["mean_prediction"], color=color, label=MODEL_LABELS[model])
    axes[0].set(title="Partial dependence on age", xlabel="Vehicle age (years), mileage held fixed",
                ylabel="Mean predicted retention")
    axes[1].set(title="Partial dependence on mileage", xlabel="Mileage, age held fixed")
    axes[1].xaxis.set_major_formatter(thousands)
    axes[1].legend(loc="upper right")
    save(fig, "fig12_partial_dependence")


# ---------------------------------------------------------------- model performance

def fig_model_comparison(results, best):
    results = results.set_index("model").loc[list(MODEL_LABELS)]
    names = [MODEL_LABELS[m] for m in results.index]
    colors = [BLUE if m == best else ("#86b6ef" if not m.startswith("baseline") else NEUTRAL)
              for m in results.index]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), layout="constrained")
    x = np.arange(len(results))
    for ax, metric, title in zip(axes, ("mae", "rmse", "r2"), ("MAE (lower is better)",
                                                                "RMSE (lower is better)", "R² (higher is better)")):
        ax.bar(x, results[f"cv_{metric}_mean"], yerr=results[f"cv_{metric}_std"], width=0.62,
               color=colors, edgecolor=MUTED, linewidth=0.4,
               error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 2})
        ax.scatter(x, results[f"test_{metric}"], marker="D", s=18, color=INK, zorder=3)
        ax.set_xticks(x, names, rotation=40, ha="right")
        ax.set_title(title)
        bar_axis(ax)
    axes[2].set_ylim(min(0, results["test_r2"].min()), 1)
    axes[0].set_ylabel("Bars: 5-fold CV mean ± SD")
    fig.legend(handles=[Patch(facecolor=NEUTRAL, edgecolor=MUTED, linewidth=0.4, label="Baselines"),
                        Patch(facecolor="#86b6ef", label="Tuned models"),
                        Patch(facecolor=BLUE, label=f"Best model ({MODEL_LABELS[best]})"),
                        Line2D([], [], marker="D", color=INK, linestyle="", markersize=4, label="Held-out test score")],
               loc="outside upper center", ncols=4)
    save(fig, "fig13_model_comparison")


def fig_predicted_vs_actual(predictions, results, best):
    pred = predictions.assign(source=predictions["listing_id"].str.split("_").str[0])
    scores = results.set_index("model").loc[best]
    fig, ax = plt.subplots(figsize=(4.8, 4.4), layout="constrained")
    for source, group in pred.groupby("source"):
        ax.scatter(group[TARGET], group[f"pred_{best}"], s=12, color=SOURCE_COLORS[source], alpha=0.7,
                   edgecolors=SURFACE, linewidths=0.4, label=f"{SOURCE_LABELS[source]} (n = {len(group)})")
    ax.plot([0, 1.35], [0, 1.35], color=INK_2, linewidth=1, label="Perfect prediction")
    ax.set(xlim=(0, 1.35), ylim=(0, 1.35), aspect="equal", xlabel="Actual retention ratio",
           ylabel="Predicted retention ratio",
           title=f"{MODEL_LABELS[best]}: predicted vs. actual (test set)")
    ax.text(0.97, 0.04, f"MAE {scores['test_mae']:.3f}\nRMSE {scores['test_rmse']:.3f}\nR² {scores['test_r2']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
    ax.legend(loc="upper left")
    save(fig, "fig14_predicted_vs_actual")


def fig_residuals(predictions, residuals, features, best):
    resid = predictions[TARGET] - predictions[f"pred_{best}"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), layout="constrained", width_ratios=(1.1, 1))
    ax = axes[0]
    ax.scatter(predictions[f"pred_{best}"], resid, s=8, color=BLUE, alpha=0.5, linewidths=0)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set(title="Residual vs. prediction", xlabel="Predicted retention ratio",
           ylabel="Residual (actual − predicted)")

    makes = features.drop_duplicates("model").set_index("model")["make"]
    by_model = residuals[(residuals["model"] == best) & (residuals["group"] == "model")].sort_values("mae")
    ax = axes[1]
    ax.barh(by_model["level"], by_model["mae"], height=0.62,
            color=[MAKE_COLORS[makes[m]] for m in by_model["level"]])
    overall = (predictions[TARGET] - predictions[f"pred_{best}"]).abs().mean()
    ax.axvline(overall, color=INK, linewidth=1)
    ax.annotate(f"overall {overall:.3f}", (overall, 0), xycoords=("data", "axes fraction"),
                xytext=(3, 3), textcoords="offset points", va="bottom", fontsize=7.5, color=INK)
    ax.tick_params(axis="y", labelsize=7)
    ax.set(title="Test MAE by vehicle model", xlabel="MAE (retention ratio)")
    ax.legend(handles=[Patch(color=c, label=m) for m, c in MAKE_COLORS.items()], loc="center right")
    bar_axis(ax, "y")
    save(fig, "fig15_residuals")


def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    integrated, features, t = load()
    best = t["model_results"].loc[t["model_results"]["selected"], "model"].iloc[0]
    importance = t["feature_importance"]
    _, test = split(features)
    assert len(test) == len(t["test_predictions"]), "test_predictions.csv is stale; re-run src.train"

    fig_distributions(integrated)
    fig_mileage_age(integrated)
    fig_listings_by_model(integrated)
    fig_retention_scatter(features)
    fig_retention_by_make(t["depreciation_curves"])
    fig_model_ranking(t["model_retention"], features)
    fig_correlation(integrated, features)

    fig_permutation_importance(importance, best)
    fig_method_ranks(importance)
    fig_grouped_importance(t["grouped_importance"])
    fig_ridge_coefficients(t["ridge_coefficients"])
    fig_partial_dependence(t["partial_dependence"])

    fig_model_comparison(t["model_results"], best)
    fig_predicted_vs_actual(t["test_predictions"], t["model_results"], best)
    fig_residuals(t["test_predictions"], t["residuals_by_group"], features, best)


if __name__ == "__main__":
    setup_logging()
    run()
