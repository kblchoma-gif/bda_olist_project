"""Phase 4 — Statistical analysis (no statsmodels dependency).

OLS regression is implemented directly with numpy + scipy.stats so that
corporate Application Control policies blocking statsmodels' compiled
extensions do not prevent the analysis from running.

Produces:
  outputs/tables/descriptive_numeric.csv
  outputs/tables/review_score_distribution.csv
  outputs/tables/hypothesis_tests.csv
  outputs/tables/correlation_pearson.csv
  outputs/tables/correlation_spearman.csv
  outputs/tables/ols_regression_summary.txt
  outputs/tables/anova_delivery_by_state.txt
  outputs/figures/*.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.config import PROC_DIR, TAB_DIR, FIG_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid", context="talk")
rng = np.random.default_rng(RANDOM_STATE)


# ------------------------------------------------------------------ load
def load_master() -> pd.DataFrame:
    p = PROC_DIR / "master.parquet"
    if not p.exists():
        p = PROC_DIR / "master.csv"
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    for c in ["order_purchase_timestamp", "order_delivered_customer_date",
              "order_estimated_delivery_date", "order_delivered_carrier_date",
              "order_approved_at"]:
        if c in df.columns and not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "is_late" in df.columns:
        df["is_late"] = pd.to_numeric(df["is_late"], errors="coerce")
    return df


# ------------------------------------------------------------------ 4.2.1 descriptive
NUMERIC_COLS = [
    "delivery_days", "estimated_days", "delay_days",
    "price_total", "freight_total", "freight_ratio",
    "payment_value", "payment_installments", "n_items",
    "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm", "review_score",
]

def descriptive_numeric(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "variable":  c,
            "n":         int(s.size),
            "mean":      round(s.mean(), 3),
            "median":    round(s.median(), 3),
            "mode":      s.mode().iloc[0] if not s.mode().empty else np.nan,
            "std":       round(s.std(), 3),
            "variance":  round(s.var(), 3),
            "min":       round(s.min(), 3),
            "p25":       round(s.quantile(0.25), 3),
            "p50":       round(s.quantile(0.50), 3),
            "p75":       round(s.quantile(0.75), 3),
            "p95":       round(s.quantile(0.95), 3),
            "max":       round(s.max(), 3),
            "skew":      round(s.skew(), 3),
            "kurtosis":  round(s.kurtosis(), 3),
        })
    return pd.DataFrame(rows)


def review_distribution(df: pd.DataFrame) -> pd.DataFrame:
    s = df["review_score"].dropna()
    vc = s.value_counts().sort_index()
    out = pd.DataFrame({
        "review_score": vc.index.astype(int),
        "count":        vc.values,
    })
    out["share_pct"]     = (100 * out["count"] / out["count"].sum()).round(2)
    out["cum_share_pct"] = out["share_pct"].cumsum().round(2)
    return out


# ------------------------------------------------------------------ 4.2.2 inference
def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = df.dropna(subset=["is_late", "review_score"]).copy()
    on_time  = sub.loc[sub["is_late"] == 0, "review_score"]
    late     = sub.loc[sub["is_late"] == 1, "review_score"]

    u_stat, p_u = stats.mannwhitneyu(late, on_time, alternative="two-sided")
    n1, n2 = len(late), len(on_time)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    t_stat, p_t = stats.ttest_ind(late, on_time, equal_var=False)

    s1, s2 = late.std(ddof=1), on_time.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (late.mean() - on_time.mean()) / pooled if pooled else np.nan

    rows += [
        {"test": "Mann-Whitney U (late vs on-time review_score)",
         "statistic": round(u_stat, 3), "p_value": p_u,
         "effect_size": round(rank_biserial, 4),
         "interpretation": "rank-biserial correlation"},
        {"test": "Welch t-test (late vs on-time review_score)",
         "statistic": round(t_stat, 3), "p_value": p_t,
         "effect_size": round(cohens_d, 4),
         "interpretation": "Cohen's d"},
    ]

    cat = sub.assign(low_review=(sub["review_score"] <= 2).astype(int))
    ct = pd.crosstab(cat["is_late"], cat["low_review"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    rows.append({
        "test": "Chi-square (is_late × low_review)",
        "statistic": round(chi2, 3), "p_value": p_chi,
        "effect_size": round(cramers_v, 4),
        "interpretation": f"Cramér's V, dof={dof}",
    })
    return pd.DataFrame(rows)


def anova_delivery_by_state(df: pd.DataFrame, top_n: int = 10) -> str:
    sub = df.dropna(subset=["delivery_days", "customer_state"]).copy()
    top_states = sub["customer_state"].value_counts().head(top_n).index
    sub = sub[sub["customer_state"].isin(top_states)]

    groups = [g["delivery_days"].values for _, g in sub.groupby("customer_state")]
    f_stat, p_val = stats.f_oneway(*groups)
    lev_w, lev_p = stats.levene(*groups)

    grand = sub["delivery_days"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total   = ((sub["delivery_days"] - grand) ** 2).sum()
    eta_sq     = ss_between / ss_total

    lines = [
        f"One-way ANOVA: delivery_days ~ customer_state (top {top_n} states)",
        f"  F-statistic : {f_stat:.3f}",
        f"  p-value     : {p_val:.6g}",
        f"  eta-squared : {eta_sq:.4f}",
        f"Levene's test for homogeneity of variance",
        f"  W           : {lev_w:.3f}",
        f"  p-value     : {lev_p:.6g}",
        "",
        "Group means (days):",
    ]
    for st, g in sub.groupby("customer_state"):
        lines.append(f"  {st}: n={len(g):>6}  mean={g['delivery_days'].mean():.2f}  "
                     f"sd={g['delivery_days'].std():.2f}")
    return "\n".join(lines)


# ------------------------------------------------------------------ 4.2.3 correlation
def correlation_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    pearson  = sub.corr(method="pearson").round(3)
    spearman = sub.corr(method="spearman").round(3)
    return pearson, spearman


# ------------------------------------------------------------------ 4.2.4 OLS (manual)
def ols_summary(df: pd.DataFrame,
                y_col: str = "review_score",
                x_cols: list[str] | None = None) -> str:
    """Ordinary Least Squares regression implemented with numpy + scipy.

    Returns a text block equivalent to statsmodels' summary table.
    """
    if x_cols is None:
        x_cols = ["is_late", "delivery_days", "freight_ratio", "n_items"]

    sub = df[[y_col] + x_cols].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[np.isfinite(sub.values).all(axis=1)]

    y = sub[y_col].to_numpy(dtype=float)
    X = sub[x_cols].to_numpy(dtype=float)
    n = len(y)

    # Add intercept column
    X_design = np.column_stack([np.ones(n), X])
    k = X_design.shape[1]                        # parameters incl. intercept
    param_names = ["const"] + x_cols

    # beta = (X'X)^-1 X'y
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X_design.T @ y)

    resid = y - X_design @ beta
    rss   = float(resid @ resid)
    tss   = float(((y - y.mean()) ** 2).sum())
    dof_resid = n - k
    sigma2 = rss / dof_resid
    se     = np.sqrt(np.diag(sigma2 * XtX_inv))

    t_stats = beta / se
    p_vals  = 2 * stats.t.sf(np.abs(t_stats), df=dof_resid)

    r2      = 1 - rss / tss
    adj_r2  = 1 - (1 - r2) * (n - 1) / (n - k)
    f_stat  = (r2 / (k - 1)) / ((1 - r2) / dof_resid)
    f_pval  = stats.f.sf(f_stat, dfn=(k - 1), dfd=dof_resid)

    # --- format
    lines = []
    lines.append("=" * 78)
    lines.append("OLS Regression Summary (numpy implementation)")
    lines.append(f"Dependent variable: {y_col}")
    lines.append("=" * 78)
    lines.append(f"No. Observations:      {n:>14,}")
    lines.append(f"Df Residuals:          {dof_resid:>14,}")
    lines.append(f"Df Model:              {k - 1:>14,}")
    lines.append(f"R-squared:             {r2:>14.4f}")
    lines.append(f"Adj. R-squared:        {adj_r2:>14.4f}")
    lines.append(f"F-statistic:           {f_stat:>14.3f}")
    lines.append(f"Prob (F-statistic):    {f_pval:>14.4g}")
    lines.append("-" * 78)
    lines.append(f"{'':<20}{'coef':>12}{'std err':>12}{'t':>10}{'P>|t|':>12}"
                 f"{'[0.025':>12}{'0.975]':>12}")
    lines.append("-" * 78)
    t_crit = stats.t.ppf(0.975, df=dof_resid)
    for name, b, s, t, p in zip(param_names, beta, se, t_stats, p_vals):
        lo, hi = b - t_crit * s, b + t_crit * s
        lines.append(f"{name:<20}{b:>12.4f}{s:>12.4f}{t:>10.3f}{p:>12.4g}"
                     f"{lo:>12.4f}{hi:>12.4f}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ------------------------------------------------------------------ 4.2.5 figures
def make_figures(df: pd.DataFrame) -> None:
    # 1. Review score bar
    fig, ax = plt.subplots(figsize=(9, 5))
    dist = review_distribution(df)
    sns.barplot(data=dist, x="review_score", y="count", ax=ax, color="#3b5b92")
    ax.set_title("Distribution of Customer Review Scores")
    ax.set_xlabel("Review score (1 = worst, 5 = best)")
    ax.set_ylabel("Number of orders")
    for i, row in dist.iterrows():
        ax.text(i, row["count"], f"{row['share_pct']:.1f}%",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_review_score_distribution.png", dpi=140)
    plt.close(fig)

    # 2. Delivery-days histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    dd = df["delivery_days"].dropna()
    cap = dd.quantile(0.99)
    sns.histplot(dd.clip(upper=cap), bins=60, ax=ax, color="#7a3b92")
    ax.set_title(f"Delivery Time Distribution (clipped at 99th pct = {cap:.1f} days)")
    ax.set_xlabel("Delivery time (days, purchase → customer)")
    ax.set_ylabel("Number of orders")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_delivery_days_hist.png", dpi=140)
    plt.close(fig)

    # 3. Review score by late/on-time
    sub = df.dropna(subset=["is_late", "review_score"]).copy()
    sub["late_label"] = sub["is_late"].map({0: "On time", 1: "Late"}).astype(str)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=sub, x="late_label", y="review_score",
                order=["On time", "Late"], ax=ax,
                palette={"On time": "#2c7a4b", "Late": "#a02020"})
    ax.set_title("Review Score by Delivery Timeliness")
    ax.set_xlabel("Delivery status")
    ax.set_ylabel("Review score (1–5)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_review_by_lateness.png", dpi=140)
    plt.close(fig)

    # 4. Correlation heatmap
    pearson, _ = correlation_matrices(df)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(pearson, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Pearson Correlation Matrix — Numeric Features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_corr_heatmap.png", dpi=140)
    plt.close(fig)

    # 5. Delay vs review score (binned scatter)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot = df.dropna(subset=["delay_days", "review_score"]).copy()
    plot = plot[plot["delay_days"].between(-30, 30)]
    sns.regplot(data=plot, x="delay_days", y="review_score",
                scatter_kws={"alpha": 0.05, "s": 6},
                line_kws={"color": "red"}, ax=ax)
    ax.set_title("Review Score vs Delay (positive = delivered after estimate)")
    ax.set_xlabel("Delay (days)")
    ax.set_ylabel("Review score (1–5)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_delay_vs_review.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    print("[INFO] Loading master table ...")
    df = load_master()
    print(f"[INFO] Master loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    print("\n=== 4.2.1 DESCRIPTIVE STATISTICS ===")
    desc = descriptive_numeric(df)
    desc.to_csv(TAB_DIR / "descriptive_numeric.csv", index=False)
    print(desc.to_string(index=False))

    print("\n=== 4.2.1b REVIEW SCORE DISTRIBUTION ===")
    rdist = review_distribution(df)
    rdist.to_csv(TAB_DIR / "review_score_distribution.csv", index=False)
    print(rdist.to_string(index=False))

    print("\n=== 4.2.2 HYPOTHESIS TESTS ===")
    ht = hypothesis_tests(df)
    ht.to_csv(TAB_DIR / "hypothesis_tests.csv", index=False)
    print(ht.to_string(index=False))

    print("\n=== 4.2.2b ANOVA — delivery_days by customer_state (top 10) ===")
    anova_txt = anova_delivery_by_state(df, top_n=10)
    (TAB_DIR / "anova_delivery_by_state.txt").write_text(anova_txt, encoding="utf-8")
    print(anova_txt)

    print("\n=== 4.2.3 CORRELATION — PEARSON ===")
    pearson, spearman = correlation_matrices(df)
    pearson.to_csv(TAB_DIR / "correlation_pearson.csv")
    spearman.to_csv(TAB_DIR / "correlation_spearman.csv")
    print(pearson.to_string())

    print("\n=== 4.2.3b CORRELATION — SPEARMAN ===")
    print(spearman.to_string())

    print("\n=== 4.2.4 OLS REGRESSION — review_score ~ lateness & covariates ===")
    ols_txt = ols_summary(df)
    (TAB_DIR / "ols_regression_summary.txt").write_text(ols_txt, encoding="utf-8")
    print(ols_txt)

    print("\n=== 4.2.5 FIGURES ===")
    make_figures(df)
    print(f"[OK] Figures written to {FIG_DIR}")