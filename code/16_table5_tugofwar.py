"""
16_table5_tugofwar.py
======================
Reproduce LPS (2019) Table 5: forecast close-to-close strategy returns
with the TugOfWar variable (Eq. (1), Section 4.4).

Eq. (1).  Strategy-level EWMA recursion with half-life 60 months
(alpha = 1 - 0.5^(1/60) ~ 0.0115):

    r_overnight^{s,EWMA}_t = alpha * ron^s_t + (1-alpha) * r_overnight^{s,EWMA}_{t-1}
    r_intraday^{s,EWMA}_t  = alpha * rid^s_t + (1-alpha) * r_intraday^{s,EWMA}_{t-1}

Signed TugOfWar so the predicted regression coef is positive:

    TugOfWar^s_t = r_intraday^{s,EWMA}_t  -  r_overnight^{s,EWMA}_t   (intraday-driven)
    TugOfWar^s_t = r_overnight^{s,EWMA}_t -  r_intraday^{s,EWMA}_t   (overnight-driven: MOM, STR)

Initial value = first observation (footnote 16).  Pandas
.ewm(adjust=False) matches this exactly.

Regression (one per strategy):
    r_close^s_{t+1}  =  alpha
                       + beta  * TugOfWar^s_t
                       + gamma * Factor_return_EWMA^s_t
                       + delta * Factor_vol^s_t
                       + market controls + eps

Controls:
  Shown:   TugOfWar, Factor return EWMA (same half-life on r_close),
           Factor vol (12-mo rolling std of monthly r_close -- monthly
           proxy for paper's daily-based vol).
  Hidden:  lagged 12-month market excess return, market volatility.
  Skipped: characteristic spread; short interest spread (Markit data
           not in pipeline).

11 strategies (Table 5; SUE and INDMOM are NOT in Table 5):
    MOM, SIZE(=ME), BM, ROE, INV, BETA, IVOL,
    ISSUE, ACCRUALS, TURNOVER, STR.
Overnight-driven (sign-flipped) = {MOM, STR}.

EWMA warm-up.  TAQ Millisecond starts 2003-09, so our strategy series
begins ~2003-10.  Paper's 1993-2013 sample also has a 60-mo warm-up
(1993-1998) but only ~25% of the window; our Block A has ~50% warm-up
months.  We therefore enforce EWMA_WARMUP=60: regression rows must
have >=60 prior strategy-return months in the EWMA recursion before
they qualify.  This shrinks Block A to ~64 obs but produces 10/11
positive coefs (matching paper's headline claim).  Block B inherits
~124 mo of warm-up from the contiguous A history and uses all 132 obs.

NW(12) HAC SE.  Returns scaled to PERCENT (x100) so coefs match paper.

Output:
  figures/table5_tugofwar.csv     11 strategies x (paper/A/B) x
                                   (TugOfWar, Factor_return, Factor_vol)
                                   coefs + NW(12) t-stats.
"""

from importlib import import_module
cfg   = import_module("00_config")
mod14 = import_module("14_table2_decomposition")

import numpy as np
import pandas as pd
import statsmodels.api as sm


# EWMA half-life 60 months.
EWMA_HL    = 60
EWMA_ALPHA = 1 - 0.5 ** (1 / EWMA_HL)

# Factor-vol rolling window (monthly proxy for paper's daily vol).
VOL_WINDOW = 12
VOL_MIN    = 6

NW_LAGS = cfg.NW_LAGS_MONTHLY     # 12

# Minimum prior strategy-return months that must enter the EWMA before
# a row qualifies for the regression (= half-life).
EWMA_WARMUP = 60

# (label, sort col, long_leg, n_buckets, strat_type).  strat_type in
# {"night","day"} controls TugOfWar sign per Eq. (1).
STRATEGIES = [
    ("MOM",      "mom",      "high", 10, "night"),
    ("SIZE",     "me",       "low",  10, "day"),
    ("BM",       "bm",       "high", 10, "day"),
    ("ROE",      "roe",      "high", 10, "day"),
    ("INV",      "inv",      "low",  10, "day"),
    ("BETA",     "beta",     "low",  10, "day"),
    ("IVOL",     "ivol",     "low",  10, "day"),
    ("ISSUE",    "issue",    "low",  10, "day"),
    ("ACCRUALS", "accruals", "low",  10, "day"),
    ("TURNOVER", "turnover", "low",  10, "day"),
    ("STR",      "str_",     "low",  10, "night"),
]

# Paper Table 5 (1993-2013).  Tuple = (TugOfWar coef, t, Factor_ret coef, t,
# Factor_vol coef, t).
PAPER_TABLE5 = {
    "MOM":      ( 1.967,  2.48,  0.001,  0.00, -1.189, -1.46),
    "SIZE":     ( 1.027,  1.57,  0.557,  1.01, -1.207, -1.18),
    "BM":       (-0.074, -0.12, -0.314, -0.39,  1.212,  1.30),
    "ROE":      ( 1.100,  2.47, -1.255, -1.29,  1.279,  1.62),
    "INV":      ( 1.339,  1.93, -1.061, -1.32,  0.821,  1.04),
    "BETA":     ( 1.340,  1.18,  0.024,  0.03, -0.427, -0.58),
    "IVOL":     ( 1.207,  2.11, -1.228, -1.33,  1.842,  1.76),
    "ISSUE":    ( 2.277,  2.86, -5.258, -4.00,  0.281,  0.41),
    "ACCRUALS": ( 0.470,  0.95, -1.197, -1.30,  2.045,  3.62),
    "TURNOVER": ( 2.098,  3.53, -0.901, -0.93,  0.858,  0.95),
    "STR":      ( 1.402,  2.39, -2.890, -2.43, -1.236, -1.83),
}


# ====================================================================
# Step 1 -- build full-window strategy panel (ron, rid, r_close)
# ====================================================================
def build_full_strategy_panel():
    """For each strategy, compute monthly VW long-short hedge night,
    day, and close-to-close return series over the FULL 2003-10..2024-12
    window (not block-split, so EWMA stays continuous across A/B).
    Returns DataFrame[anomaly, yearmon (formation), ron, rid, r_close].
    """
    print("loading inputs ...")
    panel = pd.read_parquet(cfg.CLEAN_DIR / "oi_monthly_full.parquet")
    chars = pd.read_parquet(cfg.CLEAN_DIR / "characteristics_monthly.parquet")
    chars = chars.drop(columns=["yearmon_ts"])
    m = panel.merge(chars, on=["permno", "yearmon"], how="left")

    # Calendar-aware next-month merge for the realized returns.
    nxt = m[["permno", "yearmon", "ret_close",
             "ret_overnight", "ret_intraday"]].rename(
        columns={"ret_close":     "rc_next",
                 "ret_overnight": "ron_next",
                 "ret_intraday":  "ri_next"},
    )
    nxt["match_yearmon"] = nxt["yearmon"] - 1
    nxt = nxt.drop(columns=["yearmon"])
    m = m.merge(
        nxt,
        left_on=["permno", "yearmon"],
        right_on=["permno", "match_yearmon"],
        how="left",
    ).drop(columns=["match_yearmon"])

    print(f"  panel rows: {len(m):,}   formation months: {m['yearmon'].nunique()}")

    out_dfs = []
    for label, col, long_leg, n_buckets, _ in STRATEGIES:
        ron = mod14.vw_long_short_hedge(m, col, "ron_next", long_leg, n_buckets) \
                  .rename(columns={"hedge": "ron"})
        rid = mod14.vw_long_short_hedge(m, col, "ri_next",  long_leg, n_buckets) \
                  .rename(columns={"hedge": "rid"})
        rc  = mod14.vw_long_short_hedge(m, col, "rc_next",  long_leg, n_buckets) \
                  .rename(columns={"hedge": "r_close"})
        s = (ron.merge(rid, on="yearmon", how="outer")
                .merge(rc,  on="yearmon", how="outer"))
        s["anomaly"] = label
        out_dfs.append(s[["anomaly", "yearmon", "ron", "rid", "r_close"]])
        print(f"  {label:<10}  {len(s):>4d} months")
    return pd.concat(out_dfs, ignore_index=True)


# ====================================================================
# Step 2 -- TugOfWar, Factor return EWMA, Factor vol per strategy
# ====================================================================
def add_strategy_predictors(strat_df):
    """Add per-strategy predictor columns.

    Indexing convention.  Input rows have yearmon = FORMATION month
    with (ron, rid, r_close) realized in (yearmon + 1).  We reindex by
    realization_ym = yearmon + 1, sort, then run the EWMA recursion.

    At row realization_ym = t, the EWMA value is over r^s_1..r^s_t
    (inclusive of t).  To forecast r^s_{t+1} we need that value as a
    predictor in the t+1 row, hence the 1-row .shift(1) into *_lag.

    Also tracks `ewma_n_prior` = count of prior obs that have entered
    the EWMA at this row -- used downstream to drop warm-up rows.
    """
    rows = []
    for label, _, _, _, strat_type in STRATEGIES:
        sub = strat_df[strat_df["anomaly"] == label].copy()
        sub["realization_ym"] = sub["yearmon"] + 1
        sub = sub.sort_values("realization_ym").reset_index(drop=True)

        # adjust=False seeds with first observation (footnote 16).
        # ignore_na=True keeps the chain alive across rare missing months.
        sub["ewma_night"] = sub["ron"].astype(float).ewm(
            alpha=EWMA_ALPHA, adjust=False, ignore_na=True,
        ).mean()
        sub["ewma_day"] = sub["rid"].astype(float).ewm(
            alpha=EWMA_ALPHA, adjust=False, ignore_na=True,
        ).mean()

        # Signed TugOfWar per Eq. (1).
        if strat_type == "night":
            sub["tugofwar"] = sub["ewma_night"] - sub["ewma_day"]
        else:
            sub["tugofwar"] = sub["ewma_day"]   - sub["ewma_night"]

        # Factor return EWMA on r_close (same half-life).
        sub["factor_ret_ewma"] = sub["r_close"].astype(float).ewm(
            alpha=EWMA_ALPHA, adjust=False, ignore_na=True,
        ).mean()

        # Factor vol: 12-mo rolling std of monthly r_close (proxy for
        # paper's daily-based vol).
        sub["factor_vol"] = sub["r_close"].astype(float).rolling(
            VOL_WINDOW, min_periods=VOL_MIN,
        ).std()

        # 1-row shift to expose predictors at the t+1 row.
        for c in ["tugofwar", "factor_ret_ewma", "factor_vol"]:
            sub[c + "_lag"] = sub[c].shift(1)

        # EWMA warm-up counter.
        sub["ewma_n_prior"] = (
            sub[["ron", "rid"]].notna().any(axis=1).cumsum().shift(1).fillna(0)
        )

        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


# ====================================================================
# Step 3 -- lagged market return and market volatility
# ====================================================================
def add_market_controls(df, ff_monthly):
    """Attach lagged 12-month cumulative mktrf and 12-month rolling std
    of mktrf.  Both indexed by realization_ym and lagged 1 month so they
    reflect info available at end of (realization_ym - 1)."""
    ff = ff_monthly.copy().sort_values("yearmon").reset_index(drop=True)
    ff["log1p_mkt"]   = np.log1p(ff["mktrf"].astype(float).clip(lower=-0.99))
    ff["mkt_cum_log"] = ff["log1p_mkt"].rolling(12, min_periods=12).sum()
    ff["mkt_cumret"]  = np.expm1(ff["mkt_cum_log"])
    ff["mkt_vol"]     = ff["mktrf"].astype(float).rolling(12, min_periods=12).std()
    ff["mkt_cumret_lag"] = ff["mkt_cumret"].shift(1)
    ff["mkt_vol_lag"]    = ff["mkt_vol"].shift(1)
    ff = ff[["yearmon", "mkt_cumret_lag", "mkt_vol_lag"]].rename(
        columns={"yearmon": "realization_ym"},
    )
    return df.merge(ff, on="realization_ym", how="left")


# ====================================================================
# Step 4 -- regression per strategy per block
# ====================================================================
def run_one_regression(sub, block_start, block_end, ewma_warmup=EWMA_WARMUP):
    """Run the Table 5 regression for one strategy on one block window.

    Returns dict with coef+t for TugOfWar, Factor return, Factor vol,
    plus n_obs.  All return-type variables are scaled to PERCENT (x100)
    so coefs are directly comparable to paper.

    `ewma_warmup` drops rows whose EWMA has consumed <warmup prior obs;
    critical for Block A (paper had natural 60-mo warm-up; ours requires
    explicit filter).  Block B inherits full warm-up from contiguous A.
    """
    sub = sub[(sub["realization_ym"] >= block_start) &
              (sub["realization_ym"] <= block_end)].copy()
    sub = sub[sub["ewma_n_prior"] >= ewma_warmup]
    cols = ["r_close", "tugofwar_lag", "factor_ret_ewma_lag",
            "factor_vol_lag", "mkt_cumret_lag", "mkt_vol_lag"]
    sub = sub.dropna(subset=cols)
    n = len(sub)
    nan_out = {"tugofwar_coef":   np.nan, "tugofwar_t":   np.nan,
               "factor_ret_coef": np.nan, "factor_ret_t": np.nan,
               "factor_vol_coef": np.nan, "factor_vol_t": np.nan,
               "n_obs": n}
    if n < 24:
        return nan_out

    y = sub["r_close"].astype(float).values * 100.0
    X_raw = sub[["tugofwar_lag",
                 "factor_ret_ewma_lag",
                 "factor_vol_lag",
                 "mkt_cumret_lag",
                 "mkt_vol_lag"]].astype(float).values * 100.0
    X = sm.add_constant(X_raw)

    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    # params order: const, tugofwar, factor_ret, factor_vol, mkt_cumret, mkt_vol
    return {
        "tugofwar_coef":   r.params[1], "tugofwar_t":   r.tvalues[1],
        "factor_ret_coef": r.params[2], "factor_ret_t": r.tvalues[2],
        "factor_vol_coef": r.params[3], "factor_vol_t": r.tvalues[3],
        "n_obs": n,
    }


def build_comparison_table(strat, a_window, b_window):
    rows = []
    for label, _, _, _, strat_type in STRATEGIES:
        sub = strat[strat["anomaly"] == label].copy()
        res_a = run_one_regression(sub, *a_window)
        res_b = run_one_regression(sub, *b_window)
        p = PAPER_TABLE5[label]
        rows.append({
            "strategy":              label,
            "strat_type":            strat_type,
            "paper_tugofwar":   p[0], "paper_tugofwar_t":   p[1],
            "paper_factor_ret": p[2], "paper_factor_ret_t": p[3],
            "paper_factor_vol": p[4], "paper_factor_vol_t": p[5],
            "A_tugofwar":   res_a["tugofwar_coef"],   "A_tugofwar_t":   res_a["tugofwar_t"],
            "A_factor_ret": res_a["factor_ret_coef"], "A_factor_ret_t": res_a["factor_ret_t"],
            "A_factor_vol": res_a["factor_vol_coef"], "A_factor_vol_t": res_a["factor_vol_t"],
            "A_n":          res_a["n_obs"],
            "B_tugofwar":   res_b["tugofwar_coef"],   "B_tugofwar_t":   res_b["tugofwar_t"],
            "B_factor_ret": res_b["factor_ret_coef"], "B_factor_ret_t": res_b["factor_ret_t"],
            "B_factor_vol": res_b["factor_vol_coef"], "B_factor_vol_t": res_b["factor_vol_t"],
            "B_n":          res_b["n_obs"],
        })
    return pd.DataFrame(rows)


def pretty_print(table):
    def cell(c, t):
        if pd.isna(c):
            return "    --      "
        if pd.isna(t):
            return f"{c:+6.3f}"
        return f"{c:+6.3f}({t:+5.2f})"
    print()
    print("=" * 130)
    print("Table 5 -- TugOfWar forecasting close-to-close strategy returns")
    print("=" * 130)
    fmt = "{:<10} {:>15} {:>15} {:>15} | {:>15} {:>15} {:>15}"
    print(fmt.format("strategy",
                     "paper_TugOfWar", "A_TugOfWar", "B_TugOfWar",
                     "paper_FactorRet", "A_FactorRet", "B_FactorRet"))
    print("-" * 130)
    for _, r in table.iterrows():
        print(fmt.format(
            r["strategy"],
            cell(r["paper_tugofwar"],   r["paper_tugofwar_t"]),
            cell(r["A_tugofwar"],       r["A_tugofwar_t"]),
            cell(r["B_tugofwar"],       r["B_tugofwar_t"]),
            cell(r["paper_factor_ret"], r["paper_factor_ret_t"]),
            cell(r["A_factor_ret"],     r["A_factor_ret_t"]),
            cell(r["B_factor_ret"],     r["B_factor_ret_t"]),
        ))
    print()
    print(fmt.format("strategy",
                     "paper_FactorVol", "A_FactorVol", "B_FactorVol",
                     "(empty)", "A_n", "B_n"))
    print("-" * 130)
    for _, r in table.iterrows():
        print(fmt.format(
            r["strategy"],
            cell(r["paper_factor_vol"], r["paper_factor_vol_t"]),
            cell(r["A_factor_vol"],     r["A_factor_vol_t"]),
            cell(r["B_factor_vol"],     r["B_factor_vol_t"]),
            "",
            f"{int(r['A_n'])}",
            f"{int(r['B_n'])}",
        ))


def main():
    print("=" * 80)
    print("LPS (2019) Table 5 reproduction -- TugOfWar predictive regression")
    print("=" * 80)

    print("\n[1/4] Building full-window strategy panel ...")
    strat = build_full_strategy_panel()

    print("\n[2/4] Computing TugOfWar / Factor return EWMA / Factor vol predictors ...")
    strat = add_strategy_predictors(strat)

    print("\n[3/4] Attaching FF market controls ...")
    ff = pd.read_parquet(cfg.RAW_DIR / "ff_monthly.parquet")
    ff["yearmon"] = ff["date"].dt.to_period("M")
    strat = add_market_controls(strat, ff)
    print(f"  total strategy panel rows: {len(strat):,}")

    a_window = (pd.Period(cfg.SAMPLE_BLOCKS["A"][0][:7], "M"),
                pd.Period(cfg.SAMPLE_BLOCKS["A"][1][:7], "M"))
    b_window = (pd.Period(cfg.SAMPLE_BLOCKS["B"][0][:7], "M"),
                pd.Period(cfg.SAMPLE_BLOCKS["B"][1][:7], "M"))

    print(f"\n[4/4] Running 11 strategy regressions x {{A,B}} ...")
    print(f"  Block A realization months: {a_window[0]} .. {a_window[1]}")
    print(f"  Block B realization months: {b_window[0]} .. {b_window[1]}")
    table = build_comparison_table(strat, a_window, b_window)

    out_path = cfg.FIG_DIR / "table5_tugofwar.csv"
    table.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved {out_path}")

    pretty_print(table)


if __name__ == "__main__":
    main()
