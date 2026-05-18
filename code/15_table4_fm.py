"""
15_table4_fm.py
=================
Reproduce LPS (2019) Table 4: Fama-MacBeth WLS regressions of monthly
stock returns on lagged firm characteristics.

For each formation yearmon t, run one cross-sectional WLS regression
per dependent variable {close, night, day}, weighted by lagged mktcap.
Then time-series mean of monthly coefs with Newey-West (12) SEs.

Table 4 has 5 columns:
  [1] close-to-close
  [2] overnight
  [3] intraday
  [4] = mean(b_night - b_day)                               (O - I)
  [5] = mean(b_night * 24/17.5 - b_day * 24/6.5)            (hourly scaled)
                  17.5h = 4pm close .. 9:30am next-day open
                   6.5h = 9:30am .. 4pm

14 RHS variables (paper order):
  ret_night, ret_day, ewma_night, ewma_day, mom, size, bm,
  ivol, beta, turnover, roe, inv, issue, accruals.

Conventions:
  - Returns in FRACTION inside the regression; final coefs x 100 for
    display (paper "X 100" header).
  - size = log(mktcap), same $thousand units across blocks.
  - EWMA half-life = 60 mo (alpha = 1 - 0.5^(1/60) ~ 0.0115); skip
    most recent month via 1-mo groupby shift.
  - IVOL rescaled fraction -> percent (*100) before regression so its
    coefficient is on the same unit scale as paper's.
  - WINSORIZE all 14 RHS cross-sectionally at 1%/99% each yearmon
    (standard FF/FM practice; uncontested practice but materially
    affects magnitudes via outlier suppression).
  - NW(12) on the time series of monthly coefs.
  - SUE, INDMOM, STR are NOT in Table 4 (only the 14 vars above).

Methodological deviation: paper's IVOL uses Carhart 4F with one lead
and one lag; ours uses contemporaneous Carhart only (from 13_).  This
affects magnitudes slightly but not signs.  Microcaps (where lead/lag
matters most due to non-synchronous trading) are already excluded.

Output:
  figures/table4_fm.csv     14 rows x 5 dep-var columns x paper/A/B
"""

from importlib import import_module
cfg = import_module("00_config")

import numpy as np
import pandas as pd
import statsmodels.api as sm


RHS = ["ret_night", "ret_day",
       "ewma_night", "ewma_day",
       "mom", "size", "bm",
       "ivol", "beta", "turnover",
       "roe", "inv", "issue", "accruals"]

EWMA_HL    = 60
EWMA_ALPHA = 1 - 0.5 ** (1 / EWMA_HL)        # ~ 0.01149

# Trading-hours scaling for the "scaled difference" column.
NIGHT_HRS = 17.5
DAY_HRS   = 6.5
TOTAL_HRS = 24.0

NW_LAGS = cfg.NW_LAGS_MONTHLY                # 12

# Cross-sectional winsor thresholds (each yearmon).
WIN_LO, WIN_HI = 0.01, 0.99

# Paper Table 4 (1993-2013), all values x 100 per paper header.
PAPER_TABLE4 = {
    "ret_night":  {"close": -0.161, "night":  4.585, "day": -4.792, "oi":  9.377, "scaled":  23.982},
    "ret_day":    {"close": -2.959, "night": -7.444, "day":  4.484, "oi":-11.928, "scaled": -26.766},
    "ewma_night": {"close": -4.910, "night": 16.836, "day":-21.685, "oi": 38.520, "scaled": 103.156},
    "ewma_day":   {"close": -2.456, "night":-15.564, "day": 13.583, "oi":-29.147, "scaled": -71.499},
    "mom":        {"close":  0.232, "night":  0.640, "day": -0.415, "oi":  1.056, "scaled":   2.411},
    "size":       {"close": -0.076, "night":  0.141, "day": -0.227, "oi":  0.368, "scaled":   1.031},
    "bm":         {"close":  0.028, "night":  0.148, "day": -0.120, "oi":  0.268, "scaled":   0.646},
    "ivol":       {"close": -0.045, "night":  0.165, "day": -0.149, "oi":  0.314, "scaled":   0.777},
    "beta":       {"close": -0.073, "night":  0.125, "day": -0.200, "oi":  0.325, "scaled":   0.910},
    "turnover":   {"close":  0.102, "night":  0.197, "day": -0.124, "oi":  0.322, "scaled":   0.729},
    "roe":        {"close":  0.214, "night": -0.214, "day":  0.427, "oi": -0.641, "scaled":  -1.870},
    "inv":        {"close": -0.531, "night":  0.001, "day": -0.542, "oi":  0.542, "scaled":   2.001},
    "issue":      {"close": -0.878, "night": -0.238, "day": -0.635, "oi":  0.397, "scaled":   2.019},
    "accruals":   {"close": -0.403, "night": -0.239, "day": -0.210, "oi": -0.029, "scaled":   0.447},
}


# ====================================================================
# Panel construction
# ====================================================================
def build_ewma_full():
    """EWMA night/day computed over the full 1993-2024 panel (not
    block-restricted), so EWMA at the start of each block carries a
    proper long-run history.  Returns DataFrame[permno, yearmon,
    ewma_night, ewma_day] where ewma_*[t] is the EWMA through month
    t-1 (i.e., "skip most recent month")."""
    full = pd.read_parquet(cfg.CLEAN_DIR / "oi_monthly_full.parquet")
    full = full.sort_values(["permno", "yearmon"]).reset_index(drop=True)
    # adjust=False -> recursive form seeded with first observation.
    # ignore_na=True -> single bad-data month does not zero the chain.
    full["ewma_night_now"] = full.groupby("permno")["ret_overnight"].transform(
        lambda s: s.astype(float).ewm(alpha=EWMA_ALPHA, adjust=False,
                                       ignore_na=True).mean()
    )
    full["ewma_day_now"] = full.groupby("permno")["ret_intraday"].transform(
        lambda s: s.astype(float).ewm(alpha=EWMA_ALPHA, adjust=False,
                                       ignore_na=True).mean()
    )
    # Skip-most-recent = lag by 1 within each permno.
    full["ewma_night"] = full.groupby("permno")["ewma_night_now"].shift(1)
    full["ewma_day"]   = full.groupby("permno")["ewma_day_now"].shift(1)
    return full[["permno", "yearmon", "ewma_night", "ewma_day"]]


def build_fm_panel(block):
    """Build the FM regression panel for one block.  One row per
    (permno, yearmon=t) with mktcap (WLS weight), next-month y_close /
    y_night / y_day, and the 14 RHS columns at month t.
    """
    panel = pd.read_parquet(cfg.CLEAN_DIR / f"oi_monthly_{block}.parquet")

    ewma = build_ewma_full()
    panel = panel.merge(ewma, on=["permno", "yearmon"], how="left")

    # Most-recent one-month night/day returns.
    panel["ret_night"] = panel["ret_overnight"]
    panel["ret_day"]   = panel["ret_intraday"]

    # size = log(mktcap), consistent units across blocks.
    panel["size"] = np.log(panel["mktcap"].astype(float))

    chars = pd.read_parquet(cfg.CLEAN_DIR / "characteristics_monthly.parquet")
    chars = chars[["permno", "yearmon",
                   "bm", "mom", "ivol", "beta", "turnover",
                   "roe", "inv", "issue", "accruals"]].copy()
    panel = panel.merge(chars, on=["permno", "yearmon"], how="left")

    # Calendar-aware next-month merge for the 3 dep vars.
    nxt = panel[["permno", "yearmon", "ret_close",
                 "ret_overnight", "ret_intraday"]].rename(
        columns={"ret_close":     "y_close",
                 "ret_overnight": "y_night",
                 "ret_intraday":  "y_day"},
    )
    nxt["match_yearmon"] = nxt["yearmon"] - 1
    nxt = nxt.drop(columns=["yearmon"])
    panel = panel.merge(
        nxt,
        left_on=["permno", "yearmon"],
        right_on=["permno", "match_yearmon"],
        how="left",
    ).drop(columns=["match_yearmon"])

    # IVOL fraction -> percent (matches paper unit).
    panel["ivol"] = panel["ivol"].astype(float) * 100.0

    # Cross-sectional 1%/99% winsor on the 14 RHS (applied after IVOL
    # rescaling so cutoffs are in regression units).
    panel = winsorize_cross_section(panel, RHS, lower=WIN_LO, upper=WIN_HI)

    return panel


def winsorize_cross_section(df, cols, lower=0.01, upper=0.99):
    """Per-yearmon clip of each column at its (lower, upper) cross-
    sectional quantile.  Vectorized via groupby+transform (O(N log N))."""
    out = df.copy()
    grp = out.groupby("yearmon", observed=True)
    for c in cols:
        lo = grp[c].transform(lambda s: s.astype(float).quantile(lower))
        hi = grp[c].transform(lambda s: s.astype(float).quantile(upper))
        out[c] = out[c].astype(float).clip(lower=lo, upper=hi)
    return out


# ====================================================================
# Cross-sectional WLS each month
# ====================================================================
def fm_cross_section(panel, y_col, rhs_cols, min_obs=30):
    """Per-yearmon WLS y_col ~ const + rhs_cols, weights = mktcap.
    Returns DataFrame[yearmon, const, *rhs_cols] of monthly coefs.
    Months with <min_obs valid rows (listwise) are dropped.
    """
    needed = [y_col] + rhs_cols + ["mktcap"]
    rows = []
    n_dropped_months = 0
    for ym, sub in panel.groupby("yearmon", observed=True):
        sub = sub.dropna(subset=needed)
        sub = sub[sub["mktcap"] > 0]
        for c in rhs_cols + [y_col]:
            sub = sub[np.isfinite(sub[c].astype(float))]
        if len(sub) < min_obs:
            n_dropped_months += 1
            continue
        X = sm.add_constant(sub[rhs_cols].astype(float).values)
        y = sub[y_col].astype(float).values
        w = sub["mktcap"].astype(float).values
        try:
            r = sm.WLS(y, X, weights=w).fit()
            rows.append((ym, *r.params))
        except Exception:
            n_dropped_months += 1
            continue
    cols = ["yearmon", "const"] + rhs_cols
    df = pd.DataFrame(rows, columns=cols)
    if n_dropped_months > 0:
        print(f"    (dropped {n_dropped_months} month(s) with <{min_obs} valid rows)")
    return df


# ====================================================================
# Time-series Newey-West
# ====================================================================
def nw_mean(series, lags=NW_LAGS):
    """Time-series mean with NW(lags) SE.  Returns (mean, t-stat)
    or NaN if fewer than 24 observations."""
    arr = np.asarray(pd.Series(series).dropna().astype(float).values)
    if len(arr) < 24:
        return np.nan, np.nan
    X = np.ones((len(arr), 1))
    r = sm.OLS(arr, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return r.params[0], r.tvalues[0]


# ====================================================================
# Block driver
# ====================================================================
def run_block_fm(block, label):
    """Compute all 5 Table 4 columns for one block.  Returns
    (results_df, n_obs).  Coefs x 100 for display."""
    print(f"\n=== Block {label} ===")
    panel = build_fm_panel(block)
    print(f"  panel rows: {len(panel):,}")
    valid_close = panel.dropna(subset=["y_close", "mktcap"] + RHS)
    valid_close = valid_close[valid_close["mktcap"] > 0]
    n_obs = len(valid_close)
    print(f"  rows valid for FM regression (close-to-close):  {n_obs:,}")

    print(f"  running cross-sectional WLS each month ...")
    c_close = fm_cross_section(panel, "y_close", RHS)
    c_night = fm_cross_section(panel, "y_night", RHS)
    c_day   = fm_cross_section(panel, "y_day",   RHS)
    print(f"    months with valid CS regression: "
          f"close={len(c_close)}, night={len(c_night)}, day={len(c_day)}")

    rows = []
    for var in RHS:
        # Columns 1-3: NW mean of per-month coef series.
        c_mean, c_t = nw_mean(c_close[var])
        n_mean, n_t = nw_mean(c_night[var])
        d_mean, d_t = nw_mean(c_day[var])

        # Columns 4-5: NW on the DIFFERENCE series (accounts for serial
        # correlation in (b_night - b_day)).
        merged = c_night[["yearmon", var]].merge(
            c_day[["yearmon", var]], on="yearmon", suffixes=("_n", "_d"),
        )
        diff_oi = merged[f"{var}_n"] - merged[f"{var}_d"]
        diff_sc = (merged[f"{var}_n"] * (TOTAL_HRS / NIGHT_HRS)
                   - merged[f"{var}_d"] * (TOTAL_HRS / DAY_HRS))
        oi_mean, oi_t = nw_mean(diff_oi)
        sc_mean, sc_t = nw_mean(diff_sc)

        rows.append({
            "var":         var,
            "close_coef":  c_mean * 100, "close_t":  c_t,
            "night_coef":  n_mean * 100, "night_t":  n_t,
            "day_coef":    d_mean * 100, "day_t":    d_t,
            "oi_coef":     oi_mean * 100, "oi_t":    oi_t,
            "scaled_coef": sc_mean * 100, "scaled_t": sc_t,
        })
    return pd.DataFrame(rows), n_obs


def build_comparison_table(a_df, b_df, n_a, n_b):
    """Build the paper vs A vs B comparison DataFrame."""
    a_dict = {r["var"]: r for _, r in a_df.iterrows()}
    b_dict = {r["var"]: r for _, r in b_df.iterrows()}
    rows = []
    for var in RHS:
        p = PAPER_TABLE4[var]
        a = a_dict[var]
        b = b_dict[var]
        rows.append({
            "var":           var,
            "paper_close":   p["close"],
            "A_close":       a["close_coef"], "A_close_t":  a["close_t"],
            "B_close":       b["close_coef"], "B_close_t":  b["close_t"],
            "paper_night":   p["night"],
            "A_night":       a["night_coef"], "A_night_t":  a["night_t"],
            "B_night":       b["night_coef"], "B_night_t":  b["night_t"],
            "paper_day":     p["day"],
            "A_day":         a["day_coef"],   "A_day_t":    a["day_t"],
            "B_day":         b["day_coef"],   "B_day_t":    b["day_t"],
            "paper_oi":      p["oi"],
            "A_oi":          a["oi_coef"],    "A_oi_t":     a["oi_t"],
            "B_oi":          b["oi_coef"],    "B_oi_t":     b["oi_t"],
            "paper_scaled":  p["scaled"],
            "A_scaled":      a["scaled_coef"], "A_scaled_t": a["scaled_t"],
            "B_scaled":      b["scaled_coef"], "B_scaled_t": b["scaled_t"],
        })
    out = pd.DataFrame(rows)
    out["paper_n"] = 454825
    out["A_n"] = n_a
    out["B_n"] = n_b
    return out


def pretty_print(table):
    def cell(c, t):
        if pd.isna(c):
            return "    --     "
        if pd.isna(t):
            return f"{c:+8.3f}"
        return f"{c:+8.3f}({t:+5.2f})"

    fmt = "{:<11} {:>13} {:>16} {:>16} | {:>13} {:>16} {:>16}"
    print()
    print("=" * 120)
    print("Table 4 -- Fama-MacBeth regressions (coefs x 100)")
    print(f"   Paper 1993-2013, n = 454,825   |   "
          f"Block A n = {table['A_n'].iloc[0]:,}   |   "
          f"Block B n = {table['B_n'].iloc[0]:,}")
    print("=" * 120)
    print(fmt.format("var",
                     "paper_close", "A_close", "B_close",
                     "paper_night", "A_night", "B_night"))
    print("-" * 120)
    for _, r in table.iterrows():
        print(fmt.format(
            r["var"],
            f"{r['paper_close']:+8.3f}",
            cell(r["A_close"], r["A_close_t"]),
            cell(r["B_close"], r["B_close_t"]),
            f"{r['paper_night']:+8.3f}",
            cell(r["A_night"], r["A_night_t"]),
            cell(r["B_night"], r["B_night_t"]),
        ))
    print()
    print(fmt.format("var",
                     "paper_day", "A_day", "B_day",
                     "paper_oi", "A_oi", "B_oi"))
    print("-" * 120)
    for _, r in table.iterrows():
        print(fmt.format(
            r["var"],
            f"{r['paper_day']:+8.3f}",
            cell(r["A_day"], r["A_day_t"]),
            cell(r["B_day"], r["B_day_t"]),
            f"{r['paper_oi']:+8.3f}",
            cell(r["A_oi"], r["A_oi_t"]),
            cell(r["B_oi"], r["B_oi_t"]),
        ))


def main():
    print("=" * 80)
    print("LPS (2019) Table 4 reproduction -- Fama-MacBeth WLS regressions")
    print("=" * 80)

    a_df, n_a = run_block_fm("A", "A (2003-10 ~ 2013-12)")
    b_df, n_b = run_block_fm("B", "B (2014-01 ~ 2024-12)")

    table = build_comparison_table(a_df, b_df, n_a, n_b)
    out_path = cfg.FIG_DIR / "table4_fm.csv"
    table.to_csv(out_path, index=False, float_format="%.3f")
    print(f"\nSaved {out_path}")

    pretty_print(table)


if __name__ == "__main__":
    main()
