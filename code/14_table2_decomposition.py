"""
14_table2_decomposition.py
============================
Reproduce LPS (2019) Table 2: night/day CAPM-alpha decomposition of
the CRSP equity premium plus 13 long-short anomaly strategies.

For each anomaly:
  1. At formation month t, sort sample stocks into NYSE-breakpoint
     buckets (10 deciles for all anomalies EXCEPT INDMOM which uses
     5 quintiles, per Table 2 caption).
  2. VW the realised next-month overnight (ron_{t+1}) and intraday
     (ri_{t+1}) returns within each bucket and form
        hedge = top - bottom (long-leg = "high")
              = bottom - top (long-leg = "low"),
     per Table 2 caption.
  3. Shift hedge series by +1 mo to its realisation month, merge FF,
     regress on mktrf with NW(12) SEs; report the alpha and t-stat.

CRSP row: raw VW night/day return of the sample (no sort, no CAPM,
no rf subtraction -- see vw_market_returns docstring).

Long-leg conventions (per Table 2 caption):
  ME=low, BM=high, MOM=high, SUE=high, INDMOM=high(5),
  ROE=high, INV=low, BETA=low, IVOL=low, ISSUE=low,
  ACCRUALS=low, TURNOVER=low, STR=low.

Outputs:
  figures/table2_decomposition.csv   14 rows x paper/A/B comparison
  figures/table2_strategies.parquet  monthly long-short night/day
                                     return series (for Tables 4/5)
"""

from importlib import import_module
cfg   = import_module("00_config")
mod10 = import_module("10_validate_table1")

import numpy as np
import pandas as pd
import statsmodels.api as sm


# (label, sort col, long_leg, n_buckets).  INDMOM uses 5 quintiles per
# paper; all others use 10 deciles.
ANOMALIES = [
    ("ME",       "me",       "low",  10),
    ("BM",       "bm",       "high", 10),
    ("MOM",      "mom",      "high", 10),
    ("SUE",      "sue",      "high", 10),
    ("INDMOM",   "indmom",   "high",  5),
    ("ROE",      "roe",      "high", 10),
    ("INV",      "inv",      "low",  10),
    ("BETA",     "beta",     "low",  10),
    ("IVOL",     "ivol",     "low",  10),
    ("ISSUE",    "issue",    "low",  10),
    ("ACCRUALS", "accruals", "low",  10),
    ("TURNOVER", "turnover", "low",  10),
    ("STR",      "str_",     "low",  10),
]

# Paper Table 2 (1993-2013): (night_alpha%, night_t, day_alpha%, day_t).
# CRSP row reports raw mean returns (no risk adjustment) per the caption.
PAPER_TABLE2 = {
    "CRSP":     (0.55,  3.62,  0.38,  1.87),
    "ME":       (0.11,  0.75,  0.43,  1.85),
    "BM":      (-0.10, -0.67,  0.48,  2.21),
    "MOM":      (0.98,  3.84, -0.02, -0.06),
    "SUE":      (0.56,  3.20,  0.21,  0.70),
    "INDMOM":   (1.07,  6.47, -0.63, -2.03),
    "ROE":     (-0.95, -6.25,  1.42,  5.58),
    "INV":     (-0.28, -2.10,  0.97,  4.39),
    "BETA":    (-0.49, -2.17,  0.70,  2.40),
    "IVOL":    (-1.46, -5.23,  2.48,  6.21),
    "ISSUE":   (-0.52, -3.27,  1.13,  6.13),
    "ACCRUALS":(-0.47, -3.25,  1.10,  4.73),
    "TURNOVER":(-0.29, -1.98,  0.57,  2.58),
    "STR":      (0.93,  4.28, -1.05, -3.25),
}


# ====================================================================
# Load helpers
# ====================================================================
def load_block_panel(block):
    """oi_monthly_{block} + characteristics, with next-month night/day
    returns attached via calendar-aware merge (avoids gap-jumping when
    a permno drops out for one month).
    """
    panel = pd.read_parquet(cfg.CLEAN_DIR / f"oi_monthly_{block}.parquet")
    chars = pd.read_parquet(cfg.CLEAN_DIR / "characteristics_monthly.parquet")
    chars = chars.drop(columns=["yearmon_ts"])
    m = panel.merge(chars, on=["permno", "yearmon"], how="left")

    # Calendar-aware merge: row at yearmon=t gets next-month returns
    # via match_yearmon = (next row's yearmon) - 1 = t.
    nxt = m[["permno", "yearmon", "ret_overnight", "ret_intraday"]].rename(
        columns={"ret_overnight": "ron_next",
                 "ret_intraday":  "ri_next"},
    )
    nxt["match_yearmon"] = nxt["yearmon"] - 1
    nxt = nxt.drop(columns=["yearmon"])
    m = m.merge(nxt,
                left_on=["permno", "yearmon"],
                right_on=["permno", "match_yearmon"],
                how="left").drop(columns=["match_yearmon"])
    return m


# ====================================================================
# Sort + hedge
# ====================================================================
def vw_long_short_hedge(panel, sort_col, ret_col, long_leg, n_buckets):
    """VW long-short hedge time series.  For each formation yearmon:
       - NYSE-breakpoint quantile cutoffs on sort_col,
       - bucket assignment via searchsorted (0..n_buckets-1, low to high),
       - VW ret_col within bucket,
       - hedge = top - bottom (long-leg='high') or bottom - top ('low').
    Returns DataFrame[yearmon (formation), hedge].  Caller does the +1
    shift before merging with FF.
    """
    rows = []
    for ym, sub in panel.groupby("yearmon", observed=True):
        sub = sub.dropna(subset=[sort_col, ret_col, "mktcap"])
        nyse = sub[sub["exchcd"] == 1]
        if len(nyse) < n_buckets:
            continue
        cutoffs = nyse[sort_col].quantile(
            np.linspace(0, 1, n_buckets + 1)[1:-1]
        ).values
        bucket = np.searchsorted(cutoffs, sub[sort_col].values, side="right")
        sub = sub.assign(
            bucket=bucket,
            w_ret=sub[ret_col].astype(float) * sub["mktcap"].astype(float),
        )
        num = sub.groupby("bucket")["w_ret"].sum()
        den = sub.groupby("bucket")["mktcap"].sum()
        vw = num / den
        if 0 not in vw.index or (n_buckets - 1) not in vw.index:
            continue
        top = vw[n_buckets - 1]
        bot = vw[0]
        hedge = (top - bot) if long_leg == "high" else (bot - top)
        rows.append((ym, hedge))
    return pd.DataFrame(rows, columns=["yearmon", "hedge"])


def vw_market_returns(panel):
    """VW night and day returns of the sample, one obs per yearmon.

    No lag (this is the market premium, not a sort-based hedge).
    No rf subtraction: paper text says "excess" but the reported values
    (0.55 + 0.38 = 0.93%/mo for 1993-2013) match raw VW CRSP returns,
    not excess.  We follow the magnitudes, not the text.

    `panel` is the LPS-filtered sample (microcap-excluded); since
    microcaps carry negligible VW weight, the result is comparable in
    spirit to the paper's "value-weight CRSP universe".
    """
    rows = []
    for ym, sub in panel.groupby("yearmon", observed=True):
        sub = sub.dropna(subset=["ret_overnight", "ret_intraday", "mktcap"])
        if sub.empty:
            continue
        w = sub["mktcap"].astype(float)
        wsum = w.sum()
        ron = (sub["ret_overnight"].astype(float) * w).sum() / wsum
        rid = (sub["ret_intraday"].astype(float)  * w).sum() / wsum
        rows.append((ym, ron, rid))
    return pd.DataFrame(rows, columns=["yearmon", "ron", "rid"])


# ====================================================================
# Regressions
# ====================================================================
def capm_alpha(hedge_df, ff_monthly):
    """Shift hedge series +1 month (formation -> realization), merge
    FF, run CAPM with NW(12) SEs.  Returns (alpha%, t-stat) or NaN if
    fewer than 24 months of overlap remain.
    """
    h = hedge_df.copy()
    h["yearmon"] = h["yearmon"] + 1
    h = h.merge(ff_monthly, on="yearmon")
    if len(h) < 24:
        return np.nan, np.nan
    X = sm.add_constant(np.asarray(h[["mktrf"]], dtype=float))
    r = mod10.newey_west(h["hedge"], X)
    return r.params[0] * 100, r.tvalues[0]


def raw_mean(series):
    """Time-series mean with NW(12) t-stat (CRSP row only)."""
    arr = np.asarray(series, dtype=float)
    if len(arr) < 24:
        return np.nan, np.nan
    r = mod10.newey_west(arr)
    return r.params[0] * 100, r.tvalues[0]


# ====================================================================
# Block driver
# ====================================================================
def run_block(block, ff_monthly):
    """Compute all 14 Table 2 rows for one block.  Returns
       (results, strategies_long_df) where strategies records monthly
       hedge night/day series (formation-indexed) per anomaly.
    """
    print(f"\n=== Block {block} ===")
    panel = load_block_panel(block)
    print(f"  panel rows: {len(panel):,}, "
          f"months: {panel['yearmon'].nunique()}")

    results = []
    strat_rows = []

    # CRSP equity premium row.
    crsp = vw_market_returns(panel)
    na, nt = raw_mean(crsp["ron"])
    da, dt = raw_mean(crsp["rid"])
    results.append(("CRSP", na, nt, da, dt))
    print(f"  CRSP        night={na:+6.2f}% (t={nt:+5.2f})   "
          f"day={da:+6.2f}% (t={dt:+5.2f})")
    for _, r in crsp.iterrows():
        strat_rows.append((block, "CRSP", r["yearmon"], r["ron"], r["rid"]))

    # 13 long-short anomaly rows.
    for label, col, long_leg, n_buckets in ANOMALIES:
        h_n = vw_long_short_hedge(panel, col, "ron_next", long_leg, n_buckets)
        h_d = vw_long_short_hedge(panel, col, "ri_next",  long_leg, n_buckets)
        if h_n.empty or h_d.empty:
            print(f"  {label:<10}  (no hedge data formed)")
            results.append((label, np.nan, np.nan, np.nan, np.nan))
            continue
        na, nt = capm_alpha(h_n, ff_monthly)
        da, dt = capm_alpha(h_d, ff_monthly)
        results.append((label, na, nt, da, dt))
        print(f"  {label:<10}  night={na:+6.2f}% (t={nt:+5.2f})   "
              f"day={da:+6.2f}% (t={dt:+5.2f})")
        h_both = h_n.rename(columns={"hedge": "ron"}).merge(
            h_d.rename(columns={"hedge": "rid"}), on="yearmon", how="outer"
        )
        for _, r in h_both.iterrows():
            strat_rows.append((block, label, r["yearmon"], r["ron"], r["rid"]))

    strategies = pd.DataFrame(
        strat_rows, columns=["block", "anomaly", "yearmon", "ron", "rid"]
    )
    return results, strategies


# ====================================================================
# Comparison table
# ====================================================================
def build_comparison_table(block_a_res, block_b_res):
    a_dict = {label: (na, nt, da, dt) for label, na, nt, da, dt in block_a_res}
    b_dict = {label: (na, nt, da, dt) for label, na, nt, da, dt in block_b_res}
    labels = ["CRSP"] + [a[0] for a in ANOMALIES]
    rows = []
    for label in labels:
        pna, pnt, pda, pdt = PAPER_TABLE2.get(label, (np.nan,) * 4)
        ana, ant, ada, adt = a_dict.get(label, (np.nan,) * 4)
        bna, bnt, bda, bdt = b_dict.get(label, (np.nan,) * 4)
        rows.append({
            "anomaly":            label,
            "paper_night_alpha":  pna, "paper_night_t":  pnt,
            "paper_day_alpha":    pda, "paper_day_t":    pdt,
            "A_night_alpha":      ana, "A_night_t":      ant,
            "A_day_alpha":        ada, "A_day_t":        adt,
            "B_night_alpha":      bna, "B_night_t":      bnt,
            "B_day_alpha":        bda, "B_day_t":        bdt,
        })
    return pd.DataFrame(rows)


def pretty_print(table):
    def cell(a, t):
        if pd.isna(a):
            return "    --    "
        if pd.isna(t):
            return f"{a:+5.2f}%"
        return f"{a:+5.2f}%({t:+5.2f})"

    header_fmt = "{:<10} {:>14} {:>14} | {:>14} {:>14} | {:>14} {:>14}"
    print()
    print("=" * 115)
    print("Table 2 -- Night/Day CAPM alpha decomposition")
    print("    Paper 1993-2013   |   Block A 2003-10~2013-12   |   Block B 2014-01~2024-12")
    print("=" * 115)
    print(header_fmt.format("anomaly", "paper_night", "paper_day",
                            "A_night", "A_day", "B_night", "B_day"))
    print("-" * 115)
    for _, r in table.iterrows():
        print(header_fmt.format(
            r["anomaly"],
            cell(r["paper_night_alpha"], r["paper_night_t"]),
            cell(r["paper_day_alpha"],   r["paper_day_t"]),
            cell(r["A_night_alpha"],     r["A_night_t"]),
            cell(r["A_day_alpha"],       r["A_day_t"]),
            cell(r["B_night_alpha"],     r["B_night_t"]),
            cell(r["B_day_alpha"],       r["B_day_t"]),
        ))


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 80)
    print("LPS (2019) Table 2 reproduction -- night/day decomposition")
    print("=" * 80)

    ff = pd.read_parquet(cfg.RAW_DIR / "ff_monthly.parquet")
    ff["yearmon"] = ff["date"].dt.to_period("M")

    block_a_res, strat_a = run_block("A", ff)
    block_b_res, strat_b = run_block("B", ff)

    table = build_comparison_table(block_a_res, block_b_res)
    out_csv = cfg.FIG_DIR / "table2_decomposition.csv"
    table.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"\nSaved {out_csv}")

    strat_all = pd.concat([strat_a, strat_b], ignore_index=True)
    out_pq = cfg.FIG_DIR / "table2_strategies.parquet"
    cfg.atomic_write_parquet(strat_all, out_pq, index=False)
    print(f"Saved {out_pq}  ({len(strat_all):,} rows)")

    pretty_print(table)


if __name__ == "__main__":
    main()
