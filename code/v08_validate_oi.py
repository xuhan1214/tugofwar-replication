"""
Sanity-check the daily overnight/intraday split panel produced by
08_build_overnight_intraday.py.

The single most important check is the IDENTITY:

        (1 + r_intraday)(1 + r_overnight) = 1 + r_close_to_close

If this fails on more than a handful of stock-days something is wrong
with the rolling-VWAP logic in 08.

We also:
    - confirm the share of "good" days is reasonable,
    - sanity-check distribution moments of overnight vs. intraday returns,
    - and cross-check our TAQ-derived VWAP against CRSP openprc on a
      random sample (a strong external validation that our SQL time /
      condition filters are correct).
"""
from importlib import import_module
cfg = import_module("00_config")
from _validate import Checker

import numpy as np
import pandas as pd

chk = Checker("Phase 7a — daily OI returns identity check")
chk.require_files(
    [cfg.CLEAN_DIR / "oi_daily.parquet",
     cfg.RAW_DIR   / "crsp_dsf.parquet",
     cfg.RAW_DIR   / "crsp_stocknames.parquet"],
    hint="Run `python 07_link_taq_to_crsp.py` then "
         "`python 08_build_overnight_intraday.py` first.",
)
oi = pd.read_parquet(cfg.CLEAN_DIR / "oi_daily.parquet")

# ----------------------------------------------------------------------
# 1. share of good days — broken down by sample slice
# ----------------------------------------------------------------------
# oi_daily covers the FULL CRSP DSF (1993-2024, all share codes,
# all exchanges, all prices) for downstream flexibility.  The headline
# good rate therefore looks low (~33%) because:
#   - 1993-2002 has zero TAQ-msec coverage (product launched 2003-09)
#   - DSF contains ETFs / ADRs / REITs / penny stocks LPS filters out
#
# We compute a "LPS-LIKE" good rate by attaching shrcd/exchcd via
# stocknames and filtering to (shrcd 10/11, exchcd 1/2/3, |prc|>=$5).
# This is a STRICT SUPERSET of the LPS final sample — the size filter
# (mktcap > NYSE q20) is applied at monthly granularity in 09 and is
# not replicated here to keep v08 simple.  Therefore:
#
#   - v08 "LPS-like good rate" is a LOWER BOUND
#   - true LPS sample good rate (after 09's size filter) will be 5-10
#     percentage points higher because microcaps (which have poor TAQ
#     coverage) get dropped
#   - the actual LPS row count is reported by 09 / v09
chk.section("good-day rate by sample slice")
overall = oi["good"].mean()
chk.note(f"headline good rate (entire DSF, 1993-2024): {overall*100:.1f}%  "
         f"(includes pre-TAQ years + non-common-stock; not informative)")

# Post-TAQ-launch only
post03 = oi[oi["date"] >= pd.Timestamp(cfg.SAMPLE_BLOCKS["A"][0])]
post03_rate = post03["good"].mean() if len(post03) else 0.0
chk.between("post-2003 good rate (full DSF)", post03_rate, 0.35, 0.95)

# LPS-like universe (NO size filter — see comment above)
stk = pd.read_parquet(
    cfg.RAW_DIR / "crsp_stocknames.parquet",
    columns=["permno", "namedt", "nameenddt", "shrcd", "exchcd"],
).sort_values("namedt")
oi_sorted = oi[["permno", "date", "prc", "good"]].sort_values("date").copy()
oi_sorted["prc_abs"] = oi_sorted["prc"].abs()
m = pd.merge_asof(
    oi_sorted, stk[["permno", "namedt", "nameenddt", "shrcd", "exchcd"]],
    left_on="date", right_on="namedt", by="permno", direction="backward",
)
mask = (
    m["nameenddt"].notna() & (m["date"] <= m["nameenddt"])
    & m["shrcd"].isin(cfg.SHRCD_KEEP)
    & m["exchcd"].isin(cfg.EXCHCD_KEEP)
    & (m["prc_abs"] >= cfg.PRICE_FLOOR)
    & (m["date"] >= pd.Timestamp(cfg.SAMPLE_BLOCKS["A"][0]))
)
lps_like = m.loc[mask]
lps_like_rate = lps_like["good"].mean() if len(lps_like) else 0.0
chk.note(f"LPS-like (NO size filter) sample size: {len(lps_like):,} stock-days "
         f"-- shrcd 10/11 + NYSE/AMEX/NASDAQ + |prc|>=$5, post-{cfg.SAMPLE_BLOCKS['A'][0][:7]}")
chk.note("true LPS adds 'mktcap > NYSE q20' (applied monthly in 09); "
         "the true sample's good rate will be 5-10 pp higher than the number below.")
chk.between("LPS-like good rate (lower bound on true LPS)", lps_like_rate, 0.70, 0.95)

# ----------------------------------------------------------------------
# 2. identity (1+ri)(1+ron) == (1+ret) on CONSECUTIVE good-day pairs
# ----------------------------------------------------------------------
# The identity (1+ri)(1+ron) = 1+ret is EXACT only on a good day t
# whose immediately-prior row (same permno, sorted by date) is also
# good.  When the previous row was bad (valid CRSP ret but no valid
# TAQ VWAP), our `ret_overnight` carries the rolled close(prev_good)-
# to-open(t) return, and the resulting lhs equals 1+ret_rolled, not
# 1+ret(t).  We therefore restrict the identity test to pairs where
# prev_row.good is True.  See 08_build_overnight_intraday.py for the
# rolling-VWAP construction.
oi_sorted = oi.sort_values(["permno", "date"]).copy()
oi_sorted["prev_good"] = (
    oi_sorted.groupby("permno")["good"].shift(1)
              .astype("boolean").fillna(False)
)
consec_mask = oi_sorted["good"] & oi_sorted["prev_good"]
g_consec = oi_sorted.loc[consec_mask].copy()

g_consec["lhs"] = (1.0 + g_consec["ret_intraday"]) * (1.0 + g_consec["ret_overnight"]) - 1.0
g_consec["rhs"] = g_consec["ret"]
g_consec["abs_diff"] = (g_consec["lhs"] - g_consec["rhs"]).abs()

n_good = int(oi["good"].sum())
n_consec = len(g_consec)
chk.section("identity (1+ri)(1+ron) = 1+ret on consecutive good-day pairs")
chk.note(f"consec-pair sample: {n_consec:,} of {n_good:,} good days "
         f"({n_consec/max(n_good,1)*100:.1f}%)")
chk.between("median |lhs - rhs|",       g_consec["abs_diff"].median(),       0,   1e-7)
chk.between("99th-pct |lhs - rhs|",     g_consec["abs_diff"].quantile(0.99), 0,   1e-4)
chk.at_most("worst |lhs - rhs|",        g_consec["abs_diff"].max(),               0.01)

# All good days (including post-gap) — looser sanity check.  On
# post-gap good days, lhs = 1+ret_rolled (multi-day), so diff can be
# meaningful.  The MAX absolute diff is dominated by penny-stock
# outliers (e.g., reverse splits on $0.50 stocks where one day ret >
# 1000%); these will be filtered out by 09 via the $5 floor + NYSE q20
# size cutoff, so they don't affect Table 1.  We therefore use a
# percentile-robust test (99.5th) instead of max.
g = oi[oi["good"]].copy()
g["lhs"] = (1.0 + g["ret_intraday"]) * (1.0 + g["ret_overnight"]) - 1.0
g["abs_diff"] = (g["lhs"] - g["ret"]).abs()
chk.section("identity sanity on ALL good days (post-gap days included)")
chk.at_most("99th-pct |lhs - rhs|",   g["abs_diff"].quantile(0.99),   0.20)
chk.at_most("99.5th-pct |lhs - rhs|", g["abs_diff"].quantile(0.995),  0.50)
chk.note(f"max |lhs - rhs| = {g['abs_diff'].max():.2f}  "
         f"(extreme penny-stock outliers; dropped by 09's $5 floor)")

# ----------------------------------------------------------------------
# 3. distribution moments — INFO-ONLY (not an LPS assertion)
# ----------------------------------------------------------------------
# IMPORTANT: LPS 2019 does NOT explicitly state an "intraday vol >
# overnight vol" result.  The closest mention (Section 4.2.2) is about
# the MOM portfolio's monthly overnight std (4.02%) vs close-to-close
# std (7.85%) — a different comparison entirely.  The "intraday vol >
# overnight vol" stylised fact comes from older microstructure
# literature (Fama 1965, French-Roll 1986).  We report the moments
# here as INFO and check they fall in sensible ranges, but do NOT
# assert ordering as a pass/fail predicate.
#
# Filters applied to make the comparison apples-to-apples:
#   (a) consecutive good-day pairs ONLY — on post-gap good days our
#       ret_overnight covers a multi-day rolled return whose variance
#       is inflated by construction
#   (b) LPS-like universe (shrcd 10/11 + exchcd 1/2/3 + |prc|>=$5) —
#       the unfiltered panel includes ETFs / ADRs / micro-caps with
#       persistent VWAP quality issues that distort the moments
chk.section("distribution moments (annualised %, LPS-like consec good-day pairs, INFO-ONLY)")
# Merge stocknames again to get shrcd/exchcd on each consec pair row
g_consec_lps = g_consec.merge(
    stk[["permno", "namedt", "nameenddt", "shrcd", "exchcd"]],
    on="permno", how="left",
)
g_consec_lps = g_consec_lps[
    (g_consec_lps["date"] >= g_consec_lps["namedt"])
    & (g_consec_lps["date"] <= g_consec_lps["nameenddt"])
]
g_consec_lps = g_consec_lps[
    g_consec_lps["shrcd"].isin(cfg.SHRCD_KEEP)
    & g_consec_lps["exchcd"].isin(cfg.EXCHCD_KEEP)
    & (g_consec_lps["prc"].abs() >= cfg.PRICE_FLOOR)
].drop_duplicates(subset=["permno", "date"])

ri_c  = g_consec_lps["ret_intraday"]
ron_c = g_consec_lps["ret_overnight"]
ri_std  = ri_c.std()  * np.sqrt(252) * 100
ron_std = ron_c.std() * np.sqrt(252) * 100
chk.note(f"sample: {len(g_consec_lps):,} LPS-like consec good-day pairs")
chk.between("intraday  ann. vol",  ri_std,  20, 70)
chk.between("overnight ann. vol",  ron_std, 10, 60)
chk.note(f"intraday vs overnight: {ri_std:.2f}% vs {ron_std:.2f}%  "
         f"(NOT an LPS assertion — heuristic from microstructure lit.)")

# ----------------------------------------------------------------------
# 4. coverage by era (only 2003+ for this account)
# ----------------------------------------------------------------------
chk.section("coverage")
n_total  = len(oi)
n_good   = int(oi["good"].sum())
n_recent = int(((oi["date"] >= "2010-01-01") & oi["good"]).sum())
chk.at_least("good rows total",   n_good,    1_000_000)
chk.at_least("good rows >= 2010", n_recent,    500_000)

# ----------------------------------------------------------------------
# 5. VWAP vs CRSP openprc cross-check  (THE strongest external check)
# ----------------------------------------------------------------------
chk.section("VWAP_open30 vs CRSP openprc (random sample)")
samp = g.sample(min(5000, len(g)), random_state=0)[
    ["permno", "date", "vwap_open30"]
]
dsf = pd.read_parquet(
    cfg.RAW_DIR / "crsp_dsf.parquet",
    columns=["permno", "date", "openprc"],
)
joined = samp.merge(dsf, on=["permno", "date"], how="left").dropna(
    subset=["vwap_open30", "openprc"]
)
if joined.empty:
    chk.is_true("VWAP sample joinable to CRSP openprc", False,
                "merged frame is empty -- check permno/date types")
else:
    rel_err = ((joined["vwap_open30"] - joined["openprc"]) / joined["openprc"]).abs()
    chk.between("median |VWAP - openprc| / openprc",
                rel_err.median(), 0.0, 0.02)
    chk.between("90th-pct |VWAP - openprc| / openprc",
                rel_err.quantile(0.90), 0.0, 0.10)
    chk.note(f"sample size: {len(joined):,}  (matched out of {len(samp):,})")

chk.summary()
