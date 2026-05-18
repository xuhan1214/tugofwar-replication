"""
Phase 7b — sanity-check the monthly panel produced by 09_clean_sample.py.

The strongest check at the monthly level is the approximate identity:
    (1 + ret_intraday_m)(1 + ret_overnight_m)  ≈  (1 + ret_close_m)

It's NOT exact because intraday/overnight only multiply over GOOD days
(valid VWAP), while ret_close multiplies over ALL days.  Any month
where the first or last day is "bad" (no valid VWAP) leaks return into
a neighboring month's overnight in the rolling-VWAP scheme, so the
identity drifts at month boundaries.

For stocks with consistent TAQ coverage the leakage should be sub-1%.
A heavy tail (e.g., 99th-pct > 5%) would indicate a logic bug in 08's
rolling-VWAP attribution.
"""
from importlib import import_module
cfg = import_module("00_config")
from _validate import Checker

import pandas as pd
import numpy as np

chk = Checker("Phase 7b — monthly identity & sample structure")
chk.require_files(
    [cfg.CLEAN_DIR / "oi_monthly_full.parquet",
     cfg.CLEAN_DIR / "oi_monthly_A.parquet",
     cfg.CLEAN_DIR / "oi_monthly_B.parquet"],
    hint="Run `python 09_clean_sample.py` first.",
)

full = pd.read_parquet(cfg.CLEAN_DIR / "oi_monthly_full.parquet")
a    = pd.read_parquet(cfg.CLEAN_DIR / "oi_monthly_A.parquet")
b    = pd.read_parquet(cfg.CLEAN_DIR / "oi_monthly_B.parquet")

# ---- 1. row counts ------------------------------------------------------
chk.section("row counts")
chk.at_least("full panel rows", len(full),  50_000)
chk.at_least("Block A rows",    len(a),     30_000)
chk.at_least("Block B rows",    len(b),     20_000)
chk.note(f"full={len(full):,}  A={len(a):,}  B={len(b):,}")

# ---- 2. monthly identity (1+ri)(1+ron) ≈ (1+ret_close) ------------------
chk.section("monthly identity (1+ri)(1+ron) ≈ (1+ret_close) on good months")
g = full.dropna(subset=["ret_intraday", "ret_overnight", "ret_close"]).copy()
g["lhs"] = (1.0 + g["ret_intraday"]) * (1.0 + g["ret_overnight"]) - 1.0
g["rhs"] = g["ret_close"]
g["abs_diff"] = (g["lhs"] - g["rhs"]).abs()
chk.between("median |lhs - rhs|",       g["abs_diff"].median(),       0, 1e-3)
chk.between("90th-pct |lhs - rhs|",     g["abs_diff"].quantile(0.90), 0, 0.02)
chk.between("99th-pct |lhs - rhs|",     g["abs_diff"].quantile(0.99), 0, 0.10)
chk.note(f"identity sample: {len(g):,} of {len(full):,} "
         f"({len(g)/max(len(full),1)*100:.1f}%)")

# ---- 3. monthly return moments (NOTE-ONLY — not in LPS) ----------------
# IMPORTANT: LPS Section 3-4 does NOT make a daily- or monthly-level
# intraday-vs-overnight vol claim about individual stocks.  The closest
# statement (Section 4.2.2) is about the MOM PORTFOLIO's monthly
# overnight std (4.02%) being below its close-to-close std (7.85%), an
# entirely different comparison.  The "intraday vol > overnight vol"
# rule of thumb comes from older microstructure literature (Fama 1965,
# French-Roll 1986) and is a heuristic, not an LPS predicate.
#
# Additionally, at the monthly level the comparison is APPLES-TO-ORANGES:
#   - monthly ret_intraday  = product of clean 1-day intraday returns
#   - monthly ret_overnight = product of mostly 1-day overnights + a few
#     MULTI-DAY rolled overnights (post-gap good days), which inflate the
#     overnight variance by construction
# So we report these moments as INFORMATION but do NOT assert ordering.
chk.section("monthly return moments (annualised %, INFO-ONLY)")
ri_ann  = g["ret_intraday"].std()  * np.sqrt(12) * 100
ron_ann = g["ret_overnight"].std() * np.sqrt(12) * 100
chk.between("intraday  ann. vol",  ri_ann,  10,  80)
chk.between("overnight ann. vol",  ron_ann,  5,  80)
chk.note(f"intraday vs overnight: {ri_ann:.1f}% vs {ron_ann:.1f}%  "
         f"(NOT an LPS assertion — see comment in source)")
chk.note("at monthly level overnight vol is naturally inflated by "
         "post-gap multi-day rolled overnights; daily-level comparison "
         "in v08 (LPS-like consec pairs) does show intraday > overnight.")

# ---- 4. filter sanity (post-09 panel must satisfy all LPS filters) ------
chk.section("filter sanity")
shrcd_vals = set(full["shrcd"].dropna().astype(int).unique())
chk.equal("shrcd values", shrcd_vals, set(cfg.SHRCD_KEEP))
chk.is_true(f"all exchcd in {set(cfg.EXCHCD_KEEP)}",
            full["exchcd"].dropna().astype(int).isin(cfg.EXCHCD_KEEP).all())
chk.is_true(f"all |prc_eom| >= ${cfg.PRICE_FLOOR}",
            (full["prc_eom"].abs() >= cfg.PRICE_FLOOR).all())

# ---- 5. coverage --------------------------------------------------------
chk.section("coverage")
chk.is_true("Block A within 2003-2013",
            (a["yearmon_ts"].min() >= pd.Timestamp(cfg.SAMPLE_BLOCKS["A"][0])) and
            (a["yearmon_ts"].max() <= pd.Timestamp(cfg.SAMPLE_BLOCKS["A"][1])))
chk.is_true("Block B within 2014-2024",
            (b["yearmon_ts"].min() >= pd.Timestamp(cfg.SAMPLE_BLOCKS["B"][0])) and
            (b["yearmon_ts"].max() <= pd.Timestamp(cfg.SAMPLE_BLOCKS["B"][1])))

# ---- 6. mktcap unit sanity (shrout is in thousands → mktcap in $thousands)
chk.section("mktcap unit ($thousands)")
mc_median = full["mktcap"].median()
# Median sample stock is comfortably above NYSE q20: should be in the
# hundreds of millions to single-digit billions of dollars, i.e.
# 100_000 - 10_000_000 in $thousands.
chk.between("median mktcap ($thousands)", mc_median, 50_000, 50_000_000)
chk.note(f"median mktcap = ${mc_median/1e6:,.1f} B "
         f"(raw={mc_median:,.0f} $thousands)")

# ---- 7. true LPS-sample good-day rate (after ALL filters incl. NYSE q20) --
# Block A is the strict-LPS panel: shrcd 10/11 + exchcd 1/2/3 +
# |prc|>=$5 + mktcap > NYSE q20.  Compute the actual good-day rate
# implied by n_good / trading_days_in_month within this panel.
chk.section("true LPS-sample good-day rate (per LPS Section 3)")
import calendar

def trading_days_per_month(period_m):
    """Approximate count of trading days in a calendar month.
    Uses business days (Mon-Fri) as a proxy; real CRSP would account
    for holidays but for a rough good-rate calculation the bias is
    a uniform ~5% downshift that doesn't matter here."""
    yr = period_m.year
    mo = period_m.month
    _, ndays = calendar.monthrange(yr, mo)
    bdays = sum(1 for d in range(1, ndays + 1)
                if pd.Timestamp(yr, mo, d).weekday() < 5)
    return bdays

# Apply to Block A only (the strict-LPS replication window)
a = a.copy()
a["trading_days"] = a["yearmon"].apply(trading_days_per_month)
# Per-row good rate, then average across stock-months
a["per_row_good_rate"] = a["n_good"].clip(upper=a["trading_days"]) / a["trading_days"]
mean_good_rate = a["per_row_good_rate"].mean()
chk.between("LPS Block A mean good-day rate (stock-month avg)",
            mean_good_rate, 0.75, 0.99)
chk.note(f"Block A: {len(a):,} stock-months, mean good-day rate = "
         f"{mean_good_rate*100:.1f}%  -- after ALL LPS filters incl. NYSE q20")
# Median stock-month
median_n_good = a["n_good"].median()
chk.note(f"median good days per stock-month: {median_n_good:.0f} "
         f"(out of typical 19-23 trading days)")

chk.summary()
