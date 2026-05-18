"""
Validate the CRSP DSF / MSF / stocknames pulls.

Calibration of expected ranges (1993-2024 sample):
    DSF rows           ~ 60-80 M  (32 yr * ~250 trading days * ~7800 stocks)
    MSF rows           ~ 2.8-3.5M (32 yr * 12 months * ~7800 stocks)
    STOCKNAMES rows    ~ 70-90k   (one row per (permno, name period))
    monthly permno count peaks around 1999-2000 (~9500-10000) and falls
                       toward ~4500 by 2024.

Critical AAPL check:
    CRSP 'prc' is the *raw* (unadjusted) close.  AAPL split 4:1 on
    2020-08-31, so the raw price on 2020-01-02 was ~$300 and the
    split-adjusted price (= prc / cfacpr) is ~$75.  We test BOTH, since
    they confirm two different aspects of the data.

The strongest data-integrity check is the value-weighted market index
rebuilt from DSF vs FF Mkt+RF: correlation should be > 0.99 if prc/
shrout/ret are all clean.
"""
from importlib import import_module
cfg = import_module("00_config")
from _validate import Checker

import pandas as pd
import numpy as np

chk = Checker("Phase 3 — CRSP")
chk.require_files(
    [cfg.RAW_DIR / "crsp_dsf.parquet",
     cfg.RAW_DIR / "crsp_msf.parquet",
     cfg.RAW_DIR / "crsp_stocknames.parquet",
     cfg.RAW_DIR / "crsp_dsedelist.parquet",
     cfg.RAW_DIR / "ff_daily.parquet"],
    hint="Run Phase 2 (02_download_ff_factors.py) and Phase 3 (03_download_crsp.py) first.",
)
dsf  = pd.read_parquet(cfg.RAW_DIR / "crsp_dsf.parquet",
                       columns=["permno","date","prc","ret","shrout","openprc","cfacpr"])
msf  = pd.read_parquet(cfg.RAW_DIR / "crsp_msf.parquet",
                       columns=["permno","date","prc","ret","shrout"])
stk  = pd.read_parquet(cfg.RAW_DIR / "crsp_stocknames.parquet",
                       columns=["permno","namedt","nameenddt","shrcd","exchcd"])
dlst = pd.read_parquet(cfg.RAW_DIR / "crsp_dsedelist.parquet")

# -------- 1. row counts (calibrated to 1993-2024 sample size) --------
chk.at_least("DSF rows",        len(dsf),  50_000_000)
chk.at_least("MSF rows",        len(msf),   2_500_000)
chk.at_least("STOCKNAMES rows", len(stk),      50_000)

# -------- 2. permno coverage over time --------
n_per_month = msf.groupby(msf["date"].dt.to_period("M"))["permno"].nunique()
chk.section("monthly active permnos")
chk.between("median monthly permnos", int(n_per_month.median()), 4000, 9000)
chk.between("max   monthly permnos", int(n_per_month.max()),    7000, 11000)
chk.between("min   monthly permnos", int(n_per_month.min()),    3000, 8000)

# -------- 3. shrcd / exchcd distributions (latest snapshot per permno) --------
latest = stk.sort_values("namedt").drop_duplicates("permno", keep="last")
shrcd_share  = latest["shrcd"].isin([10, 11]).mean()
exchcd_share = latest["exchcd"].isin([1, 2, 3]).mean()
chk.between("share of permnos with shrcd in {10,11}", shrcd_share,  0.55, 0.95)
chk.between("share of permnos on NYSE/AMEX/NASDAQ",   exchcd_share, 0.85, 1.0)

# -------- 4. AAPL spot check (PERMNO 14593) --------
# Two assertions on the same row:
#   (a) raw prc ~ $300 (AAPL traded at ~$300 on 2020-01-02 PRE-split)
#   (b) split-adjusted prc/cfacpr ~ $75 (matches Yahoo's adjusted view)
chk.section("AAPL (permno 14593) spot checks")
aapl = dsf[(dsf["permno"] == 14593) & (dsf["date"] == pd.Timestamp("2020-01-02"))]
if aapl.empty:
    chk.is_true("AAPL 2020-01-02 row exists", False, "no row found")
else:
    raw       = abs(aapl["prc"].iloc[0])
    cfacpr    = aapl["cfacpr"].iloc[0]
    adjusted  = raw / cfacpr if cfacpr else None
    chk.between("AAPL 2020-01-02 raw prc",      raw,      280.0, 320.0)
    chk.between("AAPL 2020-01-02 split-adj prc", adjusted, 70.0,  80.0)
    chk.note(f"raw={raw:.2f}  cfacpr={cfacpr:.4f}  adjusted={adjusted:.2f}")

# -------- 5. VW market reconstructed from DSF vs FF --------
chk.section("VW-CRSP rebuilt from DSF vs FF (single-year sample)")
ff = pd.read_parquet(cfg.RAW_DIR / "ff_daily.parquet")
sub = dsf[(dsf["date"] >= "2010-01-01") &
          (dsf["date"] <= "2010-12-31")].dropna(subset=["ret"]).copy()
sub["mcap"] = sub["prc"].abs() * sub["shrout"]
sub = sub.sort_values(["permno", "date"])
sub["mcap_lag"] = sub.groupby("permno")["mcap"].shift(1)
sub = sub.dropna(subset=["mcap_lag"])

# Vectorised value-weight: sum(w*r) / sum(w) per date — no apply()
sub["w_ret"] = sub["ret"] * sub["mcap_lag"]
num = sub.groupby("date")["w_ret"].sum()
den = sub.groupby("date")["mcap_lag"].sum()
vw = (num / den).rename("vw_ret")

ff_2010 = ff[(ff["date"] >= "2010-01-01") & (ff["date"] <= "2010-12-31")].copy()
ff_2010["mkt_total"] = ff_2010["mktrf"] + ff_2010["rf"]
joined = vw.to_frame().merge(
    ff_2010[["date", "mkt_total"]], left_index=True, right_on="date")
corr = joined["vw_ret"].corr(joined["mkt_total"])
chk.between("corr(rebuilt VW, FF Mkt) — 2010 sample", corr, 0.95, 1.001)

# -------- 6. CRSP dsedelist (used by 09 for B-M-P imputation) --------
chk.section("CRSP delisting events (dsedelist)")
# CRSP dlstcd: 100=still listed, 200-299=merger, 300-399=exchange/issue
# 400-499=going private/transfer, 500+=severe (bankruptcy, liquidation,
# regulatory).  09_clean_sample.py imputes -30% (NYSE/AMEX) / -55%
# (NASDAQ) only when dlstcd >= 500 AND dlret is NaN.
chk.equal("dsedelist columns",
          set(dlst.columns),
          {"permno", "dlstdt", "dlstcd", "dlret", "dlretx"})
chk.at_least("dsedelist rows", len(dlst), 10_000)
chk.between("dsedelist rows", len(dlst), 10_000, 100_000)

n_severe = int((dlst["dlstcd"] >= 500).sum())
chk.at_least("severe delistings (dlstcd >= 500)", n_severe, 1_000)
share_severe = n_severe / max(len(dlst), 1)
chk.between("severe delist share", share_severe, 0.05, 0.70)

# Among severe delistings, fraction with missing dlret -- B-M-P target.
sev = dlst[dlst["dlstcd"] >= 500]
nan_rate_severe = sev["dlret"].isna().mean() if len(sev) else 0.0
chk.between("dlret NaN rate among severe delists", nan_rate_severe, 0.05, 0.95)
chk.note(f"severe delists: {n_severe:,}  |  NaN-dlret share: {nan_rate_severe:.1%}")

# Date coverage -- dsedelist should span CRSP history through last year.
chk.is_true("dsedelist covers 2010+",
            dlst["dlstdt"].max() >= pd.Timestamp("2010-01-01"))
chk.is_true("dsedelist reaches back to <= 2000",
            dlst["dlstdt"].min() <= pd.Timestamp("2000-01-01"))

# Sanity: dlstcd top categories should include 100 (still listed) and
# 5xx (severe delisting).  Print top 5 codes as info.
top_codes = dlst["dlstcd"].value_counts(dropna=False).head(5)
for code, n in top_codes.items():
    chk.note(f"dlstcd={code}: {n:,}")

chk.summary()
