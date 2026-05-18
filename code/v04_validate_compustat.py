"""
v04_validate_compustat.py
=========================
Validate the Compustat funda + fundq + ccm_link pulls.

Reference values:
    funda firm-years:        ~10,000-12,000 per year, totalling ~350,000 across 1990-2024
    fundq firm-quarters:     ~30,000-40,000 per year
    AAPL (gvkey '001690')    FY2023 datadate ≈ 2023-09-30, AT (total assets) ≈ $352B
    CCM link coverage:       >= 90% of CRSP common stocks should match a gvkey
"""
from importlib import import_module
cfg = import_module("00_config")
from _validate import Checker

import pandas as pd

chk = Checker("Phase 4 — Compustat fundamentals + CCM link")
chk.require_files(
    [cfg.RAW_DIR / "funda.parquet",
     cfg.RAW_DIR / "fundq.parquet",
     cfg.RAW_DIR / "ccm_link.parquet",
     cfg.RAW_DIR / "crsp_stocknames.parquet"],
    hint="Run Phase 3 (03_download_crsp.py) and Phase 4 (04_download_compustat.py) first.",
)
funda = pd.read_parquet(cfg.RAW_DIR / "funda.parquet")
fundq = pd.read_parquet(cfg.RAW_DIR / "fundq.parquet")
ccm   = pd.read_parquet(cfg.RAW_DIR / "ccm_link.parquet")

# 1. row counts
chk.at_least("funda rows",       len(funda),  300_000)
chk.at_least("fundq rows",       len(fundq), 1_000_000)
chk.at_least("ccm  link rows",   len(ccm),       20_000)

# 2. funda annual firm count 2010-2020 should average 8k-13k
n_per_year = funda.groupby(funda["datadate"].dt.year)["gvkey"].nunique()
chk.section("funda firms-per-year")
median_firms = int(n_per_year.loc[2000:2020].median()) if not n_per_year.empty else 0
chk.between("median firms-per-year (2000-2020)", median_firms, 7000, 14000)

# 3. AAPL spot check
# Compustat reports balance-sheet items in MILLIONS of dollars, not
# billions.  AAPL's FY2023 total assets are ~$352.6B == 352,600 ($M).
chk.section("AAPL spot check (gvkey 001690)")
aapl = funda[(funda["gvkey"] == "001690") & (funda["fyear"] == 2023)]
if aapl.empty:
    chk.is_true("AAPL FY2023 row exists", False, "no row found for fyear=2023")
else:
    at_in_millions = aapl["at"].iloc[0]
    at_in_billions = at_in_millions / 1000.0
    chk.between("AAPL FY2023 total assets ($B)", at_in_billions, 300, 400)
    chk.note(f"raw at = {at_in_millions:,.1f} ($M) = {at_in_billions:,.1f} ($B)")

# 4. CCM coverage of CRSP common stocks
chk.section("CCM link coverage of CRSP common stocks")
stk = pd.read_parquet(cfg.RAW_DIR / "crsp_stocknames.parquet",
                      columns=["permno","shrcd"])
common = (stk[stk["shrcd"].isin([10, 11])]["permno"]
              .drop_duplicates())
linked = ccm["permno"].drop_duplicates()
match_rate = common.isin(linked).mean()
chk.between("CRSP common stocks with CCM link", match_rate, 0.85, 1.0)

# 5. ccm sanity — most links should be primary (linkprim in ('P','C'))
chk.section("CCM link types")
prim_share = ccm["linkprim"].isin(["P", "C"]).mean()
chk.between("share of (P,C) links", prim_share, 0.95, 1.0)

# 6. fundq: latest quarter datadate
chk.section("fundq freshness")
last_q = fundq["datadate"].max()
chk.is_true(f"fundq has 2023+ data (last datadate={last_q.date() if pd.notna(last_q) else None})",
            last_q >= pd.Timestamp("2023-12-31"))

chk.summary()
