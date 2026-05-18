"""
Validate TAQ aggregate parquet files in data/wrds_taq/.

The TAQ aggregates do NOT yet have permno attached (that happens in
07_link_taq_to_crsp.py).  So v06 only validates structural / shape
properties of the aggregates themselves:

    1. row counts plausible (~2M per full year; less in 2003)
    2. 13 buckets present and dollar-volume sum is positive
    3. intraday volume distribution is U-shaped (Fig 1 of LPS)
    4. AAPL (sym_root='AAPL', sym_suffix='') has valid VWAP rows

The cross-check of our VWAP vs CRSP openprc is moved to v08, which
runs after 07 attaches permno and so we can join on (permno, date).
"""
from importlib import import_module
cfg = import_module("00_config")
from _validate import Checker

import glob
import sys
import numpy as np
import pandas as pd

files = sorted(glob.glob(str(cfg.TAQ_DIR / "taq_agg_*.parquet")))

chk = Checker("Phase 6 — TAQ aggregates")

if not files:
    chk.is_true("at least one taq_agg_<YYYY>.parquet exists", False,
                "Run `python 06_taq_aggregate.py 2003 2024` (or a year range) first.")
    chk.summary()
    sys.exit(1)

# ---------------- 1. per-year row counts ----------------
chk.section("per-year stock-day counts")
for f in files:
    yr = int(f.split("_")[-1].split(".")[0])
    n = pd.read_parquet(f, columns=["date"]).shape[0]
    if yr == 2003:
        chk.between(f"{yr} rows (partial year)", n,  50_000, 1_500_000)
    else:
        chk.between(f"{yr} rows",                n, 750_000, 4_000_000)

# ---------------- pick a recent year for deeper checks ----------------
recent_files = [f for f in files if int(f.split("_")[-1].split(".")[0]) >= 2010]
if not recent_files:
    chk.note("no >=2010 file yet -- skipping deep checks")
    chk.summary()
    sys.exit(0)

probe_path = recent_files[-1]
probe = pd.read_parquet(probe_path)
yr = int(probe_path.split("_")[-1].split(".")[0])
chk.section(f"deep checks on year {yr}")

# ---------------- 2. schema sanity ----------------
expected_cols = {"sym_root", "sym_suffix", "date",
                 "pv_open30", "vol_open30", "vwap_open30"}
missing_cols = expected_cols - set(probe.columns)
chk.is_true("expected key columns present",
            not missing_cols,
            f"missing: {missing_cols}" if missing_cols else "")
bucket_cols = sorted(
    [c for c in probe.columns if c.startswith("bucket_dvol_")],
    key=lambda c: int(c.split("_")[-1]),
)
chk.equal("13 bucket columns",                 len(bucket_cols), 13)

# ---------------- 3. intraday volume distribution shape (Fig 1) ----------------
chk.section("intraday volume distribution shape (Fig 1)")
mean_dvol = probe[bucket_cols].mean()
b0    = float(mean_dvol.iloc[0])
bL    = float(mean_dvol.iloc[-1])
b_mid = float(mean_dvol.iloc[5:8].mean())
chk.between("first bucket / mid-day",  b0 / b_mid, 1.4, 5.0)
chk.between("last  bucket / mid-day",  bL / b_mid, 1.5, 6.0)
chk.note(f"bucket means (millions $): "
         f"first={b0/1e6:,.1f}  midday={b_mid/1e6:,.1f}  last={bL/1e6:,.1f}")

# ---------------- 4. AAPL spot check ----------------
chk.section(f"AAPL spot check in {yr} (sym_root='AAPL', sym_suffix='')")
mask = (probe["sym_root"] == "AAPL") & probe["sym_suffix"].fillna("").eq("")
aapl = probe[mask].sort_values("date")
chk.at_least("AAPL trading days in this year", len(aapl), 200)
if len(aapl):
    row = aapl.iloc[0]
    chk.is_true(f"AAPL has VWAP on {row['date'].date()}",
                pd.notna(row["vwap_open30"]) and row["vwap_open30"] > 0)
    chk.between("AAPL 9:30-10:00 first-half-hour shares",
                int(row["vol_open30"]),
                10_000, 50_000_000)
    total_dvol = sum(float(row[c]) for c in bucket_cols)
    chk.between("AAPL total intraday dollar volume ($M)",
                total_dvol / 1e6,
                100, 100_000)

chk.summary()
