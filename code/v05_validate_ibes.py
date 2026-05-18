"""
v05_validate_ibes.py
====================
Validate the IBES summary + actuals + link pulls.

Reference values:
    summary rows:          1.0M - 2.0M (1993-2024, FPI=6 only)
    actuals rows:          400k - 800k
    summary tickers/quarter: ~3,500-6,500
    AAPL ticker: 'AAPL', PERMNO 14593, should be linked
"""
from importlib import import_module
cfg = import_module("00_config")
from _validate import Checker

import pandas as pd

chk = Checker("Phase 5 — IBES")
chk.require_files(
    [cfg.RAW_DIR / "ibes_summary.parquet",
     cfg.RAW_DIR / "ibes_actuals.parquet",
     cfg.RAW_DIR / "ibes_link.parquet"],
    hint="Run `python 05_download_ibes.py` first.",
)
sumu = pd.read_parquet(cfg.RAW_DIR / "ibes_summary.parquet")
actu = pd.read_parquet(cfg.RAW_DIR / "ibes_actuals.parquet")
link = pd.read_parquet(cfg.RAW_DIR / "ibes_link.parquet")

# 1. row counts
chk.at_least("summary rows",  len(sumu),  500_000)
chk.at_least("actuals rows",  len(actu),  300_000)
chk.at_least("link rows",     len(link),   15_000)

# 2. AAPL link
chk.section("AAPL IBES <-> CRSP link")
aapl_link = link[(link["ticker"] == "AAPL") & (link["permno"] == 14593)]
chk.is_true("AAPL ticker maps to permno 14593", not aapl_link.empty)

# 3. summary tickers per quarter
chk.section("IBES summary tickers per quarter")
qstats = sumu.groupby(sumu["statpers"].dt.to_period("Q"))["ticker"].nunique()
chk.between("median tickers/quarter", qstats.median(), 3000, 7000)

# 4. summary <-> actuals matchability on (ticker, fpedats==pends)
chk.section("summary -> actuals match")
key_sum = sumu[["ticker","fpedats"]].dropna().drop_duplicates()
key_act = actu[["ticker","pends"]].dropna().drop_duplicates().rename(
    columns={"pends":"fpedats"})
matched = key_sum.merge(key_act, on=["ticker","fpedats"]).shape[0]
match_rate = matched / max(len(key_sum), 1)
chk.between("share of summary keys matched in actuals", match_rate, 0.50, 1.0)

# 5. coverage range
chk.section("coverage")
chk.is_true("summary covers 2023+",
            sumu["statpers"].max() >= pd.Timestamp("2023-01-01"))
chk.is_true("actuals covers 1995-2020",
            (actu["pends"].min() <= pd.Timestamp("1996-01-01")) and
            (actu["pends"].max() >= pd.Timestamp("2020-01-01")))

chk.summary()
