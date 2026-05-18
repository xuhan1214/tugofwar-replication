"""
v02_validate_ff.py
==================
Sanity-check the Fama-French factor pulls.

Reference values (1993-2024):
    monthly RF      mean ~ 0.20-0.25 % / month   (3 % annualised)
    monthly Mkt-RF  mean ~ 0.65-0.85 % / month   (~9 % annualised premium)
    monthly SMB     mean ~ 0.05-0.30 % / month   (size premium has shrunk)
    monthly HML     mean ~ 0.05-0.40 % / month
    monthly UMD     mean ~ 0.40-0.80 % / month   (momentum positive on average)
    daily rows ~ monthly rows * 21
"""
from importlib import import_module
cfg = import_module("00_config")
from _validate import Checker

import pandas as pd

chk = Checker("Phase 2 — Fama-French factors")
chk.require_files(
    [cfg.RAW_DIR / "ff_daily.parquet",
     cfg.RAW_DIR / "ff_monthly.parquet"],
    hint="Run `python 02_download_ff_factors.py` first.",
)
ffd = pd.read_parquet(cfg.RAW_DIR / "ff_daily.parquet")
ffm = pd.read_parquet(cfg.RAW_DIR / "ff_monthly.parquet")

# 1. column schema
expected_cols = {"date", "mktrf", "smb", "hml", "rf", "umd"}
chk.equal("daily columns", set(ffd.columns), expected_cols)
chk.equal("monthly columns", set(ffm.columns), expected_cols)

# 2. row counts
chk.at_least("daily rows", len(ffd), 7000)         # 1993-2024 ~ 8000
chk.at_least("monthly rows", len(ffm), 350)        # 1993-2024 ~ 384

# 3. coverage
chk.is_true("daily covers 1993-01-04",
            (ffd["date"].min() <= pd.Timestamp("1993-01-15")))
chk.is_true("daily covers >= 2023-12",
            (ffd["date"].max() >= pd.Timestamp("2023-12-01")))

# 4. economic magnitude — monthly means
chk.section("monthly factor means (%, per month)")
rf_mean   = ffm["rf"].mean() * 100
mkt_mean  = ffm["mktrf"].mean() * 100
smb_mean  = ffm["smb"].mean() * 100
hml_mean  = ffm["hml"].mean() * 100
umd_mean  = ffm["umd"].mean() * 100

chk.between("RF mean",     rf_mean,  0.10, 0.40)
chk.between("Mkt-RF mean", mkt_mean, 0.50, 1.20)
chk.between("SMB mean",    smb_mean, -0.20, 0.40)
chk.between("HML mean",    hml_mean, -0.20, 0.50)
chk.between("UMD mean",    umd_mean, 0.20, 1.00)

# 5. monthly factor *volatility* — should match well-known stylised facts
chk.section("monthly factor std (%, per month)")
chk.between("Mkt-RF std", ffm["mktrf"].std() * 100, 3.5, 6.0)   # ~4.4%
chk.between("SMB std",    ffm["smb"].std() * 100,  1.5, 4.0)    # ~2.6%
chk.between("HML std",    ffm["hml"].std() * 100,  1.5, 4.0)
chk.between("UMD std",    ffm["umd"].std() * 100,  3.0, 6.5)

# 6. daily/monthly count consistency
chk.section("daily-vs-monthly consistency")
ratio = len(ffd) / len(ffm)
chk.between("daily/monthly ratio", ratio, 19, 22)   # ~21 trading days/month

chk.summary()
