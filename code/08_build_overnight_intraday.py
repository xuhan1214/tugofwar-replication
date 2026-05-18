"""
Decompose every (permno, date) close-to-close return into overnight +
intraday components, per LPS Section 3:

    r_intraday(t)  = P_close(t) / VWAP_open30(t) - 1
    r_overnight(t) = (1 + r_close_to_close(t)) / (1 + r_intraday(t)) - 1

so that (1 + r_intraday)(1 + r_overnight) = 1 + r_close_to_close by
construction.

Missing-VWAP handling (LPS rule, Section 3): if VWAP is missing on day
t we hold the overnight leg from the previous valid VWAP day's close to
the next valid VWAP day, and the intraday/overnight split for the
intervening days is left undefined.

Implementation uses cumulative log returns so the rolled close-to-close
return is exactly product(1+ret_s) over the rolled window.

Memory note: this script loads the full ~3 GB CRSP DSF and ~1 GB TAQ
daily into RAM and runs a single merge + groupby cumsum.  Peak RAM ~8
GB for a 16 GB Mac.  If you hit OOM, refactor to chunk by permno_mod_N.

Output: clean/oi_daily.parquet  (permno, date, prc, ret, vol, shrout,
        vwap_open30, vol_open30, ret_intraday, ret_overnight, good)
"""

from importlib import import_module
cfg = import_module("00_config")

import numpy as np
import pandas as pd


def main():
    print("loading CRSP daily ...")
    # NOTE: `openprc` is intentionally NOT loaded here — LPS uses TAQ
    # VWAP_open30, not the CRSP first-trade open price.  openprc is
    # only useful for v08's external sanity check (VWAP vs CRSP open),
    # which loads it separately.
    dsf = pd.read_parquet(
        cfg.RAW_DIR / "crsp_dsf.parquet",
        columns=["permno", "date", "prc", "ret",
                 "vol", "shrout", "cfacpr"],
    )
    dsf["permno"] = dsf["permno"].astype("int32")
    dsf["prc"] = dsf["prc"].abs()
    dsf = dsf.sort_values(["permno", "date"]).reset_index(drop=True)

    print("loading TAQ daily ...")
    taq = pd.read_parquet(
        cfg.CLEAN_DIR / "taq_daily.parquet",
        columns=["permno", "date", "vwap_open30", "vol_open30"],
    )
    taq["permno"] = taq["permno"].astype("int32")

    df = dsf.merge(taq, on=["permno", "date"], how="left")

    df["ret_intraday"] = df["prc"] / df["vwap_open30"] - 1.0

    # SANITY FILTER on VWAP data quality (industry standard, beyond LPS's
    # explicit 1000-share rule).  Some thinly-traded stocks have valid
    # vol_open30 >= 1000 but VWAP_open30 is corrupted by 1-2 odd-lot /
    # stale-quote prints in the first 30 min, producing |ret_intraday|
    # values of 60-1000% on days when the actual close-to-close return
    # is near 0.  Empirically 99.9% of legitimate daily ret_intraday is
    # |x| < 25%; observations with |ret_intraday| > 100% are virtually
    # all VWAP data errors and would distort decile sorts if kept.
    # We mark these as bad days, letting their close-to-close return
    # roll into the next clean overnight (same treatment as missing VWAP).
    INTRADAY_ABS_CAP = 1.0   # 100% — generous bound, drops <0.05% of rows

    df["good"] = (
        df["vwap_open30"].notna()
        & (df["vol_open30"] >= cfg.MIN_SHARES_FIRST_HALF_HOUR)
        & df["ret_intraday"].notna()
        & (df["ret_intraday"].abs() < INTRADAY_ABS_CAP)
    )

    # fillna(0) before log1p: a NaN daily ret (trading halt, missing day)
    # would otherwise propagate through cumsum and break ret_overnight
    # for every subsequent row of that permno.  Treating a NaN day as
    # zero-return is a mild approximation that keeps the rolling chain
    # intact; the alternative (NaN propagation) silently zeroes-out
    # the panel after the first halt.  Delisting-day returns are
    # handled at the monthly level via B-M-P imputation in 09.
    df["log1p_ret"] = np.log1p(df["ret"].fillna(0.0).clip(lower=-0.999))
    df["cum_logret"] = df.groupby("permno")["log1p_ret"].cumsum()

    cum_at_good = df["cum_logret"].where(df["good"])
    df["cum_logret_prev_good"] = cum_at_good.groupby(df["permno"]).shift(1)
    df["cum_logret_prev_good"] = df.groupby("permno")["cum_logret_prev_good"].ffill()

    df["ret_rolled"] = np.expm1(df["cum_logret"] - df["cum_logret_prev_good"])

    df["ret_overnight"] = np.where(
        df["good"],
        (1.0 + df["ret_rolled"]) / (1.0 + df["ret_intraday"]) - 1.0,
        np.nan,
    )

    out = df[["permno", "date", "prc", "ret", "vol", "shrout",
              "vwap_open30", "vol_open30",
              "ret_intraday", "ret_overnight", "good"]]
    cfg.atomic_write_parquet(out, cfg.CLEAN_DIR / "oi_daily.parquet", index=False)
    print(f"oi_daily.parquet: {len(out):,} rows "
          f"({out['good'].mean()*100:.1f}% good days)")


if __name__ == "__main__":
    main()
