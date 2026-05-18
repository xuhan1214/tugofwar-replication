"""
Aggregate daily overnight/intraday returns to monthly and apply LPS
sample filters.

Monthly aggregation (good days only):
    ret_intraday_t  = prod_{good s in t}(1 + r_intraday_s) - 1
    ret_overnight_t = prod_{good s in t}(1 + r_overnight_s) - 1
    ret_close_t     = prod_{s in t}(1 + ret_s) - 1

Filters (LPS p.196):
    shrcd in {10, 11}
    exchcd in {1, 2, 3}
    month-end |prc| >= 5
    market cap STRICTLY above bottom NYSE size quintile

NYSE size breakpoint convention (textbook Fama-French): compute over
ALL NYSE common stocks BEFORE applying the price >= $5 floor, because
the price filter raises the q20 cutoff (penny NYSE stocks drag it
down).  Computing q20 on the already-filtered NYSE pool shrinks the
sample.  LPS does not specify their choice; we follow the Fama-French
convention.

Delisting adjustment (Beaver-McNichols-Price 2007):
    For severe delistings (dlstcd >= 500) where CRSP's dlret is NaN,
    impute -0.30 (NYSE/AMEX) or -0.55 (NASDAQ).  Apply imputed return
    to delisting month's ret_close and ret_overnight (preserving
    (1+ri)(1+ron) = 1+ret_close identity).

Outputs:
    clean/oi_monthly_full.parquet
    clean/oi_monthly_A.parquet  (2003-2013, strict replication window)
    clean/oi_monthly_B.parquet  (2014-2024, OOS extension)
"""

from importlib import import_module
cfg = import_module("00_config")

import numpy as np
import pandas as pd


def expand_stocknames(stk):
    """crsp.stocknames is one row per (permno, name period).  We expand
    it into a panel of [namedt, nameenddt] start points suitable for
    merge_asof(direction='backward').  Sorted by date GLOBALLY (not by
    permno) because pandas merge_asof requires the `on` column to be
    monotonically increasing across the whole frame, even when using
    `by=`."""
    stk = stk.copy()
    stk["date"] = stk["namedt"]
    return stk[["permno", "date", "nameenddt",
                "shrcd", "exchcd", "siccd"]].sort_values("date")


def attach_security_info(daily, stk):
    """As-of merge: for each (permno, date) find the most recent
    stocknames row whose [namedt, nameenddt] covers that date.  Rows
    with no matching stocknames record (or whose date falls beyond
    nameenddt) are dropped, since downstream filters need shrcd/exchcd.

    merge_asof requires BOTH inputs sorted by the `on` column (`date`)
    globally — sorting by (permno, date) is NOT enough since the date
    column then resets within each permno group.  After the asof merge
    we re-sort by (permno, date) so downstream groupby('permno','yearmon')
    aggregations preserve chronological order within each group (matters
    for `.last()` on prc_eom / shrout_eom)."""
    daily_sorted = daily.sort_values("date").copy()
    stk = expand_stocknames(stk)
    out = pd.merge_asof(
        daily_sorted, stk[["permno", "date", "nameenddt", "shrcd", "exchcd", "siccd"]],
        on="date", by="permno", direction="backward",
    )
    # Strict mask: only keep rows that have a valid covering stocknames
    # record.  merge_asof returns NaN for permno-dates with no prior
    # namedt row; we drop those rather than relying on downstream
    # shrcd.isin() to silently filter them.
    mask = out["nameenddt"].notna() & (out["date"] <= out["nameenddt"])
    out = out.loc[mask].drop(columns=["nameenddt"])
    # Re-sort by (permno, date) for downstream within-group chronology.
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    return out


def aggregate_to_monthly(df):
    df = df.copy()
    df["yearmon"] = df["date"].dt.to_period("M")
    df["one_plus_ret"]       = 1.0 + df["ret"].fillna(0.0)
    df["one_plus_intraday"]  = np.where(df["good"], 1.0 + df["ret_intraday"], np.nan)
    df["one_plus_overnight"] = np.where(df["good"], 1.0 + df["ret_overnight"], np.nan)

    grp = df.groupby(["permno", "yearmon"], sort=False)
    return grp.agg(
        ret_close      = ("one_plus_ret",      lambda s: s.prod() - 1.0),
        ret_intraday   = ("one_plus_intraday", lambda s: s.dropna().prod() - 1.0 if s.notna().any() else np.nan),
        ret_overnight  = ("one_plus_overnight",lambda s: s.dropna().prod() - 1.0 if s.notna().any() else np.nan),
        n_good         = ("good",   "sum"),
        shrcd          = ("shrcd",  "last"),
        exchcd         = ("exchcd", "last"),
        prc_eom        = ("prc",    "last"),
        shrout_eom     = ("shrout", "last"),
    ).reset_index()


def apply_bmp_delisting(monthly_panel):
    """Beaver-McNichols-Price (2007) delisting return imputation.

    Severe delistings = dlstcd >= 500 (bankruptcy, liquidation, etc.).
    For severe delistings with missing CRSP dlret:
        NYSE/AMEX  (exchcd 1, 2) : impute  -0.30
        NASDAQ     (exchcd 3)    : impute  -0.55
        other                    : impute   0
    Apply to delisting month's ret_close and ret_overnight, leaving
    ret_intraday untouched (delisting price drop is overnight-type).
    """
    dlst = pd.read_parquet(cfg.RAW_DIR / "crsp_dsedelist.parquet")
    severe = dlst[(dlst["dlstcd"] >= 500) & dlst["dlret"].isna()].copy()
    if severe.empty:
        return monthly_panel, 0

    stk = pd.read_parquet(
        cfg.RAW_DIR / "crsp_stocknames.parquet",
        columns=["permno", "namedt", "nameenddt", "exchcd"],
    )
    # merge_asof needs both inputs sorted by the merge key globally,
    # not by (permno, key) — see note in attach_security_info above.
    stk = stk.sort_values("namedt")
    severe = severe.sort_values("dlstdt")
    severe = pd.merge_asof(
        severe, stk,
        left_on="dlstdt", right_on="namedt",
        by="permno", direction="backward",
    )
    mask = severe["nameenddt"].isna() | (severe["dlstdt"] <= severe["nameenddt"])
    severe = severe.loc[mask]

    imp = np.where(severe["exchcd"].isin([1, 2]), -0.30,
          np.where(severe["exchcd"] == 3,         -0.55, 0.0))
    severe = severe.assign(
        dlret_imp=imp,
        yearmon=severe["dlstdt"].dt.to_period("M"),
    )[["permno", "yearmon", "dlret_imp"]]

    m = monthly_panel.merge(severe, on=["permno", "yearmon"], how="left")
    m["dlret_imp"] = m["dlret_imp"].fillna(0.0)
    n_adjusted = int((m["dlret_imp"] != 0).sum())

    m["ret_close"]     = (1 + m["ret_close"])     * (1 + m["dlret_imp"]) - 1
    m["ret_overnight"] = (1 + m["ret_overnight"]) * (1 + m["dlret_imp"]) - 1
    # ret_intraday unchanged - delisting price drop happens after close
    return m.drop(columns=["dlret_imp"]), n_adjusted


def apply_filters(month_panel):
    p = month_panel.copy()
    # mktcap on EVERYONE (needed for breakpoint AND for size filter).
    # shrout is in thousands of shares, so mktcap is in $thousands; the
    # NYSE q20 cutoff is also in $thousands, so the comparison below
    # is unit-consistent.
    p["mktcap"] = p["prc_eom"].abs() * p["shrout_eom"]

    # NYSE breakpoint over ALL NYSE common stocks BEFORE the price >= $5
    # filter (Fama-French textbook convention).  Computing q20 on the
    # already-filtered NYSE pool would shift the cutoff upward and
    # shrink the sample relative to the standard implementation.
    nyse_all = p[(p["exchcd"] == 1) & (p["shrcd"].isin(cfg.SHRCD_KEEP))]
    bp = (nyse_all.groupby("yearmon")["mktcap"]
                  .quantile(0.20).rename("nyse_q20").reset_index())

    # Now apply LPS sample filters
    p = p[p["shrcd"].isin(cfg.SHRCD_KEEP)]
    p = p[p["exchcd"].isin(cfg.EXCHCD_KEEP)]
    p = p[p["prc_eom"].abs() >= cfg.PRICE_FLOOR]

    # Strict >: exclude stocks at or below NYSE 20th percentile
    p = p.merge(bp, on="yearmon", how="left")
    p = p[p["mktcap"] > p["nyse_q20"]]
    return p.drop(columns=["nyse_q20"])


def main():
    print("loading oi_daily ...")
    daily = pd.read_parquet(cfg.CLEAN_DIR / "oi_daily.parquet")
    print(f"  daily rows: {len(daily):,}")

    print("loading stocknames ...")
    stk = pd.read_parquet(cfg.RAW_DIR / "crsp_stocknames.parquet")

    print("attaching shrcd/exchcd/siccd ...")
    daily = attach_security_info(daily, stk)

    print("aggregating to month ...")
    m = aggregate_to_monthly(daily)
    m["yearmon_ts"] = m["yearmon"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    print(f"  monthly rows pre-filter:  {len(m):,}")

    print("applying B-M-P delisting adjustment ...")
    m, n_bmp = apply_bmp_delisting(m)
    print(f"  rows adjusted by B-M-P: {n_bmp:,}")

    print("applying LPS filters ...")
    m = apply_filters(m)
    print(f"  monthly rows post-filter: {len(m):,}")

    a_lo, a_hi = cfg.SAMPLE_BLOCKS["A"]
    b_lo, b_hi = cfg.SAMPLE_BLOCKS["B"]
    block_a = m[(m["yearmon_ts"] >= a_lo) & (m["yearmon_ts"] <= a_hi)]
    block_b = m[(m["yearmon_ts"] >= b_lo) & (m["yearmon_ts"] <= b_hi)]
    print(f"  Block A ({a_lo[:7]} ~ {a_hi[:7]}): {len(block_a):,} stock-months")
    print(f"  Block B ({b_lo[:7]} ~ {b_hi[:7]}): {len(block_b):,} stock-months")

    cfg.atomic_write_parquet(m,       cfg.CLEAN_DIR / "oi_monthly_full.parquet", index=False)
    cfg.atomic_write_parquet(block_a, cfg.CLEAN_DIR / "oi_monthly_A.parquet",    index=False)
    cfg.atomic_write_parquet(block_b, cfg.CLEAN_DIR / "oi_monthly_B.parquet",    index=False)
    print("Saved oi_monthly_{full,A,B}.parquet")


if __name__ == "__main__":
    main()
