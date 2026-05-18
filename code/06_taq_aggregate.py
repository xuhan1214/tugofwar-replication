"""
Server-side TAQ aggregation.  For each stock-day we compute:

    pv_open30   sum(price * size)  over 09:30:00-10:00:00
    vol_open30  sum(size)          over 09:30:00-10:00:00
    vwap_open30 = pv_open30 / vol_open30  (derived after concat)
    bucket_dvol_0 .. bucket_dvol_12   dollar volume in 13 half-hour buckets

Two TAQ formats are handled:

    1993-2002   taq.ct_<YYYYMM>          one query per month, seconds
    2003-now    taqmsec.ctm_<YYYYMMDD>   one query per day,   millisec/usec

Trade-condition filter
----------------------
LPS Section 3 says they INCLUDE the open auction in the 9:30 bucket.
We therefore use a permissive filter: drop only sale conditions known
to mark non-continuous, late, derivatively priced, or correction trades,
and drop CORR != 0 (post-trade corrections).  We KEEP 'O' (opening
trade) and blank/regular conditions.

Pre-2007 'cond' bad set:   C  G  L  N  P  R  T  W  Z   (and 4..9)
Post-2007 'tr_scond' bad:  G  L  Q  R  T  W  Z  X  4  5  9

Output is one parquet per year in data/wrds_taq/.  Each year is cached;
re-running the script skips already-finished years.

Usage
-----
    python 06_taq_aggregate.py            # full sample, 1993-2024
    python 06_taq_aggregate.py 2003 2013  # one block
"""

from importlib import import_module
cfg = import_module("00_config")

import sys
import time
from datetime import date, timedelta
import pandas as pd


# Trade-condition filter — based on TAQ Daily Spec v4.0, WRDS IID
# Manual v2.0 (page 38) and user review (2026-05-11).
#
# KEEP (real trade prices contributing to VWAP):
#   NULL, ' ', '@'    Regular trade
#   'E'               Automatic Execution (CTS)
#   'F'               Intermarket Sweep
#   'H'               Price Variation Trade
#   'I'               CAP Election (CTS) / Odd Lot (UTP)
#   'K'               Rule 127/155 Trade
#   'O'               Market Center Opening Trade  (real open)
#   '5'               Market Center Re-Opening Trade
#   '6'               Closing Trade (real close)
#   '1'               Stopped Stock - Regular (UTP)
#   'B'               Bunched Trade (UTP)   note: CTS B = Avg Price
#                                           (acknowledged trade-off)
#   'S'               Split Trade (UTP)
#   'Y'               Yellow Flag (UTP)
#
# DROP (status duplicates, late reports, non-market prices):
#   'Q'               status duplicate of 'O'  (WRDS IID Manual p38)
#   'M'               status duplicate of '6'  (WRDS IID Manual p38)
#   'L'               Sold Last (late)
#   'P'               Prior Reference Price (late)
#   'Z'               Sold (Out of Sequence)
#   'T'               Extended Hours / Form-T
#   'U'               Extended Hours OOS
#   'W'               Average Price Trade (UTP)
#   'A','D'           Acquisition / Distribution (UTP, special)
#   'G'               Bunched Sold (UTP) / Opening Detail Discontinued (CTS)
#   'C','N','R'       Cash / Next Day / Seller (special clearing)
#   'V','X'           Contingent / Cross (special)
#   '4'               Derivatively Priced
#   '7'               Qualified Contingent Trade
#   '9'               Corrected Consolidated Close
BAD_COND = "'A','C','D','G','L','M','N','P','Q','R','T','U','V','W','X','Z','4','7','9'"
LEGACY_BAD_COND = BAD_COND
MSEC_BAD_COND   = BAD_COND

# KNOWN LIMITATION (issue #6 in pipeline review, intentionally deferred).
# `tr_scond NOT IN ('Z', ...)` below compares the WHOLE tr_scond string,
# so multi-character codes (e.g., 'CZ' = Cash + Out-of-Sequence, 'ZE')
# slip through even though they contain a bad single-char code that
# should disqualify the trade.  Per the 01c diagnostic these multi-char
# conds are <0.01% of trades on a typical day, so the impact on the
# stock-day aggregates is negligible.  A correct fix would be either
# `LEFT(tr_scond, 1) NOT IN (...)` (cheap) or a regex (precise) — both
# require a full re-aggregation of every year (~100h of WRDS time),
# which is not justified at this bias magnitude.

# tr_corr filter: keep only '00' (no correction) and '12' (correction
# record that supersedes a '01' original). DROP all error / cancel /
# stale rows.
GOOD_CORR = "'00', '12'"

BUCKET_BOUNDS = [
    "09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00",
    "12:00:00", "12:30:00", "13:00:00", "13:30:00", "14:00:00",
    "14:30:00", "15:00:00", "15:30:00", "16:00:01",
]


def _bucket_sql(time_col):
    parts = []
    for i in range(13):
        lo, hi = BUCKET_BOUNDS[i], BUCKET_BOUNDS[i + 1]
        parts.append(
            f"SUM(CASE WHEN {time_col} >= '{lo}' AND {time_col} < '{hi}' "
            f"THEN price*size ELSE 0 END) AS bucket_dvol_{i}"
        )
    return ",\n            ".join(parts)


def query_legacy_month(db, yyyymm):
    # Legacy TAQ has only `symbol` (no sym_suffix) — alias to a unified
    # (sym_root, sym_suffix) schema so the link merge is consistent.
    sql = f"""
        SELECT
            symbol AS sym_root,
            ''     AS sym_suffix,
            date,
            SUM(CASE WHEN time >= '09:30:00' AND time < '10:00:00'
                     THEN price*size ELSE 0 END) AS pv_open30,
            SUM(CASE WHEN time >= '09:30:00' AND time < '10:00:00'
                     THEN size       ELSE 0 END) AS vol_open30,
            {_bucket_sql("time")}
        FROM   taq.ct_{yyyymm}
        WHERE  price > 0 AND size > 0
          AND  COALESCE(cond, '') NOT IN ({LEGACY_BAD_COND})
          AND  COALESCE(corr, 0) IN (0, 12)
          AND  time >= '09:30:00' AND time < '16:00:01'
        GROUP BY symbol, date
    """
    return db.raw_sql(sql, date_cols=["date"])


def query_msec_day(db, yyyymmdd):
    sql = f"""
        SELECT
            sym_root,
            sym_suffix,
            DATE '{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}' AS date,
            SUM(CASE WHEN time_m >= '09:30:00' AND time_m < '10:00:00'
                     THEN price*size ELSE 0 END) AS pv_open30,
            SUM(CASE WHEN time_m >= '09:30:00' AND time_m < '10:00:00'
                     THEN size       ELSE 0 END) AS vol_open30,
            {_bucket_sql("time_m")}
        FROM   taqmsec.ctm_{yyyymmdd}
        WHERE  price > 0 AND size > 0
          AND  COALESCE(tr_scond, '') NOT IN ({MSEC_BAD_COND})
          AND  COALESCE(tr_corr, '00') IN ({GOOD_CORR})
          AND  time_m >= '09:30:00' AND time_m < '16:00:01'
        -- `date` is a SQL literal (DATE '{yyyymmdd}'), not a grouped
        -- column, so we group only by ticker columns.  Each query is
        -- scoped to a single ctm_<YYYYMMDD> table → one date by
        -- construction.
        GROUP BY sym_root, sym_suffix
    """
    try:
        return db.raw_sql(sql, date_cols=["date"])
    except Exception as e:
        if "does not exist" in str(e).lower():
            return None
        raise


def trading_days_in_month(year, month):
    """Yield weekdays in a (year, month)."""
    d = date(year, month, 1)
    next_month = date(year + (month // 12), (month % 12) + 1, 1)
    end = next_month - timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


TRANSIENT_ERR_MARKERS = (
    "timed out", "operation timed out", "could not receive data",
    "ssl syscall", "connection", "broken pipe", "server closed",
    "eof detected",
)


def query_msec_day_robust(db, yyyymmdd, max_retries=4, sleep_base=15):
    """Run query_msec_day with auto-reconnect + exponential backoff on
    transient network errors. Returns (df_or_None, db) — `db` may be a
    fresh connection after a retry."""
    for attempt in range(max_retries):
        try:
            return query_msec_day(db, yyyymmdd), db
        except Exception as e:
            err = str(e).lower()
            if "does not exist" in err:
                return None, db
            is_transient = any(m in err for m in TRANSIENT_ERR_MARKERS)
            if is_transient and attempt < max_retries - 1:
                wait = sleep_base * (attempt + 1)
                print(f"  {yyyymmdd}: transient error (try {attempt+1}/{max_retries}), "
                      f"sleep {wait}s and reconnect ...")
                try:
                    db.close()
                except Exception:
                    pass
                time.sleep(wait)
                db = cfg.get_wrds_connection()
                continue
            raise
    raise RuntimeError(f"persistent failure on {yyyymmdd}")


def run_year(db, yr):
    """Aggregate one year, with month-level checkpoints so crashes
    don't lose more than ~1 month of work."""
    out = cfg.TAQ_DIR / f"taq_agg_{yr}.parquet"
    if out.exists():
        print(f"  {yr}: cached, skip.")
        return db

    # Legacy era — single monthly query produces one chunk anyway
    if yr < 2003:
        pieces = []
        for mm in range(1, 13):
            ym = f"{yr}{mm:02d}"
            t0 = time.time()
            try:
                df = query_legacy_month(db, ym)
            except Exception as e:
                if "does not exist" in str(e).lower():
                    print(f"  {ym}: no table (skip)")
                    continue
                raise
            print(f"  {ym}: {len(df):>8,d} rows ({time.time()-t0:.0f}s)")
            pieces.append(df)
    else:
        # Millisecond era — checkpoint per month
        monthly_pieces = []
        for month in range(1, 13):
            m_ckpt = cfg.TAQ_DIR / f"taq_agg_{yr}_M{month:02d}.parquet"
            if m_ckpt.exists():
                monthly_pieces.append(pd.read_parquet(m_ckpt))
                print(f"  {yr}-{month:02d}: checkpoint cached, skip")
                continue

            day_pieces = []
            for d in trading_days_in_month(yr, month):
                ymd = d.strftime("%Y%m%d")
                df, db = query_msec_day_robust(db, ymd)
                if df is None or df.empty:
                    continue
                day_pieces.append(df)

            if not day_pieces:
                print(f"  {yr}-{month:02d}: no data")
                continue
            m_df = pd.concat(day_pieces, ignore_index=True)
            # derive vwap_open30 here so monthly checkpoints have the
            # same schema as the eventual full-year file
            m_df["vwap_open30"] = m_df["pv_open30"] / m_df["vol_open30"].where(
                m_df["vol_open30"] > 0
            )
            cfg.atomic_write_parquet(m_df, m_ckpt, index=False)
            monthly_pieces.append(m_df)
            print(f"  {yr}-{month:02d}: saved {len(m_df):>8,d} stock-days "
                  f"({len(day_pieces)} trading days)")

        pieces = monthly_pieces

    if not pieces:
        print(f"  {yr}: NO data")
        return db

    out_df = pd.concat(pieces, ignore_index=True)
    out_df["vwap_open30"] = out_df["pv_open30"] / out_df["vol_open30"].where(
        out_df["vol_open30"] > 0
    )
    cfg.atomic_write_parquet(out_df, out, index=False)
    print(f"  {yr}: SAVED {len(out_df):,} stock-days -> {out.name}")

    # Remove monthly checkpoints now that the full year is consolidated
    for month in range(1, 13):
        m_ckpt = cfg.TAQ_DIR / f"taq_agg_{yr}_M{month:02d}.parquet"
        if m_ckpt.exists():
            m_ckpt.unlink()

    return db


def main():
    if len(sys.argv) >= 3:
        yr_lo, yr_hi = int(sys.argv[1]), int(sys.argv[2])
    else:
        yr_lo, yr_hi = cfg.TAQ_START_YEAR, cfg.TAQ_END_YEAR

    if yr_lo < 2003:
        print(f"WARNING: pre-2003 TAQ Monthly data is not available to your "
              f"WRDS account; only millisecond TAQ from 2003-09-10 onward "
              f"will be aggregated.  Setting yr_lo = 2003.")
        yr_lo = 2003

    db = cfg.get_wrds_connection()
    for yr in range(yr_lo, yr_hi + 1):
        print(f"\n=== year {yr} ===")
        db = run_year(db, yr)
    db.close()


if __name__ == "__main__":
    main()
