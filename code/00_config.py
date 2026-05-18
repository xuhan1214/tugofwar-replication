"""
Project-wide configuration.  Edit PROJECT_DIR if the project sits
elsewhere than ~/Desktop/tugofwar/.

Block A = 2003-10 ~ 2013-12 = clean overlap window with Lou-Polk-
                              Skouras (2019).  Paper covers 1993-2013
                              but our TAQ Millisecond subscription
                              only starts 2003-09-10, so the only
                              feasible overlap is from 2003 onward.

                              Why 2003-10 (not 2003-01 or 2003-09)?
                              - 2003-01 ~ 2003-08: zero TAQ coverage
                                (msec product had not launched yet) →
                                100% bad days, no decompositions usable.
                              - 2003-09: partial month (TAQ launches
                                on the 10th), bad-day rate inflated to
                                ~48% from the missing Sep 2-9 dates.
                              - 2003-10 onward: bad-day rate ~26%,
                                indistinguishable from 2004.  Picking
                                2003-10 as the start gives the cleanest
                                Block A with the largest viable sample.

Block B = 2014-2024 = out-of-sample extension.
"""

from pathlib import Path

PROJECT_DIR = Path("~/Desktop/tugofwar").expanduser()

CODE_DIR  = PROJECT_DIR / "code"
RAW_DIR   = PROJECT_DIR / "data" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
TAQ_DIR   = PROJECT_DIR / "data" / "wrds_taq"
LOG_DIR   = PROJECT_DIR / "logs"
FIG_DIR   = PROJECT_DIR / "figures"

for d in (RAW_DIR, CLEAN_DIR, TAQ_DIR, LOG_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_BLOCKS = {
    "A": ("2003-10-01", "2013-12-31"),
    "B": ("2014-01-01", "2024-12-31"),
}
# We still pull CRSP/Compustat/IBES/FF back to 1993 for flexibility,
# but TAQ aggregation starts 2003 (see TAQ_START_YEAR below).
FULL_START, FULL_END = "1993-01-01", "2024-12-31"
TAQ_START_YEAR, TAQ_END_YEAR = 2003, 2024

SHRCD_KEEP  = (10, 11)
EXCHCD_KEEP = (1, 2, 3)
PRICE_FLOOR = 5.0
MIN_SHARES_FIRST_HALF_HOUR = 1000

NW_LAGS_MONTHLY = 12


def get_wrds_connection():
    import wrds
    return wrds.Connection()


TRANSIENT_ERR_MARKERS = (
    "timed out", "operation timed out", "could not receive data",
    "ssl syscall", "connection", "broken pipe", "server closed",
    "eof detected",
)


def run_query_with_retry(db, sql, *, date_cols=None,
                         max_retries=4, sleep_base=15):
    """Run db.raw_sql(sql, date_cols=...) with reconnect + exponential
    backoff on transient WRDS network errors.

    Returns (df, db).  `db` may be a freshly reconnected handle after a
    retry, so callers should rebind.  Non-transient errors (bad SQL,
    missing column, permission denied) re-raise immediately.
    """
    import time
    for attempt in range(max_retries):
        try:
            return db.raw_sql(sql, date_cols=date_cols), db
        except Exception as e:
            err = str(e).lower()
            is_transient = any(m in err for m in TRANSIENT_ERR_MARKERS)
            if is_transient and attempt < max_retries - 1:
                wait = sleep_base * (attempt + 1)
                print(f"  transient WRDS error (try {attempt+1}/{max_retries}), "
                      f"sleep {wait}s and reconnect ...")
                try:
                    db.close()
                except Exception:
                    pass
                time.sleep(wait)
                db = get_wrds_connection()
                continue
            raise
    raise RuntimeError("persistent WRDS failure after all retries")


def atomic_write_parquet(df, path, **kwargs):
    """Write df to <path>.tmp then os.replace to path.

    A single df.to_parquet(path) is NOT atomic: if the process crashes
    mid-write you can end up with a partial / unreadable file that
    later `if path.exists()` skip-if-exists checks will mistake for a
    complete cache.  Writing to a sibling .tmp file and atomically
    renaming on success guarantees that path either does not exist
    (fresh) or is the fully-written result.

    Any stale .tmp file left by a prior crash is removed before the
    new write.
    """
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, **kwargs)
    os.replace(tmp, path)
