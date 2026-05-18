"""
Pull CRSP daily + monthly stock files, security-info history, and
delisting events.

Outputs (data/raw/):
    crsp_dsf.parquet         permno/date level daily file
    crsp_msf.parquet         permno/date level monthly file
    crsp_stocknames.parquet  security history: ticker, ncusip, shrcd,
                             exchcd, siccd with [namedt, nameenddt] range
    crsp_dsedelist.parquet   delisting events (one row per permno)

DSF is pulled in 2-year chunks and written incrementally via
pyarrow ParquetWriter so peak RAM stays small.

The four output files are independent: re-running the script skips any
output that already exists, so failures part-way through don't force a
full re-download.

All four outputs are written to <name>.parquet.tmp first and atomically
renamed on success, so a mid-write crash never leaves a half-baked file
that the skip-if-exists check would mistake for a complete cache.
Stale .tmp files from a prior crash are unlinked at the start of each
fetcher.
"""

from importlib import import_module
cfg = import_module("00_config")

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time


def _tmp_for(final_path):
    return final_path.with_suffix(final_path.suffix + ".tmp")


def _cleanup_tmp(final_path):
    tmp = _tmp_for(final_path)
    if tmp.exists():
        print(f"  removing stale tmp file: {tmp.name}")
        tmp.unlink()


def chunked_years(start, end, step=2):
    s, e = int(start[:4]), int(end[:4])
    for lo in range(s, e + 1, step):
        hi = min(lo + step - 1, e)
        yield lo, hi


def pull_dsf_chunk(db, yr_lo, yr_hi):
    sql = f"""
        SELECT  permno, permco, date,
                prc, openprc, ret, retx,
                vol, shrout, cfacshr, cfacpr
        FROM    crsp.dsf
        WHERE   date BETWEEN '{yr_lo}-01-01' AND '{yr_hi}-12-31'
    """
    df, db = cfg.run_query_with_retry(db, sql, date_cols=["date"])
    df["permno"] = df["permno"].astype("int32")
    return df, db


def pull_dsf(db, dsf_path):
    _cleanup_tmp(dsf_path)
    tmp_path = _tmp_for(dsf_path)
    writer = None
    total = 0
    try:
        for yr_lo, yr_hi in chunked_years(cfg.FULL_START, cfg.FULL_END, step=2):
            t0 = time.time()
            chunk, db = pull_dsf_chunk(db, yr_lo, yr_hi)
            total += len(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            print(f"  DSF {yr_lo}-{yr_hi}: {len(chunk):>10,d} rows  ({time.time()-t0:.0f}s)")
            del chunk, table
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp_path, dsf_path)
    print(f"DSF total: {total:,} rows  ->  {dsf_path.name}")
    return db


def pull_msf(db, msf_path):
    _cleanup_tmp(msf_path)
    tmp_path = _tmp_for(msf_path)
    sql = f"""
        SELECT  permno, permco, date,
                prc, ret, retx, vol, shrout, cfacshr, cfacpr
        FROM    crsp.msf
        WHERE   date BETWEEN '{cfg.FULL_START}' AND '{cfg.FULL_END}'
    """
    msf, db = cfg.run_query_with_retry(db, sql, date_cols=["date"])
    msf["permno"] = msf["permno"].astype("int32")
    msf.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, msf_path)
    print(f"MSF rows: {len(msf):,}")
    return db


def pull_stocknames(db, stk_path):
    _cleanup_tmp(stk_path)
    tmp_path = _tmp_for(stk_path)
    # Discover columns that actually exist in crsp.stocknames; WRDS
    # has tweaked this file's schema several times so we adapt.
    info = db.describe_table("crsp", "stocknames")
    available = set(info["name"].str.lower())

    wanted = ["permno", "permco",
              "namedt", "nameenddt",
              "ticker", "ncusip", "cusip",
              "shrcd", "shrcls", "exchcd", "siccd", "hexcd",
              "tsymbol", "comnam"]
    cols = [c for c in wanted if c in available]
    missing = [c for c in wanted if c not in available]
    if missing:
        print(f"  STOCKNAMES: {missing} not in this WRDS install, skipping.")

    sql = f"SELECT {', '.join(cols)} FROM crsp.stocknames"
    date_cols = [c for c in ("namedt", "nameenddt") if c in cols]
    stk, db = cfg.run_query_with_retry(db, sql, date_cols=date_cols)
    stk["permno"] = stk["permno"].astype("int32")
    stk.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, stk_path)
    print(f"STOCKNAMES rows: {len(stk):,}  (cols: {cols})")
    return db


def pull_dsedelist(db, dlst_path):
    _cleanup_tmp(dlst_path)
    tmp_path = _tmp_for(dlst_path)
    sql = """
        SELECT  permno, dlstdt, dlstcd, dlret, dlretx
        FROM    crsp.dsedelist
    """
    dlst, db = cfg.run_query_with_retry(db, sql, date_cols=["dlstdt"])
    dlst["permno"] = dlst["permno"].astype("int32")
    dlst.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, dlst_path)
    print(f"DSEDELIST rows: {len(dlst):,}")
    return db


def main():
    db = cfg.get_wrds_connection()

    targets = [
        ("DSF",        cfg.RAW_DIR / "crsp_dsf.parquet",         pull_dsf),
        ("MSF",        cfg.RAW_DIR / "crsp_msf.parquet",         pull_msf),
        ("STOCKNAMES", cfg.RAW_DIR / "crsp_stocknames.parquet",  pull_stocknames),
        ("DSEDELIST",  cfg.RAW_DIR / "crsp_dsedelist.parquet",   pull_dsedelist),
    ]

    for name, path, fn in targets:
        if path.exists():
            print(f"{name}: cached, skip ({path.name})")
        else:
            db = fn(db, path)

    db.close()
    print("Done.")


if __name__ == "__main__":
    main()
