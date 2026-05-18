"""
Attach PERMNO to each TAQ stock-day using the WRDS-maintained link
tables.  Two link tables exist (one per TAQ era):

    1993-2014 (legacy taq.ct_*)         wrdsapps.tclink     (~259 MB)
        columns: permno, cusip, date, symbol, ...
    2003-now  (taqmsec.ctm_*)           wrdsapps.taqmclink  (~2.2 GB)
        columns: permno, cusip, ncusip, date, sym_root, sym_suffix, match_lvl

We unify both into a (date, sym_root, sym_suffix, permno) schema so the
join with the TAQ aggregates (which 06_taq_aggregate.py also writes in
that schema) is identical for both eras.

Outputs:
    raw/taq_crsp_link_legacy.parquet   for 1993-2014 (unused if no legacy access)
    raw/taq_crsp_link_msec.parquet     for 2003+
    clean/taq_daily.parquet            TAQ aggregates joined to PERMNO,
                                       collapsed to one (permno, date) row,
                                       vol_open30 >= MIN_SHARES_FIRST_HALF_HOUR
"""

from importlib import import_module
cfg = import_module("00_config")

import glob
import re
from pathlib import Path
import pandas as pd


# Match only yearly TAQ aggregates -- not monthly checkpoints like
# `taq_agg_2013_M03.parquet` that 06_taq_aggregate.py leaves behind if
# it crashes mid-year.  A monthly checkpoint slipping in would (a)
# crash `int(year)` parsing and (b) be schema-identical so it would
# silently double-count that month's stock-days against the year file.
_YEAR_FILE_RE = re.compile(r"taq_agg_\d{4}$")


def fetch_link_legacy(db):
    # tclink uses single 'symbol' column; coerce sym_suffix='' for unity.
    # score: 1=exact CUSIP, 2=ticker+date, 3=fuzzy, 4=manual.  Keep <= 2.
    sql = """
        SELECT  date,
                symbol AS sym_root,
                ''     AS sym_suffix,
                permno
        FROM    wrdsapps.tclink
        WHERE   permno IS NOT NULL
          AND   score <= 2
    """
    df, db = cfg.run_query_with_retry(db, sql, date_cols=["date"])
    df["permno"] = df["permno"].astype("int32")
    return df, db


def fetch_link_msec(db):
    # taqmclink already has sym_root/sym_suffix natively.
    # match_lvl: 1=exact CUSIP, 2=ticker+date, 3+=lower quality.  Keep <= 2.
    sql = """
        SELECT  date,
                sym_root,
                COALESCE(sym_suffix, '') AS sym_suffix,
                permno
        FROM    wrdsapps.taqmclink
        WHERE   permno IS NOT NULL
          AND   match_lvl <= 2
    """
    df, db = cfg.run_query_with_retry(db, sql, date_cols=["date"])
    df["permno"] = df["permno"].astype("int32")
    return df, db


def _is_missing_table(err_msg):
    m = err_msg.lower()
    return ("does not exist" in m
            or "no such" in m
            or "permission denied" in m
            or "not found" in m)


def fetch_links():
    print("loading TAQ-CRSP link tables ...")
    db = cfg.get_wrds_connection()

    leg_path = cfg.RAW_DIR / "taq_crsp_link_legacy.parquet"
    if leg_path.exists():
        print(f"  legacy link: cached, skip ({leg_path.name})")
        legacy = pd.read_parquet(leg_path)
    else:
        try:
            legacy, db = fetch_link_legacy(db)
            cfg.atomic_write_parquet(legacy, leg_path, index=False)
            print(f"  wrdsapps.tclink:    {len(legacy):>10,} rows")
        except Exception as e:
            # legacy access (wrdsapps.tclink) is paid-tier; many academic
            # subscriptions don't include it.  Only fall back silently
            # if the table is missing / inaccessible.
            if _is_missing_table(str(e)):
                print(f"  wrdsapps.tclink unavailable ({str(e)[:80]}); legacy link empty")
                legacy = pd.DataFrame(columns=["date", "sym_root", "sym_suffix", "permno"])
            else:
                raise

    msec_path = cfg.RAW_DIR / "taq_crsp_link_msec.parquet"
    if msec_path.exists():
        print(f"  msec link: cached, skip ({msec_path.name})")
        msec = pd.read_parquet(msec_path)
    else:
        msec, db = fetch_link_msec(db)
        cfg.atomic_write_parquet(msec, msec_path, index=False)
        print(f"  wrdsapps.taqmclink: {len(msec):>10,} rows")

    db.close()
    return legacy, msec


def attach_permno(taq, link):
    """Inner join on (date, sym_root, sym_suffix)."""
    if "sym_suffix" not in taq.columns:
        taq["sym_suffix"] = ""
    taq["sym_suffix"] = taq["sym_suffix"].fillna("")
    link["sym_suffix"] = link["sym_suffix"].fillna("")
    return taq.merge(link, on=["date", "sym_root", "sym_suffix"], how="inner")


def collapse_share_classes(df):
    bucket_cols = [c for c in df.columns if c.startswith("bucket_dvol_")]
    agg = df.groupby(["permno", "date"], as_index=False).agg(
        pv_open30=("pv_open30", "sum"),
        vol_open30=("vol_open30", "sum"),
        **{c: (c, "sum") for c in bucket_cols},
    )
    agg["vwap_open30"] = agg["pv_open30"] / agg["vol_open30"].where(agg["vol_open30"] > 0)
    return agg


def list_yearly_aggregate_files():
    """Return all `taq_agg_YYYY.parquet` files, ignoring per-month
    checkpoint files that 06 leaves behind on a mid-year crash."""
    all_files = glob.glob(str(cfg.TAQ_DIR / "taq_agg_*.parquet"))
    yearly = [f for f in all_files if _YEAR_FILE_RE.fullmatch(Path(f).stem)]
    skipped = [Path(f).name for f in all_files if not _YEAR_FILE_RE.fullmatch(Path(f).stem)]
    if skipped:
        print(f"  ignoring {len(skipped)} non-yearly checkpoint file(s): {skipped[:3]}...")
    return sorted(yearly)


def main():
    legacy, msec = fetch_links()

    files = list_yearly_aggregate_files()
    if not files:
        raise SystemExit("No TAQ aggregate files found. Run 06_taq_aggregate.py first.")

    pieces = []
    for f in files:
        yr = int(Path(f).stem.split("_")[-1])
        link = legacy if yr < 2003 else msec
        df = pd.read_parquet(f)
        before = len(df)
        df = attach_permno(df, link)
        after = len(df)
        pct = 100 * after / before if before else 0
        print(f"{yr}: {before:,} -> {after:,} matched ({pct:.1f}%)")
        if after:
            pieces.append(collapse_share_classes(df))

    if not pieces:
        raise SystemExit("No rows matched after permno linkage; aborting.")

    final = pd.concat(pieces, ignore_index=True)
    final = final[final["vol_open30"] >= cfg.MIN_SHARES_FIRST_HALF_HOUR].copy()
    cfg.atomic_write_parquet(final, cfg.CLEAN_DIR / "taq_daily.parquet", index=False)
    print(f"\nFinal taq_daily.parquet rows: {len(final):,}")


if __name__ == "__main__":
    main()
