"""
Pull Fama-French + Carhart UMD factors from WRDS for risk adjustment.

Output:
    raw/ff_daily.parquet
    raw/ff_monthly.parquet
Columns: date, mktrf, smb, hml, rf, umd.

Note: ff.factors_monthly's `date` column is end-of-month (e.g.
2010-01-31), not the YYYYMM int format from Ken French's web CSVs.
Downstream code converts to period[M] before merging, so this is
transparent — but worth noting if you ever compare directly against
the Dartmouth library files.
"""

from importlib import import_module
cfg = import_module("00_config")


def main():
    db = cfg.get_wrds_connection()

    for freq, table in [("daily", "factors_daily"),
                        ("monthly", "factors_monthly")]:
        out_path = cfg.RAW_DIR / f"ff_{freq}.parquet"
        if out_path.exists():
            print(f"FF {freq}: cached, skip ({out_path.name})")
            continue
        sql = f"""
            SELECT date, mktrf, smb, hml, rf, umd
            FROM   ff.{table}
            WHERE  date BETWEEN '{cfg.FULL_START}' AND '{cfg.FULL_END}'
            ORDER BY date
        """
        df, db = cfg.run_query_with_retry(db, sql, date_cols=["date"])
        print(f"FF {freq:7s} rows: {len(df):,}")
        df.to_parquet(out_path, index=False)

    db.close()
    print("Done.")


if __name__ == "__main__":
    main()
