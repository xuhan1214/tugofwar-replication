"""
Pull IBES consensus EPS forecasts, actuals, and the IBES-CRSP link.

Outputs (data/raw/):
    ibes_summary.parquet  ibes.statsumu_epsus, FPI='6' (next quarter)
    ibes_actuals.parquet  ibes.actu_epsus (quarterly EPS actuals)
    ibes_link.parquet     wrdsapps.ibcrsphist, score <= 3

We discover the actual columns of each table at run time and select
the intersection of (what we want) and (what exists).  This makes the
script robust to small WRDS schema differences across deployments.

ibcrsphist score: 1 = CUSIP exact, 2 = 8-digit CUSIP + name, 3 = 6-digit
CUSIP, 4 = name + ticker (no CUSIP), 5/6 = name-only / fuzzy.  We keep
<= 3 so every retained link is CUSIP-validated, matching the strictness
of the TAQ link filter (match_lvl <= 2).
"""

from importlib import import_module
cfg = import_module("00_config")


def existing_cols(db, schema, table):
    info = db.describe_table(schema, table)
    return set(info["name"].str.lower())


def select_existing(wanted, available, schema_table):
    cols = [c for c in wanted if c in available]
    missing = [c for c in wanted if c not in available]
    if missing:
        print(f"  {schema_table}: {missing} not present, skipping.")
    return cols


def main():
    db = cfg.get_wrds_connection()

    # ---------------------------------------------------------------
    # 1.  Consensus summary (next-quarter forecasts)
    # ---------------------------------------------------------------
    if (cfg.RAW_DIR / "ibes_summary.parquet").exists():
        print("ibes_summary.parquet: cached, skip.")
    else:
        avail = existing_cols(db, "ibes", "statsumu_epsus")
        wanted = ["ticker", "statpers", "fpedats", "fpi",
                  "numest", "meanest", "medest", "stdev", "actual"]
        cols = select_existing(wanted, avail, "ibes.statsumu_epsus")
        sql = f"""
            SELECT {", ".join(cols)}
            FROM   ibes.statsumu_epsus
            WHERE  fpi = '6'
              AND  statpers BETWEEN '{cfg.FULL_START}' AND '{cfg.FULL_END}'
        """
        date_cols = [c for c in ("statpers", "fpedats") if c in cols]
        sumu, db = cfg.run_query_with_retry(db, sql, date_cols=date_cols)
        print(f"IBES summary rows: {len(sumu):,}")
        sumu.to_parquet(cfg.RAW_DIR / "ibes_summary.parquet", index=False)

    # ---------------------------------------------------------------
    # 2.  Quarterly actuals
    # ---------------------------------------------------------------
    if (cfg.RAW_DIR / "ibes_actuals.parquet").exists():
        print("ibes_actuals.parquet: cached, skip.")
    else:
        avail = existing_cols(db, "ibes", "actu_epsus")
        wanted = ["ticker", "pends", "anndats", "value", "pdicity"]
        cols = select_existing(wanted, avail, "ibes.actu_epsus")
        # Build SELECT list, aliasing for downstream consumers
        select_parts = []
        for c in cols:
            if c == "anndats":
                select_parts.append("anndats AS rdq_ibes")
            elif c == "value":
                select_parts.append("value AS actual_eps")
            else:
                select_parts.append(c)
        where = [f"pends BETWEEN '{cfg.FULL_START}' AND '{cfg.FULL_END}'"]
        if "pdicity" in cols:
            where.append("pdicity = 'QTR'")
        sql = f"""
            SELECT {", ".join(select_parts)}
            FROM   ibes.actu_epsus
            WHERE  {" AND ".join(where)}
        """
        date_cols = []
        if "pends"   in cols: date_cols.append("pends")
        if "anndats" in cols: date_cols.append("rdq_ibes")
        actu, db = cfg.run_query_with_retry(db, sql, date_cols=date_cols)
        print(f"IBES actuals rows: {len(actu):,}")
        actu.to_parquet(cfg.RAW_DIR / "ibes_actuals.parquet", index=False)

    # ---------------------------------------------------------------
    # 3.  IBES-CRSP link (try the alias schema first)
    # ---------------------------------------------------------------
    if (cfg.RAW_DIR / "ibes_link.parquet").exists():
        print("ibes_link.parquet: cached, skip.")
    else:
        link = None
        last_err = None
        for schema in ("wrdsapps", "wrdsapps_link_crsp_ibes"):
            try:
                avail = existing_cols(db, schema, "ibcrsphist")
                wanted = ["ticker", "permno", "ncusip", "sdate", "edate", "score"]
                cols = select_existing(wanted, avail, f"{schema}.ibcrsphist")
                sql = f"""
                    SELECT {", ".join(cols)}
                    FROM   {schema}.ibcrsphist
                    WHERE  score <= 3
                """
                date_cols = [c for c in ("sdate", "edate") if c in cols]
                link, db = cfg.run_query_with_retry(db, sql, date_cols=date_cols)
                print(f"IBES link source: {schema}.ibcrsphist")
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # only fall through to next schema if the relation is
                # missing / unavailable.  Re-raise any other error.
                if ("does not exist" in msg
                        or "no such" in msg
                        or "permission denied" in msg
                        or "not found" in msg):
                    continue
                raise
        if link is None:
            raise RuntimeError(
                "Could not locate ibcrsphist in any IBES-link schema. "
                f"Last error: {last_err}"
            )
        print(f"IBES link rows: {len(link):,}")
        link.to_parquet(cfg.RAW_DIR / "ibes_link.parquet", index=False)

    db.close()
    print("Done.")


if __name__ == "__main__":
    main()
