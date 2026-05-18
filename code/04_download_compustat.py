"""
Pull Compustat fundamentals + the CRSP-Compustat link table.

Outputs (data/raw/):
    funda.parquet     annual fundamentals
    fundq.parquet     quarterly fundamentals (for ROE / SUE)
    ccm_link.parquet  ccmxpf_lnkhist filtered to canonical link rows

We pull raw fields; characteristic construction (BM, ROE, INV, ISSUE,
ACCRUALS, etc.) happens in a later script so we can iterate definitions
without re-downloading.

The funda WHERE clause uses '1990-01-01' (3 years before FULL_START) so
that lag-requiring features (BM uses last year's book equity; INV uses
two prior years of assets) have data on day one of the sample.  Pulled
once, ignored if the parquet already exists.
"""

from importlib import import_module
cfg = import_module("00_config")

import pandas as pd


COMPUSTAT_FY_BUFFER = "1990-01-01"  # 3-year buffer before cfg.FULL_START for lagged features


FUNDA_COLS = [
    "gvkey", "datadate", "fyear", "fyr",
    "at", "lt", "ceq", "seq",
    "pstk", "pstkrv", "pstkl",
    "txditc",
    "ib", "ni", "revt", "cogs", "xsga", "sale",
    "csho", "prcc_f",
    "act", "che", "lct", "dlc", "dltt", "dp", "txp", "wcap",
    "sstk", "prstkc",
]

# cshprq = shares used to compute basic EPS (can differ from total
# outstanding if treasury stock / partial-quarter issuance).
# cshoq  = total common shares outstanding at quarter-end — needed for
# any BE/share or ME-as-of-FYQE computation.  We pull both.
FUNDQ_COLS = [
    "gvkey", "datadate", "fyearq", "fqtr", "rdq",
    "ibq", "epspxq", "epsfxq",
    "ceqq", "atq", "ltq", "saleq",
    "cshfdq", "cshprq", "cshoq",
    "pstkq", "pstkrq",
    "txditcq",
]


def _is_missing_table(err_msg):
    """True if the WRDS error indicates a non-existent / inaccessible
    relation, so it's safe to try the next schema in a fallback list.
    Any other exception (typo in column name, network error) should
    propagate."""
    m = err_msg.lower()
    return ("does not exist" in m
            or "no such" in m
            or "permission denied" in m
            or "not found" in m)


def pull_funda(db, out_path):
    sql = f"""
        SELECT {", ".join(FUNDA_COLS)}
        FROM   comp.funda
        WHERE  indfmt = 'INDL'
          AND  datafmt = 'STD'
          AND  popsrc  = 'D'
          AND  consol  = 'C'
          AND  datadate BETWEEN '{COMPUSTAT_FY_BUFFER}' AND '{cfg.FULL_END}'
    """
    funda, db = cfg.run_query_with_retry(db, sql, date_cols=["datadate"])
    print(f"FUNDA rows: {len(funda):,}")
    funda.to_parquet(out_path, index=False)
    return db


def pull_fundq(db, out_path):
    sql = f"""
        SELECT {", ".join(FUNDQ_COLS)}
        FROM   comp.fundq
        WHERE  indfmt = 'INDL'
          AND  datafmt = 'STD'
          AND  popsrc  = 'D'
          AND  consol  = 'C'
          AND  datadate BETWEEN '{COMPUSTAT_FY_BUFFER}' AND '{cfg.FULL_END}'
    """
    fundq, db = cfg.run_query_with_retry(db, sql,
                                         date_cols=["datadate", "rdq"])
    print(f"FUNDQ rows: {len(fundq):,}")
    fundq.to_parquet(out_path, index=False)
    return db


def pull_ccm(db, out_path):
    # The CCM link table lives under the 'crsp' schema in modern WRDS
    # Postgres deployments (crsp.ccmxpf_lnkhist).  Older configs use
    # 'ccm.ccmxpf_lnkhist'; we try crsp first, fall back to ccm.
    base_sql = """
        SELECT  gvkey, lpermno AS permno, lpermco AS permco,
                linkdt, linkenddt, linktype, linkprim
        FROM    {schema}.ccmxpf_lnkhist
        WHERE   linktype IN ('LU','LC')
          AND   linkprim IN ('P','C')
    """
    ccm = None
    last_err = None
    for schema in ("crsp", "crsp_a_ccm", "ccm"):
        try:
            ccm, db = cfg.run_query_with_retry(
                db, base_sql.format(schema=schema),
                date_cols=["linkdt", "linkenddt"],
            )
            print(f"CCM source: {schema}.ccmxpf_lnkhist")
            break
        except Exception as e:
            last_err = e
            if _is_missing_table(str(e)):
                # try the next candidate schema
                continue
            # bad SQL / unexpected error -- re-raise so user sees the real problem
            raise
    if ccm is None:
        raise RuntimeError(
            "Could not locate ccmxpf_lnkhist in any of crsp / crsp_a_ccm / ccm "
            f"schemas. Last error: {last_err}"
        )
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    ccm["permno"] = ccm["permno"].astype("Int64")
    print(f"CCM link rows: {len(ccm):,}")
    ccm.to_parquet(out_path, index=False)
    return db


def main():
    db = cfg.get_wrds_connection()

    targets = [
        ("FUNDA", cfg.RAW_DIR / "funda.parquet",    pull_funda),
        ("FUNDQ", cfg.RAW_DIR / "fundq.parquet",    pull_fundq),
        ("CCM",   cfg.RAW_DIR / "ccm_link.parquet", pull_ccm),
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
