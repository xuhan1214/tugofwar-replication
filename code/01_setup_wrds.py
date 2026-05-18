"""
One-time WRDS connection check.  Run this FIRST.

Prompts for WRDS username/password on first call; offers to write
~/.pgpass for passwordless reuse.  Then lists which WRDS schemas are
visible and the row counts of the tables we depend on.
"""

from importlib import import_module
cfg = import_module("00_config")

import sys


def main():
    try:
        import wrds
    except ImportError:
        sys.exit("wrds-py not installed.  pip install wrds")

    try:
        from importlib.metadata import version
        print(f"wrds-py version: {version('wrds')}\n")
    except Exception:
        print("wrds-py version: (unknown)\n")
    db = wrds.Connection()
    print("Connected.\n")

    needed = ["crsp", "comp", "ibes", "taq", "taqmsec", "ff", "ccm", "wrdsapps"]
    libs = set(db.list_libraries())
    print(f"{'library':<12}  available?")
    print("-" * 26)
    for lib in needed:
        ok = "yes" if lib in libs else "no  <-- missing"
        print(f"{lib:<12}  {ok}")

    print("\nKey table row counts:")
    for schema, table in [
        ("crsp",     "dsf"),
        ("crsp",     "msf"),
        ("crsp",     "stocknames"),
        ("crsp",     "dsedelist"),
        ("crsp",     "ccmxpf_lnkhist"),
        ("comp",     "funda"),
        ("comp",     "fundq"),
        ("ff",       "factors_daily"),
        ("ff",       "factors_monthly"),
        ("wrdsapps", "ibcrsphist"),
        ("wrdsapps", "tclink"),
        ("wrdsapps", "taqmclink"),
    ]:
        try:
            n = db.get_row_count(schema, table)
            print(f"  {schema}.{table:<22} rows = {n:,}")
        except Exception as e:
            print(f"  {schema}.{table:<22} ERROR: {e}")

    db.close()
    print("\nSetup OK.")


if __name__ == "__main__":
    main()
