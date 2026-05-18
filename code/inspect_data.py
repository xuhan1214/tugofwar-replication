"""
Quick inspection of all parquet files in data/.

Usage:
    python inspect_data.py             # show summary of every file
    python inspect_data.py crsp_msf    # show one specific file in detail
    python inspect_data.py --interactive  # drop into a REPL with all DFs loaded
"""

from importlib import import_module
cfg = import_module("00_config")

import sys
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


FILES = {
    # Fama-French
    "ff_daily":          cfg.RAW_DIR  / "ff_daily.parquet",
    "ff_monthly":        cfg.RAW_DIR  / "ff_monthly.parquet",
    # CRSP
    "crsp_dsf":          cfg.RAW_DIR  / "crsp_dsf.parquet",
    "crsp_msf":          cfg.RAW_DIR  / "crsp_msf.parquet",
    "crsp_stocknames":   cfg.RAW_DIR  / "crsp_stocknames.parquet",
    "crsp_dsedelist":    cfg.RAW_DIR  / "crsp_dsedelist.parquet",
    # Compustat
    "funda":             cfg.RAW_DIR  / "funda.parquet",
    "fundq":             cfg.RAW_DIR  / "fundq.parquet",
    "ccm_link":          cfg.RAW_DIR  / "ccm_link.parquet",
    # IBES
    "ibes_summary":      cfg.RAW_DIR  / "ibes_summary.parquet",
    "ibes_actuals":      cfg.RAW_DIR  / "ibes_actuals.parquet",
    "ibes_link":         cfg.RAW_DIR  / "ibes_link.parquet",
}


def show_summary(name, path):
    if not path.exists():
        print(f"  ✗ {name:20s}  (not downloaded yet)")
        return
    size_mb = path.stat().st_size / 1024 / 1024
    # use pyarrow to just read the metadata first (cheap)
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(path)
    n_rows = meta.metadata.num_rows
    n_cols = len(meta.schema.names)
    print(f"  ✓ {name:20s}  {n_rows:>15,d} rows   {n_cols:>3d} cols   {size_mb:>8,.1f} MB")


def show_detail(name, path):
    df = pd.read_parquet(path)
    print(f"\n{'='*70}")
    print(f"  {name}  ({path.name})")
    print(f"{'='*70}")
    print(f"shape: {df.shape}")
    print(f"\ncolumns + dtypes:")
    for c, d in df.dtypes.items():
        print(f"  {c:<25s} {str(d):<12s} {df[c].notna().sum():>15,d} non-null")
    print(f"\nfirst 5 rows:")
    print(df.head().to_string())
    print(f"\nlast 5 rows:")
    print(df.tail().to_string())
    print(f"\nnumeric columns summary:")
    num = df.select_dtypes(include="number")
    if len(num.columns):
        print(num.describe().to_string())


def show_taq_progress():
    """List all per-year TAQ aggregates that exist."""
    print(f"\nTAQ aggregates (data/wrds_taq/):")
    paths = sorted(cfg.TAQ_DIR.glob("taq_agg_*.parquet"))
    if not paths:
        print("  (none yet)")
        return
    for p in paths:
        size_mb = p.stat().st_size / 1024 / 1024
        try:
            import pyarrow.parquet as pq
            n = pq.ParquetFile(p).metadata.num_rows
            print(f"  {p.name:<35s} {n:>12,d} rows   {size_mb:>8,.1f} MB")
        except Exception as e:
            print(f"  {p.name:<35s} (error reading: {e})")


def interactive():
    """Load all available DFs into local vars for an iPython session."""
    print("Loading DFs into memory ...")
    dfs = {}
    for name, path in FILES.items():
        if path.exists():
            dfs[name] = pd.read_parquet(path)
            print(f"  ✓ {name}: {dfs[name].shape}")
    # Also load taq
    taq_paths = sorted(cfg.TAQ_DIR.glob("taq_agg_*.parquet"))
    if taq_paths:
        dfs["taq_all"] = pd.concat(
            [pd.read_parquet(p) for p in taq_paths], ignore_index=True
        )
        print(f"  ✓ taq_all: {dfs['taq_all'].shape}")
    print("\nDFs available:")
    for k in dfs:
        print(f"  - {k}")
    print("\nDropping into Python REPL.  Press Ctrl-D to exit.")
    try:
        from IPython import embed
        embed(user_ns={**dfs, "pd": pd})
    except ImportError:
        import code
        code.interact(local={**dfs, "pd": pd})


def main():
    if len(sys.argv) == 1:
        print("="*70)
        print(f"  Data inventory  ({cfg.RAW_DIR.parent})")
        print("="*70)
        for name, path in FILES.items():
            show_summary(name, path)
        show_taq_progress()
        print(f"\nUsage:")
        print(f"  python inspect_data.py <name>      # detail on one file")
        print(f"  python inspect_data.py --interactive  # IPython with all DFs loaded")
        return

    if sys.argv[1] == "--interactive":
        interactive()
        return

    name = sys.argv[1]
    if name in FILES:
        show_detail(name, FILES[name])
    elif name.startswith("taq_agg_"):
        p = cfg.TAQ_DIR / f"{name}.parquet"
        if p.exists():
            show_detail(name, p)
        else:
            print(f"file not found: {p}")
    else:
        print(f"unknown name: {name}")
        print(f"available: {list(FILES.keys())} + taq_agg_<year>")


if __name__ == "__main__":
    main()
