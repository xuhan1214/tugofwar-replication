"""
Robustness check — re-run Table 1 Panel A on Block A AND Block B under
different judgment-call specifications.

Three sources of researcher discretion are tested (others held at
baseline):
    1.  Intraday outlier cap (|ret_intraday| < CAP for daily good=True)
        - inf  (no cap, i.e. as-LPS-text)
        - 2.0  (200%; loose)
        - 1.0  (100%; baseline — used in v08/09/10)
        - 0.5  (50%; tight)
    2.  Beaver-McNichols-Price delisting adjustment
        - Yes (baseline) / No
    3.  (Held fixed for compute budget) match_lvl<=2 on taqmclink.
        Varying this would require re-pulling the 2.2 GB link table
        from WRDS, which is out of scope for a local robustness sweep.

For each (cap, BMP) cell we rebuild oi_daily → oi_monthly → compute
Table 1 stats on BOTH Block A (2003-10~2013-12, strict replication)
AND Block B (2014-01~2024-12, out-of-sample).

Output:
    figures/robustness_table1.csv  — tidy CSV of all spec results
    stdout                         — side-by-side comparison tables
"""
from importlib import import_module
cfg = import_module("00_config")
mod9  = import_module("09_clean_sample")
mod10 = import_module("10_validate_table1")

import numpy as np
import pandas as pd


def build_oi_daily(dsf, taq, intraday_cap):
    """Replicate 08_build_overnight_intraday.py in-memory with a tunable
    intraday cap.  Returns a daily panel with ret_intraday, ret_overnight,
    and good."""
    df = dsf.merge(taq, on=["permno", "date"], how="left")
    df["ret_intraday"] = df["prc"] / df["vwap_open30"] - 1.0

    base_good = (
        df["vwap_open30"].notna()
        & (df["vol_open30"] >= cfg.MIN_SHARES_FIRST_HALF_HOUR)
        & df["ret_intraday"].notna()
    )
    if np.isfinite(intraday_cap):
        df["good"] = base_good & (df["ret_intraday"].abs() < intraday_cap)
    else:
        df["good"] = base_good

    df["log1p_ret"] = np.log1p(df["ret"].fillna(0.0).clip(lower=-0.999))
    df["cum_logret"] = df.groupby("permno")["log1p_ret"].cumsum()

    cum_at_good = df["cum_logret"].where(df["good"])
    df["cum_logret_prev_good"] = cum_at_good.groupby(df["permno"]).shift(1)
    df["cum_logret_prev_good"] = df.groupby("permno")["cum_logret_prev_good"].ffill()
    df["ret_rolled"] = np.expm1(df["cum_logret"] - df["cum_logret_prev_good"])
    df["ret_overnight"] = np.where(
        df["good"],
        (1.0 + df["ret_rolled"]) / (1.0 + df["ret_intraday"]) - 1.0,
        np.nan,
    )
    return df[["permno", "date", "prc", "ret", "vol", "shrout",
               "vwap_open30", "vol_open30",
               "ret_intraday", "ret_overnight", "good"]]


def build_monthly_panel(daily, stk, apply_bmp):
    """Replicate 09_clean_sample.py in-memory.  Returns the filtered
    monthly panel (full sample — caller slices to A or B)."""
    daily = mod9.attach_security_info(daily, stk)
    m = mod9.aggregate_to_monthly(daily)
    m["yearmon_ts"] = m["yearmon"].dt.to_timestamp() + pd.offsets.MonthEnd(0)

    if apply_bmp:
        m, _ = mod9.apply_bmp_delisting(m)

    m = mod9.apply_filters(m)
    return m


def compute_block_stats(monthly, block_key, ff):
    """Slice monthly to the given block, run Table 1 hedge regressions,
    return (n_stock_months, n_months, night_3f, night_t, day_3f, day_t)."""
    lo, hi = cfg.SAMPLE_BLOCKS[block_key]
    block = monthly[(monthly["yearmon_ts"] >= lo) &
                    (monthly["yearmon_ts"] <= hi)].copy()
    n_stockmo = len(block)
    if n_stockmo == 0:
        return n_stockmo, 0, np.nan, np.nan, np.nan, np.nan

    block = block.sort_values(["permno", "yearmon"])
    nxt = block[["permno", "yearmon", "ret_overnight", "ret_intraday"]].rename(
        columns={"ret_overnight": "ret_overnight_next",
                 "ret_intraday":  "ret_intraday_next"}
    )
    nxt["match_yearmon"] = nxt["yearmon"] - 1
    nxt = nxt.drop(columns=["yearmon"])
    panel = block.merge(
        nxt, left_on=["permno", "yearmon"],
        right_on=["permno", "match_yearmon"], how="left",
    ).drop(columns=["match_yearmon"])
    panel = panel.dropna(subset=["ret_overnight", "mktcap"])

    night = mod10.vw_decile_returns(panel, "ret_overnight", "ret_overnight_next")
    day   = mod10.vw_decile_returns(panel, "ret_overnight", "ret_intraday_next")
    night_w = night.pivot(index="yearmon", columns="decile", values="vw_ret")
    day_w   = day.pivot(index="yearmon",  columns="decile", values="vw_ret")
    hedge_n = (night_w[9] - night_w[0]).rename("hedge")
    hedge_d = (day_w[9]   - day_w[0]  ).rename("hedge")

    df_ff = pd.DataFrame({"hedge_night": hedge_n, "hedge_day": hedge_d}).dropna()
    df_ff.index = df_ff.index + 1
    df_ff.index.name = "yearmon"
    df_ff = df_ff.reset_index().merge(ff, on="yearmon")

    n_months = len(df_ff)
    _, _, _, _, night_3f, night_t = mod10.alphas(df_ff["hedge_night"], df_ff)
    _, _, _, _, day_3f,   day_t   = mod10.alphas(df_ff["hedge_day"],   df_ff)
    return n_stockmo, n_months, night_3f, night_t, day_3f, day_t


def main():
    print("Loading CRSP DSF + TAQ daily + stocknames + FF ...")
    dsf = pd.read_parquet(
        cfg.RAW_DIR / "crsp_dsf.parquet",
        columns=["permno", "date", "prc", "ret",
                 "vol", "shrout", "cfacpr"],
    )
    dsf["permno"] = dsf["permno"].astype("int32")
    dsf["prc"] = dsf["prc"].abs()
    dsf = dsf.sort_values(["permno", "date"]).reset_index(drop=True)

    taq = pd.read_parquet(
        cfg.CLEAN_DIR / "taq_daily.parquet",
        columns=["permno", "date", "vwap_open30", "vol_open30"],
    )
    taq["permno"] = taq["permno"].astype("int32")

    stk = pd.read_parquet(cfg.RAW_DIR / "crsp_stocknames.parquet")
    ff = pd.read_parquet(cfg.RAW_DIR / "ff_monthly.parquet")
    ff["yearmon"] = ff["date"].dt.to_period("M")

    print(f"  DSF rows: {len(dsf):,}   TAQ daily rows: {len(taq):,}")

    specs = [
        # (intraday_cap, bmp_on, label)
        (np.inf, True,  "no cap (∞)"),
        (np.inf, False, "no cap (∞)"),
        (2.0,    True,  "cap 200%"),
        (2.0,    False, "cap 200%"),
        (1.0,    True,  "cap 100% [baseline]"),
        (1.0,    False, "cap 100% [baseline]"),
        (0.5,    True,  "cap 50%"),
        (0.5,    False, "cap 50%"),
    ]

    results = []
    cap_to_daily = {}  # cache daily panel per cap
    for cap, bmp, label in specs:
        print(f"\n--- spec: {label}  |  BMP={'on' if bmp else 'off'} ---")
        if cap not in cap_to_daily:
            print(f"  building oi_daily (cap={cap}) ...")
            cap_to_daily[cap] = build_oi_daily(dsf, taq, cap)
        daily = cap_to_daily[cap]
        print(f"  good day rate: {daily['good'].mean()*100:.1f}%")
        print(f"  building monthly panel (BMP={'on' if bmp else 'off'}) ...")
        monthly = build_monthly_panel(daily, stk, apply_bmp=bmp)
        print(f"  monthly rows post-filter: {len(monthly):,}")

        for block_key in ("A", "B"):
            n_stockmo, n_months, n3f, nt, d3f, dt = compute_block_stats(
                monthly, block_key, ff
            )
            print(f"  Block {block_key}: night α = {n3f:+.2f}% (t={nt:+.2f})   "
                  f"day α = {d3f:+.2f}% (t={dt:+.2f})   "
                  f"[{n_stockmo:,} stk-mo, {n_months} mo]")
            results.append({
                "cap": cap,
                "bmp": bmp,
                "label": label,
                "block": block_key,
                "n_stock_months": n_stockmo,
                "n_months": n_months,
                "night_3f": n3f, "night_t": nt,
                "day_3f": d3f,   "day_t": dt,
            })
        del monthly

    print_results(results, "A")
    print_results(results, "B")
    save_csv(results)


def print_results(results, block_key):
    GREEN = "\033[92m"; YEL = "\033[93m"; END = "\033[0m"; BOLD = "\033[1m"
    line = "=" * 90
    sub  = "-" * 90

    block_results = [r for r in results if r["block"] == block_key]
    if block_key == "A":
        block_label = f"Block A  (2003-10 ~ 2013-12, strict overlap)"
        paper_str = ("Paper LPS 2019 (1993-2013):  "
                     "night α = +3.47% (t=+16.83)  "
                     "day α = -3.02% (t=-9.74)")
    else:
        block_label = f"Block B  (2014-01 ~ 2024-12, out-of-sample)"
        paper_str = "No paper benchmark (genuinely OOS)"

    print(f"\n{line}\n{BOLD}  Robustness — Table 1 Panel A on {block_label}{END}\n{line}")
    print(f"  {paper_str}\n")

    hdr = "{:<23} {:<5} {:>9} {:>8} {:>10} {:>10}    {:>10} {:>10}"
    print(hdr.format("Intraday cap", "BMP", "stk-mo", "months",
                     "night α", "t", "day α", "t"))
    print(sub)
    for r in block_results:
        bmp = "on" if r["bmp"] else "off"
        is_baseline = (r["cap"] == 1.0 and r["bmp"])
        marker = f" {YEL}★{END}" if is_baseline else "  "
        print(hdr.format(
            r["label"], bmp,
            f"{r['n_stock_months']:,}",
            f"{r['n_months']:,}",
            f"{r['night_3f']:+.2f}%",
            f"({r['night_t']:+.2f})",
            f"{r['day_3f']:+.2f}%",
            f"({r['day_t']:+.2f})",
        ) + marker)
    print(sub)

    night_vals = [r["night_3f"] for r in block_results]
    day_vals   = [r["day_3f"]   for r in block_results]
    night_ts   = [r["night_t"]  for r in block_results]
    day_ts     = [r["day_t"]    for r in block_results]
    baseline = next(r for r in block_results if r["cap"] == 1.0 and r["bmp"])
    print(f"{BOLD}Range across all 8 specs ({block_label}):{END}")
    print(f"  night 3F α: {min(night_vals):+.2f}% to {max(night_vals):+.2f}%  "
          f"(spread {max(night_vals)-min(night_vals):.2f}pp, "
          f"baseline = {baseline['night_3f']:+.2f}%)")
    print(f"  night |t|:   {min(night_ts):+.2f} to {max(night_ts):+.2f}")
    print(f"  day 3F α:   {min(day_vals):+.2f}% to {max(day_vals):+.2f}%  "
          f"(spread {max(day_vals)-min(day_vals):.2f}pp, "
          f"baseline = {baseline['day_3f']:+.2f}%)")
    print(f"  day |t|:     {min(day_ts):+.2f} to {max(day_ts):+.2f}")

    all_night_positive = all(r["night_3f"] > 0 and r["night_t"] > 2 for r in block_results)
    all_day_negative   = all(r["day_3f"] < 0 and r["day_t"] < -2 for r in block_results)
    print(f"\n{YEL}INTERPRETATION ({block_label}):{END}")
    if all_night_positive and all_day_negative:
        print(f"  ✓ Direction (night +, day -) and |t|>2 PRESERVED across all 8 specs.")
    else:
        unsig_n = sum(1 for r in block_results if not (r["night_3f"] > 0 and r["night_t"] > 2))
        unsig_d = sum(1 for r in block_results if not (r["day_3f"] < 0 and r["day_t"] < -2))
        if unsig_n: print(f"  ✗ Night α loses direction/significance in {unsig_n}/8 specs")
        if unsig_d: print(f"  ✗ Day α loses direction/significance in {unsig_d}/8 specs")
    print(line)


def save_csv(results):
    df = pd.DataFrame(results)
    out_path = cfg.FIG_DIR / "robustness_table1.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote tidy CSV: {out_path}")


if __name__ == "__main__":
    main()
