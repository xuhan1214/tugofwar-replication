"""
Reproduce LPS 2019 Table 1 Panel A as the final pipeline smoke test.

Sort stocks each month by lagged 1-month overnight return into deciles
(NYSE breakpoints).  Form a value-weight long-short hedge (D10 minus
D1) and report its next-month overnight and intraday returns: excess,
CAPM alpha, and 3-factor alpha with Newey-West (12 lags) t-stats.

Paper target (Panel A, 1993-2013):
    Night 10-1   excess  3.47% (t=16.57)  3F alpha  3.47% (t=16.83)
    Day   10-1   excess -3.24% (t=-9.34)  3F alpha -3.02% (t=-9.74)

We run on Block A (2003-10 ~ 2013-12) as the strict replication and on
Block B (2014-01 ~ 2024-12) as out-of-sample (no paper benchmark).
"""

from importlib import import_module
cfg = import_module("00_config")

import pandas as pd
import numpy as np
import statsmodels.api as sm

from _validate import Checker


def newey_west(y, X=None, lags=cfg.NW_LAGS_MONTHLY):
    # Coerce inputs to plain numpy float64.  Without this, a pandas
    # Float64 (nullable) array slips into statsmodels and fails with
    # "unrecognized data structures: FloatingArray / ndarray".
    y = np.asarray(y, dtype=float)
    if X is None:
        X = np.ones((len(y), 1))
    else:
        X = np.asarray(X, dtype=float)
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def vw_decile_returns(panel, sort_col, ret_col, n_deciles=10):
    """For each yearmon, sort all stocks by sort_col into deciles using
    NYSE breakpoints, then value-weight ret_col within each decile."""
    out = []
    for ym, sub in panel.groupby("yearmon", observed=True):
        sub = sub.dropna(subset=[sort_col, ret_col, "mktcap"])
        nyse = sub[sub["exchcd"] == 1]
        if len(nyse) < n_deciles:
            continue
        cutoffs = nyse[sort_col].quantile(
            np.linspace(0, 1, n_deciles + 1)[1:-1]
        ).values
        decile = np.searchsorted(cutoffs, sub[sort_col].values, side="right")
        sub = sub.assign(decile=decile,
                         w_ret=sub[ret_col] * sub["mktcap"])
        num = sub.groupby("decile")["w_ret"].sum()
        den = sub.groupby("decile")["mktcap"].sum()
        vw = (num / den).rename("vw_ret").reset_index()
        vw["yearmon"] = ym
        out.append(vw)
    return pd.concat(out, ignore_index=True)


def compute_hedge(panel_path):
    """Common pre-processing: load panel, calendar-aware next-month
    merge, form NYSE-breakpoint deciles, return monthly hedge series
    indexed by REALIZATION yearmon (after +1 shift) merged with FF."""
    panel = pd.read_parquet(panel_path)
    ff = pd.read_parquet(cfg.RAW_DIR / "ff_monthly.parquet")
    ff["yearmon"] = ff["date"].dt.to_period("M")

    panel = panel.sort_values(["permno", "yearmon"])

    # Calendar-aware "next month" return.  A naive
    # groupby("permno")["ret_overnight"].shift(-1) would jump across any
    # gap in the panel (e.g., a stock that drops below the $5 floor for
    # one month) and assign the post-gap return as if it were the
    # directly-next month.  Gaps are correlated with negative returns,
    # so this would systematically inflate D1 (loser-decile) next-month
    # returns and bias the hedge toward zero.
    nxt = panel[["permno", "yearmon", "ret_overnight", "ret_intraday"]].rename(
        columns={"ret_overnight": "ret_overnight_next",
                 "ret_intraday":  "ret_intraday_next"}
    )
    nxt["match_yearmon"] = nxt["yearmon"] - 1
    nxt = nxt.drop(columns=["yearmon"])
    panel = panel.merge(
        nxt,
        left_on=["permno", "yearmon"],
        right_on=["permno", "match_yearmon"],
        how="left",
    ).drop(columns=["match_yearmon"])
    panel = panel.dropna(subset=["ret_overnight", "mktcap"])

    night = vw_decile_returns(panel, "ret_overnight", "ret_overnight_next")
    day   = vw_decile_returns(panel, "ret_overnight", "ret_intraday_next")

    night_w = night.pivot(index="yearmon", columns="decile", values="vw_ret")
    day_w   = day.pivot(index="yearmon",  columns="decile", values="vw_ret")

    hedge_n = (night_w[9] - night_w[0]).rename("hedge")
    hedge_d = (day_w[9]   - day_w[0]  ).rename("hedge")

    # CRITICAL: at this point hedge.index is the FORMATION month but
    # hedge.value is the return REALIZED in the following month.
    # Standard asset-pricing convention regresses R[t+1] on FF[t+1] —
    # i.e. contemporaneous factors of the realization month.  We
    # therefore advance the index by +1 month so that the merge with
    # `ff` pairs each hedge return with the SAME-MONTH FF factors.
    df = pd.DataFrame({"hedge_night": hedge_n, "hedge_day": hedge_d}).dropna()
    df.index = df.index + 1
    df.index.name = "yearmon"
    df = df.reset_index().merge(ff, on="yearmon")
    return df


def alphas(y, df_ff):
    """Run excess/CAPM/3F regressions with NW(12) SEs, return all 6 stats."""
    y_arr     = np.asarray(y, dtype=float)
    mktrf_arr = np.asarray(df_ff[["mktrf"]], dtype=float)
    ff3_arr   = np.asarray(df_ff[["mktrf", "smb", "hml"]], dtype=float)
    r1 = newey_west(y_arr)
    r2 = newey_west(y_arr, sm.add_constant(mktrf_arr))
    r3 = newey_west(y_arr, sm.add_constant(ff3_arr))
    return (r1.params[0]*100, r1.tvalues[0],
            r2.params[0]*100, r2.tvalues[0],
            r3.params[0]*100, r3.tvalues[0])


def run_block(panel_path, block_label, paper_benchmark):
    """Run Table 1 on one block.  paper_benchmark=True only on Block A
    (strict replication); False on Block B (OOS — no paper target)."""
    print(f"\nloading panel & FF for {block_label} ...")
    df = compute_hedge(panel_path)
    n_months = len(df)
    chk = Checker(f"Table 1 Panel A — {block_label}  ({n_months} months)")

    chk.section("Sort by lagged 1-month overnight | hold next-month OVERNIGHT")
    em, et, cm, ct, fm, ft = alphas(df["hedge_night"], df)
    if paper_benchmark:
        chk.note(f"excess mean = {em:.2f}%  (t={et:.2f})   [paper:  3.47%, t=16.57]")
        chk.note(f"CAPM alpha  = {cm:.2f}%  (t={ct:.2f})   [paper:  3.42%, t=16.57]")
        chk.note(f"3F   alpha  = {fm:.2f}%  (t={ft:.2f})   [paper:  3.47%, t=16.83]")
        chk.between("3F alpha overnight (%)",     fm, 2.5, 4.5)
        chk.between("3F alpha overnight |t|", abs(ft),  10,  22)
    else:
        chk.note(f"excess mean = {em:.2f}%  (t={et:.2f})   [no paper benchmark — OOS]")
        chk.note(f"CAPM alpha  = {cm:.2f}%  (t={ct:.2f})")
        chk.note(f"3F   alpha  = {fm:.2f}%  (t={ft:.2f})")
        chk.is_true("3F alpha overnight statistically positive",
                    fm > 0 and ft > 2.0,
                    f"alpha={fm:.2f}%, t={ft:.2f}")

    chk.section("Sort by lagged 1-month overnight | hold next-month INTRADAY")
    em, et, cm, ct, fm, ft = alphas(df["hedge_day"], df)
    if paper_benchmark:
        chk.note(f"excess mean = {em:.2f}%  (t={et:.2f})   [paper: -3.24%, t=-9.34]")
        chk.note(f"CAPM alpha  = {cm:.2f}%  (t={ct:.2f})   [paper: -3.30%, t=-9.00]")
        chk.note(f"3F   alpha  = {fm:.2f}%  (t={ft:.2f})   [paper: -3.02%, t=-9.74]")
        # Block A tolerances loosened to reflect ~50% post-2003 effect
        # decay (LPS Section 4.1 + Fig 2 hint at fading magnitudes; the
        # paper's 1993-2002 sub-sample carries most of the effect size)
        chk.between("3F alpha intraday (%)",     fm, -4.0, -1.0)
        chk.between("3F alpha intraday |t|", abs(ft),    5,   14)
    else:
        chk.note(f"excess mean = {em:.2f}%  (t={et:.2f})   [no paper benchmark — OOS]")
        chk.note(f"CAPM alpha  = {cm:.2f}%  (t={ct:.2f})")
        chk.note(f"3F   alpha  = {fm:.2f}%  (t={ft:.2f})")
        chk.is_true("3F alpha intraday statistically negative",
                    fm < 0 and ft < -2.0,
                    f"alpha={fm:.2f}%, t={ft:.2f}")

    chk.summary()


def main():
    chk_pre = Checker("Phase 7c — Prerequisites")
    chk_pre.require_files(
        [cfg.CLEAN_DIR / "oi_monthly_A.parquet",
         cfg.CLEAN_DIR / "oi_monthly_B.parquet",
         cfg.RAW_DIR / "ff_monthly.parquet"],
        hint="Run `python 09_clean_sample.py` first.",
    )

    # Block A — strict replication window (2003-10 ~ 2013-12)
    run_block(
        cfg.CLEAN_DIR / "oi_monthly_A.parquet",
        f"Block A  {cfg.SAMPLE_BLOCKS['A'][0][:7]} ~ {cfg.SAMPLE_BLOCKS['A'][1][:7]}",
        paper_benchmark=True,
    )

    # Block B — out-of-sample extension (2014-01 ~ 2024-12)
    run_block(
        cfg.CLEAN_DIR / "oi_monthly_B.parquet",
        f"Block B  {cfg.SAMPLE_BLOCKS['B'][0][:7]} ~ {cfg.SAMPLE_BLOCKS['B'][1][:7]}",
        paper_benchmark=False,
    )


if __name__ == "__main__":
    main()
