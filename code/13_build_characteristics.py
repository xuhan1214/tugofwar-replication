"""
13_build_characteristics.py
============================
Build the 13 sortable firm-level characteristics from LPS (2019)
Table 2 at the (permno, yearmon=formation) panel level.  Each value
uses only information available at the end of the formation month.

  col       definition                                             source
  ----      ------------------------------------------------       --------
  me        |prc_eom| * shrout_eom                                 msf
  bm        FF1992 BE/ME; July(y+1)..June(y+2) tile of FY(y)       funda+msf
  mom       cumret t-12..t-2, skip most recent month               msf
  sue       (actual - consensus) / prc_eom                         ibes
  indmom    12-mo VW return of stock's MG99 industry (t-11..t)     msf+siccd
  roe       ibq / lagged quarterly BE (HXZ); as-of merge on rdq    fundq
  inv       at_y / at_{y-1} - 1; FF1992 tile                       funda
  beta      Dimson 3-lag sum, prior 252 daily obs                  dsf+ff
  ivol      std of Carhart 4F daily residual, prior 252 obs        dsf+ff
  issue     log(adj_shrout_t / adj_shrout_{t-12}) (Pontiff-Wood.)  msf
  accruals  Sloan 1996 operating accruals / avg(TA); FF1992 tile   funda
  turnover  mean(vol/shrout) over prior 252 trading days           dsf
  str_      ret at month t                                         msf

Output: clean/characteristics_monthly.parquet  (CRSP row of Table 2
is not a sortable characteristic and is built at hedge time.)
"""

from importlib import import_module
cfg = import_module("00_config")

import numpy as np
import pandas as pd


# Formation-month coverage (padded both sides of Block A/B).
CHAR_START_YM = pd.Period("2003-01", freq="M")
CHAR_END_YM   = pd.Period("2024-12", freq="M")

# Daily-window params for BETA/IVOL/TURNOVER ("prior year" = 252 days).
DAILY_LOOKBACK = 252
DAILY_MIN_OBS  = 120

OUT_PATH = cfg.CLEAN_DIR / "characteristics_monthly.parquet"


# ====================================================================
# Industry mapping (Moskowitz-Grinblatt 1999, Table I): 20 industries
# keyed on 2-digit SIC.  Codes outside the listed ranges return NaN.
# ====================================================================
def _sic_to_mg20(s):
    """4-digit SIC -> MG99 industry id (1..20) or NaN.  Bucket = s // 100."""
    if pd.isna(s):
        return np.nan
    s2 = int(s) // 100
    if 10 <= s2 <= 14: return  1   # Mining
    if s2 == 20:       return  2   # Food
    if 22 <= s2 <= 23: return  3   # Apparel
    if s2 == 26:       return  4   # Paper
    if s2 == 28:       return  5   # Chemical
    if s2 == 29:       return  6   # Petroleum
    if s2 == 32:       return  7   # Construction materials
    if s2 == 33:       return  8   # Primary metals
    if s2 == 34:       return  9   # Fabricated metals
    if s2 == 35:       return 10   # Machinery
    if s2 == 36:       return 11   # Electrical equipment
    if s2 == 37:       return 12   # Transportation equipment
    if 38 <= s2 <= 39: return 13   # Other manufacturing
    if s2 == 40:       return 14   # Railroads
    if 41 <= s2 <= 47: return 15   # Other transportation
    if s2 == 49:       return 16   # Utilities
    if s2 == 53:       return 17   # Department stores
    if (50 <= s2 <= 52) or (54 <= s2 <= 59): return 18   # Retail
    if 60 <= s2 <= 69: return 19   # Financial
    if 70 <= s2 <= 89: return 20   # Other services
    return np.nan


def attach_siccd_monthly(panel, stocknames):
    """As-of merge of siccd onto (permno, yearmon_ts) panel via
    [namedt, nameenddt].  Uses '_namedt' alias so it never collides
    with an existing 'date' column in `panel`."""
    stk = stocknames[["permno", "namedt", "nameenddt", "siccd"]].copy()
    stk = stk.rename(columns={"namedt": "_namedt",
                              "nameenddt": "_nameenddt"}).sort_values("_namedt")
    p = panel.sort_values("yearmon_ts").copy()
    out = pd.merge_asof(p, stk, left_on="yearmon_ts", right_on="_namedt",
                        by="permno", direction="backward")
    mask = out["_nameenddt"].notna() & (out["yearmon_ts"] <= out["_nameenddt"])
    out.loc[~mask, "siccd"] = np.nan
    return out.drop(columns=["_namedt", "_nameenddt"])


# ====================================================================
# CCM-link helper: attach permno to (gvkey, datadate) rows.
# ====================================================================
def link_gvkey_to_permno(df, ccm, date_col="datadate"):
    """Join permno onto `df` via ccmxpf_lnkhist, keeping only rows where
    df[date_col] falls inside [linkdt, linkenddt].  Prefer linkprim='P'
    on duplicates."""
    m = df.merge(
        ccm[["gvkey", "permno", "linkdt", "linkenddt", "linkprim"]],
        on="gvkey", how="inner",
    )
    valid = (m[date_col] >= m["linkdt"]) & (m[date_col] <= m["linkenddt"])
    m = m.loc[valid].copy()
    m["pri_order"] = (m["linkprim"] != "P").astype(int)   # P=0, others=1
    m = m.sort_values(["gvkey", date_col, "pri_order"])
    m = m.drop_duplicates(subset=["gvkey", date_col], keep="first")
    return m.drop(columns=["linkdt", "linkenddt", "linkprim", "pri_order"])


# ====================================================================
# CRSP-monthly chars: ME, MOM, ISSUE, STR
# ====================================================================
def build_crsp_monthly_chars(msf):
    """ME, MOM (t-12..t-2 skip-1), STR (this-month ret), ISSUE (12-mo
    log change of split-adjusted shares)."""
    m = msf[["permno", "date", "prc", "ret", "shrout", "cfacshr"]].copy()
    m["yearmon"] = m["date"].dt.to_period("M")
    m["me"]      = m["prc"].abs() * m["shrout"]
    m["adj_sh"]  = m["shrout"] * m["cfacshr"]
    m = m.sort_values(["permno", "yearmon"]).reset_index(drop=True)

    # MOM via cumulative log-return trick.  Treat NaN as 0 so a single
    # missing month does not nullify the entire chain.
    m["log1p_ret"] = np.log1p(m["ret"].fillna(0).clip(lower=-0.999))
    g = m.groupby("permno", sort=False)
    m["cum_log"] = g["log1p_ret"].cumsum()
    m["mom"] = np.expm1(g["cum_log"].shift(2) - g["cum_log"].shift(13))

    # ISSUE = log(adj_sh_t / adj_sh_{t-12}).  Guard divide-by-zero on
    # corporate events where adj_sh becomes 0.
    adj_lag12 = g["adj_sh"].shift(12).astype(float).to_numpy()
    adj_now   = m["adj_sh"].astype(float).to_numpy()
    valid_issue = (adj_now > 0) & (adj_lag12 > 0)
    issue_vals = np.full(len(m), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        issue_vals[valid_issue] = np.log(adj_now[valid_issue] / adj_lag12[valid_issue])
    m["issue"] = issue_vals

    m["str_"] = m["ret"]

    out = m[["permno", "yearmon", "me", "mom", "issue", "str_"]].copy()
    out["permno"] = out["permno"].astype("int32")
    return out


# ====================================================================
# INDMOM: trailing 12-mo VW return of stock's MG20 industry.
# ====================================================================
def build_indmom(msf, stocknames):
    """For each (permno, yearmon=t), assign its industry's trailing
    12-month VW cumulative return ending at month t."""
    m = msf[["permno", "date", "prc", "ret", "shrout"]].copy()
    m["yearmon"]    = m["date"].dt.to_period("M")
    m["yearmon_ts"] = m["yearmon"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    m["mktcap"]     = m["prc"].abs() * m["shrout"]

    m = attach_siccd_monthly(m, stocknames)
    m["industry"] = m["siccd"].apply(_sic_to_mg20)
    m = m.dropna(subset=["industry", "ret", "mktcap"])
    m["industry"] = m["industry"].astype(int)

    # Lagged-mktcap-weighted monthly industry return.
    m = m.sort_values(["permno", "yearmon"])
    m["mktcap_lag"] = m.groupby("permno")["mktcap"].shift(1)
    w = m.dropna(subset=["mktcap_lag"]).copy()
    w["w_ret"] = w["mktcap_lag"] * w["ret"]
    grp = w.groupby(["industry", "yearmon"], observed=True)
    ind_ret = (grp["w_ret"].sum() / grp["mktcap_lag"].sum()).rename("ind_ret").reset_index()

    # Trailing 12-mo cumret via cumulative-log trick.
    ind_ret = ind_ret.sort_values(["industry", "yearmon"])
    ind_ret["log1p_ind"] = np.log1p(ind_ret["ind_ret"].fillna(0).clip(lower=-0.999))
    ind_ret["cum_log_ind"] = ind_ret.groupby("industry")["log1p_ind"].cumsum()
    ind_ret["cum_log_ind_lag12"] = ind_ret.groupby("industry")["cum_log_ind"].shift(12)
    ind_ret["indmom_industry"] = np.expm1(ind_ret["cum_log_ind"] - ind_ret["cum_log_ind_lag12"])

    m_out = m[["permno", "yearmon", "industry"]].drop_duplicates()
    out = m_out.merge(
        ind_ret[["industry", "yearmon", "indmom_industry"]],
        on=["industry", "yearmon"], how="left",
    ).rename(columns={"indmom_industry": "indmom"})
    return out[["permno", "yearmon", "indmom"]]


# ====================================================================
# Year-level value -> monthly July(y+1)..June(y+2) tile (FF1992).
# ====================================================================
def _expand_annual_to_july_june(df_year, col):
    """Given (permno, year, <col>), tile to (permno, yearmon) over
    July(y+1)..June(y+2)."""
    if len(df_year) == 0:
        return pd.DataFrame(columns=["permno", "yearmon", col])
    months = pd.DataFrame({
        "year_delta": [1]*6 + [2]*6,
        "month":      [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6],
    })
    exp = df_year.merge(months, how="cross")
    exp["calendar_year"] = exp["year"] + exp["year_delta"]
    exp["yearmon"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=exp["calendar_year"],
                            month=exp["month"], day=1)),
        freq="M",
    )
    exp["permno"] = exp["permno"].astype("int32")
    return exp[["permno", "yearmon", col]]


# ====================================================================
# BM: Fama-French (1992).  BE(FY ending y) / ME(Dec y), tile Jul-Jun.
# ====================================================================
def build_bm(funda, ccm, msf):
    f = funda[["gvkey", "datadate", "ceq", "txditc", "pstk", "pstkrv", "pstkl"]].copy()
    # Preferred-stock priority: pstkrv -> pstkl -> pstk
    pstk_pref = f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"]).fillna(0.0)
    f["be"] = f["ceq"].fillna(0.0) + f["txditc"].fillna(0.0) - pstk_pref
    f.loc[f["be"] <= 0, "be"] = np.nan
    f = f.dropna(subset=["be"])
    f["year"] = f["datadate"].dt.year
    # Multiple datadates within one calendar year (FY change) -> keep last.
    f = f.sort_values(["gvkey", "datadate"])
    f = f.drop_duplicates(subset=["gvkey", "year"], keep="last")

    f = link_gvkey_to_permno(f[["gvkey", "datadate", "year", "be"]], ccm, "datadate")
    f = f[["permno", "year", "be"]]

    # ME at last December observation of year y.
    dec = msf[["permno", "date", "prc", "shrout"]].copy()
    dec = dec[dec["date"].dt.month == 12]
    dec["year"] = dec["date"].dt.year
    dec["me_dec"] = dec["prc"].abs() * dec["shrout"]
    dec = dec.sort_values(["permno", "year", "date"])
    dec = dec.drop_duplicates(subset=["permno", "year"], keep="last")
    dec = dec[["permno", "year", "me_dec"]]

    bm_yearly = f.merge(dec, on=["permno", "year"], how="inner")
    # Unit fix: BE in $millions; me_dec = prc * shrout(thousands) is in
    # $thousands.  Scale BE by 1000 to put both in $thousands.
    bm_yearly["bm"] = (bm_yearly["be"] * 1000.0) / bm_yearly["me_dec"]
    return _expand_annual_to_july_june(bm_yearly[["permno", "year", "bm"]], "bm")


# ====================================================================
# INV: annual asset growth, FF1992 tile.
# ====================================================================
def build_inv(funda, ccm):
    f = funda[["gvkey", "datadate", "at"]].copy()
    f["year"] = f["datadate"].dt.year
    f = f.dropna(subset=["at"])
    f = f[f["at"] > 0]
    f = f.sort_values(["gvkey", "datadate"])
    f = f.drop_duplicates(subset=["gvkey", "year"], keep="last")
    f["at_lag"] = f.groupby("gvkey")["at"].shift(1)
    f["inv"] = f["at"] / f["at_lag"] - 1
    f = f.dropna(subset=["inv"])
    f = link_gvkey_to_permno(f[["gvkey", "datadate", "year", "inv"]], ccm, "datadate")
    return _expand_annual_to_july_june(f[["permno", "year", "inv"]], "inv")


# ====================================================================
# ACCRUALS: Sloan (1996) operating accruals / avg(TA), FF1992 tile.
# ====================================================================
def build_accruals(funda, ccm):
    cols = ["gvkey", "datadate", "act", "che", "lct", "dlc", "txp", "dp", "at"]
    f = funda[cols].copy()
    f["year"] = f["datadate"].dt.year
    f = f.sort_values(["gvkey", "datadate"])
    f = f.drop_duplicates(subset=["gvkey", "year"], keep="last")
    for c in ["act", "che", "lct", "dlc", "txp", "at"]:
        f[c + "_lag"] = f.groupby("gvkey")[c].shift(1)

    # Treat individual missing components as 0-change (Sloan's
    # robustness footnote).  Require non-NaN AT and AT_lag for scaling.
    d_ca   = (f["act"] - f["act_lag"]).fillna(0.0)
    d_cash = (f["che"] - f["che_lag"]).fillna(0.0)
    d_cl   = (f["lct"] - f["lct_lag"]).fillna(0.0)
    d_std  = (f["dlc"] - f["dlc_lag"]).fillna(0.0)
    d_tp   = (f["txp"] - f["txp_lag"]).fillna(0.0)
    dep    = f["dp"].fillna(0.0)
    accr   = (d_ca - d_cash) - (d_cl - d_std - d_tp) - dep
    avg_ta = (f["at"] + f["at_lag"]) / 2.0
    f["accruals"] = accr / avg_ta.where(avg_ta > 0)
    f = f.dropna(subset=["accruals"])
    f = link_gvkey_to_permno(f[["gvkey", "datadate", "year", "accruals"]], ccm, "datadate")
    return _expand_annual_to_july_june(f[["permno", "year", "accruals"]], "accruals")


# ====================================================================
# ROE: ibq / lagged quarterly BE (HXZ 2015), as-of merged on rdq.
# ====================================================================
def build_roe(fundq, ccm):
    """Returns (permno, info_date, roe).  info_date = rdq if known,
    else datadate + 90 days (conservative fallback for early WRDS)."""
    q = fundq[["gvkey", "datadate", "rdq", "ibq",
               "ceqq", "txditcq", "pstkq", "pstkrq"]].copy()
    pstk_pref = q["pstkq"].fillna(q["pstkrq"]).fillna(0.0)
    q["beq"] = q["ceqq"].fillna(0.0) + q["txditcq"].fillna(0.0) - pstk_pref
    q.loc[q["beq"] <= 0, "beq"] = np.nan
    q = q.dropna(subset=["beq", "ibq", "datadate"])
    q = q.sort_values(["gvkey", "datadate"])
    q["beq_lag"] = q.groupby("gvkey")["beq"].shift(1)
    q["roe"] = q["ibq"] / q["beq_lag"]
    q = q.dropna(subset=["roe"])
    q["info_date"] = q["rdq"].fillna(q["datadate"] + pd.Timedelta(days=90))
    q = link_gvkey_to_permno(q[["gvkey", "datadate", "info_date", "roe"]],
                             ccm, "datadate")
    q = q[["permno", "info_date", "roe"]].copy()
    q = q.sort_values(["permno", "info_date"])
    q = q.drop_duplicates(subset=["permno", "info_date"], keep="last")
    q["permno"] = q["permno"].astype("int32")
    return q


# ====================================================================
# SUE: IBES earnings surprise, pre-announcement consensus.
# ====================================================================
def build_sue(actuals, summary, ibes_link):
    """Returns (permno, rdq_ibes, sue_raw = actual - consensus).
    For each (ticker, pends) actual, take the latest statpers strictly
    before rdq_ibes with matching fpedats.  Scale by prc downstream."""
    a = actuals.dropna(subset=["actual_eps", "rdq_ibes", "pends", "ticker"]).copy()
    s = summary.dropna(subset=["meanest", "statpers", "fpedats", "ticker"]).copy()
    a = a.sort_values("rdq_ibes")
    s = s.rename(columns={"fpedats": "pends"}).sort_values("statpers")
    # Strictly-before via allow_exact_matches=False.
    merged = pd.merge_asof(
        a, s[["ticker", "pends", "statpers", "meanest"]],
        left_on="rdq_ibes", right_on="statpers",
        by=["ticker", "pends"],
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.dropna(subset=["meanest"])
    merged["sue_raw"] = merged["actual_eps"] - merged["meanest"]

    L = ibes_link[["ticker", "permno", "sdate", "edate"]].copy()
    m = merged.merge(L, on="ticker", how="inner")
    m = m[(m["rdq_ibes"] >= m["sdate"]) & (m["rdq_ibes"] <= m["edate"])]
    m = m[["permno", "rdq_ibes", "sue_raw"]].copy()
    m = m.sort_values(["permno", "rdq_ibes"])
    m = m.drop_duplicates(subset=["permno", "rdq_ibes"], keep="last")
    m["permno"] = m["permno"].astype("int32")
    return m


# ====================================================================
# BETA + IVOL + TURNOVER (daily, rolling 252).
# ====================================================================
def build_daily_chars(dsf, ff_daily, yearmons):
    """For each (permno, yearmon=t), use the trailing 252 daily obs
    ending in month t to compute:
      beta     = sum(coefs[1..4]) from r_ex = a + sum_k b_k * mktrf_{t-k}
                 (Dimson 1979, k=0..3)
      ivol     = std(resid) from r_ex = a + b1*MKT + b2*SMB + b3*HML + b4*UMD
      turnover = mean(vol/shrout)
    Closed-form OLS via np.linalg.lstsq -- no statsmodels overhead.
    """
    ff = ff_daily.copy()
    ff["mktrf_l1"] = ff["mktrf"].shift(1)
    ff["mktrf_l2"] = ff["mktrf"].shift(2)
    ff["mktrf_l3"] = ff["mktrf"].shift(3)

    d = dsf[["permno", "date", "ret", "vol", "shrout"]].copy()
    d["permno"] = d["permno"].astype("int32")
    d = d.merge(ff, on="date", how="inner")
    d["ret_excess"] = d["ret"] - d["rf"]
    d = d.dropna(subset=["ret_excess", "mktrf_l3"])   # need 3 lags
    d["turn_daily"] = d["vol"] / d["shrout"]
    d = d.sort_values(["permno", "date"]).reset_index(drop=True)

    ym_ts = {ym: (ym.to_timestamp() + pd.offsets.MonthEnd(0)) for ym in yearmons}
    ym_list = list(yearmons)

    BETA_X_COLS = ["mktrf", "mktrf_l1", "mktrf_l2", "mktrf_l3"]
    IVOL_X_COLS = ["mktrf", "smb", "hml", "umd"]

    permno_groups = d.groupby("permno", sort=False)
    out_rows = []
    n_done = 0
    n_total = permno_groups.ngroups
    for permno, sub in permno_groups:
        sub = sub.reset_index(drop=True)
        sub_dates = sub["date"].values
        y    = sub["ret_excess"].to_numpy(dtype=float)
        Xb   = sub[BETA_X_COLS].to_numpy(dtype=float)
        Xi   = sub[IVOL_X_COLS].to_numpy(dtype=float)
        turn = sub["turn_daily"].to_numpy(dtype=float)

        for ym in ym_list:
            ym_end = np.datetime64(ym_ts[ym].to_datetime64())
            i_end = np.searchsorted(sub_dates, ym_end, side="right")
            if i_end < DAILY_MIN_OBS:
                continue
            # Require an obs IN the formation month -- avoids using stale
            # data for delisted stocks.
            ym_start = np.datetime64(ym.to_timestamp().to_datetime64())
            if sub_dates[i_end - 1] < ym_start:
                continue
            i_start = max(0, i_end - DAILY_LOOKBACK)
            n_obs = i_end - i_start
            if n_obs < DAILY_MIN_OBS:
                continue

            y_w    = y[i_start:i_end]
            Xb_w   = Xb[i_start:i_end]
            Xi_w   = Xi[i_start:i_end]
            turn_w = turn[i_start:i_end]
            if np.isnan(y_w).any() or np.isnan(Xb_w).any() or np.isnan(Xi_w).any():
                continue

            # Dimson beta = sum of 4 lag coefs.
            Xb_const = np.column_stack([np.ones(n_obs), Xb_w])
            try:
                coefs_b, *_ = np.linalg.lstsq(Xb_const, y_w, rcond=None)
                beta_sum = coefs_b[1] + coefs_b[2] + coefs_b[3] + coefs_b[4]
            except np.linalg.LinAlgError:
                beta_sum = np.nan

            # IVOL = std of Carhart 4F residual.
            Xi_const = np.column_stack([np.ones(n_obs), Xi_w])
            try:
                coefs_i, *_ = np.linalg.lstsq(Xi_const, y_w, rcond=None)
                resid = y_w - Xi_const @ coefs_i
                ivol_val = float(np.std(resid, ddof=1))
            except np.linalg.LinAlgError:
                ivol_val = np.nan

            turnover_val = float(np.nanmean(turn_w))
            out_rows.append((int(permno), ym, beta_sum, ivol_val, turnover_val))

        n_done += 1
        if n_done % 500 == 0:
            print(f"  daily chars: {n_done:>5d}/{n_total} permnos done", flush=True)

    out = pd.DataFrame(out_rows, columns=["permno", "yearmon",
                                           "beta", "ivol", "turnover"])
    out["permno"] = out["permno"].astype("int32")
    return out


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 70)
    print("Building 13 anomaly characteristics (LPS 2019 Table 2)")
    print("=" * 70)

    print("\nloading inputs ...")
    msf = pd.read_parquet(cfg.RAW_DIR / "crsp_msf.parquet")
    print(f"  msf:   {len(msf):>10,d}")
    stocknames = pd.read_parquet(cfg.RAW_DIR / "crsp_stocknames.parquet")
    print(f"  stocknames: {len(stocknames):>5,d}")
    funda = pd.read_parquet(cfg.RAW_DIR / "funda.parquet")
    print(f"  funda: {len(funda):>10,d}")
    fundq = pd.read_parquet(cfg.RAW_DIR / "fundq.parquet")
    print(f"  fundq: {len(fundq):>10,d}")
    ccm = pd.read_parquet(cfg.RAW_DIR / "ccm_link.parquet")
    print(f"  ccm:   {len(ccm):>10,d}")

    ibes_summary = pd.read_parquet(cfg.RAW_DIR / "ibes_summary.parquet")
    ibes_actuals = pd.read_parquet(cfg.RAW_DIR / "ibes_actuals.parquet")
    ibes_link    = pd.read_parquet(cfg.RAW_DIR / "ibes_link.parquet")
    print(f"  ibes:  summary={len(ibes_summary):,}  actuals={len(ibes_actuals):,}  link={len(ibes_link):,}")

    ff_daily = pd.read_parquet(cfg.RAW_DIR / "ff_daily.parquet")
    print(f"  ff_daily:   {len(ff_daily):>5,d}")

    # Need ~12 mo before earliest formation month for rolling lookbacks.
    daily_window_start = pd.Timestamp("2002-01-01")
    target_ym = pd.period_range(CHAR_START_YM, CHAR_END_YM, freq="M")

    print("\nbuilding ME / MOM / STR / ISSUE ...")
    crsp_chars = build_crsp_monthly_chars(msf)
    print(f"  rows: {len(crsp_chars):,}")

    print("\nbuilding INDMOM ...")
    indmom = build_indmom(msf, stocknames)
    print(f"  rows: {len(indmom):,}")

    print("\nbuilding BM (FF 1992) ...")
    bm = build_bm(funda, ccm, msf)
    print(f"  rows: {len(bm):,}")

    print("\nbuilding INV (asset growth) ...")
    inv = build_inv(funda, ccm)
    print(f"  rows: {len(inv):,}")

    print("\nbuilding ACCRUALS (Sloan 1996) ...")
    accr = build_accruals(funda, ccm)
    print(f"  rows: {len(accr):,}")

    print("\nbuilding ROE (HXZ quarterly) ...")
    roe_events = build_roe(fundq, ccm)
    print(f"  events: {len(roe_events):,}")

    print("\nbuilding SUE (IBES) ...")
    sue_events = build_sue(ibes_actuals, ibes_summary, ibes_link)
    print(f"  events: {len(sue_events):,}")

    print("\nbuilding BETA / IVOL / TURNOVER from daily CRSP ...")
    dsf = pd.read_parquet(
        cfg.RAW_DIR / "crsp_dsf.parquet",
        columns=["permno", "date", "ret", "vol", "shrout"],
    )
    dsf = dsf[dsf["date"] >= daily_window_start].copy()
    print(f"  dsf rows (post {daily_window_start.date()}): {len(dsf):,}")
    daily_chars = build_daily_chars(dsf, ff_daily, target_ym)
    print(f"  rows: {len(daily_chars):,}")
    del dsf

    print("\nassembling final panel ...")
    skel = msf[["permno", "date", "prc"]].copy()
    skel["yearmon"] = skel["date"].dt.to_period("M")
    skel = skel[(skel["yearmon"] >= CHAR_START_YM) & (skel["yearmon"] <= CHAR_END_YM)]
    skel["prc_eom"] = skel["prc"].abs()
    skel = skel[["permno", "yearmon", "prc_eom"]].copy()
    skel["permno"] = skel["permno"].astype("int32")

    out = (skel
           .merge(crsp_chars, on=["permno", "yearmon"], how="left")
           .merge(indmom,     on=["permno", "yearmon"], how="left")
           .merge(bm,         on=["permno", "yearmon"], how="left")
           .merge(inv,        on=["permno", "yearmon"], how="left")
           .merge(accr,       on=["permno", "yearmon"], how="left")
           .merge(daily_chars, on=["permno", "yearmon"], how="left"))

    # ROE: as-of merge on info_date <= month-end.
    out["yearmon_ts"] = out["yearmon"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    out = out.sort_values("yearmon_ts")
    roe_events = roe_events.sort_values("info_date")
    out = pd.merge_asof(
        out, roe_events,
        left_on="yearmon_ts", right_on="info_date",
        by="permno", direction="backward",
    )
    out = out.drop(columns=["info_date"])

    # SUE: as-of merge on rdq_ibes <= month-end, then scale by prc.
    # Stale earnings (> 180 days old) are dropped.
    out = out.sort_values("yearmon_ts")
    sue_events = sue_events.sort_values("rdq_ibes")
    out = pd.merge_asof(
        out, sue_events,
        left_on="yearmon_ts", right_on="rdq_ibes",
        by="permno", direction="backward",
    )
    age_days = (out["yearmon_ts"] - out["rdq_ibes"]).dt.days
    out.loc[age_days > 180, "sue_raw"] = np.nan
    out["sue"] = out["sue_raw"] / out["prc_eom"].where(out["prc_eom"] > 0)
    out = out.drop(columns=["rdq_ibes", "sue_raw"])

    cols = ["permno", "yearmon", "yearmon_ts",
            "me", "bm", "mom", "sue", "indmom", "roe", "inv",
            "beta", "ivol", "issue", "accruals", "turnover", "str_"]
    out = out[cols].sort_values(["permno", "yearmon"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["permno", "yearmon"], keep="last")

    print(f"\nfinal rows: {len(out):,}")
    print("non-NaN counts per char:")
    for c in cols[3:]:
        print(f"  {c:<10s} {out[c].notna().sum():>10,d}")

    cfg.atomic_write_parquet(out, OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
