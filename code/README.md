# Tug of War -- Replication Pipeline

Replication code for **Lou, Polk & Skouras (2019), "A tug of war: Overnight versus intraday expected returns"**, *Journal of Financial Economics* 134, 192-213.

This directory contains the **full pipeline**: data extraction + cleaning + replication of Tables 1, 2, 4, and 5.

## Sample windows

We keep two versions of the cleaned monthly panel:

- **Block A: 2003-10 ~ 2013-12** -- the overlap between LPS's 1993-2013 paper window and our WRDS TAQ Millisecond availability (starts 2003-09-10).  Used as the strict replication block.
- **Block B: 2014-01 ~ 2024-12** -- out-of-sample extension.  Post-publication decay / strengthening of the tug-of-war effect is the headline extension question.

Block A and Block B are stored as `oi_monthly_A.parquet` and `oi_monthly_B.parquet` respectively; `oi_monthly_full.parquet` is the un-split monthly panel.

## Output structure

```
data/
|-- raw/                       # one-time WRDS pulls, never edited
|   |-- ff_daily.parquet
|   |-- ff_monthly.parquet
|   |-- crsp_dsf.parquet
|   |-- crsp_msf.parquet
|   |-- crsp_stocknames.parquet
|   |-- crsp_dsedelist.parquet
|   |-- funda.parquet
|   |-- fundq.parquet
|   |-- ccm_link.parquet
|   |-- ibes_summary.parquet
|   |-- ibes_actuals.parquet
|   |-- ibes_link.parquet
|   `-- taq_crsp_link_msec.parquet
|-- wrds_taq/                  # one parquet per year, server-side aggregates
|   |-- taq_agg_2003.parquet
|   |-- taq_agg_2004.parquet
|   `-- ...
`-- clean/                     # ready-to-use panels
    |-- taq_daily.parquet              # TAQ aggregates joined to PERMNO
    |-- oi_daily.parquet               # daily intraday / overnight returns
    |-- oi_monthly_full.parquet        # 2003-2024 monthly panel (post LPS filters)
    |-- oi_monthly_A.parquet           # 2003-10 ~ 2013-12 (strict replication)
    |-- oi_monthly_B.parquet           # 2014-01 ~ 2024-12 (OOS extension)
    `-- characteristics_monthly.parquet # 13 anomaly characteristics for Table 2/4/5

figures/
|-- table1_replication.csv             # Paper / Block A / Block B comparison (Table 1)
|-- robustness_table1.csv              # 16-cell robustness check
|-- table2_decomposition.csv           # 14 rows x night/day CAPM alpha
|-- table2_strategies.parquet          # monthly long-short night/day return series
|-- table2_block_b_analysis.csv        # Block B classification + decay metrics
|-- table4_fm.csv                      # 14 RHS x 5 dep-var columns Fama-MacBeth
`-- table5_tugofwar.csv                # 11 strategies x (paper / A / B) TugOfWar coefs
```

## Run order

### Phase 1 -- Data extraction

| Step | Script | What it does | Wall time |
|---|---|---|---|
| 0 | edit `00_config.py` | set `PROJECT_DIR` | 1 min |
| 1 | `01_setup_wrds.py` | check connection & subscriptions | 1 min |
| 2 | `02_download_ff_factors.py` | Fama-French daily + monthly | < 1 min |
| 3 | `03_download_crsp.py` | CRSP DSF / MSF / stocknames / delist | 30-60 min |
| 4 | `04_download_compustat.py` | Compustat funda / fundq + CCM link | 5 min |
| 5 | `05_download_ibes.py` | IBES summary + actuals + link | 2 min |
| 6 | `06_taq_aggregate.py 2003 2024` | TAQ server-side aggregation (**slowest**) | 4-12 hours |
| 7 | `07_link_taq_to_crsp.py` | attach PERMNO to TAQ stock-days | 5-10 min |
| 8 | `08_build_overnight_intraday.py` | apply LPS Eq. (3)-(5) to build daily OI returns | 5-10 min |
| 9 | `09_clean_sample.py` | filters + monthly aggregation + B-M-P delisting adj. | 2-5 min |

### Phase 2 -- Table 1 (overnight/intraday persistence)

| Step | Script | What it does | Wall time |
|---|---|---|---|
| 10 | `10_validate_table1.py` | reproduce Table 1 Panel A (Block A + Block B) | < 1 min |
| 11 | `11_summary_table1.py` | 3-column comparison CSV (Paper vs A vs B) | < 1 min |
| 12 | `12_robustness.py` | 8 specs x 2 blocks = 16 cells robustness check | 2-3 min |

### Phase 3 -- Tables 2 / 4 / 5 (anomaly decomposition + predictive regressions)

| Step | Script | What it does | Wall time |
|---|---|---|---|
| 13 | `13_build_characteristics.py` | build 13 anomaly characteristics (ME, BM, MOM, SUE, INDMOM, ROE, INV, BETA, IVOL, ISSUE, ACCRUALS, TURNOVER, STR) | 20-30 min (BETA/IVOL/TURNOVER pass is the bottleneck) |
| 14 | `14_table2_decomposition.py` | Table 2: night/day CAPM-alpha for CRSP + 13 long-short anomalies | < 1 min |
| 15 | `15_table4_fm.py` | Table 4: Fama-MacBeth WLS regression with 14 RHS x 5 dep-var columns | 1-2 min |
| 16 | `16_table5_tugofwar.py` | Table 5: TugOfWar (Eq. 1) predictive regression for 11 strategies | 1-2 min |

### Validation helpers (any-order, optional)

| Script | What it does |
|---|---|
| `_validate.py` | `Checker` PASS/FAIL framework used by the `v*` scripts |
| `v02_validate_ff.py` | sanity-check FF factors |
| `v03_validate_crsp.py` | spot-check CRSP price levels (Apple, etc.) |
| `v04_validate_compustat.py` | spot-check Compustat fundamentals |
| `v05_validate_ibes.py` | spot-check IBES summary + actuals |
| `v06_validate_taq.py` | VWAP cross-check + bucket shape vs Fig. 1 |
| `v08_validate_oi.py` | overnight/intraday identity test |
| `v09_validate_monthly.py` | monthly-panel filters and coverage |
| `inspect_data.py` | quick parquet summary / interactive REPL |

## TAQ aggregation notes

Step 6 (`06_taq_aggregate.py`) is the bottleneck.  Post-2003 TAQ Millisecond (`taqmsec.ctm_<YYYYMMDD>`) aggregates one query per trading day (~5-30 s each, depending on WRDS server load).  Run overnight, **per year**, and resume on failure -- the script caches `taq_agg_<YYYY>.parquet` and skips completed years.

The pre-2003 legacy TAQ aggregation code is in the script but unused, because we have no Millisecond TAQ before 2003-09-10 and the legacy second-level data produces a different VWAP scale.  Hence the project starts at Block A = 2003-10.

## Sample-filter judgment calls (LPS-not-specified)

| Filter | Value | Rationale |
|---|---|---|
| `shrcd in (10, 11)` | common stock only | standard FF; paper implies it |
| `exchcd in (1, 2, 3)` | NYSE / AMEX / NASDAQ | paper uses NYSE breakpoints |
| TAQ-CRSP link `match_lvl <= 2` | CUSIP-validated only | conservative |
| IBES-CRSP link `score <= 3` | CUSIP-validated only | aligned with TAQ link |
| `INTRADAY_ABS_CAP = 1.0` | drop daily abs(ret_intraday) above 100 percent | filter VWAP data errors |
| B-M-P delisting adj. | impute -30 percent (NYSE/AMEX) or -55 percent (NASDAQ) for severe delistings (dlstcd at or above 500) | classical (Beaver-McNichols-Price 2007) |
| NYSE q20 computed before the 5-dollar price filter | exclude bottom NYSE size quintile, then apply price filter | FF textbook convention |

All judgment calls are documented in `HANDOFF.md` Section 5.  See also `PIPELINE_AUDIT_REVIEW.md` for the original sign-off checklist.

## Known sources of small drift vs LPS 2019

1. **TAQ master / link table** -- WRDS link table has been updated multiple times since 2017; ticker reassignments differ across vintages.
2. **Sale-condition filter** -- LPS only state "exclude observations with fewer than 1,000 shares in the first half hour"; they do not list exact `cond` codes.  We use a Bessembinder-style filter (see `06_taq_aggregate.py`).
3. **Open-auction inclusion** -- LPS write that the 9:30 bucket *includes* the open auction.  The legacy TAQ flag for open auction is 'O' for Nasdaq and 'B' for NYSE; we keep them in the VWAP.
4. **CRSP delisting** -- we apply Beaver-McNichols-Price (2007) imputation; LPS do not specify their treatment.
5. **TAQ Millisecond starts 2003-09-10** -- we cannot reproduce the 1993-2003 portion of paper Table 1; our Block A is the second half of LPS's window.

## Dependencies

```bash
pip install wrds pandas numpy pyarrow statsmodels
```

`~/.pgpass` should contain a line like:

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:<your_username>:<your_password>
```

(After `01_setup_wrds.py` runs once interactively, it will offer to create this file for you.)

## Outputs at a glance

| Table | Headline result (Block A vs paper) |
|---|---|
| Table 1 -- `figures/table1_replication.csv` | Night 3F alpha 2.55 pct (t=10.94) vs paper 3.47 pct (t=16.83); Day -1.51 pct (t=-6.01) vs paper -3.02 pct (t=-9.74) |
| Table 2 -- `figures/table2_decomposition.csv` | 9 of 13 anomalies match paper direction on both night and day legs in Block A |
| Table 4 -- `figures/table4_fm.csv` | 89 pct sign-match; 14 of 15 paper-key cells reproduce in Block A |
| Table 5 -- `figures/table5_tugofwar.csv` | 10 of 11 positive coefs in Block A post-warmup, matching paper's "all but one" headline claim |

Block B results in the same CSVs document the OOS decay / persistence of each effect post-2014.
