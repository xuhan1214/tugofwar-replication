# Tug of War -- Replication and OOS Extension

Replication code for **Lou, Polk, and Skouras (2019), "A tug of war: Overnight versus intraday expected returns"**, *Journal of Financial Economics* 134, 192-213.

This repository extends the paper's 1993-2013 results out of sample to 2024 and reports the OOS decay / persistence of every published table.

## Status

| Paper Table | Reproduced in | Sample window |
|---|---|---|
| Table 1 -- overnight/intraday persistence | `figures/table1_replication.csv` | Block A: 2003-10..2013-12 (TAQ-Millisecond-constrained); Block B: 2014-01..2024-12 |
| Table 2 -- 14-anomaly night/day decomposition | `figures/table2_decomposition.csv` | same |
| Table 4 -- Fama-MacBeth WLS regression | `figures/table4_fm.csv` | same |
| Table 5 -- TugOfWar (Eq. 1) predictive regression | `figures/table5_tugofwar.csv` | same |

Tables 3 (futures), 6-8 (institutional holdings) are not reproduced -- they require TRTH / Thomson 13-F data outside our WRDS subscription.

## Headline results (Block A vs paper)

- **Table 1**: Night 3F alpha 2.55 pct (t=10.94) vs paper 3.47 pct (t=16.83); Day -1.51 pct (t=-6.01) vs paper -3.02 pct (t=-9.74).  Direction, significance, and ~50-73 pct of paper's magnitude reproduce.
- **Table 2**: 9 of 13 anomalies match paper direction on **both** night and day legs.  INDMOM and ISSUE match the paper's magnitudes nearly cell-for-cell.
- **Table 4**: 89 pct sign-match with paper across 70 cells; 14 of 15 paper-key cells reproduce.
- **Table 5**: 10 of 11 strategies have a positive TugOfWar coefficient in Block A (post EWMA warm-up), matching the paper's "all but one" headline claim.

Block B (2014-2024 OOS) tells a more nuanced story: momentum-family anomalies strengthen; risk- and accounting-anomalies largely decay or reverse.  See the CSV outputs for the full breakdown.

## Data

### Why two blocks?

The paper's window is 1993-2013, but my institution's data access of WRDS **TAQ Millisecond starts 2003-09-10**.  Before that date there is no millisecond-precision intraday data, so the paper's 9:30-10:00 VWAP "open price" cannot be reproduced. I therefore split the analysable window into:

- **Block A: 2003-10 to 2013-12** -- the overlap with LPS, used as the strict replication block.  About half of the paper's window in months (122 vs 252) and the half during which most anomalies are weaker, so magnitude shortfalls vs paper are expected.
- **Block B: 2014-01 to 2024-12** -- out-of-sample extension, used to study post-publication decay / persistence.

The pre-2003 legacy second-precision TAQ tables (`taq.ct_<YYYYMM>`) **are** available in WRDS, but their VWAP construction differs materially from the millisecond product; we intentionally do not mix them.

### Required raw data (all licensed; not in this repo)

| Source | What we pull | Approx. local size |
|---|---|---|
| `crsp.dsf` | daily stock file, 1993-2024 | ~1.2 GB |
| `crsp.msf` | monthly stock file, 1993-2024 | ~80 MB |
| `crsp.stocknames` + `crsp.dsedelist` | security history + delisting events | ~4 MB |
| `comp.funda` + `comp.fundq` | Compustat annual + quarterly fundamentals, 1990-2024 | ~85 MB |
| `crsp.ccmxpf_lnkhist` | CRSP-Compustat gvkey-permno link | ~1 MB |
| `ibes.statsumu_epsus` + `ibes.actu_epsus` | IBES consensus + actuals (for SUE) | ~10 MB |
| `wrdsapps.ibcrsphist` | IBES-CRSP link, `score <= 3` | ~1 MB |
| `wrdsapps.taqmclink` | TAQ-CRSP link, `match_lvl <= 2` | ~50 MB |
| `taqmsec.ctm_<YYYYMMDD>` | TAQ Millisecond consolidated trades, 2003-09 to 2024-12 (one parquet per year after server-side aggregation) | ~3.2 GB |
| `ff.factors_daily` + `ff.factors_monthly` | Fama-French 3 factors + UMD | <1 MB |

Total local cache after running the full pipeline: roughly **9.4 GB**.

### After cleaning

After applying LPS sample filters (`shrcd in {10,11}`, `exchcd in {1,2,3}`, price floor 5 USD, exclude bottom NYSE size quintile) the monthly stock-month panel sizes are:

- Block A: 220,971 stock-months over 122 months
- Block B: 256,755 stock-months over 131 months

(Paper Table 4 reports 454,825 stock-months over 1993-2013; our two blocks combined are 477,726.)

### How to obtain the data

- **WRDS subscription** with access to the listed schemas is the only way to reproduce.
- The Fama-French factors are also free from Ken French's website if you do not want to pull them via WRDS.
- The paper PDF is behind the *Journal of Financial Economics* paywall and is not redistributable.

To rebuild every parquet from scratch with the pipeline scripts, see `code/README.md`.

## Repository layout

```
code/
|-- README.md                          # full pipeline documentation
|-- 00_config.py                       # paths, filter params, helpers
|-- 01..09_*.py                        # data extraction + cleaning
|-- 10..12_*.py                        # Table 1 replication + robustness
|-- 13_build_characteristics.py        # 13 anomaly characteristics
|-- 14_table2_decomposition.py         # Table 2
|-- 15_table4_fm.py                    # Table 4
|-- 16_table5_tugofwar.py              # Table 5
|-- v0[2-9]_validate_*.py              # per-stage sanity checkers
|-- _validate.py                       # Checker PASS/FAIL framework
`-- inspect_data.py                    # parquet inventory / REPL helper

figures/
|-- table1_replication.csv             # paper vs A vs B comparison
|-- robustness_table1.csv              # 16-cell robustness
|-- table2_decomposition.csv           # 14 rows x night/day CAPM alpha
|-- table2_strategies.parquet          # monthly long-short return series
|-- table2_block_b_analysis.csv        # Block B classification + decay
|-- table4_fm.csv                      # 14 RHS x 5 columns FM regression
`-- table5_tugofwar.csv                # 11 strategies x (paper / A / B)
```

## Quick start

```bash
pip install wrds pandas numpy pyarrow statsmodels
cd code
python 01_setup_wrds.py
# ... then follow the run order in code/README.md
```

A full pipeline run (Phase 1 + 2 + 3) takes roughly 8-14 hours, dominated by the TAQ aggregation step (`06_taq_aggregate.py`).

## Citation

If you build on this code, please cite the original paper:

> Lou, D., Polk, C., and Skouras, S. (2019).  A tug of war: Overnight versus intraday expected returns.  *Journal of Financial Economics*, 134(1), 192-213.

## Reuse

This repository ships **code and aggregated results** only.  The raw datasets accessed via WRDS (CRSP, Compustat, IBES, TAQ) remain subject to WRDS's own licensing terms and cannot be redistributed.
