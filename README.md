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

## Important: data not included

This repository contains **code and aggregated results only**.  The underlying raw data are licensed and cannot be redistributed:

- **CRSP**, **Compustat**, **IBES**, **TAQ Millisecond** -- WRDS-licensed.
- **Fama-French factors** -- available via WRDS or Ken French's website.
- **Original paper PDF** -- copyrighted by *Journal of Financial Economics*.

To run the pipeline end-to-end you need a WRDS subscription with access to the listed schemas.  See `code/README.md` for the full setup and run order.

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
