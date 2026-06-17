# #354 PIT fundamentals panel — session notes (2026-06-17)

Supervised Bloomberg session, branch `claude/phase1a-casy-bloomberg-pull`. The "big" Bloomberg-gated
item (pull-now-or-never), approved this session. **Staging only — NOT integrated into the monoliths and
NOT wired into the connector.** The #354 accessor (`as_of` on `get_fundamentals`, threaded from
`wheel_runner`) is a decision-layer/trio change for a later §2-reviewed session; this persists the DATA.

## Why
The engine reads the single **dateless** `data/bloomberg/sp500_fundamentals.csv` for the EV-path
fundamental fields, so every historical backtest reads 2026 values (lookahead). The existing
`sp500_historical_fundamentals.csv` is dated but only carries `pe_ratio/eps/revenue/ebitda/bvps` —
**not** the EV-consumed fields (notably `eqy_dvd_yld_12m` → BSM carry `q`). This panel fills that gap.

## What
`sp500_fundamentals_pit.csv` — monthly (`Per=M`) dated panel, 2015-01-02→2026-05-31, 503 names,
**72,937 rows**. Columns: `date, ticker` + the EV-consumed fields
`eqy_dvd_yld_12m, beta_raw_overridable, pe_ratio, best_pe_ratio, cur_mkt_cap, return_com_eqy,
tot_debt_to_tot_eqy, free_cash_flow_yield`. Ticker in the snapshot's full `<TICKER> <exch> Equity`
format (so a future PIT accessor keys consistently). IV/realized-vol excluded (covered by the vol_iv
PIT panel); `gics_*` excluded (categorical/quasi-static).

## Validation (latest monthly row ≈ current dateless snapshot)
| name | PIT latest (2026-05-31) eqy_dvd_yld_12m | snapshot (~06-04) | note |
|---|---|---|---|
| AAPL UW Equity | 0.3365 | 0.3416 | ~1.5% — ~4-day recency gap |
| XOM UN Equity | 2.8088 | 2.7195 | ~3% — recency gap |
| CAG UN Equity | 10.5422 | 10.7692 | ~2% — recency gap |

Field/units confirmed (dividend yield in %, same basis as snapshot); the small deltas are the
month-end-vs-snapshot recency gap, not a units/field error. `eqy_dvd_yld_12m` non-null coverage 69.2%
(the rest are non-dividend-payers or names not yet listed in early-window months → NaN, expected).

## NOT done (held — per runbook + rails)
No connector wiring / `as_of` accessor (trio change, §2 review). No integration into monoliths.
No re-baseline. This file is the persisted Bloomberg-gated artifact for the later #354 trio session.
