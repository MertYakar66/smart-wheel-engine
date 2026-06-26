# Theta Enrichment Program — Runbook (2026-06-17, autonomous run)

Operational record for the 5-phase Theta enrichment launched 2026-06-17. The
supervisor + name lists live under `logs/` (gitignored), so they are reproduced
here verbatim. Decision + entitlement/depth evidence:
`docs/THETA_ENTITLEMENT_RETEST_2026-06-17.md`.

## Why this shape
- **Entitlement re-test (live):** greeks/IV history = **404 (not served)**, intraday
  quote/ohlc = **410**, tick `trade` = available but **out of scope** (CLAUDE.md §3).
  EOD all-strikes + open_interest is the only path. Per-call window max = **365 days**
  (`--lookback-days 364`; 365 → 366-day window → HTTP 400).
- **Depth finding:** a flat 365-day re-pull of names already held is mostly
  empty-window rescanning (90d and 180d returned byte-identical data for a
  recently-listed NVDA contract) and **times out at 180s on the heaviest names**.
  Real incremental data exists only for contracts traded >90d before expiry
  (term-structure / LEAP history) — low value for a 30–45 DTE wheel, feeds the
  dormant NS/skew subsystem. So the top-20 "depth" is run to a **staging dir** with
  a raised 300s timeout, and the genuine top-20 win is **BRKB** (the missing #10).

## Phases (sequential; Theta caps at 4 conns)

| Ph | Tickers | Cadence | Lookback | Out-dir | Role |
|----|---------|---------|----------|---------|------|
| A | BRKB | all | 90 | `option_history/` (LIVE) | fill missing top-20 #10 → tradeable larder |
| B | top-20 | all | 364 | `option_history_deep365/` (staging) | full 0–365 DTE term-structure depth |
| C | SPX,NDX,RUT,VIX,XSP | weekly | 90 | `index_reference/option_history/` | index GEX reference (out of ranker) |
| D | 45 delisted | all | 90 | `option_history_delisted/` (staging) | survivor-bias tail calibration (out of ranker) |
| E | next-50 by mkt cap | all | 90 | `option_history/` (LIVE) | universe 150→200 tradeable |

All phases: `--all-strikes --include-oi --start 2016-01-01 --end <today> --resume`,
atomic per-partition writes, supervisor restarts Theta on degradation.
B/C use `--read-timeout 300` (heavy chains); A/D/E use 180.

Staging dirs (`option_history_deep365`, `option_history_delisted`) are promoted /
wired deliberately after review — they **never auto-enter** `rank_candidates_by_ev`.

## Outcome on disk (verified 2026-06-25)

What actually landed (disk-verified — see `docs/DATA_INVENTORY.md` §3 for full counts):

| Ph | Planned | On disk | Status |
|----|---------|---------|--------|
| A | BRKB → `option_history/` | BRKB present; `option_history/` = **154** dirs / 390.1M rows | ✅ done |
| B | top-20 → `option_history_deep365/` | **8/20** names (AAPL, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA); 17.5M rows | ⚠ partial (heavy-name timeouts, §6) |
| C | SPX,NDX,RUT,VIX,XSP → `index_reference/` | **4/5** roots (SPX, NDX, RUT, XSP) + SPY/QQQ from Phase-2 = 6 total; **VIX absent**; 38.9M rows | ⚠ partial |
| D | 45 delisted → `option_history_delisted/` | **10/45** names (ABMD, ATVI, FRC, PXD, RE, SBNY, SGEN, SIVB, SPLK, TWTR); 9.7M rows | ⚠ partial |
| E | next-50 → `option_history/` (150→200) | net **+0** beyond the larder+BRKB (154 dirs) | ⏸ not landed |

## Name lists (verbatim)

**top-20 by market cap** (`logs/larder_top20_deep.txt`):
```
NVDA,AAPL,GOOGL,GOOG,MSFT,AMZN,META,AVGO,TSLA,BRKB,WMT,LLY,JPM,XOM,V,JNJ,MU,MA,COST,ORCL
```
**next-50 by market cap, not already in larder** (`logs/larder_next50.txt`):
```
AMD,GEV,ANET,APH,UBER,APP,ETN,BX,WELL,SYK,CB,PH,COF,IBKR,HCA,MCK,SNDK,GLW,SO,DELL,CEG,NOC,DUK,GD,WM,HWM,TT,ICE,STX,WMB,MRSH,PWR,PNC,KKR,USB,JCI,BK,MCO,MSI,ITW,CMI,KMI,ECL,MNST,EMR,CTAS,VLO,RCL,PSX,AON
```
**delisted survivor-bias set, all probed serving** (`logs/larder_delisted.txt`):
```
SIVB,FRC,SBNY,ATVI,TWTR,SGEN,PXD,SPLK,RE,ABMD,XLNX,CTXS,MXIM,ALXN,FLIR,TIF,CXO,NBL,ETFC,AGN,CELG,RHT,APC,WLTW,INFO,CERN,NLSN,PBCT,KSU,DRE,CTLT,MYL,DISCA,DISCK,VIAB,WCG,ARNC,ANTM,COG,FBHS,RTN,AABA,STI,BBT,CBS
```
**index roots:** `SPX,NDX,RUT,VIX,XSP`

## Monitoring / completion
- Supervisor: `logs/_RUN_THETA_ENRICH.sh`; rolling log `logs/theta_enrich.log`;
  per-phase flags `logs/theta_enrich_{A,B,C,D,E}_*.flag`; whole-program
  `logs/theta_enrich_DONE.flag`.
- Checkpoints appended to `logs/theta_pull_checkpoints.log`.
- On completion: validate row counts per phase; **promote** staging dirs only after
  review (decide wiring of deep365 → NS/skew subsystem and delisted → tail/forward-dist).
