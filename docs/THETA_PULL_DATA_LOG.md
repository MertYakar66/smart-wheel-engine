# Theta Option-History Pull — Data Log

A running, **prepend** log of *what data has been pulled* — **names, titles, and dates only**
(no actual option data). One snapshot is added on **top** every ~4 hours.

- **Data title:** EOD option-history chains — *all strikes, calls + puts, daily open interest*
  (no IV / Greeks). One "chain" = one (name, expiration) end-of-day series across the run window.
- **Source:** `scripts/pull_theta_option_history.py`, run `top-150 Mag7-first, 2016..2026-06-01,
  all-strikes, 90d`. Universe = 150 tickers (`logs/larder_top150.txt`), Mag7 first.
- **Names** = ticker symbols. **Titles** = the data series type. **Dates** = expiration span captured.

**Maintenance:** entries are added by the **30-min Theta health-monitor loop**, which prepends a
fresh snapshot the first time it runs after each 4h boundary (~03 / 07 / 11 / 15 / 19 / 23 local).
This replaced a standalone 4h cron (job `76fdeb6c`, retired 2026-06-03 07:13) that kept missing its
idle window. Read-only; never touches the pull. The loop is **session-only** — if this Claude
session exits, both the monitor and these updates stop until restarted. Newest entry is inserted
directly below the marker line.

---

# THETA DATA — FULL INVENTORY (everything we have on disk, for Bloomberg matching)

> **Canonical inventory is now `docs/DATA_INVENTORY.md`** (Bloomberg + Theta + derived, with
> per-dataset column schemas/dtypes). This table is the Theta-only quick reference; keep it in sync
> with that doc. **Refreshed 2026-06-25** from `scripts/inventory_data.py` (footer/row-group stats).

_Total Theta footprint **≈ 12 GB** local-only (git-ignored, not on GitHub). This is a static reference
(distinct from the rolling 4-hourly log below). Use it to see which Theta datasets/names/dates you
already hold so you can match/avoid re-pulling on Bloomberg._

**Shared universe:** unless noted, a dataset covers the **S&P 500 (~500 names)** = *Universe A* (listed
once at the bottom). Small/special universes are spelled out inline.

| # | Dataset (`data_processed/theta/…`) | Type / title | Symbols | Date coverage | Granularity | Files / size / rows | Status |
|---|---|---|---|---|---|---|---|
| 1 | `option_history/` | **Full-depth EOD option chains** (ALL strikes, C+P) + open interest | **154** (149 larder + BRKB + 4 orphans) | exp **2016-01-08 … 2026-08-21**; bars ~2016 → 2026-06-17 | one (ticker,expiration) = full chain EOD | 71,027 files / 7.7 GB / **390,119,692** | **COMPLETE** (larder pull, DONE flag 2026-06-17) |
| 2 | `option_history_banded_backup_2026-06-01/` | **EOD option chains, Δ-BANDED strikes** (not full depth) + OI | **503** (Universe A) | exp **2016-01-15 … 2026-05-22** | one (ticker,expiration) | 51,729 files / 2.2 GB / 66,574,386 | COMPLETE (prior pull, backed up 06-01) |
| 2a | `option_history_deep365/` ⭑NEW | EOD chains, **0–365 DTE term-structure depth** (Phase B); **staging, out-of-ranker** | **8** (AAPL, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA) | exp **2016-01-08 … 2026-06-18** | one (ticker,expiration) | 1,682 files / 368 MB / 17,528,832 | COMPLETE (staging) |
| 2b | `option_history_delisted/` ⭑NEW | EOD chains for **delisted / acquired names** (Phase D survivor-bias); **staging, out-of-ranker** | **10** (ABMD, ATVI, FRC, PXD, RE, SBNY, SGEN, SIVB, SPLK, TWTR) | exp **2016-01-08 … 2024-02-16** | one (ticker,expiration) | 1,834 files / 164 MB / 9,707,709 | COMPLETE (staging) |
| 2c | `index_reference/option_history/` ⭑NEW | **Index / ETF GEX-reference chains** (Phase C + SPY/QQQ Phase-2); **out-of-ranker** | **6** (NDX, QQQ, RUT, SPX, SPY, XSP) | exp **2016-01-08 … 2026-07-31** | one (ticker,expiration) | 1,853 files / 737 MB / 38,854,575 | COMPLETE |
| 3 | `chains/` | Full option-chain **snapshots** w/ greeks+IV+quotes+OI | 495 | snaps **2026-04-23, 05-24, 06-01, 06-05** | point-in-time chain | 1,521 files / 32 MB / 116,214 | COMPLETE (4 snapshots) |
| 4 | `iv_surface/` | Per-name **IV surface snapshots** (strike×right×δ×iv×mid×dte) | 502 (Univ A + 8 ETFs) | snaps **04-23, 05-24, 06-01** | point-in-time surface | 558 files / 14 MB / 364,192 | COMPLETE |
| 5 | `iv_surface_history/` | **IV surface TIME-SERIES** (daily, back-solve) | **4 only** (A, AAPL, ABBV, ABNB) | **2026-04-13 … 2026-06-03** | daily, ticker/year/date | 108 files / 1.5 MB / 53,725 | PILOT (4 names) |
| 6 | `iv_history/` | **ATM-IV TIME-SERIES** (daily `iv_atm`) | 497 (Universe A) | **2015-01-02 … 2026-03-20** (~11 yr) | daily per name | 497 files / 25 MB / 1,291,775 | COMPLETE — long history |
| 7 | `index_options_chains/` | Index-option full-chain snapshots (greeks+IV+OI) | 8 indices* | snaps **04-23, 05-24, 06-01** | point-in-time | 21 files / 964 KB / 8,508 | COMPLETE |
| 8 | `index_options_surfaces/` | Index-option IV surfaces | 8 indices* | snaps **04-23, 05-24, 06-01** | point-in-time | 21 files / 1.6 MB / 66,230 | COMPLETE |
| 9 | `option_ohlc/` | **Per-contract daily OHLC + bid/ask** (selected contracts) | 502 | bars **~2026-02-23 … 05-22**; 4 expirations | per-contract daily | 1,507 files / 15 MB / 46,886 | COMPLETE — narrow window |
| 10 | `stocks_eod/` | **Equity EOD OHLCV** (underlying stock bars) | 493 (Universe A) | **2024-04-23 … 2026-03-20** (~2 yr) | daily per name | 493 files / 12 MB / 233,912 | COMPLETE |
| 11 | `vix_family/` | VIX term-structure index OHLC | VIX, VIX3M, VIX6M, VIX9D | **2023-04-24 … 2026-04-22** | daily | 1 file / 52 KB / 3,030 | COMPLETE |
| 12 | `../vol_indices.parquet` (+ `_wide`) | **Vol-index time series** (long + wide) | VIX, VVIX, SKEW, MOVE, GVZ, OVX, VXN, VIX3M/6M/9D | daily (35,062 long / 3,783 wide rows) | daily | 548 KB | COMPLETE |

\* **Index universe (8):** SPX, SPXW, NDX, RUT, DJX, VIX, XSP, AAPL.  
**iv_surface ETF add-ons (8):** SPY, QQQ, DIA, IWM, XLE, XLF, XLK, XLV.  
**⭑NEW staging trees (2a/2b/2c)** were added by the 2026-06-17 enrichment run
(`docs/THETA_ENRICH_RUNBOOK_2026-06-17.md`); they feed dormant subsystems (skew / tail-calibration /
dealer-GEX) and **never enter `rank_candidates_by_ev`**. On-disk rosters are partial vs plan
(deep365 8/20, delisted 10/45, index_reference 6 roots; VIX root not yet present).  
**Empty / not pulled:** Theta corp-actions never produced output (404); there is **no**
`corporate_actions/` dir under `data_processed/theta/`.

### Field schemas (so you know which Bloomberg fields to match)

- **option_history, banded_backup, deep365, delisted, index_reference** (22, identical schema): `symbol, expiration, strike, right, created, last_trade, open, high, low, close, volume, count, bid_size, bid_exchange, bid, bid_condition, ask_size, ask_exchange, ask, ask_condition, open_interest, ticker` — EOD OHLC + quote (bid/ask w/ size, exchange, condition) + volume/count + OI. **No greeks/IV** in these (greeks/IV history is 404/not-entitled at this tier).
- **chains** (22 data fields + 1 `__index_level_0__` pandas index = 23 cols on disk): `symbol, expiration, strike, right, delta, theta, vega, rho, epsilon, lambda, iv, iv_error, underlying_timestamp, underlying_price, bid, ask, bid_size, ask_size, open_interest, mid, ticker, snapshot_date` — greeks (Δ,Θ,V,ρ,ε,λ) + IV + quotes + OI. **Snapshot only** (4 dates).
- **iv_surface / index_options_surfaces** (9 data fields + 1 `__index_level_0__` pandas index = 10 cols on disk): `strike, right, delta, iv, mid, expiration, dte, ticker, snapshot_date`.
- **iv_surface_history** (8): `date, ticker, expiration, dte, strike, right, iv, mid`.
- **iv_history** (4): `iv_atm, ticker, source, date` — single ATM-IV per day, **11-yr history**.
- **option_ohlc** (12): `open, high, low, close, volume, bid, ask, ticker, expiration, strike, right, date`.
- **stocks_eod** (7): `open, high, low, close, volume, ticker, date`.
- **vix_family / vol_indices** (7): `open, high, low, close, symbol, source, date`; `vol_indices_wide` has per-index close columns (`vix_close, vvix_close, skew_close, move_close, gvz_close, ovx_close, vxn_close, vix3m_close, vix6m_close, vix9d_close`).

### Quick read for Bloomberg matching
- **Deepest historical asset:** the **full-depth larder (#1)** — 154 names, 2016→2026, all-strikes EOD OHLC+quotes+OI (390.1M rows), **no greeks/IV**; the **Δ-banded backup (#2)** covers all 503 names at banded strike depth (same fields). Both pulls are **complete** (larder DONE 2026-06-17).
- **Greeks/IV exist only as recent snapshots** (#3,#4,#7,#8 — three dates Apr–Jun 2026) plus the **11-yr ATM-IV series (#6)** and a **4-name daily IV-surface pilot (#5)**. If you need historical greeks/IV across the universe, Bloomberg is the gap-filler.
- **Equity OHLCV (#10)** only goes back ~2 yr (2024-04→2026-03); **ATM-IV (#6)** and **vol indices (#12)** go back to 2015.

### Universe A — the ~500 S&P names (datasets #2, #6, #10 and the base of #3,#4,#9)
`A, AAPL, ABBV, ABNB, ABT, ACGL, ACN, ADBE, ADI, ADM, ADP, ADSK, AEE, AEP, AES, AFL, AIG, AIZ, AJG, AKAM, ALB, ALGN, ALL, ALLE, AMAT, AMCR, AMD, AME, AMGN, AMP, AMT, AMZN, ANET, AON, AOS, APA, APD, APH, APO, APP, APTV, ARE, ATO, AVB, AVGO, AVY, AWK, AXON, AXP, AZO, BA, BAC, BALL, BAX, BBY, BDX, BEN, BG, BIIB, BK, BKNG, BKR, BLDR, BLK, BMY, BR, BRO, BSX, BX, BXP, C, CAG, CAH, CARR, CAT, CB, CBOE, CBRE, CCI, CCL, CDNS, CDW, CEG, CF, CFG, CHD, CHRW, CHTR, CI, CINF, CL, CLX, CMCSA, CME, CMG, CMI, CMS, CNC, CNP, COF, COIN, COO, COP, COR, COST, CPAY, CPB, CPRT, CPT, CRL, CRM, CRWD, CSCO, CSGP, CSX, CTAS, CTRA, CTSH, CTVA, CVS, CVX, D, DAL, DASH, DAY, DD, DDOG, DE, DECK, DELL, DG, DGX, DHI, DHR, DIS, DLR, DLTR, DOC, DOV, DOW, DPZ, DRI, DTE, DUK, DVA, DVN, DXCM, EA, EBAY, ECL, ED, EFX, EG, EIX, EL, ELV, EME, EMR, EOG, EPAM, EQIX, EQR, EQT, ERIE, ES, ESS, ETN, ETR, EVRG, EW, EXC, EXE, EXPD, EXPE, EXR, F, FANG, FAST, FCX, FDS, FDX, FE, FFIV, FICO, FIS, FISV, FITB, FOX, FOXA, FRT, FSLR, FTNT, FTV, GD, GDDY, GE, GEHC, GEN, GEV, GILD, GIS, GL, GLW, GM, GNRC, GOOG, GOOGL, GPC, GPN, GRMN, GS, GWW, HAL, HAS, HBAN, HCA, HD, HIG, HII, HLT, HOLX, HON, HOOD, HPE, HPQ, HRL, HSIC, HST, HSY, HUBB, HUM, HWM, IBKR, IBM, ICE, IDXX, IEX, IFF, INCY, INTC, INTU, INVH, IP, IPG, IQV, IR, IRM, ISRG, IT, ITW, IVZ, J, JBHT, JBL, JCI, JKHY, JNJ, JPM, K, KDP, KEY, KEYS, KHC, KIM, KKR, KLAC, KMB, KMI, KO, KR, KVUE, L, LDOS, LEN, LH, LHX, LII, LIN, LKQ, LLY, LMT, LNT, LOW, LRCX, LULU, LUV, LVS, LW, LYB, LYV, MA, MAA, MAR, MAS, MCD, MCHP, MCK, MCO, MDLZ, MDT, MET, META, MGM, MHK, MKC, MLM, MMC, MMM, MNST, MO, MOH, MOS, MPC, MPWR, MRK, MRNA, MS, MSCI, MSFT, MSI, MTB, MTCH, MTD, MU, NCLH, NDAQ, NDSN, NEE, NEM, NFLX, NI, NKE, NOC, NOW, NRG, NSC, NTAP, NTRS, NUE, NVDA, NVR, NWS, NWSA, NXPI, O, ODFL, OKE, OMC, ON, ORCL, ORLY, OTIS, OXY, PANW, PAYC, PAYX, PCAR, PCG, PEG, PEP, PFE, PFG, PG, PGR, PH, PHM, PKG, PLD, PLTR, PM, PNC, PNR, PNW, PODD, POOL, PPG, PPL, PRU, PSA, PSKY, PSX, PTC, PWR, PYPL, Q, QCOM, RCL, REG, REGN, RF, RJF, RL, RMD, ROK, ROL, ROP, ROST, RSG, RTX, RVTY, SBAC, SBUX, SCHW, SHW, SJM, SLB, SMCI, SNA, SNPS, SO, SOLS, SOLV, SPG, SPGI, SPY, SRE, STE, STLD, STT, STX, STZ, SW, SWK, SWKS, SYF, SYK, SYY, T, TAP, TDG, TDY, TECH, TEL, TER, TFC, TGT, TJX, TKO, TMO, TMUS, TPL, TPR, TRGP, TRMB, TROW, TRV, TSCO, TSLA, TSN, TT, TTD, TTWO, TXN, TXT, TYL, UAL, UBER, UDR, UHS, ULTA, UNH, UNP, UPS, URI, USB, V, VICI, VLO, VLTO, VMC, VRSK, VRSN, VRTX, VST, VTR, VTRS, VZ, WAB, WAT, WBD, WDAY, WDC, WEC, WELL, WFC, WM, WMB, WMT, WRB, WSM, WST, WTW, WY, WYNN, XEL, XOM, XYL, XYZ, YUM, ZBH, ZBRA, ZTS`
_(503 names in the banded backup; individual datasets differ by a handful as membership changed — e.g. stocks_eod has 493, iv_history 497, chains 495, iv_surface/option_ohlc 502 incl. ETFs.)_

---

<!-- NEW ENTRIES GO DIRECTLY BELOW THIS LINE -->

## 2026-06-17 15:55 local — *** THETA LARDER PULL COMPLETE (Phase-1 + Phase-2) ***

The full 2026-06-01 re-scoped Theta pull is **DONE** (	heta_full_2026-06-01_DONE.flag written 2026-06-17 15:55:29 · `=== THETA LARDER COMPLETE ===`).

**Phase-1 — wheel larder (option_history/):** 149 of 150 names on disk (BRK/B unpullable — symbol-format bug, `BRKB` queued for the next pull via PR #413; see docs/THETA_PULL_AUDIT_2026-06-15.md). ~69,196 (name,expiration) EOD chains, all-strikes + OI, 2016-01 → 2026-06. Final worker run: 7,028 expirations pulled / 62,096 skipped (resume) / 68 failed (expected 472 no-data); **742,452 contracts · 33,051,399 rows**; completed 2026-06-16 23:14 (~27.9h). Status mix C=140 · N=0 · X=9 (+4 non-larder orphans AES/AFL/AMD/AOS).

**Phase-2 — SPY/QQQ index-reference surface (index_reference/):** auto-started 2026-06-16 23:14, completed 2026-06-17 15:55 (~16.7h). 2/2 tickers, **852 weekly expirations / 0 skip / 0 fail**, all-strikes + OI, 45-day lookback, 2018-01 → 2026-06; **275,938 contracts · 10,746,549 rows** (852 partitions). Separate dir — reference only, never enters the candidate ranker.

**Grand total ≈ 43.8M rows.** Pull is complete; the 30-min monitor loop has stopped. (Per-name option_history inventory unchanged from the 2026-06-17 08:00 entry below — static since Phase-1 finished.)


## 2026-06-17 08:00 local (logged 08:02) · ~69,196 chains · 149 names · +0 in 8h · Phase-1 complete; Phase-2 running (idxref=576)

**Phase-1 (the wheel larder) remains COMPLETE and static** — 149 larder names, ~69,196 chains (33.05M rows), no new option_history writes this 8h window (+0). Status mix unchanged: **C=140 · N=0 · X=9** (+4 non-larder orphans excluded). BRK/B is the one unpullable name (BRKB queued for the next pull, PR #413).

**Phase-2 (SPY/QQQ index-reference surface) still RUNNING** — worker PID 18120 alive, **576 partitions** written so far. Climbing the weekly chain: **SPY→2023-02, QQQ→2024-04** (works oldest→newest from 2018). Growth ~25-30 partitions/30min, slowing as it reaches the heavier 2024-26 SPY chains. Separate index_reference/ dir (never enters the candidate ranker). P2_FLAG / DONE flag still absent — estimated ~2-4h remaining (QQQ leads SPY; both must reach ~2026-06).

**Note:** option_history is static; health is judged by index_reference writes (idxref rising). When Phase-2 sets P2_FLAG and the supervisor writes the DONE flag, the whole pull is complete.

| Ticker | Chains | MinExp | MaxExp | New(8h) | St |
|---|---------:|---|---|---------:|---|
| AAPL | 581 | 2016-01 | 2026-06 | 0 | C |
| ABBV | 544 | 2016-01 | 2026-06 | 0 | C |
| ABNB | 286 | 2020-12 | 2026-06 | 0 | C |
| ABT | 532 | 2016-01 | 2026-06 | 0 | C |
| ACN | 544 | 2016-01 | 2026-06 | 0 | C |
| ADBE | 486 | 2016-01 | 2026-06 | 0 | C |
| ADI | 460 | 2016-01 | 2026-06 | 0 | C |
| ADP | 480 | 2016-01 | 2026-06 | 0 | C |
| ADSK | 479 | 2016-01 | 2026-06 | 0 | C |
| AEP | 125 | 2016-01 | 2026-05 | 0 | X |
| AMAT | 544 | 2016-01 | 2026-06 | 0 | C |
| AMGN | 544 | 2016-01 | 2026-06 | 0 | C |
| AMT | 366 | 2016-01 | 2026-05 | 0 | X |
| AMZN | 581 | 2016-01 | 2026-06 | 0 | C |
| AVGO | 580 | 2016-01 | 2026-06 | 0 | C |
| AXP | 544 | 2016-01 | 2026-06 | 0 | C |
| BA | 544 | 2016-01 | 2026-06 | 0 | C |
| BAC | 544 | 2016-01 | 2026-06 | 0 | C |
| BIIB | 544 | 2016-01 | 2026-06 | 0 | C |
| BKNG | 432 | 2018-03 | 2026-06 | 0 | C |
| BLK | 459 | 2016-01 | 2026-06 | 0 | C |
| BMY | 544 | 2016-01 | 2026-06 | 0 | C |
| BSX | 442 | 2016-01 | 2026-06 | 0 | C |
| C | 544 | 2016-01 | 2026-06 | 0 | C |
| CAT | 544 | 2016-01 | 2026-06 | 0 | C |
| CDNS | 159 | 2016-01 | 2026-06 | 0 | C |
| CHTR | 528 | 2016-01 | 2026-06 | 0 | C |
| CI | 502 | 2016-01 | 2026-06 | 0 | C |
| CL | 511 | 2016-01 | 2026-06 | 0 | C |
| CMCSA | 544 | 2016-01 | 2026-06 | 0 | C |
| CME | 484 | 2016-01 | 2026-06 | 0 | C |
| COIN | 267 | 2021-04 | 2026-06 | 0 | C |
| COP | 544 | 2016-01 | 2026-06 | 0 | C |
| COST | 544 | 2016-01 | 2026-06 | 0 | C |
| CRM | 544 | 2016-01 | 2026-06 | 0 | C |
| CRWD | 357 | 2019-06 | 2026-06 | 0 | C |
| CSCO | 544 | 2016-01 | 2026-06 | 0 | C |
| CSX | 487 | 2016-01 | 2026-06 | 0 | C |
| CVS | 544 | 2016-01 | 2026-06 | 0 | C |
| CVX | 544 | 2016-01 | 2026-06 | 0 | C |
| DASH | 286 | 2020-12 | 2026-06 | 0 | C |
| DDOG | 320 | 2019-10 | 2026-06 | 0 | C |
| DE | 544 | 2016-01 | 2026-06 | 0 | C |
| DHR | 312 | 2016-01 | 2026-06 | 0 | C |
| DIS | 544 | 2016-01 | 2026-06 | 0 | C |
| DLTR | 499 | 2016-01 | 2026-06 | 0 | C |
| EA | 536 | 2016-01 | 2026-06 | 0 | C |
| EBAY | 544 | 2016-01 | 2026-06 | 0 | C |
| ELV | 118 | 2022-07 | 2026-05 | 0 | X |
| EOG | 544 | 2016-01 | 2026-06 | 0 | C |
| EQIX | 125 | 2016-01 | 2026-05 | 0 | X |
| EXC | 125 | 2016-01 | 2026-05 | 0 | C |
| EXPE | 544 | 2016-01 | 2026-06 | 0 | C |
| F | 544 | 2016-01 | 2026-06 | 0 | C |
| FCX | 544 | 2016-01 | 2026-06 | 0 | C |
| FDX | 544 | 2016-01 | 2026-06 | 0 | C |
| FISV | 237 | 2016-01 | 2026-06 | 0 | C |
| FTNT | 247 | 2016-01 | 2026-06 | 0 | C |
| GE | 544 | 2016-01 | 2026-06 | 0 | C |
| GILD | 544 | 2016-01 | 2026-06 | 0 | C |
| GM | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOG | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOGL | 579 | 2016-01 | 2026-06 | 0 | C |
| GS | 544 | 2016-01 | 2026-06 | 0 | C |
| HD | 544 | 2016-01 | 2026-06 | 0 | C |
| HON | 475 | 2016-01 | 2026-06 | 0 | C |
| HOOD | 252 | 2021-08 | 2026-06 | 0 | C |
| IBM | 544 | 2016-01 | 2026-06 | 0 | C |
| INTC | 544 | 2016-01 | 2026-06 | 0 | C |
| INTU | 414 | 2016-01 | 2026-06 | 0 | C |
| ISRG | 544 | 2016-01 | 2026-06 | 0 | C |
| JNJ | 544 | 2016-01 | 2026-06 | 0 | C |
| JPM | 544 | 2016-01 | 2026-06 | 0 | C |
| KLAC | 317 | 2016-01 | 2026-05 | 0 | X |
| KMB | 463 | 2016-01 | 2026-06 | 0 | C |
| KO | 544 | 2016-01 | 2026-06 | 0 | C |
| LIN | 114 | 2018-11 | 2026-05 | 0 | X |
| LLY | 524 | 2016-01 | 2026-06 | 0 | C |
| LMT | 502 | 2016-01 | 2026-06 | 0 | C |
| LOW | 505 | 2016-01 | 2026-06 | 0 | C |
| LRCX | 460 | 2016-01 | 2026-06 | 0 | C |
| LULU | 523 | 2016-01 | 2026-06 | 0 | C |
| MA | 524 | 2016-01 | 2026-06 | 0 | C |
| MAR | 496 | 2016-01 | 2026-06 | 0 | C |
| MCD | 524 | 2016-01 | 2026-06 | 0 | C |
| MCHP | 309 | 2016-01 | 2026-06 | 0 | C |
| MDLZ | 523 | 2016-01 | 2026-06 | 0 | C |
| MDT | 544 | 2016-01 | 2026-06 | 0 | C |
| META | 254 | 2021-07 | 2026-06 | 0 | C |
| MMM | 544 | 2016-01 | 2026-06 | 0 | C |
| MO | 544 | 2016-01 | 2026-06 | 0 | C |
| MPC | 513 | 2016-01 | 2026-05 | 0 | C |
| MRK | 544 | 2016-01 | 2026-06 | 0 | C |
| MS | 544 | 2016-01 | 2026-06 | 0 | C |
| MSFT | 580 | 2016-01 | 2026-06 | 0 | C |
| MU | 544 | 2016-01 | 2026-06 | 0 | C |
| NEE | 240 | 2016-01 | 2026-06 | 0 | C |
| NEM | 544 | 2016-01 | 2026-06 | 0 | C |
| NFLX | 544 | 2016-01 | 2026-06 | 0 | C |
| NKE | 544 | 2016-01 | 2026-06 | 0 | C |
| NOW | 537 | 2016-01 | 2026-06 | 0 | C |
| NVDA | 580 | 2016-01 | 2026-06 | 0 | C |
| NXPI | 527 | 2016-01 | 2026-05 | 0 | X |
| ON | 348 | 2016-01 | 2026-06 | 0 | C |
| ORCL | 544 | 2016-01 | 2026-06 | 0 | C |
| ORLY | 125 | 2016-01 | 2026-05 | 0 | X |
| OXY | 544 | 2016-01 | 2026-06 | 0 | C |
| PANW | 544 | 2016-01 | 2026-06 | 0 | C |
| PEP | 544 | 2016-01 | 2026-06 | 0 | C |
| PFE | 544 | 2016-01 | 2026-06 | 0 | C |
| PG | 544 | 2016-01 | 2026-06 | 0 | C |
| PGR | 191 | 2016-01 | 2026-06 | 0 | C |
| PLD | 125 | 2016-01 | 2026-05 | 0 | C |
| PLTR | 295 | 2020-10 | 2026-06 | 0 | C |
| PM | 544 | 2016-01 | 2026-06 | 0 | C |
| QCOM | 544 | 2016-01 | 2026-06 | 0 | C |
| REGN | 516 | 2016-01 | 2026-06 | 0 | C |
| ROP | 125 | 2016-01 | 2026-05 | 0 | X |
| ROST | 367 | 2016-01 | 2026-06 | 0 | C |
| RTX | 319 | 2020-04 | 2026-06 | 0 | C |
| SBUX | 544 | 2016-01 | 2026-06 | 0 | C |
| SCHW | 516 | 2016-01 | 2026-06 | 0 | C |
| SHW | 221 | 2016-01 | 2026-05 | 0 | C |
| SLB | 544 | 2016-01 | 2026-06 | 0 | C |
| SMCI | 237 | 2016-01 | 2026-06 | 0 | C |
| SNPS | 181 | 2016-01 | 2026-06 | 0 | C |
| SPGI | 458 | 2016-05 | 2026-06 | 0 | C |
| T | 544 | 2016-01 | 2026-06 | 0 | C |
| TGT | 544 | 2016-01 | 2026-06 | 0 | C |
| TJX | 367 | 2016-01 | 2026-06 | 0 | C |
| TMO | 280 | 2016-01 | 2026-06 | 0 | C |
| TMUS | 544 | 2016-01 | 2026-06 | 0 | C |
| TSLA | 581 | 2016-01 | 2026-06 | 0 | C |
| TTD | 436 | 2017-04 | 2026-06 | 0 | C |
| TXN | 544 | 2016-01 | 2026-06 | 0 | C |
| UAL | 544 | 2016-01 | 2026-06 | 0 | C |
| ULTA | 528 | 2016-01 | 2026-06 | 0 | C |
| UNH | 544 | 2016-01 | 2026-06 | 0 | C |
| UNP | 544 | 2016-01 | 2026-06 | 0 | C |
| UPS | 544 | 2016-01 | 2026-06 | 0 | C |
| V | 544 | 2016-01 | 2026-06 | 0 | C |
| VRTX | 544 | 2016-01 | 2026-06 | 0 | C |
| VZ | 544 | 2016-01 | 2026-06 | 0 | C |
| WDAY | 544 | 2016-01 | 2026-06 | 0 | C |
| WDC | 544 | 2016-01 | 2026-06 | 0 | C |
| WFC | 544 | 2016-01 | 2026-06 | 0 | C |
| WMT | 544 | 2016-01 | 2026-06 | 0 | C |
| XOM | 544 | 2016-01 | 2026-06 | 0 | C |
| XYZ | 72 | 2025-01 | 2026-06 | 0 | C |



## 2026-06-17 00:00 local (logged 00:07) · ~69,196 chains · 149 names · +182 in 8h · PHASE-1 COMPLETE; Phase-2 running

### *** PHASE-1 LARDER COMPLETE — 2026-06-16 23:14:43 ***
The full-depth EOD option-history larder is DONE. Final run stats (from the log):
- **150/150 tickers** processed; **7,028 expirations** pulled this run / 62,096 skipped (resume) / 68 failed (expected 472 no-data dates).
- **742,452 contracts · 33,051,399 rows** written this run; worker elapsed ~27.9h (PID 57396, exited at 23:14:43; === PHASE1 DONE rc=0 ===).
- **149 of 150 larder names on disk** (~69,196 chains, all-strikes + OI, exp 2016-01 → 2026-06). ULTA was the final name, completing to 2026-06.
- **BRK/B is the one unpullable name** — a symbol-format bug (Theta v3 wants BRKB, not BRK.B); fix queued for the next pull in **PR #413**. Not a data gap, a code fix.

**Status mix:** C(complete)=140 · N(active)=0 · X(ceiling, maxExp~2026-05-15)=9. (+4 non-larder orphans AES/AFL/AMD/AOS excluded.) +182 chains since the 16:00 boundary (the ULTA tail fill).

### *** PHASE-2 STARTED — 2026-06-16 23:14:48 ***
The supervisor auto-launched Phase-2 (the armed nable_reference_phase2.flag):
- **SPY + QQQ**, weekly expirations, all-strikes + OI, range 2018-01-01 → 2026-06-10, 45-day lookback, 2 workers → separate index_reference/ dir (NEVER enters the candidate ranker — index skew / dealer-GEX reference only).
- As of this boundary: **78 partitions** written, climbing through 2018 (SPY→2018-08, QQQ→2018-11). Worker PID 18120 alive. SPY's huge chain makes this slow (~110s/call, throttled to 2 workers) — running it fully per the "don't leave useful data behind" directive. P2_FLAG / DONE flag still absent.

**Note:** option_history is now STATIC (Phase-1 done writing there); health is tracked via index_reference writes from here.

| Ticker | Chains | MinExp | MaxExp | New(8h) | St |
|---|---------:|---|---|---------:|---|
| AAPL | 581 | 2016-01 | 2026-06 | 0 | C |
| ABBV | 544 | 2016-01 | 2026-06 | 0 | C |
| ABNB | 286 | 2020-12 | 2026-06 | 0 | C |
| ABT | 532 | 2016-01 | 2026-06 | 0 | C |
| ACN | 544 | 2016-01 | 2026-06 | 0 | C |
| ADBE | 486 | 2016-01 | 2026-06 | 0 | C |
| ADI | 460 | 2016-01 | 2026-06 | 0 | C |
| ADP | 480 | 2016-01 | 2026-06 | 0 | C |
| ADSK | 479 | 2016-01 | 2026-06 | 0 | C |
| AEP | 125 | 2016-01 | 2026-05 | 0 | X |
| AMAT | 544 | 2016-01 | 2026-06 | 0 | C |
| AMGN | 544 | 2016-01 | 2026-06 | 0 | C |
| AMT | 366 | 2016-01 | 2026-05 | 0 | X |
| AMZN | 581 | 2016-01 | 2026-06 | 0 | C |
| AVGO | 580 | 2016-01 | 2026-06 | 0 | C |
| AXP | 544 | 2016-01 | 2026-06 | 0 | C |
| BA | 544 | 2016-01 | 2026-06 | 0 | C |
| BAC | 544 | 2016-01 | 2026-06 | 0 | C |
| BIIB | 544 | 2016-01 | 2026-06 | 0 | C |
| BKNG | 432 | 2018-03 | 2026-06 | 0 | C |
| BLK | 459 | 2016-01 | 2026-06 | 0 | C |
| BMY | 544 | 2016-01 | 2026-06 | 0 | C |
| BSX | 442 | 2016-01 | 2026-06 | 0 | C |
| C | 544 | 2016-01 | 2026-06 | 0 | C |
| CAT | 544 | 2016-01 | 2026-06 | 0 | C |
| CDNS | 159 | 2016-01 | 2026-06 | 0 | C |
| CHTR | 528 | 2016-01 | 2026-06 | 0 | C |
| CI | 502 | 2016-01 | 2026-06 | 0 | C |
| CL | 511 | 2016-01 | 2026-06 | 0 | C |
| CMCSA | 544 | 2016-01 | 2026-06 | 0 | C |
| CME | 484 | 2016-01 | 2026-06 | 0 | C |
| COIN | 267 | 2021-04 | 2026-06 | 0 | C |
| COP | 544 | 2016-01 | 2026-06 | 0 | C |
| COST | 544 | 2016-01 | 2026-06 | 0 | C |
| CRM | 544 | 2016-01 | 2026-06 | 0 | C |
| CRWD | 357 | 2019-06 | 2026-06 | 0 | C |
| CSCO | 544 | 2016-01 | 2026-06 | 0 | C |
| CSX | 487 | 2016-01 | 2026-06 | 0 | C |
| CVS | 544 | 2016-01 | 2026-06 | 0 | C |
| CVX | 544 | 2016-01 | 2026-06 | 0 | C |
| DASH | 286 | 2020-12 | 2026-06 | 0 | C |
| DDOG | 320 | 2019-10 | 2026-06 | 0 | C |
| DE | 544 | 2016-01 | 2026-06 | 0 | C |
| DHR | 312 | 2016-01 | 2026-06 | 0 | C |
| DIS | 544 | 2016-01 | 2026-06 | 0 | C |
| DLTR | 499 | 2016-01 | 2026-06 | 0 | C |
| EA | 536 | 2016-01 | 2026-06 | 0 | C |
| EBAY | 544 | 2016-01 | 2026-06 | 0 | C |
| ELV | 118 | 2022-07 | 2026-05 | 0 | X |
| EOG | 544 | 2016-01 | 2026-06 | 0 | C |
| EQIX | 125 | 2016-01 | 2026-05 | 0 | X |
| EXC | 125 | 2016-01 | 2026-05 | 0 | C |
| EXPE | 544 | 2016-01 | 2026-06 | 0 | C |
| F | 544 | 2016-01 | 2026-06 | 0 | C |
| FCX | 544 | 2016-01 | 2026-06 | 0 | C |
| FDX | 544 | 2016-01 | 2026-06 | 0 | C |
| FISV | 237 | 2016-01 | 2026-06 | 0 | C |
| FTNT | 247 | 2016-01 | 2026-06 | 0 | C |
| GE | 544 | 2016-01 | 2026-06 | 0 | C |
| GILD | 544 | 2016-01 | 2026-06 | 0 | C |
| GM | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOG | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOGL | 579 | 2016-01 | 2026-06 | 0 | C |
| GS | 544 | 2016-01 | 2026-06 | 0 | C |
| HD | 544 | 2016-01 | 2026-06 | 0 | C |
| HON | 475 | 2016-01 | 2026-06 | 0 | C |
| HOOD | 252 | 2021-08 | 2026-06 | 0 | C |
| IBM | 544 | 2016-01 | 2026-06 | 0 | C |
| INTC | 544 | 2016-01 | 2026-06 | 0 | C |
| INTU | 414 | 2016-01 | 2026-06 | 0 | C |
| ISRG | 544 | 2016-01 | 2026-06 | 0 | C |
| JNJ | 544 | 2016-01 | 2026-06 | 0 | C |
| JPM | 544 | 2016-01 | 2026-06 | 0 | C |
| KLAC | 317 | 2016-01 | 2026-05 | 0 | X |
| KMB | 463 | 2016-01 | 2026-06 | 0 | C |
| KO | 544 | 2016-01 | 2026-06 | 0 | C |
| LIN | 114 | 2018-11 | 2026-05 | 0 | X |
| LLY | 524 | 2016-01 | 2026-06 | 0 | C |
| LMT | 502 | 2016-01 | 2026-06 | 0 | C |
| LOW | 505 | 2016-01 | 2026-06 | 0 | C |
| LRCX | 460 | 2016-01 | 2026-06 | 0 | C |
| LULU | 523 | 2016-01 | 2026-06 | 0 | C |
| MA | 524 | 2016-01 | 2026-06 | 0 | C |
| MAR | 496 | 2016-01 | 2026-06 | 0 | C |
| MCD | 524 | 2016-01 | 2026-06 | 0 | C |
| MCHP | 309 | 2016-01 | 2026-06 | 0 | C |
| MDLZ | 523 | 2016-01 | 2026-06 | 0 | C |
| MDT | 544 | 2016-01 | 2026-06 | 0 | C |
| META | 254 | 2021-07 | 2026-06 | 0 | C |
| MMM | 544 | 2016-01 | 2026-06 | 0 | C |
| MO | 544 | 2016-01 | 2026-06 | 0 | C |
| MPC | 513 | 2016-01 | 2026-05 | 0 | C |
| MRK | 544 | 2016-01 | 2026-06 | 0 | C |
| MS | 544 | 2016-01 | 2026-06 | 0 | C |
| MSFT | 580 | 2016-01 | 2026-06 | 0 | C |
| MU | 544 | 2016-01 | 2026-06 | 0 | C |
| NEE | 240 | 2016-01 | 2026-06 | 0 | C |
| NEM | 544 | 2016-01 | 2026-06 | 0 | C |
| NFLX | 544 | 2016-01 | 2026-06 | 0 | C |
| NKE | 544 | 2016-01 | 2026-06 | 0 | C |
| NOW | 537 | 2016-01 | 2026-06 | 0 | C |
| NVDA | 580 | 2016-01 | 2026-06 | 0 | C |
| NXPI | 527 | 2016-01 | 2026-05 | 0 | X |
| ON | 348 | 2016-01 | 2026-06 | 0 | C |
| ORCL | 544 | 2016-01 | 2026-06 | 0 | C |
| ORLY | 125 | 2016-01 | 2026-05 | 0 | X |
| OXY | 544 | 2016-01 | 2026-06 | 0 | C |
| PANW | 544 | 2016-01 | 2026-06 | 0 | C |
| PEP | 544 | 2016-01 | 2026-06 | 0 | C |
| PFE | 544 | 2016-01 | 2026-06 | 0 | C |
| PG | 544 | 2016-01 | 2026-06 | 0 | C |
| PGR | 191 | 2016-01 | 2026-06 | 0 | C |
| PLD | 125 | 2016-01 | 2026-05 | 0 | C |
| PLTR | 295 | 2020-10 | 2026-06 | 0 | C |
| PM | 544 | 2016-01 | 2026-06 | 0 | C |
| QCOM | 544 | 2016-01 | 2026-06 | 0 | C |
| REGN | 516 | 2016-01 | 2026-06 | 0 | C |
| ROP | 125 | 2016-01 | 2026-05 | 0 | X |
| ROST | 367 | 2016-01 | 2026-06 | 0 | C |
| RTX | 319 | 2020-04 | 2026-06 | 0 | C |
| SBUX | 544 | 2016-01 | 2026-06 | 0 | C |
| SCHW | 516 | 2016-01 | 2026-06 | 0 | C |
| SHW | 221 | 2016-01 | 2026-05 | 0 | C |
| SLB | 544 | 2016-01 | 2026-06 | 0 | C |
| SMCI | 237 | 2016-01 | 2026-06 | 0 | C |
| SNPS | 181 | 2016-01 | 2026-06 | 0 | C |
| SPGI | 458 | 2016-05 | 2026-06 | 0 | C |
| T | 544 | 2016-01 | 2026-06 | 0 | C |
| TGT | 544 | 2016-01 | 2026-06 | 0 | C |
| TJX | 367 | 2016-01 | 2026-06 | 0 | C |
| TMO | 280 | 2016-01 | 2026-06 | 0 | C |
| TMUS | 544 | 2016-01 | 2026-06 | 0 | C |
| TSLA | 581 | 2016-01 | 2026-06 | 0 | C |
| TTD | 436 | 2017-04 | 2026-06 | 0 | C |
| TXN | 544 | 2016-01 | 2026-06 | 0 | C |
| UAL | 544 | 2016-01 | 2026-06 | 0 | C |
| ULTA | 528 | 2016-01 | 2026-06 | 187 | C |
| UNH | 544 | 2016-01 | 2026-06 | 0 | C |
| UNP | 544 | 2016-01 | 2026-06 | 0 | C |
| UPS | 544 | 2016-01 | 2026-06 | 0 | C |
| V | 544 | 2016-01 | 2026-06 | 0 | C |
| VRTX | 544 | 2016-01 | 2026-06 | 0 | C |
| VZ | 544 | 2016-01 | 2026-06 | 0 | C |
| WDAY | 544 | 2016-01 | 2026-06 | 0 | C |
| WDC | 544 | 2016-01 | 2026-06 | 0 | C |
| WFC | 544 | 2016-01 | 2026-06 | 0 | C |
| WMT | 544 | 2016-01 | 2026-06 | 0 | C |
| XOM | 544 | 2016-01 | 2026-06 | 0 | C |
| XYZ | 72 | 2025-01 | 2026-06 | 0 | C |



## 2026-06-16 16:00 local (logged 16:10) · ~69,014 chains · 149 names · +1,039 in 8h · worker alive

**Snapshot:** 149 larder names on disk (+4 non-larder orphans AES/AFL/AMD/AOS, excluded). Total partitions ~69,014 (larder 68,941 + orphans 72). Delta vs the 08:00 boundary (~67,975): **+1,039 chains** (the pull's tail — only deep-history backfill of the last few names remained). Worker 57396 alive, ~0.5/min, heavy probe OK(2/2), P1_FLAG absent.

**Status mix:** C(complete)=139 . N(active)=1 . X(ceiling, maxExp~2026-05-15)=9.

**New completions since the 08:00 boundary (4):** EXC(#142), CL(#143), DLTR(#144), EOG(#145). (EXPE#141 was logged at the 08:00 boundary.)

**Active(N)=1 — the pull is on its LAST name:** ULTA (the 149th and final pullable name) is deep-backfilling its history from 2017 toward present; at this boundary it has reached 2022-12 (climbing ~4 months per 30 min). All other 148 larder names are complete. When ULTA reaches ~2026-06 and the worker exits, Phase-1 is DONE.

**Ceiling(X)=9:** AEP, AMT, ELV, EQIX, KLAC, LIN, NXPI, ORLY, ROP.

**Pull status:** 148/149 pullable larder names complete; ULTA in final backfill (est. ~3-4h to present). BRK/B is the unpullable 150th larder slot (symbol-format bug — BRKB queued for the next pull via PR #413). Phase-2 (SPY/QQQ reference) not yet started (index_reference empty); it auto-starts when P1_FLAG is written.

| Ticker | Chains | MinExp | MaxExp | New(8h) | St |
|---|---------:|---|---|---------:|---|
| AAPL | 581 | 2016-01 | 2026-06 | 0 | C |
| ABBV | 544 | 2016-01 | 2026-06 | 0 | C |
| ABNB | 286 | 2020-12 | 2026-06 | 0 | C |
| ABT | 532 | 2016-01 | 2026-06 | 0 | C |
| ACN | 544 | 2016-01 | 2026-06 | 0 | C |
| ADBE | 486 | 2016-01 | 2026-06 | 0 | C |
| ADI | 460 | 2016-01 | 2026-06 | 0 | C |
| ADP | 480 | 2016-01 | 2026-06 | 0 | C |
| ADSK | 479 | 2016-01 | 2026-06 | 0 | C |
| AEP | 125 | 2016-01 | 2026-05 | 0 | X |
| AMAT | 544 | 2016-01 | 2026-06 | 0 | C |
| AMGN | 544 | 2016-01 | 2026-06 | 0 | C |
| AMT | 366 | 2016-01 | 2026-05 | 0 | X |
| AMZN | 581 | 2016-01 | 2026-06 | 0 | C |
| AVGO | 580 | 2016-01 | 2026-06 | 0 | C |
| AXP | 544 | 2016-01 | 2026-06 | 0 | C |
| BA | 544 | 2016-01 | 2026-06 | 0 | C |
| BAC | 544 | 2016-01 | 2026-06 | 0 | C |
| BIIB | 544 | 2016-01 | 2026-06 | 0 | C |
| BKNG | 432 | 2018-03 | 2026-06 | 0 | C |
| BLK | 459 | 2016-01 | 2026-06 | 0 | C |
| BMY | 544 | 2016-01 | 2026-06 | 0 | C |
| BSX | 442 | 2016-01 | 2026-06 | 0 | C |
| C | 544 | 2016-01 | 2026-06 | 0 | C |
| CAT | 544 | 2016-01 | 2026-06 | 0 | C |
| CDNS | 159 | 2016-01 | 2026-06 | 0 | C |
| CHTR | 528 | 2016-01 | 2026-06 | 0 | C |
| CI | 502 | 2016-01 | 2026-06 | 0 | C |
| CL | 511 | 2016-01 | 2026-06 | 275 | C |
| CMCSA | 544 | 2016-01 | 2026-06 | 0 | C |
| CME | 484 | 2016-01 | 2026-06 | 0 | C |
| COIN | 267 | 2021-04 | 2026-06 | 0 | C |
| COP | 544 | 2016-01 | 2026-06 | 0 | C |
| COST | 544 | 2016-01 | 2026-06 | 0 | C |
| CRM | 544 | 2016-01 | 2026-06 | 0 | C |
| CRWD | 357 | 2019-06 | 2026-06 | 0 | C |
| CSCO | 544 | 2016-01 | 2026-06 | 0 | C |
| CSX | 487 | 2016-01 | 2026-06 | 0 | C |
| CVS | 544 | 2016-01 | 2026-06 | 0 | C |
| CVX | 544 | 2016-01 | 2026-06 | 0 | C |
| DASH | 286 | 2020-12 | 2026-06 | 0 | C |
| DDOG | 320 | 2019-10 | 2026-06 | 0 | C |
| DE | 544 | 2016-01 | 2026-06 | 0 | C |
| DHR | 312 | 2016-01 | 2026-06 | 0 | C |
| DIS | 544 | 2016-01 | 2026-06 | 0 | C |
| DLTR | 499 | 2016-01 | 2026-06 | 225 | C |
| EA | 536 | 2016-01 | 2026-06 | 0 | C |
| EBAY | 544 | 2016-01 | 2026-06 | 0 | C |
| ELV | 118 | 2022-07 | 2026-05 | 0 | X |
| EOG | 544 | 2016-01 | 2026-06 | 212 | C |
| EQIX | 125 | 2016-01 | 2026-05 | 0 | X |
| EXC | 125 | 2016-01 | 2026-05 | 66 | C |
| EXPE | 544 | 2016-01 | 2026-06 | 0 | C |
| F | 544 | 2016-01 | 2026-06 | 0 | C |
| FCX | 544 | 2016-01 | 2026-06 | 0 | C |
| FDX | 544 | 2016-01 | 2026-06 | 0 | C |
| FISV | 237 | 2016-01 | 2026-06 | 0 | C |
| FTNT | 247 | 2016-01 | 2026-06 | 0 | C |
| GE | 544 | 2016-01 | 2026-06 | 0 | C |
| GILD | 544 | 2016-01 | 2026-06 | 0 | C |
| GM | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOG | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOGL | 579 | 2016-01 | 2026-06 | 0 | C |
| GS | 544 | 2016-01 | 2026-06 | 0 | C |
| HD | 544 | 2016-01 | 2026-06 | 0 | C |
| HON | 475 | 2016-01 | 2026-06 | 0 | C |
| HOOD | 252 | 2021-08 | 2026-06 | 0 | C |
| IBM | 544 | 2016-01 | 2026-06 | 0 | C |
| INTC | 544 | 2016-01 | 2026-06 | 0 | C |
| INTU | 414 | 2016-01 | 2026-06 | 0 | C |
| ISRG | 544 | 2016-01 | 2026-06 | 0 | C |
| JNJ | 544 | 2016-01 | 2026-06 | 0 | C |
| JPM | 544 | 2016-01 | 2026-06 | 0 | C |
| KLAC | 317 | 2016-01 | 2026-05 | 0 | X |
| KMB | 463 | 2016-01 | 2026-06 | 0 | C |
| KO | 544 | 2016-01 | 2026-06 | 0 | C |
| LIN | 114 | 2018-11 | 2026-05 | 0 | X |
| LLY | 524 | 2016-01 | 2026-06 | 0 | C |
| LMT | 502 | 2016-01 | 2026-06 | 0 | C |
| LOW | 505 | 2016-01 | 2026-06 | 0 | C |
| LRCX | 460 | 2016-01 | 2026-06 | 0 | C |
| LULU | 523 | 2016-01 | 2026-06 | 0 | C |
| MA | 524 | 2016-01 | 2026-06 | 0 | C |
| MAR | 496 | 2016-01 | 2026-06 | 0 | C |
| MCD | 524 | 2016-01 | 2026-06 | 0 | C |
| MCHP | 309 | 2016-01 | 2026-06 | 0 | C |
| MDLZ | 523 | 2016-01 | 2026-06 | 0 | C |
| MDT | 544 | 2016-01 | 2026-06 | 0 | C |
| META | 254 | 2021-07 | 2026-06 | 0 | C |
| MMM | 544 | 2016-01 | 2026-06 | 0 | C |
| MO | 544 | 2016-01 | 2026-06 | 0 | C |
| MPC | 513 | 2016-01 | 2026-05 | 0 | C |
| MRK | 544 | 2016-01 | 2026-06 | 0 | C |
| MS | 544 | 2016-01 | 2026-06 | 0 | C |
| MSFT | 580 | 2016-01 | 2026-06 | 0 | C |
| MU | 544 | 2016-01 | 2026-06 | 0 | C |
| NEE | 240 | 2016-01 | 2026-06 | 0 | C |
| NEM | 544 | 2016-01 | 2026-06 | 0 | C |
| NFLX | 544 | 2016-01 | 2026-06 | 0 | C |
| NKE | 544 | 2016-01 | 2026-06 | 0 | C |
| NOW | 537 | 2016-01 | 2026-06 | 0 | C |
| NVDA | 580 | 2016-01 | 2026-06 | 0 | C |
| NXPI | 527 | 2016-01 | 2026-05 | 0 | X |
| ON | 348 | 2016-01 | 2026-06 | 0 | C |
| ORCL | 544 | 2016-01 | 2026-06 | 0 | C |
| ORLY | 125 | 2016-01 | 2026-05 | 0 | X |
| OXY | 544 | 2016-01 | 2026-06 | 0 | C |
| PANW | 544 | 2016-01 | 2026-06 | 0 | C |
| PEP | 544 | 2016-01 | 2026-06 | 0 | C |
| PFE | 544 | 2016-01 | 2026-06 | 0 | C |
| PG | 544 | 2016-01 | 2026-06 | 0 | C |
| PGR | 191 | 2016-01 | 2026-06 | 0 | C |
| PLD | 125 | 2016-01 | 2026-05 | 0 | C |
| PLTR | 295 | 2020-10 | 2026-06 | 0 | C |
| PM | 544 | 2016-01 | 2026-06 | 0 | C |
| QCOM | 544 | 2016-01 | 2026-06 | 0 | C |
| REGN | 516 | 2016-01 | 2026-06 | 0 | C |
| ROP | 125 | 2016-01 | 2026-05 | 0 | X |
| ROST | 367 | 2016-01 | 2026-06 | 0 | C |
| RTX | 319 | 2020-04 | 2026-06 | 0 | C |
| SBUX | 544 | 2016-01 | 2026-06 | 0 | C |
| SCHW | 516 | 2016-01 | 2026-06 | 0 | C |
| SHW | 221 | 2016-01 | 2026-05 | 0 | C |
| SLB | 544 | 2016-01 | 2026-06 | 0 | C |
| SMCI | 237 | 2016-01 | 2026-06 | 0 | C |
| SNPS | 181 | 2016-01 | 2026-06 | 0 | C |
| SPGI | 458 | 2016-05 | 2026-06 | 0 | C |
| T | 544 | 2016-01 | 2026-06 | 0 | C |
| TGT | 544 | 2016-01 | 2026-06 | 0 | C |
| TJX | 367 | 2016-01 | 2026-06 | 0 | C |
| TMO | 280 | 2016-01 | 2026-06 | 0 | C |
| TMUS | 544 | 2016-01 | 2026-06 | 0 | C |
| TSLA | 581 | 2016-01 | 2026-06 | 0 | C |
| TTD | 436 | 2017-04 | 2026-06 | 0 | C |
| TXN | 544 | 2016-01 | 2026-06 | 0 | C |
| UAL | 544 | 2016-01 | 2026-06 | 0 | C |
| ULTA | 345 | 2016-01 | 2022-12 | 345 | N |
| UNH | 544 | 2016-01 | 2026-06 | 0 | C |
| UNP | 544 | 2016-01 | 2026-06 | 0 | C |
| UPS | 544 | 2016-01 | 2026-06 | 0 | C |
| V | 544 | 2016-01 | 2026-06 | 0 | C |
| VRTX | 544 | 2016-01 | 2026-06 | 0 | C |
| VZ | 544 | 2016-01 | 2026-06 | 0 | C |
| WDAY | 544 | 2016-01 | 2026-06 | 0 | C |
| WDC | 544 | 2016-01 | 2026-06 | 0 | C |
| WFC | 544 | 2016-01 | 2026-06 | 0 | C |
| WMT | 544 | 2016-01 | 2026-06 | 0 | C |
| XOM | 544 | 2016-01 | 2026-06 | 0 | C |
| XYZ | 72 | 2025-01 | 2026-06 | 0 | C |



## 2026-06-16 08:00 local (logged 08:11) · ~67,975 chains · 148 names · +3,085 in 8h · worker alive

**Snapshot:** 148 larder names on disk (+4 non-larder orphans AES/AFL/AMD/AOS, excluded). Total partitions ~67,975 (larder 67,903 + orphans 72). Delta vs the 00:00 boundary (64,890): **+3,085 chains**; 3,141 partitions written since 00:00. Worker 57396 alive, ~9/min, heavy probe OK(2/2), P1_FLAG absent. Phase-1 still finishing the last tail names; Phase-2 not started (index_reference empty).

**Status mix:** C(complete)=135 . N(active)=4 . X(ceiling, maxExp~2026-05-15)=9.

**New completions since the 00:00 boundary (8):** BSX(#134), MPC(#135), SHW(#136), FTNT(#137), PGR(#138), ROST(#139), FDX(#140), EXPE(#141).

**New tail dirs opened since 00:00 (144->152, +8):** FTNT, EXPE, ROST, PGR, EOG, DLTR, CL, EXC (EXPE/FDX/ROST/FTNT/PGR now complete to 2026-06; EOG/DLTR/CL/EXC still filling).

**Active(N)=4:** CL->2021-01, DLTR->2022-06, EOG->2022-07, EXC->2023-04 (just opened, hitting expected 472s on no-data expirations).

**Ceiling(X)=9:** AEP, AMT, ELV, EQIX, KLAC, LIN, NXPI, ORLY, ROP.

**Larder tail remaining (1):** ULTA — opens once the 4 active finish. After ULTA, Phase-1 larder is complete (BRK/B is the unpullable 150th slot — symbol-format bug, BRKB queued for next pull via PR #413).

**Pull status:** ~148/150 larder names on disk; Phase-1 expected to set P1_FLAG within the next ~1-2h, after which the supervisor auto-starts Phase-2 (SPY/QQQ reference, separate index_reference/ dir).

| Ticker | Chains | MinExp | MaxExp | New(8h) | St |
|---|---------:|---|---|---------:|---|
| AAPL | 581 | 2016-01 | 2026-06 | 0 | C |
| ABBV | 544 | 2016-01 | 2026-06 | 0 | C |
| ABNB | 286 | 2020-12 | 2026-06 | 0 | C |
| ABT | 532 | 2016-01 | 2026-06 | 0 | C |
| ACN | 544 | 2016-01 | 2026-06 | 0 | C |
| ADBE | 486 | 2016-01 | 2026-06 | 0 | C |
| ADI | 460 | 2016-01 | 2026-06 | 0 | C |
| ADP | 480 | 2016-01 | 2026-06 | 0 | C |
| ADSK | 479 | 2016-01 | 2026-06 | 0 | C |
| AEP | 125 | 2016-01 | 2026-05 | 0 | X |
| AMAT | 544 | 2016-01 | 2026-06 | 0 | C |
| AMGN | 544 | 2016-01 | 2026-06 | 0 | C |
| AMT | 366 | 2016-01 | 2026-05 | 0 | X |
| AMZN | 581 | 2016-01 | 2026-06 | 0 | C |
| AVGO | 580 | 2016-01 | 2026-06 | 0 | C |
| AXP | 544 | 2016-01 | 2026-06 | 0 | C |
| BA | 544 | 2016-01 | 2026-06 | 0 | C |
| BAC | 544 | 2016-01 | 2026-06 | 0 | C |
| BIIB | 544 | 2016-01 | 2026-06 | 0 | C |
| BKNG | 432 | 2018-03 | 2026-06 | 0 | C |
| BLK | 459 | 2016-01 | 2026-06 | 0 | C |
| BMY | 544 | 2016-01 | 2026-06 | 0 | C |
| BSX | 442 | 2016-01 | 2026-06 | 78 | C |
| C | 544 | 2016-01 | 2026-06 | 0 | C |
| CAT | 544 | 2016-01 | 2026-06 | 0 | C |
| CDNS | 159 | 2016-01 | 2026-06 | 0 | C |
| CHTR | 528 | 2016-01 | 2026-06 | 0 | C |
| CI | 502 | 2016-01 | 2026-06 | 0 | C |
| CL | 267 | 2016-01 | 2021-02 | 267 | N |
| CMCSA | 544 | 2016-01 | 2026-06 | 0 | C |
| CME | 484 | 2016-01 | 2026-06 | 0 | C |
| COIN | 267 | 2021-04 | 2026-06 | 0 | C |
| COP | 544 | 2016-01 | 2026-06 | 0 | C |
| COST | 544 | 2016-01 | 2026-06 | 0 | C |
| CRM | 544 | 2016-01 | 2026-06 | 0 | C |
| CRWD | 357 | 2019-06 | 2026-06 | 0 | C |
| CSCO | 544 | 2016-01 | 2026-06 | 0 | C |
| CSX | 487 | 2016-01 | 2026-06 | 0 | C |
| CVS | 544 | 2016-01 | 2026-06 | 0 | C |
| CVX | 544 | 2016-01 | 2026-06 | 0 | C |
| DASH | 286 | 2020-12 | 2026-06 | 0 | C |
| DDOG | 320 | 2019-10 | 2026-06 | 0 | C |
| DE | 544 | 2016-01 | 2026-06 | 0 | C |
| DHR | 312 | 2016-01 | 2026-06 | 0 | C |
| DIS | 544 | 2016-01 | 2026-06 | 0 | C |
| DLTR | 294 | 2016-01 | 2022-07 | 294 | N |
| EA | 536 | 2016-01 | 2026-06 | 0 | C |
| EBAY | 544 | 2016-01 | 2026-06 | 0 | C |
| ELV | 118 | 2022-07 | 2026-05 | 0 | X |
| EOG | 343 | 2016-01 | 2022-07 | 343 | N |
| EQIX | 125 | 2016-01 | 2026-05 | 0 | X |
| EXC | 89 | 2016-01 | 2023-05 | 89 | N |
| EXPE | 544 | 2016-01 | 2026-06 | 544 | C |
| F | 544 | 2016-01 | 2026-06 | 0 | C |
| FCX | 544 | 2016-01 | 2026-06 | 0 | C |
| FDX | 544 | 2016-01 | 2026-06 | 496 | C |
| FISV | 237 | 2016-01 | 2026-06 | 0 | C |
| FTNT | 247 | 2016-01 | 2026-06 | 247 | C |
| GE | 544 | 2016-01 | 2026-06 | 0 | C |
| GILD | 544 | 2016-01 | 2026-06 | 0 | C |
| GM | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOG | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOGL | 579 | 2016-01 | 2026-06 | 0 | C |
| GS | 544 | 2016-01 | 2026-06 | 0 | C |
| HD | 544 | 2016-01 | 2026-06 | 0 | C |
| HON | 475 | 2016-01 | 2026-06 | 0 | C |
| HOOD | 252 | 2021-08 | 2026-06 | 0 | C |
| IBM | 544 | 2016-01 | 2026-06 | 0 | C |
| INTC | 544 | 2016-01 | 2026-06 | 0 | C |
| INTU | 414 | 2016-01 | 2026-06 | 0 | C |
| ISRG | 544 | 2016-01 | 2026-06 | 0 | C |
| JNJ | 544 | 2016-01 | 2026-06 | 0 | C |
| JPM | 544 | 2016-01 | 2026-06 | 0 | C |
| KLAC | 317 | 2016-01 | 2026-05 | 0 | X |
| KMB | 463 | 2016-01 | 2026-06 | 0 | C |
| KO | 544 | 2016-01 | 2026-06 | 0 | C |
| LIN | 114 | 2018-11 | 2026-05 | 0 | X |
| LLY | 524 | 2016-01 | 2026-06 | 0 | C |
| LMT | 502 | 2016-01 | 2026-06 | 0 | C |
| LOW | 505 | 2016-01 | 2026-06 | 0 | C |
| LRCX | 460 | 2016-01 | 2026-06 | 0 | C |
| LULU | 523 | 2016-01 | 2026-06 | 0 | C |
| MA | 524 | 2016-01 | 2026-06 | 0 | C |
| MAR | 496 | 2016-01 | 2026-06 | 0 | C |
| MCD | 524 | 2016-01 | 2026-06 | 0 | C |
| MCHP | 309 | 2016-01 | 2026-06 | 0 | C |
| MDLZ | 523 | 2016-01 | 2026-06 | 0 | C |
| MDT | 544 | 2016-01 | 2026-06 | 0 | C |
| META | 254 | 2021-07 | 2026-06 | 0 | C |
| MMM | 544 | 2016-01 | 2026-06 | 0 | C |
| MO | 544 | 2016-01 | 2026-06 | 0 | C |
| MPC | 513 | 2016-01 | 2026-05 | 117 | C |
| MRK | 544 | 2016-01 | 2026-06 | 0 | C |
| MS | 544 | 2016-01 | 2026-06 | 0 | C |
| MSFT | 580 | 2016-01 | 2026-06 | 0 | C |
| MU | 544 | 2016-01 | 2026-06 | 0 | C |
| NEE | 240 | 2016-01 | 2026-06 | 0 | C |
| NEM | 544 | 2016-01 | 2026-06 | 0 | C |
| NFLX | 544 | 2016-01 | 2026-06 | 0 | C |
| NKE | 544 | 2016-01 | 2026-06 | 0 | C |
| NOW | 537 | 2016-01 | 2026-06 | 0 | C |
| NVDA | 580 | 2016-01 | 2026-06 | 0 | C |
| NXPI | 527 | 2016-01 | 2026-05 | 0 | X |
| ON | 348 | 2016-01 | 2026-06 | 0 | C |
| ORCL | 544 | 2016-01 | 2026-06 | 0 | C |
| ORLY | 125 | 2016-01 | 2026-05 | 0 | X |
| OXY | 544 | 2016-01 | 2026-06 | 0 | C |
| PANW | 544 | 2016-01 | 2026-06 | 0 | C |
| PEP | 544 | 2016-01 | 2026-06 | 0 | C |
| PFE | 544 | 2016-01 | 2026-06 | 0 | C |
| PG | 544 | 2016-01 | 2026-06 | 0 | C |
| PGR | 191 | 2016-01 | 2026-06 | 191 | C |
| PLD | 125 | 2016-01 | 2026-05 | 0 | C |
| PLTR | 295 | 2020-10 | 2026-06 | 0 | C |
| PM | 544 | 2016-01 | 2026-06 | 0 | C |
| QCOM | 544 | 2016-01 | 2026-06 | 0 | C |
| REGN | 516 | 2016-01 | 2026-06 | 0 | C |
| ROP | 125 | 2016-01 | 2026-05 | 0 | X |
| ROST | 367 | 2016-01 | 2026-06 | 367 | C |
| RTX | 319 | 2020-04 | 2026-06 | 0 | C |
| SBUX | 544 | 2016-01 | 2026-06 | 0 | C |
| SCHW | 516 | 2016-01 | 2026-06 | 0 | C |
| SHW | 221 | 2016-01 | 2026-05 | 115 | C |
| SLB | 544 | 2016-01 | 2026-06 | 0 | C |
| SMCI | 237 | 2016-01 | 2026-06 | 0 | C |
| SNPS | 181 | 2016-01 | 2026-06 | 0 | C |
| SPGI | 458 | 2016-05 | 2026-06 | 0 | C |
| T | 544 | 2016-01 | 2026-06 | 0 | C |
| TGT | 544 | 2016-01 | 2026-06 | 0 | C |
| TJX | 367 | 2016-01 | 2026-06 | 0 | C |
| TMO | 280 | 2016-01 | 2026-06 | 0 | C |
| TMUS | 544 | 2016-01 | 2026-06 | 0 | C |
| TSLA | 581 | 2016-01 | 2026-06 | 0 | C |
| TTD | 436 | 2017-04 | 2026-06 | 0 | C |
| TXN | 544 | 2016-01 | 2026-06 | 0 | C |
| UAL | 544 | 2016-01 | 2026-06 | 0 | C |
| UNH | 544 | 2016-01 | 2026-06 | 0 | C |
| UNP | 544 | 2016-01 | 2026-06 | 0 | C |
| UPS | 544 | 2016-01 | 2026-06 | 0 | C |
| V | 544 | 2016-01 | 2026-06 | 0 | C |
| VRTX | 544 | 2016-01 | 2026-06 | 0 | C |
| VZ | 544 | 2016-01 | 2026-06 | 0 | C |
| WDAY | 544 | 2016-01 | 2026-06 | 0 | C |
| WDC | 544 | 2016-01 | 2026-06 | 0 | C |
| WFC | 544 | 2016-01 | 2026-06 | 0 | C |
| WMT | 544 | 2016-01 | 2026-06 | 0 | C |
| XOM | 544 | 2016-01 | 2026-06 | 0 | C |
| XYZ | 72 | 2025-01 | 2026-06 | 0 | C |



## 2026-06-16 00:00 local (logged 00:13) · 64,890 chains · 140 names · +3,345 in 8h · worker alive

**Snapshot:** 140 larder names on disk (+4 non-larder orphans AES/AFL/AMD/AOS, excluded from all counts below). Total partitions ~64,890 (larder 64,823). Delta vs the 16:00 boundary (61,545): **+3,345 chains**; 3,359 partitions written since 16:00. Worker 57396 alive, ~6.6/min, heavy probe OK(2/2), P1_FLAG absent.

**Status mix:** C(complete)=127 . N(active)=4 . X(ceiling, maxExp~2026-05-15)=9.

**New completions since the 16:00 boundary (7):** NEM(#127), MO(#128), TTD(#129), WDC(#130), PLD(#131), FCX(#132), CI(#133).

**New dirs opened since 16:00 (136->144, +8):** CDNS, CI, FCX, MPC, PLD, BSX, SHW, FDX (CI/FCX/CDNS/PLD now complete; MPC/BSX/SHW/FDX still filling).

**Active(N)=4:** MPC->2023-09, BSX->2025-03, SHW->2019-11, FDX->2017-05.

**Ceiling(X)=9:** AEP, AMT, ELV, EQIX, KLAC, LIN, NXPI, ORLY, ROP.

**Larder tail not yet started (9):** FTNT, EXPE, ROST, PGR, EOG, DLTR, CL, EXC, ULTA - open as the 4 active names finish.

**Audit note (2026-06-15):** BRK/B was a symbol-format bug (script sent BRK.B -> 472; correct Theta v3 symbol is BRKB). Removed from this run's larder; BRKB queued for the **next** pull via PR #413. Current run unaffected. Full audit: docs/THETA_PULL_AUDIT_2026-06-15.md.

| Ticker | Chains | MinExp | MaxExp | New(8h) | St |
|---|---------:|---|---|---------:|---|
| AAPL | 581 | 2016-01 | 2026-06 | 0 | C |
| ABBV | 544 | 2016-01 | 2026-06 | 0 | C |
| ABNB | 286 | 2020-12 | 2026-06 | 0 | C |
| ABT | 532 | 2016-01 | 2026-06 | 0 | C |
| ACN | 544 | 2016-01 | 2026-06 | 0 | C |
| ADBE | 486 | 2016-01 | 2026-06 | 0 | C |
| ADI | 460 | 2016-01 | 2026-06 | 0 | C |
| ADP | 480 | 2016-01 | 2026-06 | 0 | C |
| ADSK | 479 | 2016-01 | 2026-06 | 0 | C |
| AEP | 125 | 2016-01 | 2026-05 | 0 | X |
| AMAT | 544 | 2016-01 | 2026-06 | 0 | C |
| AMGN | 544 | 2016-01 | 2026-06 | 0 | C |
| AMT | 366 | 2016-01 | 2026-05 | 0 | X |
| AMZN | 581 | 2016-01 | 2026-06 | 0 | C |
| AVGO | 580 | 2016-01 | 2026-06 | 0 | C |
| AXP | 544 | 2016-01 | 2026-06 | 0 | C |
| BA | 544 | 2016-01 | 2026-06 | 0 | C |
| BAC | 544 | 2016-01 | 2026-06 | 0 | C |
| BIIB | 544 | 2016-01 | 2026-06 | 0 | C |
| BKNG | 432 | 2018-03 | 2026-06 | 10 | C |
| BLK | 459 | 2016-01 | 2026-06 | 0 | C |
| BMY | 544 | 2016-01 | 2026-06 | 0 | C |
| BSX | 380 | 2016-01 | 2025-03 | 380 | N |
| C | 544 | 2016-01 | 2026-06 | 0 | C |
| CAT | 544 | 2016-01 | 2026-06 | 0 | C |
| CDNS | 159 | 2016-01 | 2026-06 | 159 | C |
| CHTR | 528 | 2016-01 | 2026-06 | 0 | C |
| CI | 502 | 2016-01 | 2026-06 | 502 | C |
| CMCSA | 544 | 2016-01 | 2026-06 | 0 | C |
| CME | 484 | 2016-01 | 2026-06 | 0 | C |
| COIN | 267 | 2021-04 | 2026-06 | 0 | C |
| COP | 544 | 2016-01 | 2026-06 | 0 | C |
| COST | 544 | 2016-01 | 2026-06 | 0 | C |
| CRM | 544 | 2016-01 | 2026-06 | 0 | C |
| CRWD | 357 | 2019-06 | 2026-06 | 0 | C |
| CSCO | 544 | 2016-01 | 2026-06 | 0 | C |
| CSX | 487 | 2016-01 | 2026-06 | 0 | C |
| CVS | 544 | 2016-01 | 2026-06 | 0 | C |
| CVX | 544 | 2016-01 | 2026-06 | 0 | C |
| DASH | 286 | 2020-12 | 2026-06 | 0 | C |
| DDOG | 320 | 2019-10 | 2026-06 | 0 | C |
| DE | 544 | 2016-01 | 2026-06 | 0 | C |
| DHR | 312 | 2016-01 | 2026-06 | 0 | C |
| DIS | 544 | 2016-01 | 2026-06 | 0 | C |
| EA | 536 | 2016-01 | 2026-06 | 0 | C |
| EBAY | 544 | 2016-01 | 2026-06 | 1 | C |
| ELV | 118 | 2022-07 | 2026-05 | 0 | X |
| EQIX | 125 | 2016-01 | 2026-05 | 0 | X |
| F | 544 | 2016-01 | 2026-06 | 0 | C |
| FCX | 544 | 2016-01 | 2026-06 | 544 | C |
| FDX | 72 | 2016-01 | 2017-05 | 72 | N |
| FISV | 237 | 2016-01 | 2026-06 | 0 | C |
| GE | 544 | 2016-01 | 2026-06 | 0 | C |
| GILD | 544 | 2016-01 | 2026-06 | 0 | C |
| GM | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOG | 544 | 2016-01 | 2026-06 | 0 | C |
| GOOGL | 579 | 2016-01 | 2026-06 | 0 | C |
| GS | 544 | 2016-01 | 2026-06 | 0 | C |
| HD | 544 | 2016-01 | 2026-06 | 0 | C |
| HON | 475 | 2016-01 | 2026-06 | 0 | C |
| HOOD | 252 | 2021-08 | 2026-06 | 0 | C |
| IBM | 544 | 2016-01 | 2026-06 | 0 | C |
| INTC | 544 | 2016-01 | 2026-06 | 0 | C |
| INTU | 414 | 2016-01 | 2026-06 | 0 | C |
| ISRG | 544 | 2016-01 | 2026-06 | 0 | C |
| JNJ | 544 | 2016-01 | 2026-06 | 0 | C |
| JPM | 544 | 2016-01 | 2026-06 | 0 | C |
| KLAC | 317 | 2016-01 | 2026-05 | 0 | X |
| KMB | 463 | 2016-01 | 2026-06 | 0 | C |
| KO | 544 | 2016-01 | 2026-06 | 0 | C |
| LIN | 114 | 2018-11 | 2026-05 | 0 | X |
| LLY | 524 | 2016-01 | 2026-06 | 0 | C |
| LMT | 502 | 2016-01 | 2026-06 | 0 | C |
| LOW | 505 | 2016-01 | 2026-06 | 0 | C |
| LRCX | 460 | 2016-01 | 2026-06 | 0 | C |
| LULU | 523 | 2016-01 | 2026-06 | 0 | C |
| MA | 524 | 2016-01 | 2026-06 | 0 | C |
| MAR | 496 | 2016-01 | 2026-06 | 0 | C |
| MCD | 524 | 2016-01 | 2026-06 | 0 | C |
| MCHP | 309 | 2016-01 | 2026-06 | 0 | C |
| MDLZ | 523 | 2016-01 | 2026-06 | 0 | C |
| MDT | 544 | 2016-01 | 2026-06 | 0 | C |
| META | 254 | 2021-07 | 2026-06 | 0 | C |
| MMM | 544 | 2016-01 | 2026-06 | 0 | C |
| MO | 544 | 2016-01 | 2026-06 | 171 | C |
| MPC | 405 | 2016-01 | 2023-10 | 405 | N |
| MRK | 544 | 2016-01 | 2026-06 | 0 | C |
| MS | 544 | 2016-01 | 2026-06 | 0 | C |
| MSFT | 580 | 2016-01 | 2026-06 | 0 | C |
| MU | 544 | 2016-01 | 2026-06 | 0 | C |
| NEE | 240 | 2016-01 | 2026-06 | 0 | C |
| NEM | 544 | 2016-01 | 2026-06 | 159 | C |
| NFLX | 544 | 2016-01 | 2026-06 | 0 | C |
| NKE | 544 | 2016-01 | 2026-06 | 0 | C |
| NOW | 537 | 2016-01 | 2026-06 | 0 | C |
| NVDA | 580 | 2016-01 | 2026-06 | 0 | C |
| NXPI | 527 | 2016-01 | 2026-05 | 0 | X |
| ON | 348 | 2016-01 | 2026-06 | 0 | C |
| ORCL | 544 | 2016-01 | 2026-06 | 0 | C |
| ORLY | 125 | 2016-01 | 2026-05 | 0 | X |
| OXY | 544 | 2016-01 | 2026-06 | 0 | C |
| PANW | 544 | 2016-01 | 2026-06 | 0 | C |
| PEP | 544 | 2016-01 | 2026-06 | 0 | C |
| PFE | 544 | 2016-01 | 2026-06 | 0 | C |
| PG | 544 | 2016-01 | 2026-06 | 0 | C |
| PLD | 125 | 2016-01 | 2026-05 | 125 | C |
| PLTR | 295 | 2020-10 | 2026-06 | 0 | C |
| PM | 544 | 2016-01 | 2026-06 | 0 | C |
| QCOM | 544 | 2016-01 | 2026-06 | 0 | C |
| REGN | 516 | 2016-01 | 2026-06 | 0 | C |
| ROP | 125 | 2016-01 | 2026-05 | 0 | X |
| RTX | 319 | 2020-04 | 2026-06 | 0 | C |
| SBUX | 544 | 2016-01 | 2026-06 | 0 | C |
| SCHW | 516 | 2016-01 | 2026-06 | 0 | C |
| SHW | 118 | 2016-01 | 2019-11 | 118 | N |
| SLB | 544 | 2016-01 | 2026-06 | 0 | C |
| SMCI | 237 | 2016-01 | 2026-06 | 0 | C |
| SNPS | 181 | 2016-01 | 2026-06 | 0 | C |
| SPGI | 458 | 2016-05 | 2026-06 | 0 | C |
| T | 544 | 2016-01 | 2026-06 | 0 | C |
| TGT | 544 | 2016-01 | 2026-06 | 0 | C |
| TJX | 367 | 2016-01 | 2026-06 | 1 | C |
| TMO | 280 | 2016-01 | 2026-06 | 2 | C |
| TMUS | 544 | 2016-01 | 2026-06 | 0 | C |
| TSLA | 581 | 2016-01 | 2026-06 | 0 | C |
| TTD | 436 | 2017-04 | 2026-06 | 307 | C |
| TXN | 544 | 2016-01 | 2026-06 | 0 | C |
| UAL | 544 | 2016-01 | 2026-06 | 0 | C |
| UNH | 544 | 2016-01 | 2026-06 | 0 | C |
| UNP | 544 | 2016-01 | 2026-06 | 0 | C |
| UPS | 544 | 2016-01 | 2026-06 | 0 | C |
| V | 544 | 2016-01 | 2026-06 | 0 | C |
| VRTX | 544 | 2016-01 | 2026-06 | 0 | C |
| VZ | 544 | 2016-01 | 2026-06 | 0 | C |
| WDAY | 544 | 2016-01 | 2026-06 | 0 | C |
| WDC | 544 | 2016-01 | 2026-06 | 408 | C |
| WFC | 544 | 2016-01 | 2026-06 | 0 | C |
| WMT | 544 | 2016-01 | 2026-06 | 0 | C |
| XOM | 544 | 2016-01 | 2026-06 | 0 | C |
| XYZ | 72 | 2025-01 | 2026-06 | 0 | C |



## 2026-06-15 16:00 local (logged 16:02) · 61545 chains · 136 names · +2683 in 8h · worker alive

**Completed this 8h:** ON(#122) MAR(#123) SLB(#124) OXY(#125) AEP(#126 X-ceil) · **New dirs opened:** MO NEM AEP WDC TTD (131→136)
**Active(N)=4:** MO, NEM, TTD, WDC · **Ceiling(X)=9:** AEP, AMT, ELV, EQIX, KLAC, LIN, NXPI, ORLY, ROP
**Notable New(8h):** OXY=510(C) SLB=452(C) NEM=388(N) MO=377(N) ON=316(C) MAR=243(C) WDC=141(N) TTD=131(N) AEP=125(X)

| ticker | partitions | maxExp | status |
|--------|----------:|----------:|:------:|
| AAPL | 581 | 2026-06-10 | C |
| ABBV | 544 | 2026-06-05 | C |
| ABNB | 286 | 2026-06-05 | C |
| ABT | 532 | 2026-06-05 | C |
| ACN | 544 | 2026-06-05 | C |
| ADBE | 486 | 2026-06-05 | C |
| ADI | 460 | 2026-06-05 | C |
| ADP | 480 | 2026-06-05 | C |
| ADSK | 479 | 2026-06-05 | C |
| AEP | 125(+125) | 2026-05-15 | X |
| AES | 12 | 2016-12-16 | C |
| AFL | 11 | 2016-11-18 | C |
| AMAT | 544 | 2026-06-05 | C |
| AMD | 29 | 2016-07-22 | C |
| AMGN | 544 | 2026-06-05 | C |
| AMT | 366 | 2026-05-15 | X |
| AMZN | 581 | 2026-06-10 | C |
| AOS | 20 | 2017-08-18 | C |
| AVGO | 580 | 2026-06-10 | C |
| AXP | 544 | 2026-06-05 | C |
| BA | 544 | 2026-06-05 | C |
| BAC | 544 | 2026-06-05 | C |
| BIIB | 544 | 2026-06-05 | C |
| BKNG | 422 | 2026-06-05 | C |
| BLK | 459 | 2026-06-05 | C |
| BMY | 544 | 2026-06-05 | C |
| C | 544 | 2026-06-05 | C |
| CAT | 544 | 2026-06-05 | C |
| CHTR | 528 | 2026-06-05 | C |
| CMCSA | 544 | 2026-06-05 | C |
| CME | 484 | 2026-06-05 | C |
| COIN | 267 | 2026-06-05 | C |
| COP | 544 | 2026-06-05 | C |
| COST | 544 | 2026-06-05 | C |
| CRM | 544 | 2026-06-05 | C |
| CRWD | 357 | 2026-06-05 | C |
| CSCO | 544 | 2026-06-05 | C |
| CSX | 487 | 2026-06-05 | C |
| CVS | 544 | 2026-06-05 | C |
| CVX | 544 | 2026-06-05 | C |
| DASH | 286 | 2026-06-05 | C |
| DDOG | 320 | 2026-06-05 | C |
| DE | 544 | 2026-06-05 | C |
| DHR | 312 | 2026-06-05 | C |
| DIS | 544 | 2026-06-05 | C |
| EA | 536 | 2026-06-05 | C |
| EBAY | 543 | 2026-06-05 | C |
| ELV | 118 | 2026-05-15 | X |
| EQIX | 125 | 2026-05-15 | X |
| F | 544 | 2026-06-05 | C |
| FISV | 237 | 2026-06-05 | C |
| GE | 544 | 2026-06-05 | C |
| GILD | 544 | 2026-06-05 | C |
| GM | 544 | 2026-06-05 | C |
| GOOG | 544 | 2026-06-05 | C |
| GOOGL | 579 | 2026-06-10 | C |
| GS | 544 | 2026-06-05 | C |
| HD | 544 | 2026-06-05 | C |
| HON | 475 | 2026-06-05 | C |
| HOOD | 252 | 2026-06-05 | C |
| IBM | 544 | 2026-06-05 | C |
| INTC | 544 | 2026-06-05 | C |
| INTU | 414 | 2026-06-05 | C |
| ISRG | 544 | 2026-06-05 | C |
| JNJ | 544 | 2026-06-05 | C |
| JPM | 544 | 2026-06-05 | C |
| KLAC | 317 | 2026-05-15 | X |
| KMB | 463 | 2026-06-05 | C |
| KO | 544 | 2026-06-05 | C |
| LIN | 114 | 2026-05-15 | X |
| LLY | 524 | 2026-06-05 | C |
| LMT | 502 | 2026-06-05 | C |
| LOW | 505 | 2026-06-05 | C |
| LRCX | 460 | 2026-06-05 | C |
| LULU | 523 | 2026-06-05 | C |
| MA | 524 | 2026-06-05 | C |
| MAR | 496(+243) | 2026-06-05 | C |
| MCD | 524 | 2026-06-05 | C |
| MCHP | 309 | 2026-06-05 | C |
| MDLZ | 523 | 2026-06-05 | C |
| MDT | 544 | 2026-06-05 | C |
| META | 254 | 2026-06-10 | C |
| MMM | 544 | 2026-06-05 | C |
| MO | 377(+377) | 2023-03-24 | N |
| MRK | 544 | 2026-06-05 | C |
| MS | 544 | 2026-06-05 | C |
| MSFT | 580 | 2026-06-10 | C |
| MU | 544 | 2026-06-05 | C |
| NEE | 240 | 2026-06-05 | C |
| NEM | 388(+388) | 2023-06-09 | N |
| NFLX | 544 | 2026-06-05 | C |
| NKE | 544 | 2026-06-05 | C |
| NOW | 537 | 2026-06-05 | C |
| NVDA | 580 | 2026-06-10 | C |
| NXPI | 527 | 2026-05-15 | X |
| ON | 348(+316) | 2026-06-05 | C |
| ORCL | 544 | 2026-06-05 | C |
| ORLY | 125 | 2026-05-15 | X |
| OXY | 544(+510) | 2026-06-05 | C |
| PANW | 544 | 2026-06-05 | C |
| PEP | 544 | 2026-06-05 | C |
| PFE | 544 | 2026-06-05 | C |
| PG | 544 | 2026-06-05 | C |
| PLTR | 295 | 2026-06-05 | C |
| PM | 544 | 2026-06-05 | C |
| QCOM | 544 | 2026-06-05 | C |
| REGN | 516 | 2026-06-05 | C |
| ROP | 125 | 2026-05-15 | X |
| RTX | 319 | 2026-06-05 | C |
| SBUX | 544 | 2026-06-05 | C |
| SCHW | 516 | 2026-06-05 | C |
| SLB | 544(+452) | 2026-06-05 | C |
| SMCI | 237 | 2026-06-05 | C |
| SNPS | 181 | 2026-06-05 | C |
| SPGI | 458 | 2026-06-05 | C |
| T | 544 | 2026-06-05 | C |
| TGT | 544 | 2026-06-05 | C |
| TJX | 366 | 2026-06-05 | C |
| TMO | 278 | 2026-06-05 | C |
| TMUS | 544 | 2026-06-05 | C |
| TSLA | 581 | 2026-06-10 | C |
| TTD | 131(+131) | 2020-07-31 | N |
| TXN | 544 | 2026-06-05 | C |
| UAL | 544 | 2026-06-05 | C |
| UNH | 544 | 2026-06-05 | C |
| UNP | 544 | 2026-06-05 | C |
| UPS | 544 | 2026-06-05 | C |
| V | 544 | 2026-06-05 | C |
| VRTX | 544 | 2026-06-05 | C |
| VZ | 544 | 2026-06-05 | C |
| WDAY | 544 | 2026-06-05 | C |
| WDC | 141(+141) | 2018-09-14 | N |
| WFC | 544 | 2026-06-05 | C |
| WMT | 544 | 2026-06-05 | C |
| XOM | 544 | 2026-06-05 | C |
| XYZ | 72 | 2026-06-05 | C |


## 2026-06-15 08:00 local (logged 08:21) · 59,023 chains · 131 names · +2,502 in 8h · worker alive

Notable New(8h): EBAY=543(C) F=421(C) AMT=366(X-ceil) MAR=292(N) TJX=208(C) BLK=136(C) SLB=131(N) EQIX=125(X-ceil) ON=97(N) OXY=74(N) ELV=55(X-ceil) DE=54(C). Active(N)=4: MAR ON OXY SLB. Ceiling(X)=8: AMT ELV EQIX KLAC LIN NXPI ORLY ROP.

| Ticker | St | Parts | New(8h) | MaxExp |
| -------: || ------: | ------: | ------: |
| AAPL | C | 581 | 0 | 20260610 |
| ABBV | C | 544 | 0 | 20260605 |
| ABNB | C | 286 | 0 | 20260605 |
| ABT | C | 532 | 0 | 20260605 |
| ACN | C | 544 | 0 | 20260605 |
| ADBE | C | 486 | 0 | 20260605 |
| ADI | C | 460 | 0 | 20260605 |
| ADP | C | 480 | 0 | 20260605 |
| ADSK | C | 479 | 0 | 20260605 |
| AES | C | 12 | 0 | 20161216 |
| AFL | C | 11 | 0 | 20161118 |
| AMAT | C | 544 | 0 | 20260605 |
| AMD | C | 29 | 0 | 20160722 |
| AMGN | C | 544 | 0 | 20260605 |
| AMT | X | 366 | 366 | 20260515 |
| AMZN | C | 581 | 0 | 20260610 |
| AOS | C | 20 | 0 | 20170818 |
| AVGO | C | 580 | 0 | 20260610 |
| AXP | C | 544 | 0 | 20260605 |
| BA | C | 544 | 0 | 20260605 |
| BAC | C | 544 | 0 | 20260605 |
| BIIB | C | 544 | 0 | 20260605 |
| BKNG | C | 422 | 0 | 20260605 |
| BLK | C | 459 | 136 | 20260605 |
| BMY | C | 544 | 0 | 20260605 |
| C | C | 544 | 0 | 20260605 |
| CAT | C | 544 | 0 | 20260605 |
| CHTR | C | 528 | 0 | 20260605 |
| CMCSA | C | 544 | 0 | 20260605 |
| CME | C | 484 | 0 | 20260605 |
| COIN | C | 267 | 0 | 20260605 |
| COP | C | 544 | 0 | 20260605 |
| COST | C | 544 | 0 | 20260605 |
| CRM | C | 544 | 0 | 20260605 |
| CRWD | C | 357 | 0 | 20260605 |
| CSCO | C | 544 | 0 | 20260605 |
| CSX | C | 487 | 0 | 20260605 |
| CVS | C | 544 | 0 | 20260605 |
| CVX | C | 544 | 0 | 20260605 |
| DASH | C | 286 | 0 | 20260605 |
| DDOG | C | 320 | 0 | 20260605 |
| DE | C | 544 | 54 | 20260605 |
| DHR | C | 312 | 0 | 20260605 |
| DIS | C | 544 | 0 | 20260605 |
| EA | C | 536 | 0 | 20260605 |
| EBAY | C | 543 | 543 | 20260605 |
| ELV | X | 118 | 55 | 20260515 |
| EQIX | X | 125 | 125 | 20260515 |
| F | C | 544 | 421 | 20260605 |
| FISV | C | 237 | 0 | 20260605 |
| GE | C | 544 | 0 | 20260605 |
| GILD | C | 544 | 0 | 20260605 |
| GM | C | 544 | 0 | 20260605 |
| GOOG | C | 544 | 0 | 20260605 |
| GOOGL | C | 579 | 0 | 20260610 |
| GS | C | 544 | 0 | 20260605 |
| HD | C | 544 | 0 | 20260605 |
| HON | C | 475 | 0 | 20260605 |
| HOOD | C | 252 | 0 | 20260605 |
| IBM | C | 544 | 0 | 20260605 |
| INTC | C | 544 | 0 | 20260605 |
| INTU | C | 414 | 0 | 20260605 |
| ISRG | C | 544 | 0 | 20260605 |
| JNJ | C | 544 | 0 | 20260605 |
| JPM | C | 544 | 0 | 20260605 |
| KLAC | X | 317 | 0 | 20260515 |
| KMB | C | 463 | 0 | 20260605 |
| KO | C | 544 | 0 | 20260605 |
| LIN | X | 114 | 0 | 20260515 |
| LLY | C | 524 | 0 | 20260605 |
| LMT | C | 502 | 0 | 20260605 |
| LOW | C | 505 | 0 | 20260605 |
| LRCX | C | 460 | 0 | 20260605 |
| LULU | C | 523 | 0 | 20260605 |
| MA | C | 524 | 0 | 20260605 |
| MAR | N | 292 | 292 | 20220708 |
| MCD | C | 524 | 0 | 20260605 |
| MCHP | C | 309 | 0 | 20260605 |
| MDLZ | C | 523 | 0 | 20260605 |
| MDT | C | 544 | 0 | 20260605 |
| META | C | 254 | 0 | 20260610 |
| MMM | C | 544 | 0 | 20260605 |
| MRK | C | 544 | 0 | 20260605 |
| MS | C | 544 | 0 | 20260605 |
| MSFT | C | 580 | 0 | 20260610 |
| MU | C | 544 | 0 | 20260605 |
| NEE | C | 240 | 0 | 20260605 |
| NFLX | C | 544 | 0 | 20260605 |
| NKE | C | 544 | 0 | 20260605 |
| NOW | C | 537 | 0 | 20260605 |
| NVDA | C | 580 | 0 | 20260610 |
| NXPI | X | 527 | 0 | 20260515 |
| ON | N | 97 | 97 | 20210813 |
| ORCL | C | 544 | 0 | 20260605 |
| ORLY | X | 125 | 0 | 20260515 |
| OXY | N | 74 | 74 | 20170602 |
| PANW | C | 544 | 0 | 20260605 |
| PEP | C | 544 | 0 | 20260605 |
| PFE | C | 544 | 0 | 20260605 |
| PG | C | 544 | 0 | 20260605 |
| PLTR | C | 295 | 0 | 20260605 |
| PM | C | 544 | 0 | 20260605 |
| QCOM | C | 544 | 0 | 20260605 |
| REGN | C | 516 | 0 | 20260605 |
| ROP | X | 125 | 0 | 20260515 |
| RTX | C | 319 | 0 | 20260605 |
| SBUX | C | 544 | 0 | 20260605 |
| SCHW | C | 516 | 0 | 20260605 |
| SLB | N | 131 | 131 | 20180706 |
| SMCI | C | 237 | 0 | 20260605 |
| SNPS | C | 181 | 0 | 20260605 |
| SPGI | C | 458 | 0 | 20260605 |
| T | C | 544 | 0 | 20260605 |
| TGT | C | 544 | 0 | 20260605 |
| TJX | C | 366 | 208 | 20260605 |
| TMO | C | 278 | 0 | 20260605 |
| TMUS | C | 544 | 0 | 20260605 |
| TSLA | C | 581 | 0 | 20260610 |
| TXN | C | 544 | 0 | 20260605 |
| UAL | C | 544 | 0 | 20260605 |
| UNH | C | 544 | 0 | 20260605 |
| UNP | C | 544 | 0 | 20260605 |
| UPS | C | 544 | 0 | 20260605 |
| V | C | 544 | 0 | 20260605 |
| VRTX | C | 544 | 0 | 20260605 |
| VZ | C | 544 | 0 | 20260605 |
| WDAY | C | 544 | 0 | 20260605 |
| WFC | C | 544 | 0 | 20260605 |
| WMT | C | 544 | 0 | 20260605 |
| XOM | C | 544 | 0 | 20260605 |
| XYZ | C | 72 | 0 | 20260605 |


## 2026-06-15 00:00 local (logged 00:18) · 56,621 chains · 124 names · +1,476 in 8h · worker alive

| Ticker | St | Parts | New(8h) | MaxExp |
| -------: || ------: | ------: | ------: |
| AAPL | C | 581 | 0 | 20260610 |
| ABBV | C | 544 | 0 | 20260605 |
| ABNB | C | 286 | 0 | 20260605 |
| ABT | C | 532 | 0 | 20260605 |
| ACN | C | 544 | 0 | 20260605 |
| ADBE | C | 486 | 0 | 20260605 |
| ADI | C | 460 | 0 | 20260605 |
| ADP | C | 480 | 0 | 20260605 |
| ADSK | C | 479 | 339 | 20260605 |
| AES | N | 12 | 0 | 20161216 |
| AFL | N | 11 | 0 | 20161118 |
| AMAT | C | 544 | 0 | 20260605 |
| AMD | N | 29 | 0 | 20160722 |
| AMGN | C | 544 | 0 | 20260605 |
| AMZN | C | 581 | 0 | 20260610 |
| AOS | N | 20 | 0 | 20170818 |
| AVGO | C | 580 | 0 | 20260610 |
| AXP | C | 544 | 0 | 20260605 |
| BA | C | 544 | 0 | 20260605 |
| BAC | C | 544 | 0 | 20260605 |
| BIIB | C | 544 | 175 | 20260605 |
| BKNG | C | 422 | 0 | 20260605 |
| BLK | N | 334 | 259 | 20240112 |
| BMY | C | 544 | 0 | 20260605 |
| C | C | 544 | 0 | 20260605 |
| CAT | C | 544 | 0 | 20260605 |
| CHTR | C | 528 | 0 | 20260605 |
| CMCSA | C | 544 | 0 | 20260605 |
| CME | C | 484 | 0 | 20260605 |
| COIN | C | 267 | 0 | 20260605 |
| COP | C | 544 | 0 | 20260605 |
| COST | C | 544 | 0 | 20260605 |
| CRM | C | 544 | 0 | 20260605 |
| CRWD | C | 357 | 0 | 20260605 |
| CSCO | C | 544 | 0 | 20260605 |
| CSX | C | 487 | 0 | 20260605 |
| CVS | C | 544 | 0 | 20260605 |
| CVX | C | 544 | 0 | 20260605 |
| DASH | C | 286 | 0 | 20260605 |
| DDOG | C | 320 | 0 | 20260605 |
| DE | N | 501 | 333 | 20250808 |
| DHR | C | 312 | 0 | 20260605 |
| DIS | C | 544 | 0 | 20260605 |
| EA | C | 536 | 0 | 20260605 |
| ELV | N | 79 | 79 | 20231229 |
| F | N | 166 | 166 | 20190308 |
| FISV | C | 237 | 0 | 20260605 |
| GE | C | 544 | 0 | 20260605 |
| GILD | C | 544 | 0 | 20260605 |
| GM | C | 544 | 0 | 20260605 |
| GOOG | C | 544 | 0 | 20260605 |
| GOOGL | C | 579 | 0 | 20260610 |
| GS | C | 544 | 0 | 20260605 |
| HD | C | 544 | 0 | 20260605 |
| HON | C | 475 | 0 | 20260605 |
| HOOD | C | 252 | 0 | 20260605 |
| IBM | C | 544 | 0 | 20260605 |
| INTC | C | 544 | 0 | 20260605 |
| INTU | C | 414 | 0 | 20260605 |
| ISRG | C | 544 | 0 | 20260605 |
| JNJ | C | 544 | 0 | 20260605 |
| JPM | C | 544 | 0 | 20260605 |
| KLAC | N | 317 | 0 | 20260515 |
| KMB | C | 463 | 0 | 20260605 |
| KO | C | 544 | 0 | 20260605 |
| LIN | N | 114 | 0 | 20260515 |
| LLY | C | 524 | 0 | 20260605 |
| LMT | C | 502 | 0 | 20260605 |
| LOW | C | 505 | 0 | 20260605 |
| LRCX | C | 460 | 0 | 20260605 |
| LULU | C | 523 | 0 | 20260605 |
| MA | C | 524 | 0 | 20260605 |
| MCD | C | 524 | 0 | 20260605 |
| MCHP | C | 309 | 0 | 20260605 |
| MDLZ | C | 523 | 0 | 20260605 |
| MDT | C | 544 | 0 | 20260605 |
| META | C | 254 | 0 | 20260610 |
| MMM | C | 544 | 0 | 20260605 |
| MRK | C | 544 | 0 | 20260605 |
| MS | C | 544 | 0 | 20260605 |
| MSFT | C | 580 | 0 | 20260610 |
| MU | C | 544 | 0 | 20260605 |
| NEE | C | 240 | 0 | 20260605 |
| NFLX | C | 544 | 0 | 20260605 |
| NKE | C | 544 | 0 | 20260605 |
| NOW | C | 537 | 0 | 20260605 |
| NVDA | C | 580 | 0 | 20260610 |
| NXPI | N | 527 | 0 | 20260515 |
| ORCL | C | 544 | 0 | 20260605 |
| ORLY | N | 125 | 0 | 20260515 |
| PANW | C | 544 | 0 | 20260605 |
| PEP | C | 544 | 0 | 20260605 |
| PFE | C | 544 | 0 | 20260605 |
| PG | C | 544 | 0 | 20260605 |
| PLTR | C | 295 | 0 | 20260605 |
| PM | C | 544 | 0 | 20260605 |
| QCOM | C | 544 | 0 | 20260605 |
| REGN | C | 516 | 0 | 20260605 |
| ROP | N | 125 | 125 | 20260515 |
| RTX | C | 319 | 0 | 20260605 |
| SBUX | C | 544 | 0 | 20260605 |
| SCHW | C | 516 | 0 | 20260605 |
| SMCI | C | 237 | 0 | 20260605 |
| SNPS | C | 181 | 0 | 20260605 |
| SPGI | C | 458 | 0 | 20260605 |
| T | C | 544 | 0 | 20260605 |
| TGT | C | 544 | 0 | 20260605 |
| TJX | C | 158 | 0 | 20260605 |
| TMO | C | 278 | 0 | 20260605 |
| TMUS | C | 544 | 0 | 20260605 |
| TSLA | C | 581 | 0 | 20260610 |
| TXN | C | 544 | 0 | 20260605 |
| UAL | C | 544 | 0 | 20260605 |
| UNH | C | 544 | 0 | 20260605 |
| UNP | C | 544 | 0 | 20260605 |
| UPS | C | 544 | 0 | 20260605 |
| V | C | 544 | 0 | 20260605 |
| VRTX | C | 544 | 0 | 20260605 |
| VZ | C | 544 | 0 | 20260605 |
| WDAY | C | 544 | 0 | 20260605 |
| WFC | C | 544 | 0 | 20260605 |
| WMT | C | 544 | 0 | 20260605 |
| XOM | C | 544 | 0 | 20260605 |
| XYZ | C | 72 | 0 | 20260605 |


## 2026-06-14 16:00 local (logged 16:17) · 55,199 chains · 121 names · +2,296 in 8h · worker alive

| Ticker | St | Parts | New(8h) | MaxExp |
| -------: || ------: | ------: | ------: |
| AAPL | C | 581 | 0 | 20260610 |
| ABBV | C | 544 | 0 | 20260605 |
| ABNB | C | 286 | 0 | 20260605 |
| ABT | C | 532 | 0 | 20260605 |
| ACN | C | 544 | 0 | 20260605 |
| ADBE | C | 486 | 0 | 20260605 |
| ADI | C | 460 | 0 | 20260605 |
| ADP | C | 480 | 0 | 20260605 |
| ADSK | N | 157 | 157 | 20200403 |
| AES | N | 12 | 0 | 20161216 |
| AFL | N | 11 | 0 | 20161118 |
| AMAT | C | 544 | 0 | 20260605 |
| AMD | N | 29 | 0 | 20160722 |
| AMGN | C | 544 | 0 | 20260605 |
| AMZN | C | 581 | 0 | 20260610 |
| AOS | N | 20 | 0 | 20170818 |
| AVGO | C | 580 | 0 | 20260610 |
| AXP | C | 544 | 0 | 20260605 |
| BA | C | 544 | 0 | 20260605 |
| BAC | C | 544 | 0 | 20260605 |
| BIIB | N | 374 | 374 | 20230303 |
| BKNG | C | 422 | 0 | 20260605 |
| BLK | N | 87 | 87 | 20190418 |
| BMY | C | 544 | 0 | 20260605 |
| C | C | 544 | 0 | 20260605 |
| CAT | C | 544 | 0 | 20260605 |
| CHTR | C | 528 | 0 | 20260605 |
| CMCSA | C | 544 | 0 | 20260605 |
| CME | C | 484 | 195 | 20260605 |
| COIN | C | 267 | 0 | 20260605 |
| COP | C | 544 | 0 | 20260605 |
| COST | C | 544 | 0 | 20260605 |
| CRM | C | 544 | 0 | 20260605 |
| CRWD | C | 357 | 0 | 20260605 |
| CSCO | C | 544 | 0 | 20260605 |
| CSX | C | 487 | 0 | 20260605 |
| CVS | C | 544 | 495 | 20260605 |
| CVX | C | 544 | 0 | 20260605 |
| DASH | C | 286 | 0 | 20260605 |
| DDOG | C | 320 | 0 | 20260605 |
| DE | N | 187 | 187 | 20190802 |
| DHR | C | 312 | 0 | 20260605 |
| DIS | C | 544 | 0 | 20260605 |
| EA | C | 536 | 476 | 20260605 |
| FISV | C | 237 | 0 | 20260605 |
| GE | C | 544 | 0 | 20260605 |
| GILD | C | 544 | 0 | 20260605 |
| GM | C | 544 | 0 | 20260605 |
| GOOG | C | 544 | 0 | 20260605 |
| GOOGL | C | 579 | 0 | 20260610 |
| GS | C | 544 | 0 | 20260605 |
| HD | C | 544 | 0 | 20260605 |
| HON | C | 475 | 0 | 20260605 |
| HOOD | C | 252 | 0 | 20260605 |
| IBM | C | 544 | 0 | 20260605 |
| INTC | C | 544 | 0 | 20260605 |
| INTU | C | 414 | 0 | 20260605 |
| ISRG | C | 544 | 0 | 20260605 |
| JNJ | C | 544 | 0 | 20260605 |
| JPM | C | 544 | 0 | 20260605 |
| KLAC | N | 317 | 0 | 20260515 |
| KMB | C | 463 | 0 | 20260605 |
| KO | C | 544 | 0 | 20260605 |
| LIN | N | 114 | 0 | 20260515 |
| LLY | C | 524 | 0 | 20260605 |
| LMT | C | 502 | 0 | 20260605 |
| LOW | C | 505 | 0 | 20260605 |
| LRCX | C | 460 | 0 | 20260605 |
| LULU | C | 523 | 0 | 20260605 |
| MA | C | 524 | 0 | 20260605 |
| MCD | C | 524 | 0 | 20260605 |
| MCHP | C | 309 | 0 | 20260605 |
| MDLZ | C | 523 | 0 | 20260605 |
| MDT | C | 544 | 0 | 20260605 |
| META | C | 254 | 0 | 20260610 |
| MMM | C | 544 | 0 | 20260605 |
| MRK | C | 544 | 0 | 20260605 |
| MS | C | 544 | 0 | 20260605 |
| MSFT | C | 580 | 0 | 20260610 |
| MU | C | 544 | 0 | 20260605 |
| NEE | C | 240 | 0 | 20260605 |
| NFLX | C | 544 | 0 | 20260605 |
| NKE | C | 544 | 0 | 20260605 |
| NOW | C | 537 | 0 | 20260605 |
| NVDA | C | 580 | 0 | 20260610 |
| NXPI | N | 527 | 0 | 20260515 |
| ORCL | C | 544 | 0 | 20260605 |
| ORLY | N | 125 | 125 | 20260515 |
| PANW | C | 544 | 0 | 20260605 |
| PEP | C | 544 | 0 | 20260605 |
| PFE | C | 544 | 0 | 20260605 |
| PG | C | 544 | 0 | 20260605 |
| PLTR | C | 295 | 0 | 20260605 |
| PM | C | 544 | 0 | 20260605 |
| QCOM | C | 544 | 0 | 20260605 |
| REGN | C | 516 | 0 | 20260605 |
| RTX | C | 319 | 0 | 20260605 |
| SBUX | C | 544 | 0 | 20260605 |
| SCHW | C | 516 | 0 | 20260605 |
| SMCI | C | 237 | 0 | 20260605 |
| SNPS | C | 181 | 181 | 20260605 |
| SPGI | C | 458 | 0 | 20260605 |
| T | C | 544 | 0 | 20260605 |
| TGT | C | 544 | 0 | 20260605 |
| TJX | C | 158 | 0 | 20260605 |
| TMO | C | 278 | 0 | 20260605 |
| TMUS | C | 544 | 0 | 20260605 |
| TSLA | C | 581 | 0 | 20260610 |
| TXN | C | 544 | 0 | 20260605 |
| UAL | C | 544 | 144 | 20260605 |
| UNH | C | 544 | 0 | 20260605 |
| UNP | C | 544 | 0 | 20260605 |
| UPS | C | 544 | 0 | 20260605 |
| V | C | 544 | 0 | 20260605 |
| VRTX | C | 544 | 0 | 20260605 |
| VZ | C | 544 | 0 | 20260605 |
| WDAY | C | 544 | 0 | 20260605 |
| WFC | C | 544 | 0 | 20260605 |
| WMT | C | 544 | 0 | 20260605 |
| XOM | C | 544 | 0 | 20260605 |
| XYZ | C | 72 | 0 | 20260605 |


## 2026-06-14 08:00 local (logged 08:15) · 52,878 chains · 115 names · +2,715 in 8h · worker alive

| Ticker | St | Parts | New(8h) | MaxExp |
|--------|:--:|------:|--------:|--------|
| AAPL | C | 581 | 0 | 2026-06 |
| ABBV | C | 544 | 0 | 2026-06 |
| ABNB | C | 286 | 0 | 2026-06 |
| ABT | C | 532 | 0 | 2026-06 |
| ACN | C | 544 | 0 | 2026-06 |
| ADBE | C | 486 | 0 | 2026-06 |
| ADI | C | 460 | 0 | 2026-06 |
| ADP | C | 480 | 0 | 2026-06 |
| AES | N | 12 | 0 | 2016-12 |
| AFL | N | 11 | 0 | 2016-11 |
| AMAT | C | 544 | 0 | 2026-06 |
| AMD | N | 29 | 0 | 2016-07 |
| AMGN | C | 544 | 0 | 2026-06 |
| AMZN | C | 581 | 0 | 2026-06 |
| AOS | N | 20 | 0 | 2017-08 |
| AVGO | C | 580 | 0 | 2026-06 |
| AXP | C | 544 | 0 | 2026-06 |
| BA | C | 544 | 0 | 2026-06 |
| BAC | C | 544 | 0 | 2026-06 |
| BKNG | C | 422 | 0 | 2026-06 |
| BMY | C | 544 | 0 | 2026-06 |
| C | C | 544 | 0 | 2026-06 |
| CAT | C | 544 | 0 | 2026-06 |
| CHTR | C | 528 | 0 | 2026-06 |
| CMCSA | C | 544 | 0 | 2026-06 |
| CME | N | 319 | 320 | 2022-02 |
| COIN | C | 267 | 0 | 2026-06 |
| COP | C | 544 | 0 | 2026-06 |
| COST | C | 544 | 0 | 2026-06 |
| CRM | C | 544 | 0 | 2026-06 |
| CRWD | C | 357 | 0 | 2026-06 |
| CSCO | C | 544 | 0 | 2026-06 |
| CSX | C | 487 | 301 | 2026-06 |
| CVS | N | 73 | 74 | 2017-05 |
| CVX | C | 544 | 0 | 2026-06 |
| DASH | C | 286 | 0 | 2026-06 |
| DDOG | C | 320 | 6 | 2026-06 |
| DHR | C | 312 | 0 | 2026-06 |
| DIS | C | 544 | 0 | 2026-06 |
| EA | N | 81 | 82 | 2017-07 |
| FISV | C | 237 | 0 | 2026-06 |
| GE | C | 544 | 0 | 2026-06 |
| GILD | C | 544 | 0 | 2026-06 |
| GM | C | 544 | 544 | 2026-06 |
| GOOG | C | 544 | 0 | 2026-06 |
| GOOGL | C | 579 | 0 | 2026-06 |
| GS | C | 544 | 0 | 2026-06 |
| HD | C | 544 | 0 | 2026-06 |
| HON | C | 475 | 0 | 2026-06 |
| HOOD | C | 252 | 0 | 2026-06 |
| IBM | C | 544 | 0 | 2026-06 |
| INTC | C | 544 | 0 | 2026-06 |
| INTU | C | 414 | 0 | 2026-06 |
| ISRG | C | 544 | 0 | 2026-06 |
| JNJ | C | 544 | 0 | 2026-06 |
| JPM | C | 544 | 0 | 2026-06 |
| KLAC | N | 317 | 0 | 2026-05 |
| KMB | C | 463 | 0 | 2026-06 |
| KO | C | 544 | 0 | 2026-06 |
| LIN | N | 114 | 0 | 2026-05 |
| LLY | C | 524 | 0 | 2026-06 |
| LMT | C | 502 | 0 | 2026-06 |
| LOW | C | 505 | 0 | 2026-06 |
| LRCX | C | 460 | 0 | 2026-06 |
| LULU | C | 523 | 0 | 2026-06 |
| MA | C | 524 | 0 | 2026-06 |
| MCD | C | 524 | 0 | 2026-06 |
| MCHP | C | 309 | 0 | 2026-06 |
| MDLZ | C | 523 | 0 | 2026-06 |
| MDT | C | 544 | 0 | 2026-06 |
| META | C | 254 | 0 | 2026-06 |
| MMM | C | 544 | 0 | 2026-06 |
| MRK | C | 544 | 0 | 2026-06 |
| MS | C | 544 | 0 | 2026-06 |
| MSFT | C | 580 | 0 | 2026-06 |
| MU | C | 544 | 0 | 2026-06 |
| NEE | C | 240 | 0 | 2026-06 |
| NFLX | C | 544 | 0 | 2026-06 |
| NKE | C | 544 | 0 | 2026-06 |
| NOW | C | 537 | 0 | 2026-06 |
| NVDA | C | 580 | 0 | 2026-06 |
| NXPI | N | 527 | 0 | 2026-05 |
| ORCL | C | 544 | 0 | 2026-06 |
| PANW | C | 544 | 0 | 2026-06 |
| PEP | C | 544 | 0 | 2026-06 |
| PFE | C | 544 | 0 | 2026-06 |
| PG | C | 544 | 0 | 2026-06 |
| PLTR | C | 295 | 0 | 2026-06 |
| PM | C | 544 | 0 | 2026-06 |
| QCOM | C | 544 | 0 | 2026-06 |
| REGN | C | 516 | 0 | 2026-06 |
| RTX | C | 319 | 0 | 2026-06 |
| SBUX | C | 544 | 0 | 2026-06 |
| SCHW | C | 516 | 418 | 2026-06 |
| SMCI | C | 237 | 0 | 2026-06 |
| SPGI | C | 458 | 0 | 2026-06 |
| T | C | 544 | 0 | 2026-06 |
| TGT | C | 544 | 78 | 2026-06 |
| TJX | C | 158 | 0 | 2026-06 |
| TMO | C | 278 | 0 | 2026-06 |
| TMUS | C | 544 | 0 | 2026-06 |
| TSLA | C | 581 | 0 | 2026-06 |
| TXN | C | 544 | 0 | 2026-06 |
| UAL | N | 427 | 428 | 2024-03 |
| UNH | C | 544 | 0 | 2026-06 |
| UNP | C | 544 | 0 | 2026-06 |
| UPS | C | 544 | 544 | 2026-06 |
| V | C | 544 | 0 | 2026-06 |
| VRTX | C | 544 | 0 | 2026-06 |
| VZ | C | 544 | 0 | 2026-06 |
| WDAY | C | 544 | 0 | 2026-06 |
| WFC | C | 544 | 0 | 2026-06 |
| WMT | C | 544 | 0 | 2026-06 |
| XOM | C | 544 | 0 | 2026-06 |
| XYZ | C | 72 | 0 | 2026-06 |

_Complete: 104 / 115 · Not-done: 11 (AES/AFL/AMD/AOS=early-only tiny; KLAC/LIN/NXPI=2026-05 ceiling; CME/CVS/EA/UAL=active)_


## 2026-06-14 00:00 local (logged 00:16) · 50163 chains · 110 names · +1849 in 8h · worker alive

| Ticker | St | Parts | New(8h) | MaxExp |
|--------|:--:|------:|--------:|--------|
| AAPL | C | 581 | 0 | 2026-06 |
| ABBV | C | 544 | 0 | 2026-06 |
| ABNB | C | 286 | 0 | 2026-06 |
| ABT | C | 532 | 0 | 2026-06 |
| ACN | C | 544 | 0 | 2026-06 |
| ADBE | C | 486 | 0 | 2026-06 |
| ADI | C | 460 | 0 | 2026-06 |
| ADP | C | 480 | 0 | 2026-06 |
| AES | N | 12 | 0 | 2016-12 |
| AFL | N | 11 | 0 | 2016-11 |
| AMAT | C | 544 | 0 | 2026-06 |
| AMD | N | 29 | 0 | 2016-07 |
| AMGN | C | 544 | 0 | 2026-06 |
| AMZN | C | 581 | 0 | 2026-06 |
| AOS | N | 20 | 0 | 2017-08 |
| AVGO | C | 580 | 0 | 2026-06 |
| AXP | C | 544 | 15 | 2026-06 |
| BA | C | 544 | 0 | 2026-06 |
| BAC | C | 544 | 0 | 2026-06 |
| BKNG | C | 422 | 0 | 2026-06 |
| BMY | C | 544 | 0 | 2026-06 |
| C | C | 544 | 0 | 2026-06 |
| CAT | C | 544 | 0 | 2026-06 |
| CHTR | C | 528 | 0 | 2026-06 |
| CMCSA | C | 544 | 0 | 2026-06 |
| COIN | C | 267 | 0 | 2026-06 |
| COP | C | 544 | 0 | 2026-06 |
| COST | C | 544 | 0 | 2026-06 |
| CRM | C | 544 | 0 | 2026-06 |
| CRWD | C | 357 | 0 | 2026-06 |
| CSCO | C | 544 | 0 | 2026-06 |
| CSX | N | 206 | 206 | 2021-01 |
| CVX | C | 544 | 0 | 2026-06 |
| DASH | C | 286 | 0 | 2026-06 |
| DDOG | C | 320 | 320 | 2026-06 |
| DHR | C | 312 | 0 | 2026-06 |
| DIS | C | 544 | 0 | 2026-06 |
| FISV | C | 237 | 0 | 2026-06 |
| GE | C | 544 | 0 | 2026-06 |
| GILD | C | 544 | 0 | 2026-06 |
| GOOG | C | 544 | 0 | 2026-06 |
| GOOGL | C | 579 | 0 | 2026-06 |
| GS | C | 544 | 0 | 2026-06 |
| HD | C | 544 | 0 | 2026-06 |
| HON | C | 475 | 0 | 2026-06 |
| HOOD | C | 252 | 0 | 2026-06 |
| IBM | C | 544 | 0 | 2026-06 |
| INTC | C | 544 | 0 | 2026-06 |
| INTU | C | 414 | 0 | 2026-06 |
| ISRG | C | 544 | 0 | 2026-06 |
| JNJ | C | 544 | 0 | 2026-06 |
| JPM | C | 544 | 0 | 2026-06 |
| KLAC | N | 317 | 0 | 2026-05 |
| KMB | C | 463 | 0 | 2026-06 |
| KO | C | 544 | 0 | 2026-06 |
| LIN | N | 114 | 0 | 2026-05 |
| LLY | C | 524 | 0 | 2026-06 |
| LMT | C | 502 | 251 | 2026-06 |
| LOW | C | 505 | 0 | 2026-06 |
| LRCX | C | 460 | 0 | 2026-06 |
| LULU | C | 523 | 0 | 2026-06 |
| MA | C | 524 | 0 | 2026-06 |
| MCD | C | 524 | 0 | 2026-06 |
| MCHP | C | 309 | 95 | 2026-06 |
| MDLZ | C | 523 | 0 | 2026-06 |
| MDT | C | 544 | 0 | 2026-06 |
| META | C | 254 | 0 | 2026-06 |
| MMM | C | 544 | 352 | 2026-06 |
| MRK | C | 544 | 0 | 2026-06 |
| MS | C | 544 | 0 | 2026-06 |
| MSFT | C | 580 | 0 | 2026-06 |
| MU | C | 544 | 0 | 2026-06 |
| NEE | C | 240 | 0 | 2026-06 |
| NFLX | C | 544 | 0 | 2026-06 |
| NKE | C | 544 | 0 | 2026-06 |
| NOW | C | 537 | 0 | 2026-06 |
| NVDA | C | 580 | 0 | 2026-06 |
| NXPI | N | 527 | 0 | 2026-05 |
| ORCL | C | 544 | 0 | 2026-06 |
| PANW | C | 544 | 0 | 2026-06 |
| PEP | C | 544 | 0 | 2026-06 |
| PFE | C | 544 | 0 | 2026-06 |
| PG | C | 544 | 0 | 2026-06 |
| PLTR | C | 295 | 0 | 2026-06 |
| PM | C | 544 | 0 | 2026-06 |
| QCOM | C | 544 | 0 | 2026-06 |
| REGN | C | 516 | 0 | 2026-06 |
| RTX | C | 319 | 0 | 2026-06 |
| SBUX | C | 544 | 0 | 2026-06 |
| SCHW | N | 121 | 121 | 2018-11 |
| SMCI | C | 237 | 0 | 2026-06 |
| SPGI | C | 458 | 0 | 2026-06 |
| T | C | 544 | 0 | 2026-06 |
| TGT | N | 479 | 479 | 2025-03 |
| TJX | C | 158 | 0 | 2026-06 |
| TMO | C | 278 | 0 | 2026-06 |
| TMUS | C | 544 | 0 | 2026-06 |
| TSLA | C | 581 | 0 | 2026-06 |
| TXN | C | 544 | 0 | 2026-06 |
| UNH | C | 544 | 0 | 2026-06 |
| UNP | C | 544 | 0 | 2026-06 |
| UPS | N | 10 | 10 | 2016-03 |
| V | C | 544 | 0 | 2026-06 |
| VRTX | C | 544 | 0 | 2026-06 |
| VZ | C | 544 | 0 | 2026-06 |
| WDAY | C | 544 | 0 | 2026-06 |
| WFC | C | 544 | 0 | 2026-06 |
| WMT | C | 544 | 0 | 2026-06 |
| XOM | C | 544 | 0 | 2026-06 |
| XYZ | C | 72 | 0 | 2026-06 |

_Complete: 99 / 110 · Not-done: 11 (AES/AFL/AMD/AOS/UPS=early-only; KLAC/LIN/NXPI=2026-05 ceiling; CSX/SCHW/TGT=active)_


## 2026-06-13 16:00 local (logged 16:11) · 48,360 chains · 105 names · +2,201 in 8h · worker alive

| Ticker | Parts | MaxExp | New(8h) | Done |
|---------:|------:|-------:|--------:|-----:|
| AAPL | 581 | 20260610 | 0 | Y |
| ABBV | 544 | 20260605 | 0 | Y |
| ABNB | 286 | 20260605 | 0 | Y |
| ABT | 532 | 20260605 | 0 | Y |
| ACN | 544 | 20260605 | 0 | Y |
| ADBE | 486 | 20260605 | 0 | Y |
| ADI | 460 | 20260605 | 0 | Y |
| ADP | 480 | 20260605 | 0 | Y |
| AES | 12 | 20161216 | 0 | N |
| AFL | 11 | 20161118 | 0 | N |
| AMAT | 544 | 20260605 | 0 | Y |
| AMD | 29 | 20160722 | 0 | N |
| AMGN | 544 | 20260605 | 0 | Y |
| AMZN | 581 | 20260610 | 0 | Y |
| AOS | 20 | 20170818 | 0 | N |
| AVGO | 580 | 20260610 | 0 | Y |
| AXP | 537 | 20260417 | 526 | N |
| BA | 544 | 20260605 | 0 | Y |
| BAC | 544 | 20260605 | 0 | Y |
| BKNG | 422 | 20260605 | 0 | Y |
| BMY | 544 | 20260605 | 0 | Y |
| C | 544 | 20260605 | 0 | Y |
| CAT | 544 | 20260605 | 0 | Y |
| CHTR | 528 | 20260605 | 0 | Y |
| CMCSA | 544 | 20260605 | 0 | Y |
| COIN | 267 | 20260605 | 0 | Y |
| COP | 544 | 20260605 | 0 | Y |
| COST | 544 | 20260605 | 0 | Y |
| CRM | 544 | 20260605 | 0 | Y |
| CRWD | 357 | 20260605 | 0 | Y |
| CSCO | 544 | 20260605 | 0 | Y |
| CVX | 544 | 20260605 | 0 | Y |
| DASH | 286 | 20260605 | 0 | Y |
| DHR | 312 | 20260605 | 0 | Y |
| DIS | 544 | 20260605 | 0 | Y |
| FISV | 237 | 20260605 | 237 | Y |
| GE | 544 | 20260605 | 0 | Y |
| GILD | 544 | 20260605 | 0 | Y |
| GOOG | 544 | 20260605 | 0 | Y |
| GOOGL | 579 | 20260610 | 0 | Y |
| GS | 544 | 20260605 | 0 | Y |
| HD | 544 | 20260605 | 0 | Y |
| HON | 475 | 20260605 | 0 | Y |
| HOOD | 252 | 20260605 | 0 | Y |
| IBM | 544 | 20260605 | 0 | Y |
| INTC | 544 | 20260605 | 0 | Y |
| INTU | 414 | 20260605 | 0 | Y |
| ISRG | 544 | 20260605 | 0 | Y |
| JNJ | 544 | 20260605 | 0 | Y |
| JPM | 544 | 20260605 | 0 | Y |
| KLAC | 317 | 20260515 | 0 | N |
| KMB | 463 | 20260605 | 0 | Y |
| KO | 544 | 20260605 | 0 | Y |
| LIN | 114 | 20260515 | 0 | N |
| LLY | 524 | 20260605 | 0 | Y |
| LMT | 260 | 20211015 | 260 | N |
| LOW | 505 | 20260605 | 0 | Y |
| LRCX | 460 | 20260605 | 0 | Y |
| LULU | 523 | 20260605 | 0 | Y |
| MA | 524 | 20260605 | 0 | Y |
| MCD | 524 | 20260605 | 0 | Y |
| MCHP | 227 | 20240621 | 227 | N |
| MDLZ | 523 | 20260605 | 0 | Y |
| MDT | 544 | 20260605 | 266 | Y |
| META | 254 | 20260610 | 0 | Y |
| MMM | 208 | 20191227 | 183 | N |
| MRK | 544 | 20260605 | 0 | Y |
| MS | 544 | 20260605 | 0 | Y |
| MSFT | 580 | 20260610 | 0 | Y |
| MU | 544 | 20260605 | 0 | Y |
| NEE | 240 | 20260605 | 0 | Y |
| NFLX | 544 | 20260605 | 0 | Y |
| NKE | 544 | 20260605 | 0 | Y |
| NOW | 537 | 20260605 | 0 | Y |
| NVDA | 580 | 20260610 | 0 | Y |
| NXPI | 527 | 20260515 | 0 | N |
| ORCL | 544 | 20260605 | 0 | Y |
| PANW | 544 | 20260605 | 0 | Y |
| PEP | 544 | 20260605 | 0 | Y |
| PFE | 544 | 20260605 | 0 | Y |
| PG | 544 | 20260605 | 0 | Y |
| PLTR | 295 | 20260605 | 0 | Y |
| PM | 544 | 20260605 | 0 | Y |
| QCOM | 544 | 20260605 | 0 | Y |
| REGN | 516 | 20260605 | 0 | Y |
| RTX | 319 | 20260605 | 0 | Y |
| SBUX | 544 | 20260605 | 0 | Y |
| SMCI | 237 | 20260605 | 0 | Y |
| SPGI | 458 | 20260605 | 207 | Y |
| T | 544 | 20260605 | 0 | Y |
| TJX | 158 | 20260605 | 0 | Y |
| TMO | 278 | 20260605 | 0 | Y |
| TMUS | 544 | 20260605 | 0 | Y |
| TSLA | 581 | 20260610 | 0 | Y |
| TXN | 544 | 20260605 | 0 | Y |
| UNH | 544 | 20260605 | 0 | Y |
| UNP | 544 | 20260605 | 0 | Y |
| V | 544 | 20260605 | 0 | Y |
| VRTX | 544 | 20260605 | 0 | Y |
| VZ | 544 | 20260605 | 0 | Y |
| WDAY | 544 | 20260605 | 295 | Y |
| WFC | 544 | 20260605 | 0 | Y |
| WMT | 544 | 20260605 | 0 | Y |
| XOM | 544 | 20260605 | 0 | Y |
| XYZ | 72 | 20260605 | 0 | Y |

_Active workers: AXP (2026-04, ~2mo left), MCHP (2024-08, ~22mo left), MMM (2020-01, ~77mo left), LMT (2021-10, ~56mo left). KLAC/LIN/NXPI stuck at 2026-05 (Theta data ceiling). AES/AFL/AMD/AOS tiny early-only data (Theta coverage limited)._


## 2026-06-13 08:00 local (logged 08:08) · 46,206 chains · 102 names · +1,976 in 8h · worker alive

| Ticker | Partitions | MaxExp | Complete |
|---------:|----------:|----------:|:---:|
| AAPL | 581 | 2026-06-10 | Y |
| ABBV | 544 | 2026-06-05 | Y |
| ABNB | 286 | 2026-06-05 | Y |
| ABT | 532 | 2026-06-05 | Y |
| ACN | 544 | 2026-06-05 | Y |
| ADBE | 486 | 2026-06-05 | Y |
| ADI | 460 | 2026-06-05 | Y |
| ADP | 480 | 2026-06-05 | Y |
| AES | 12 | 2016-12-16 | N |
| AFL | 11 | 2016-11-18 | N |
| AMAT | 544 | 2026-06-05 | Y |
| AMD | 29 | 2016-07-22 | N |
| AMGN | 544 | 2026-06-05 | Y |
| AMZN | 581 | 2026-06-10 | Y |
| AOS | 20 | 2017-08-18 | N |
| AVGO | 580 | 2026-06-10 | Y |
| AXP | 24 | 2016-06-17 | N |
| BA | 544 | 2026-06-05 | Y |
| BAC | 544 | 2026-06-05 | Y |
| BKNG | 422 | 2026-06-05 | Y |
| BMY | 544 | 2026-06-05 | Y |
| C | 544 | 2026-06-05 | Y |
| CAT | 544 | 2026-06-05 | Y |
| CHTR | 528 | 2026-06-05 | Y |
| CMCSA | 544 | 2026-06-05 | Y |
| COIN | 267 | 2026-06-05 | Y |
| COP | 544 | 2026-06-05 | Y |
| COST | 544 | 2026-06-05 | Y |
| CRM | 544 | 2026-06-05 | Y |
| CRWD | 357 | 2026-06-05 | Y |
| CSCO | 544 | 2026-06-05 | Y |
| CVX | 544 | 2026-06-05 | Y |
| DASH | 286 | 2026-06-05 | Y |
| DHR | 312 | 2026-06-05 | Y |
| DIS | 544 | 2026-06-05 | Y |
| GE | 544 | 2026-06-05 | Y |
| GILD | 544 | 2026-06-05 | Y |
| GOOG | 544 | 2026-06-05 | Y |
| GOOGL | 579 | 2026-06-10 | Y |
| GS | 544 | 2026-06-05 | Y |
| HD | 544 | 2026-06-05 | Y |
| HON | 475 | 2026-06-05 | Y |
| HOOD | 252 | 2026-06-05 | Y |
| IBM | 544 | 2026-06-05 | Y |
| INTC | 544 | 2026-06-05 | Y |
| INTU | 414 | 2026-06-05 | Y |
| ISRG | 544 | 2026-06-05 | Y |
| JNJ | 544 | 2026-06-05 | Y |
| JPM | 544 | 2026-06-05 | Y |
| KLAC | 317 | 2026-05-15 | N |
| KMB | 463 | 2026-06-05 | Y |
| KO | 544 | 2026-06-05 | Y |
| LIN | 114 | 2026-05-15 | N |
| LLY | 524 | 2026-06-05 | Y |
| LOW | 505 | 2026-06-05 | Y |
| LRCX | 460 | 2026-06-05 | Y |
| LULU | 523 | 2026-06-05 | Y |
| MA | 524 | 2026-06-05 | Y |
| MCD | 524 | 2026-06-05 | Y |
| MDLZ | 523 | 2026-06-05 | Y |
| MDT | 294 | 2021-08-20 | N |
| META | 254 | 2026-06-10 | Y |
| MMM | 25 | 2016-06-24 | N |
| MRK | 544 | 2026-06-05 | Y |
| MS | 544 | 2026-06-05 | Y |
| MSFT | 580 | 2026-06-10 | Y |
| MU | 544 | 2026-06-05 | Y |
| NEE | 240 | 2026-06-05 | Y |
| NFLX | 544 | 2026-06-05 | Y |
| NKE | 544 | 2026-06-05 | Y |
| NOW | 537 | 2026-06-05 | Y |
| NVDA | 580 | 2026-06-10 | Y |
| NXPI | 527 | 2026-05-15 | N |
| ORCL | 544 | 2026-06-05 | Y |
| PANW | 544 | 2026-06-05 | Y |
| PEP | 544 | 2026-06-05 | Y |
| PFE | 544 | 2026-06-05 | Y |
| PG | 544 | 2026-06-05 | Y |
| PLTR | 295 | 2026-06-05 | Y |
| PM | 544 | 2026-06-05 | Y |
| QCOM | 544 | 2026-06-05 | Y |
| REGN | 516 | 2026-06-05 | Y |
| RTX | 319 | 2026-06-05 | Y |
| SBUX | 544 | 2026-06-05 | Y |
| SMCI | 237 | 2026-06-05 | Y |
| SPGI | 262 | 2022-09-02 | N |
| T | 544 | 2026-06-05 | Y |
| TJX | 158 | 2026-06-05 | Y |
| TMO | 278 | 2026-06-05 | Y |
| TMUS | 544 | 2026-06-05 | Y |
| TSLA | 581 | 2026-06-10 | Y |
| TXN | 544 | 2026-06-05 | Y |
| UNH | 544 | 2026-06-05 | Y |
| UNP | 544 | 2026-06-05 | Y |
| V | 544 | 2026-06-05 | Y |
| VRTX | 544 | 2026-06-05 | Y |
| VZ | 544 | 2026-06-05 | Y |
| WDAY | 260 | 2020-12-24 | N |
| WFC | 544 | 2026-06-05 | Y |
| WMT | 544 | 2026-06-05 | Y |
| XOM | 544 | 2026-06-05 | Y |
| XYZ | 72 | 2026-06-05 | Y |


## 2026-06-13 00:00 local (logged 00:08) · 44,260 chains · 97 names · +1,483 in 8h · worker alive

| Ticker | Partitions | MaxExp | Complete |
|---------:|----------:|----------:|:---:|
| AAPL | 581 | 2026-06-10 | Y |
| ABBV | 544 | 2026-06-05 | Y |
| ABNB | 286 | 2026-06-05 | Y |
| ABT | 532 | 2026-06-05 | Y |
| ACN | 544 | 2026-06-05 | Y |
| ADBE | 486 | 2026-06-05 | Y |
| ADI | 460 | 2026-06-05 | Y |
| ADP | 267 | 2022-05-06 | N |
| AES | 12 | 2016-12-16 | N |
| AFL | 11 | 2016-11-18 | N |
| AMAT | 544 | 2026-06-05 | Y |
| AMD | 29 | 2016-07-22 | N |
| AMGN | 544 | 2026-06-05 | Y |
| AMZN | 581 | 2026-06-10 | Y |
| AOS | 20 | 2017-08-18 | N |
| AVGO | 580 | 2026-06-10 | Y |
| BA | 544 | 2026-06-05 | Y |
| BAC | 544 | 2026-06-05 | Y |
| BKNG | 422 | 2026-06-05 | Y |
| BMY | 544 | 2026-06-05 | Y |
| C | 544 | 2026-06-05 | Y |
| CAT | 544 | 2026-06-05 | Y |
| CHTR | 528 | 2026-06-05 | Y |
| CMCSA | 544 | 2026-06-05 | Y |
| COIN | 267 | 2026-06-05 | Y |
| COP | 544 | 2026-06-05 | Y |
| COST | 544 | 2026-06-05 | Y |
| CRM | 544 | 2026-06-05 | Y |
| CRWD | 357 | 2026-06-05 | Y |
| CSCO | 544 | 2026-06-05 | Y |
| CVX | 544 | 2026-06-05 | Y |
| DASH | 286 | 2026-06-05 | Y |
| DHR | 312 | 2026-06-05 | Y |
| DIS | 544 | 2026-06-05 | Y |
| GE | 544 | 2026-06-05 | Y |
| GILD | 544 | 2026-06-05 | Y |
| GOOG | 544 | 2026-06-05 | Y |
| GOOGL | 579 | 2026-06-10 | Y |
| GS | 544 | 2026-06-05 | Y |
| HD | 544 | 2026-06-05 | Y |
| HON | 475 | 2026-06-05 | Y |
| HOOD | 252 | 2026-06-05 | Y |
| IBM | 544 | 2026-06-05 | Y |
| INTC | 544 | 2026-06-05 | Y |
| INTU | 414 | 2026-06-05 | Y |
| ISRG | 544 | 2026-06-05 | Y |
| JNJ | 544 | 2026-06-05 | Y |
| JPM | 544 | 2026-06-05 | Y |
| KLAC | 317 | 2026-05-15 | N |
| KO | 544 | 2026-06-05 | Y |
| LIN | 114 | 2026-05-15 | N |
| LLY | 524 | 2026-06-05 | Y |
| LOW | 505 | 2026-06-05 | Y |
| LRCX | 460 | 2026-06-05 | Y |
| LULU | 489 | 2025-10-10 | N |
| MA | 524 | 2026-06-05 | Y |
| MCD | 524 | 2026-06-05 | Y |
| MDLZ | 523 | 2026-06-05 | Y |
| META | 254 | 2026-06-10 | Y |
| MMM | 25 | 2016-06-24 | N |
| MRK | 544 | 2026-06-05 | Y |
| MS | 544 | 2026-06-05 | Y |
| MSFT | 580 | 2026-06-10 | Y |
| MU | 544 | 2026-06-05 | Y |
| NEE | 240 | 2026-06-05 | Y |
| NFLX | 544 | 2026-06-05 | Y |
| NKE | 544 | 2026-06-05 | Y |
| NOW | 537 | 2026-06-05 | Y |
| NVDA | 580 | 2026-06-10 | Y |
| NXPI | 527 | 2026-05-15 | N |
| ORCL | 544 | 2026-06-05 | Y |
| PANW | 544 | 2026-06-05 | Y |
| PEP | 544 | 2026-06-05 | Y |
| PFE | 544 | 2026-06-05 | Y |
| PG | 544 | 2026-06-05 | Y |
| PLTR | 295 | 2026-06-05 | Y |
| PM | 544 | 2026-06-05 | Y |
| QCOM | 544 | 2026-06-05 | Y |
| REGN | 516 | 2026-06-05 | Y |
| RTX | 75 | 2023-02-10 | N |
| SBUX | 544 | 2026-06-05 | Y |
| SMCI | 237 | 2026-06-05 | Y |
| T | 544 | 2026-06-05 | Y |
| TJX | 158 | 2026-06-05 | Y |
| TMO | 278 | 2026-06-05 | Y |
| TMUS | 544 | 2026-06-05 | Y |
| TSLA | 581 | 2026-06-10 | Y |
| TXN | 544 | 2026-06-05 | Y |
| UNH | 544 | 2026-06-05 | Y |
| UNP | 544 | 2026-06-05 | Y |
| V | 544 | 2026-06-05 | Y |
| VRTX | 392 | 2023-07-07 | N |
| VZ | 544 | 2026-06-05 | Y |
| WFC | 544 | 2026-06-05 | Y |
| WMT | 544 | 2026-06-05 | Y |
| XOM | 544 | 2026-06-05 | Y |
| XYZ | 72 | 2026-06-05 | Y |


## 2026-06-12 16:00 local (logged 16:09) · 42,801 chains · 96 names · +1,255 in 8h · worker alive

| Ticker | Partitions | MaxExp | Complete |
|---------:|----------:|----------:|:---:|
| AAPL | 581 | 2026-06-10 | Y |
| ABBV | 544 | 2026-06-05 | Y |
| ABNB | 286 | 2026-06-05 | Y |
| ABT | 532 | 2026-06-05 | Y |
| ACN | 544 | 2026-06-05 | Y |
| ADBE | 486 | 2026-06-05 | Y |
| ADI | 460 | 2026-06-05 | Y |
| AES | 12 | 2016-12-16 | N |
| AFL | 11 | 2016-11-18 | N |
| AMAT | 544 | 2026-06-05 | Y |
| AMD | 29 | 2016-07-22 | N |
| AMGN | 544 | 2026-06-05 | Y |
| AMZN | 581 | 2026-06-10 | Y |
| AOS | 20 | 2017-08-18 | N |
| AVGO | 580 | 2026-06-10 | Y |
| BA | 544 | 2026-06-05 | Y |
| BAC | 544 | 2026-06-05 | Y |
| BKNG | 422 | 2026-06-05 | Y |
| BMY | 544 | 2026-06-05 | Y |
| C | 544 | 2026-06-05 | Y |
| CAT | 544 | 2026-06-05 | Y |
| CHTR | 528 | 2026-06-05 | Y |
| CMCSA | 544 | 2026-06-05 | Y |
| COIN | 267 | 2026-06-05 | Y |
| COP | 544 | 2026-06-05 | Y |
| COST | 544 | 2026-06-05 | Y |
| CRM | 544 | 2026-06-05 | Y |
| CRWD | 357 | 2026-06-05 | Y |
| CSCO | 544 | 2026-06-05 | Y |
| CVX | 544 | 2026-06-05 | Y |
| DASH | 286 | 2026-06-05 | Y |
| DHR | 169 | 2023-09-08 | N |
| DIS | 544 | 2026-06-05 | Y |
| GE | 544 | 2026-06-05 | Y |
| GILD | 544 | 2026-06-05 | Y |
| GOOG | 544 | 2026-06-05 | Y |
| GOOGL | 579 | 2026-06-10 | Y |
| GS | 544 | 2026-06-05 | Y |
| HD | 544 | 2026-06-05 | Y |
| HON | 475 | 2026-06-05 | Y |
| HOOD | 252 | 2026-06-05 | Y |
| IBM | 544 | 2026-06-05 | Y |
| INTC | 544 | 2026-06-05 | Y |
| INTU | 414 | 2026-06-05 | Y |
| ISRG | 544 | 2026-06-05 | Y |
| JNJ | 544 | 2026-06-05 | Y |
| JPM | 544 | 2026-06-05 | Y |
| KLAC | 317 | 2026-05-15 | N |
| KO | 544 | 2026-06-05 | Y |
| LIN | 114 | 2026-05-15 | N |
| LLY | 524 | 2026-06-05 | Y |
| LOW | 505 | 2026-06-05 | Y |
| LRCX | 460 | 2026-06-05 | Y |
| LULU | 273 | 2021-08-20 | N |
| MA | 524 | 2026-06-05 | Y |
| MCD | 524 | 2026-06-05 | Y |
| MDLZ | 69 | 2017-09-22 | N |
| META | 254 | 2026-06-10 | Y |
| MMM | 25 | 2016-06-24 | N |
| MRK | 544 | 2026-06-05 | Y |
| MS | 544 | 2026-06-05 | Y |
| MSFT | 580 | 2026-06-10 | Y |
| MU | 544 | 2026-06-05 | Y |
| NEE | 240 | 2026-06-05 | Y |
| NFLX | 544 | 2026-06-05 | Y |
| NKE | 544 | 2026-06-05 | Y |
| NOW | 537 | 2026-06-05 | Y |
| NVDA | 580 | 2026-06-10 | Y |
| NXPI | 527 | 2026-05-15 | N |
| ORCL | 544 | 2026-06-05 | Y |
| PANW | 544 | 2026-06-05 | Y |
| PEP | 544 | 2026-06-05 | Y |
| PFE | 544 | 2026-06-05 | Y |
| PG | 544 | 2026-06-05 | Y |
| PLTR | 295 | 2026-06-05 | Y |
| PM | 544 | 2026-06-05 | Y |
| QCOM | 544 | 2026-06-05 | Y |
| REGN | 516 | 2026-06-05 | Y |
| RTX | 18 | 2023-02-10 | N |
| SBUX | 544 | 2026-06-05 | Y |
| SMCI | 237 | 2026-06-05 | Y |
| T | 544 | 2026-06-05 | Y |
| TJX | 158 | 2026-06-05 | Y |
| TMO | 278 | 2026-06-05 | Y |
| TMUS | 544 | 2026-06-05 | Y |
| TSLA | 581 | 2026-06-10 | Y |
| TXN | 544 | 2026-06-05 | Y |
| UNH | 544 | 2026-06-05 | Y |
| UNP | 544 | 2026-06-05 | Y |
| V | 544 | 2026-06-05 | Y |
| VRTX | 66 | 2017-04-07 | N |
| VZ | 544 | 2026-06-05 | Y |
| WFC | 544 | 2026-06-05 | Y |
| WMT | 544 | 2026-06-05 | Y |
| XOM | 544 | 2026-06-05 | Y |
| XYZ | 72 | 2026-06-05 | Y |


## 2026-06-12 08:00 local (logged 08:09) · 41,561 chains · 91 names · +1,386 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 12508 + supervisor(3) + java(1) alive; heavy 2/2 OK (len=126791).
Frontier: Two completions this window — **LOW** (63 new chains → 505 total, at 2026-06-05) and **NEE** (240 new chains → 240 total, at 2026-06-05; NEE completed entire 2016→2026 range during this 8h). Active in-progress: **UNP** (357 new chains since midnight, 494 total, at 2025-06-20 — ~1hr from done), **COP** (373 chains all-new since 02:02, at 2023-02-24), **REGN** (139 new, 374 total, at 2023-09-15), **PM** (212 new, 316 total, at 2024-07-26 old frontier — now entering new territory 2024-07→2026-06). **78 complete** of 91 dirs; 13 not yet at 2026-06-01 frontier.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|------|-------|------------------------------|-------:|---------:|
| AAPL | AAPL option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| AMZN | AMZN option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| TSLA | TSLA option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| MSFT | MSFT option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| NVDA | NVDA option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| AVGO | AVGO option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-10 | 579 | 0 (complete) |
| TMUS | TMUS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD | HD option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| IBM | IBM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC | INTC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ISRG | ISRG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ | JNJ option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT | WMT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO | KO option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK | MRK option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MS | MS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC | WFC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU | MU option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN | TXN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX | NFLX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| VZ | VZ option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL | ORCL option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GS | GS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP | PEP option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V | V option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE | PFE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH | UNH option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG | PG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM | QCOM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| SBUX | SBUX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T | T option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NKE | NKE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW | PANW option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM | JPM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GILD | GILD option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV | ABBV option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ACN | ACN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT | AMAT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN | AMGN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA | BA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC | BAC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BMY | BMY option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG | GOOG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CAT | CAT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST | COST option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C | C option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM | XOM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS | DIS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX | CVX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO | CSCO option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GE | GE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM | CRM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NOW | NOW option-history EOD | 2016-01-15 … 2026-06-05 | 537 | 0 (complete) |
| ABT | ABT option-history EOD | 2016-01-15 … 2026-06-05 | 532 | 2 (complete) |
| CHTR | CHTR option-history EOD | 2016-01-15 … 2026-06-05 | 528 | 0 (complete) |
| NXPI | NXPI option-history EOD | 2016-01-08 … 2026-05-15 | 527 | 0 |
| LLY | LLY option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MCD | MCD option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA | MA option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| LOW | LOW option-history EOD | 2016-01-15 … 2026-06-05 | 505 | 63 (complete) |
| UNP | UNP option-history EOD | 2016-01-08 … 2025-06-20 | 494 | 357 |
| ADBE | ADBE option-history EOD | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON | HON option-history EOD | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| ADI | ADI option-history EOD | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| LRCX | LRCX option-history EOD | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| BKNG | BKNG option-history EOD | 2018-03-02 … 2026-06-05 | 422 | 0 (complete) |
| INTU | INTU option-history EOD | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| REGN | REGN option-history EOD | 2016-01-08 … 2023-09-15 | 374 | 139 |
| COP | COP option-history EOD | 2016-01-08 … 2023-02-24 | 373 | 373 |
| CRWD | CRWD option-history EOD | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| KLAC | KLAC option-history EOD | 2016-01-15 … 2026-05-15 | 317 | 0 |
| PM | PM option-history EOD | 2016-01-08 … 2024-07-26 | 316 | 212 |
| PLTR | PLTR option-history EOD | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| ABNB | ABNB option-history EOD | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| DASH | DASH option-history EOD | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| TMO | TMO option-history EOD | 2016-01-15 … 2026-06-05 | 278 | 0 (complete) |
| COIN | COIN option-history EOD | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META | META option-history EOD | 2021-07-16 … 2026-06-10 | 254 | 0 (complete) |
| HOOD | HOOD option-history EOD | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| NEE | NEE option-history EOD | 2016-01-15 … 2026-06-05 | 240 | 240 (complete) |
| SMCI | SMCI option-history EOD | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| TJX | TJX option-history EOD | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LIN | LIN option-history EOD | 2018-11-02 … 2026-05-15 | 114 | 0 |
| AMD | AMD option-history EOD | 2016-01-08 … 2016-07-22 | 29 | 0 |
| MMM | MMM option-history EOD | 2016-01-08 … 2016-06-24 | 25 | 0 |
| AOS | AOS option-history EOD | 2016-01-15 … 2017-08-18 | 20 | 0 |
| RTX | RTX option-history EOD | 2020-04-17 … 2023-02-10 | 18 | 0 |
| AES | AES option-history EOD | 2016-01-15 … 2016-12-16 | 12 | 0 |
| AFL | AFL option-history EOD | 2016-01-15 … 2016-11-18 | 11 | 0 |
## 2026-06-12 00:00 local (logged 00:13) · 40,207 chains · 90 names · +1,158 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 12508 + supervisor(3) + java(1) alive; heavy 2/2 OK (len=126791, one probe-contention timeout).
Frontier: Three completions this window — **ABT** (326 new chains → 532 total, at 2026-06-05), **NOW** (70 new → 537 total, at 2026-06-05), **MS** (41 new → 544 total, at 2026-06-05). Three new names opened since 16:00: **REGN** (all 237 chains written since 16:00, currently at 2020-07-17), **UNP** (145 chains written, at 2018-10-12), **NEE** (10 chains, just opened at 2016-10-21). Active in-progress: LOW→2025-04-17 (323 new, 446 chains), REGN→2020-07, UNP→2018-10, NEE→2016-10, PM→2024-07 (104 chains stalled, no activity this window). **76 complete** of 90 dirs on disk; 14 names not yet at 2026-06-01 frontier (NXPI/KLAC/LIN stalled at 2026-05-15; LOW/REGN/UNP/NEE/PM/RTX/AMD/MMM/AOS/AES/AFL active or stalled early).

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|------|-------|------------------------------|-------:|---------:|
| AAPL | AAPL option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| AMZN | AMZN option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| TSLA | TSLA option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| MSFT | MSFT option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| NVDA | NVDA option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| AVGO | AVGO option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-10 | 579 | 0 (complete) |
| TMUS | TMUS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD | HD option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| IBM | IBM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC | INTC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ISRG | ISRG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ | JNJ option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT | WMT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO | KO option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK | MRK option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MS | MS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 41 (complete) |
| WFC | WFC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU | MU option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN | TXN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX | NFLX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| VZ | VZ option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL | ORCL option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GS | GS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP | PEP option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V | V option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE | PFE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH | UNH option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG | PG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM | QCOM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| SBUX | SBUX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T | T option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NKE | NKE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW | PANW option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM | JPM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GILD | GILD option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV | ABBV option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ACN | ACN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT | AMAT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN | AMGN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA | BA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC | BAC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BMY | BMY option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG | GOOG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C | C option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CAT | CAT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST | COST option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM | XOM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS | DIS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GE | GE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX | CVX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO | CSCO option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM | CRM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NOW | NOW option-history EOD | 2016-01-15 … 2026-06-05 | 537 | 70 (complete) |
| ABT | ABT option-history EOD | 2016-01-15 … 2026-06-05 | 532 | 326 (complete) |
| CHTR | CHTR option-history EOD | 2016-01-15 … 2026-06-05 | 528 | 0 (complete) |
| NXPI | NXPI option-history EOD | 2016-01-08 … 2026-05-15 | 527 | 0 |
| LLY | LLY option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MCD | MCD option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA | MA option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE | ADBE option-history EOD | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON | HON option-history EOD | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| ADI | ADI option-history EOD | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| LRCX | LRCX option-history EOD | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| LOW | LOW option-history EOD | 2016-01-15 … 2025-04-17 | 446 | 323 |
| BKNG | BKNG option-history EOD | 2018-03-02 … 2026-06-05 | 422 | 0 (complete) |
| INTU | INTU option-history EOD | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| CRWD | CRWD option-history EOD | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| KLAC | KLAC option-history EOD | 2016-01-15 … 2026-05-15 | 317 | 0 |
| PLTR | PLTR option-history EOD | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| ABNB | ABNB option-history EOD | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| DASH | DASH option-history EOD | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| TMO | TMO option-history EOD | 2016-01-15 … 2026-06-05 | 278 | 0 (complete) |
| COIN | COIN option-history EOD | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META | META option-history EOD | 2021-07-16 … 2026-06-10 | 254 | 0 (complete) |
| HOOD | HOOD option-history EOD | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| REGN | REGN option-history EOD | 2016-01-08 … 2020-07-17 | 237 | 237 |
| SMCI | SMCI option-history EOD | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| TJX | TJX option-history EOD | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| UNP | UNP option-history EOD | 2016-01-08 … 2018-10-12 | 145 | 145 |
| LIN | LIN option-history EOD | 2018-11-02 … 2026-05-15 | 114 | 0 |
| PM | PM option-history EOD | 2016-01-08 … 2024-07-26 | 104 | 0 |
| AMD | AMD option-history EOD | 2016-01-08 … 2016-07-22 | 29 | 0 |
| MMM | MMM option-history EOD | 2016-01-08 … 2016-06-24 | 25 | 0 |
| AOS | AOS option-history EOD | 2016-01-15 … 2017-08-18 | 20 | 0 |
| RTX | RTX option-history EOD | 2020-04-17 … 2023-02-10 | 18 | 0 |
| AES | AES option-history EOD | 2016-01-15 … 2016-12-16 | 12 | 0 |
| AFL | AFL option-history EOD | 2016-01-15 … 2016-11-18 | 11 | 0 |
| NEE | NEE option-history EOD | 2016-01-15 … 2016-10-21 | 10 | 10 |
## 2026-06-11 16:00 local (logged 16:03) · 39,052 chains · 87 names · +1,117 since 08:00 · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 12508 + supervisor(3) + java(1) alive; heavy 2/2 OK (len=126791).
Frontier: **BMY** closed out to 2026-06-05 this window (169 new → **73 complete** total). Five lanes still active: **MS** (504, 2016→2025-08, 338 new), **NXPI** (527, 2016→2026-05-15, 153 new — one step from the frontier), **NOW** (468, 2016→2025-02, 139 new), **ABT** (206, 2016→2020-03, 194 new — opened fresh this window), **LOW** (124, 2016→2018-09, 124 new — opened fresh this window). Rate ~1.3–2.1/min; softer than prior windows as ABT + LOW grind through thin 2016–2020 history. KLAC stalled at 2026-05-15 (not in frontier; still needs last ~6 months to reach 2026-06-01). Two more lanes (PM at 2024-07, RTX at 2023-02) partially started but not yet active.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|------|-------|------------------------------|-------:|---------:|
| TSLA | TSLA option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| AAPL | AAPL option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| AMZN | AMZN option-history EOD | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| MSFT | MSFT option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| NVDA | NVDA option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| AVGO | AVGO option-history EOD | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-10 | 579 | 0 (complete) |
| NFLX | NFLX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO | KO option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC | INTC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST | COST option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP | PEP option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V | V option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW | PANW option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL | ORCL option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GE | GE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA | BA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT | WMT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD | HD option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BMY | BMY option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 169 (complete) |
| SBUX | SBUX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG | PG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM | JPM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN | TXN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE | PFE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NKE | NKE option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM | CRM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH | UNH option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS | DIS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV | ABBV option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM | QCOM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC | BAC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| VZ | VZ option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ISRG | ISRG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| IBM | IBM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T | T option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN | AMGN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CAT | CAT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU | MU option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM | XOM option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ACN | ACN option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC | WFC option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT | AMAT option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK | MRK option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO | CSCO option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GS | GS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG | GOOG option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GILD | GILD option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TMUS | TMUS option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX | CVX option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C | C option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ | JNJ option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CHTR | CHTR option-history EOD | 2016-01-15 … 2026-06-05 | 528 | 0 (complete) |
| NXPI | NXPI option-history EOD | 2016-01-08 … 2026-05-15 | 527 | 153 |
| LLY | LLY option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA | MA option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MCD | MCD option-history EOD | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MS | MS option-history EOD | 2016-01-08 … 2025-08-29 | 504 | 338 |
| ADBE | ADBE option-history EOD | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON | HON option-history EOD | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| NOW | NOW option-history EOD | 2016-01-15 … 2025-02-07 | 468 | 139 |
| ADI | ADI option-history EOD | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| LRCX | LRCX option-history EOD | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| BKNG | BKNG option-history EOD | 2018-03-02 … 2026-06-05 | 422 | 0 (complete) |
| INTU | INTU option-history EOD | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| CRWD | CRWD option-history EOD | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| KLAC | KLAC option-history EOD | 2016-01-15 … 2026-05-15 | 317 | 0 |
| PLTR | PLTR option-history EOD | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| DASH | DASH option-history EOD | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| ABNB | ABNB option-history EOD | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| TMO | TMO option-history EOD | 2016-01-15 … 2026-06-05 | 278 | 0 (complete) |
| COIN | COIN option-history EOD | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META | META option-history EOD | 2021-07-16 … 2026-06-10 | 254 | 0 (complete) |
| HOOD | HOOD option-history EOD | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| SMCI | SMCI option-history EOD | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| ABT | ABT option-history EOD | 2016-01-15 … 2020-03-06 | 206 | 194 |
| TJX | TJX option-history EOD | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LOW | LOW option-history EOD | 2016-01-15 … 2018-09-21 | 124 | 124 |
| LIN | LIN option-history EOD | 2018-11-02 … 2026-05-15 | 114 | 0 |
| PM | PM option-history EOD | 2016-01-08 … 2024-07-26 | 104 | 0 |
| AMD | AMD option-history EOD | 2016-01-08 … 2016-07-22 | 29 | 0 |
| MMM | MMM option-history EOD | 2016-01-08 … 2016-06-24 | 25 | 0 |
| AOS | AOS option-history EOD | 2016-01-15 … 2017-08-18 | 20 | 0 |
| RTX | RTX option-history EOD | 2020-04-17 … 2023-02-10 | 18 | 0 |
| AES | AES option-history EOD | 2016-01-15 … 2016-12-16 | 12 | 0 |
| AFL | AFL option-history EOD | 2016-01-15 … 2016-11-18 | 11 | 0 |

## 2026-06-11 08:00 local (logged 08:18) · 37,981 chains · 86 names · +1,320 since 00:00 · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 12508 + supervisor(3) + java(2) alive; heavy 2/2 OK (len=126791).
Frontier: BMY→2023-05, NXPI→2023-04, NOW→2022-08, KLAC→2026-05, MS→2019-08. **72 names complete** to the 2026-06-05/10 frontier (+1 since 00:00: GE closed out to present this window, 544 chains) — plus LIN terminal-complete at 2026-05-15 (≈73 effective). Clean, undisrupted 8h window, ~1.9–3.5/min 4-wide (heavy-probe timeouts this window were all probe-vs-worker connection contention, never a stall). Five deep lanes in flight: **BMY** (383, 2016→2023-05, fresh fill), **NXPI** (380, 2016→2023-04), **NOW** (337, 2016→2022-08), **KLAC** (317, 2016→2026-05-15 — one step from the frontier, finishing its last expirations), and **MS** (190, 2016→2019-08, newest lane). Correction to the 00:00→08:00 narrative: KLAC was NOT complete at ~05:13 (it kept filling in a concurrent worker slot while MS opened) — it is the near-complete lane above. 86/150 names touched; ~64 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-10 | 579 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T     | T option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GS    | GS option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GE    | GE option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 48 (complete) |
| NKE   | NKE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| VZ    | VZ option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CAT   | CAT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO    | KO option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ISRG  | ISRG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TMUS  | TMUS option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| IBM   | IBM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| SBUX  | SBUX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GILD  | GILD option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CHTR  | CHTR option-history EOD  | 2016-01-15 … 2026-06-05 | 528 | 0 (complete) |
| MCD   | MCD option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON   | HON option-history EOD   | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| ADI   | ADI option-history EOD   | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2026-06-05 | 422 | 0 (complete) |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| BMY   | BMY option-history EOD   | 2016-01-08 … 2023-05-05 | 383 | 383 |
| NXPI  | NXPI option-history EOD  | 2016-01-08 … 2023-04-14 | 380 | 343 |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| NOW   | NOW option-history EOD   | 2016-01-15 … 2022-08-05 | 337 | 261 |
| KLAC  | KLAC option-history EOD  | 2016-01-15 … 2026-05-15 | 317 | 95 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| DASH  | DASH option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| ABNB  | ABNB option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| TMO   | TMO option-history EOD   | 2016-01-15 … 2026-06-05 | 278 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-10 | 254 | 0 (complete) |
| HOOD  | HOOD option-history EOD  | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| SMCI  | SMCI option-history EOD  | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| MS    | MS option-history EOD    | 2016-01-08 … 2019-08-23 | 190 | 190 |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (terminal-complete 2026-05-15) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29 | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25 | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20 | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18 | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12 | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12 | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11 | 0 |

## 2026-06-11 00:00 local (logged 23:32) · 36,603 chains · 84 names · +768 since 16:00 · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 12508 + supervisor(3) + java(2) alive; heavy 2/2 OK (len=126791).
Frontier: GE→2025-03, KLAC→2023-01, NOW→2017-04, NXPI→2016-05. **71 names complete** to the 2026-06-05/10 frontier (+2 since 20:09: TMO and BKNG both closed out to present this window) — plus LIN terminal-complete at 2026-05-15 (≈72 effective). Clean, undisrupted ~7.5h window, ~1.3–2.5/min 4-wide (rate dips on dense recent-year chains, recovers on sparse early-year ones; a few heavy-probe timeouts that were all probe-vs-worker connection contention, never a stall). The deep lane **GE** raced from 2023→2025-03 (481 chains, +291) and is nearly at the frontier; **KLAC** (213, 2016→2023-01) is mid-fill; two fresh lanes opened — **NOW** (59, 2016→2017-04) and **NXPI** (20, 2016→2016-05), both early. 84/150 names touched; ~66 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-10 | 579 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NKE   | NKE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TMUS  | TMUS option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GILD  | GILD option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO    | KO option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CAT   | CAT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 12 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ISRG  | ISRG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| VZ    | VZ option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| SBUX  | SBUX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GS    | GS option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T     | T option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| IBM   | IBM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CHTR  | CHTR option-history EOD  | 2016-01-15 … 2026-06-05 | 528 | 0 (complete) |
| MCD   | MCD option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| GE    | GE option-history EOD    | 2016-01-08 … 2025-03-21 | 481 | 291 |
| HON   | HON option-history EOD   | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| ADI   | ADI option-history EOD   | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2026-06-05 | 422 | 21 (complete) |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| ABNB  | ABNB option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| DASH  | DASH option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| TMO   | TMO option-history EOD   | 2016-01-15 … 2026-06-05 | 278 | 152 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-10 | 254 | 0 (complete) |
| HOOD  | HOOD option-history EOD  | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| SMCI  | SMCI option-history EOD  | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| KLAC  | KLAC option-history EOD  | 2016-01-15 … 2023-01-27 | 213 | 213 |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (terminal-complete 2026-05-15) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| NOW   | NOW option-history EOD   | 2016-01-15 … 2017-04-07 | 59 | 59 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29 | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25 | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20 | 0 |
| NXPI  | NXPI option-history EOD  | 2016-01-08 … 2016-05-20 | 20 | 20 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18 | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12 | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12 | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11 | 0 |

## 2026-06-10 20:09 local (16:00 boundary, catch-up) · 36,296 chains · 82 names · +1,123 since 08:00 · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 12508 + supervisor(3) + java(2) alive; heavy 2/2 OK (len=126791).
Frontier: GE→2023-05, TMO→2025-06, KLAC→2021-07, BKNG→2026-04. **69 names complete** to the 2026-06-05/10 frontier (+3 since 00:00: ISRG, ACN, CHTR all closed out to present this window; CAT completed too) — plus LIN terminal-complete at its last listed expiration 2026-05-15 (≈70 effective). **Note:** the terminal crashed ~01:28 and was relaunched at ~09:46 (Theta re-warmed, supervisor relaunched via login shell, `--resume` skipped the ~35,170 done partitions — only wall-clock lost, no data loss). This is the catch-up snapshot for the 08:00+16:00 boundaries that the restart displaced; New-column counts files written since the 08:00 boundary (≈ the 09:46→20:09 post-restart window). After finishing the four deep in-flight lanes (ISRG/ACN/CHTR/CAT), the worker advanced into fresh secondary names: GE (386, 2016→2023-05 so far), TMO (228, 2016→2025-06), KLAC (135, 2016→2021-07) are mid-fill; BKNG (417, →2026-04) nearly at frontier. 82/150 names touched; ~68 secondary names still ahead. Mag7 re-touched +2 each on resume (newest expiration now 2026-06-10).

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 2 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 2 (complete) |
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-10 | 581 | 2 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 2 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 2 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-10 | 580 | 2 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-10 | 579 | 2 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NKE   | NKE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO    | KO option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| IBM   | IBM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TMUS  | TMUS option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CAT   | CAT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 109 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GS    | GS option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GILD  | GILD option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T     | T option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ISRG  | ISRG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 53 (complete) |
| VZ    | VZ option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 61 (complete) |
| SBUX  | SBUX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CHTR  | CHTR option-history EOD  | 2016-01-15 … 2026-06-05 | 528 | 93 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MCD   | MCD option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON   | HON option-history EOD   | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| ADI   | ADI option-history EOD   | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2026-04-02 | 417 | 42 |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| GE    | GE option-history EOD    | 2016-01-08 … 2023-05-26 | 386 | 386 |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| ABNB  | ABNB option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| DASH  | DASH option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-10 | 254 | 2 (complete) |
| HOOD  | HOOD option-history EOD  | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| SMCI  | SMCI option-history EOD  | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| TMO   | TMO option-history EOD   | 2016-01-15 … 2025-06-20 | 228 | 228 |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| KLAC  | KLAC option-history EOD  | 2016-01-15 … 2021-07-30 | 135 | 135 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (terminal-complete 2026-05-15) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29 | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25 | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20 | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18 | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12 | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12 | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11 | 0 |

## 2026-06-10 00:00 local · 34,995 chains · 79 names · +1,522 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive; heavy 2/2 OK.
Frontier: ISRG→2024-10, ACN→2024-03, CHTR→2023-10, CAT→2023-06. 66 names complete (NKE+IBM closed out to the 2026-06-05 frontier this window). Clean, undisrupted 8h window, ~2.4–4.2/min 4-wide (one transient heavy-probe timeout at 16:53 that cleared by 17:22; rate dips on the dense recent-year chains the four deep lanes are grinding). NKE and IBM both reached present day (full-depth 2016→2026). The four active lanes — ISRG, ACN, CHTR (new this window, 2016→2023 so far), CAT (new, 2016→2023) — are all mid-2023/2024 and climbing toward the frontier; none complete yet. 79/150 names touched; ~71 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GS    | GS option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T     | T option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TMUS  | TMUS option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO    | KO option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| VZ    | VZ option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| SBUX  | SBUX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GILD  | GILD option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| IBM   | IBM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 62 (complete) |
| NKE   | NKE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 77 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MCD   | MCD option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON   | HON option-history EOD   | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| ADI   | ADI option-history EOD   | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| ISRG  | ISRG option-history EOD  | 2016-01-08 … 2024-10-11 | 458 | 260 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2024-03-15 | 428 | 414 |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| CHTR  | CHTR option-history EOD  | 2016-01-15 … 2023-10-27 | 392 | 321 |
| CAT   | CAT option-history EOD   | 2016-01-08 … 2023-06-09 | 388 | 388 |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2025-05-02 | 375 | 0 |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| DASH  | DASH option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| ABNB  | ABNB option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| HOOD  | HOOD option-history EOD  | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| SMCI  | SMCI option-history EOD  | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29 | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25 | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20 | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18 | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12 | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12 | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11 | 0 |

## 2026-06-09 16:00 local · 33,541 chains · 78 names · +1,896 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive; heavy 2/2 OK.
Frontier: IBM→2025-08, NKE→2025-05, ISRG→2019/2020, CHTR→2017-12. 64 names complete (GILD/GS/MCD/ADI all closed out to the 2026-06-05 frontier this window; GILD alone +499 from its 2016 floor). Clean, undisrupted 8h window, ~2.7–3.8/min 4-wide (rate dips on the dense recent-year chains the deep lanes are now grinding through). GILD finished its full-depth backfill (2016→2026), and the high-depth tech/financial lanes GS, MCD, ADI all reached present day. Four lanes mid-flight: IBM (into 2025), NKE (into 2025, terminal ~2025-05), ISRG (2019→2020), and the freshly-opened CHTR (Charter, 2016→2017 so far). 78/150 names touched; ~72 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GS    | GS option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 42 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T     | T option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TMUS  | TMUS option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO    | KO option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| VZ    | VZ option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| SBUX  | SBUX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GILD  | GILD option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 499 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MCD   | MCD option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 38 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| IBM   | IBM option-history EOD   | 2016-01-08 … 2025-08-08 | 501 | 501 |
| NKE   | NKE option-history EOD   | 2016-01-08 … 2025-05-02 | 487 | 487 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON   | HON option-history EOD   | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| ADI   | ADI option-history EOD   | 2016-01-15 … 2026-06-05 | 460 | 31 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2025-05-02 | 375 | 0 |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| DASH  | DASH option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| ABNB  | ABNB option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| HOOD  | HOOD option-history EOD  | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| SMCI  | SMCI option-history EOD  | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| ISRG  | ISRG option-history EOD  | 2016-01-08 … 2020-01-10 | 210 | 210 |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| CHTR  | CHTR option-history EOD  | 2016-01-15 … 2017-12-29 | 88 | 88 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29 | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25 | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20 | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18 | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14 | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12 | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12 | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11 | 0 |

## 2026-06-09 08:00 local · 31,700 chains · 74 names · +2,206 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive; heavy 2/2 OK.
Frontier: MCD→2026-01, ADI→2026-02, GS→2025-10, GILD→2017-03. 60 names complete (VZ/TMUS/ABBV/DASH finished full-depth this window). Clean, undisrupted 8h window, ~3.5–7/min 4-wide (rate dips on dense recent-year chains). The telecom/pharma trio VZ, TMUS, ABBV closed out to present-day and DASH finished its post-IPO ramp; new high-depth lanes GS, MCD, ADI are all racing toward the 2026-06 frontier (near complete), plus GILD (Gilead, 2016 floor) just opened. 74/150 names touched; ~76 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO    | KO option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| VZ    | VZ option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 168 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| SBUX  | SBUX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T     | T option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TMUS  | TMUS option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 174 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 187 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| GS    | GS option-history EOD    | 2016-01-08 … 2025-10-03 | 509 | 435 |
| MCD   | MCD option-history EOD   | 2016-01-08 … 2026-01-09 | 503 | 503 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON   | HON option-history EOD   | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| ADI   | ADI option-history EOD   | 2016-01-15 … 2026-02-20 | 445 | 445 |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2025-05-02 | 375 | 0 |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| ABNB  | ABNB option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 0 (complete) |
| DASH  | DASH option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 286 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| HOOD  | HOOD option-history EOD  | 2021-08-13 … 2026-06-05 | 252 | 0 (complete) |
| SMCI  | SMCI option-history EOD  | 2016-01-15 … 2026-06-05 | 237 | 0 (complete) |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| GILD  | GILD option-history EOD  | 2016-01-08 … 2017-03-24 | 64  | 64 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-09 00:00 local · 29,494 chains · 70 names · +2,431 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive; heavy 2/2 OK.
Frontier: GS→2017-10, VZ→2023-06, TMUS→2023-04, ABBV→2023-02. 56 names complete (T/SBUX/SMCI/HOOD/ABNB finished full-depth this window). Clean, undisrupted 8h window, steady ~5–7/min 4-wide. T (AT&T) reached present-day frontier and SBUX/SMCI/HOOD/ABNB closed their post-IPO ramps; new lanes opened — GS (2016 floor) plus the telecom/pharma trio VZ, TMUS, ABBV all backfilling through their dense 2018–2023 chains. 70/150 names touched; ~80 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO    | KO option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| SBUX  | SBUX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 326 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| T     | T option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 544 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON   | HON option-history EOD   | 2016-01-15 … 2026-06-05 | 475 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2026-06-05 | 414 | 0 (complete) |
| VZ    | VZ option-history EOD    | 2016-01-08 … 2023-06-09 | 388 | 388 |
| TMUS  | TMUS option-history EOD  | 2016-01-08 … 2023-04-14 | 380 | 380 |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2025-05-02 | 375 | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2023-02-10 | 371 | 351 |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| ABNB  | ABNB option-history EOD  | 2020-12-18 … 2026-06-05 | 286 | 172 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| HOOD  | HOOD option-history EOD  | 2021-08-13 … 2026-06-05 | 252 | 100 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| SMCI  | SMCI option-history EOD  | 2016-01-15 … 2026-06-05 | 237 | 103 (complete) |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| GS    | GS option-history EOD    | 2016-01-08 … 2017-10-13 | 93  | 93 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-08 16:00 local · 27,063 chains · 66 names · +1,568 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive; heavy 2/2 OK.
Frontier: BKNG→2025-05, SBUX→2020-04, HOOD→2024-09, SMCI→2024-06, ABNB→2023-03. 51 names complete (KO/HON/INTU finished full-depth this window). Clean, undisrupted 8h window, steady ~2.5–5/min 4-wide. KO, HON, INTU reached present-day frontier; new lanes opened — SBUX (2016 floor), SMCI/HOOD/ABNB (post-2020 IPO floors) — plus BKNG still backfilling. 66/150 names touched; ~84 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| KO    | KO option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 463 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| HON   | HON option-history EOD   | 2016-01-15 … 2026-06-05 | 475 | 325 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2026-06-05 | 414 | 120 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2025-05-02 | 375 | 42 |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| SBUX  | SBUX option-history EOD  | 2016-01-08 … 2020-04-17 | 224 | 224 |
| HOOD  | HOOD option-history EOD  | 2021-08-13 … 2024-09-20 | 163 | 163 |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| SMCI  | SMCI option-history EOD  | 2016-01-15 … 2024-06-28 | 136 | 136 |
| ABNB  | ABNB option-history EOD  | 2020-12-18 … 2023-03-17 | 118 | 118 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-05-20 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-08 08:00 local · 25,495 chains · 62 names · +1,335 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive; heavy 2/2 OK.
Frontier: BKNG→2024-07, INTU→2024-03, HON→2020-04, KO→2017-10. 48 names complete (MRK/AMGN/CRWD finished full-depth this window). Clean, undisrupted 8h window, steady ~2–4/min 4-wide (rate dips as lanes work dense recent-year chains). MRK, AMGN, CRWD reached present-day frontier; three new lanes opened — INTU, HON, KO (all 2016-era floors) — plus BKNG still backfilling. 62/150 names touched; ~88 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 143 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 338 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 0 (complete) |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2026-06-05 | 357 | 239 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2024-07-19 | 334 | 76 |
| INTU  | INTU option-history EOD  | 2016-01-15 … 2024-03-22 | 299 | 299 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| HON   | HON option-history EOD   | 2016-01-15 … 2020-04-17 | 155 | 155 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| KO    | KO option-history EOD    | 2016-01-08 … 2017-10-13 | 93  | 93 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-05-20 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-08 00:00 local · 24,160 chains · 59 names · +1,991 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive; heavy 2/2 OK.
Frontier: MRK→2023-09, AMGN→2020-01, CRWD→2021-11, BKNG→2023-02. 45 names complete (PFE/C/DIS finished full-depth this window). Clean, undisrupted 8h window, steady ~3.5–4.5/min 4-wide; PFE+DIS+C completed, then four new lanes opened — MRK, AMGN, CRWD (post-2019 IPO floor) and BKNG backfilling. 59/150 names touched; ~91 secondary names still ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PFE   | PFE option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 544 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 191 (complete) |
| C     | C option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 440 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-06-05 | 460 | 9 (complete) |
| MRK   | MRK option-history EOD   | 2016-01-08 … 2023-09-22 | 403 | 403 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2023-02-03 | 258 | 104 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| AMGN  | AMGN option-history EOD  | 2016-01-08 … 2020-01-10 | 210 | 210 |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| CRWD  | CRWD option-history EOD  | 2019-06-21 … 2021-11-19 | 120 | 120 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-05-20 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-07 16:07 local · 22,169 chains · 55 names · +2,305 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive.
Frontier: C→2018-03, DIS→2022-12, LRCX→2026-05, BKNG→2021-02, PM→2024-07. 41 names complete (WFC/ORCL/CMCSA/HD newly done this window). Clean undisrupted 8h window, steady ~4-5/min 4-wide; new names DIS/C filling, LRCX finishing. 55/150 names touched so far; the still-untouched secondary names remain ahead.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WFC   | WFC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +531 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +238 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CMCSA | CMCSA option-history EOD | 2016-01-08 … 2026-06-05 | 544 | +544 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2026-05-22 | 458 | +396 |
| DIS   | DIS option-history EOD   | 2016-01-08 … 2022-12-16 | 363 | +363 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2021-02-19 | 156 | +118 |
| C     | C option-history EOD     | 2016-01-08 … 2018-03-16 | 115 | +115 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-05-20 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-07 08:02 local · 19,819 chains · 51 names · +2,594 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive.
Frontier: ORCL→2021-07, HD→2026-05, LRCX→2018-03, BKNG→2018-11, PM→2024-07. 37 names complete (PANW/AMAT/CVX/CRM/MA newly done this window). First FULL undisrupted 8h window post-recovery — solid +2,594 (vs +629 last window, which spanned the ~3.9h ThetaData outage). Steady ~5.5/min 4-wide; new names ORCL/LRCX/BKNG filling, HD finishing.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +463 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +75 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +157 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +457 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| HD    | HD option-history EOD    | 2016-01-08 … 2026-05-08 | 540 | +540 |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | 0 (complete) |
| MA    | MA option-history EOD    | 2016-01-08 … 2026-06-05 | 524 | +524 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| ORCL  | ORCL option-history EOD  | 2016-01-08 … 2021-07-30 | 291 | +291 |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | 0 (complete) |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | 0 |
| LRCX  | LRCX option-history EOD  | 2016-01-15 … 2018-03-09 | 50  | +50 |
| BKNG  | BKNG option-history EOD  | 2018-03-02 … 2018-11-09 | 37  | +37 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-05-20 | 20  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-07 00:02 local · 17,225 chains · 46 names · +629 in 8h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 9292 + supervisor(4) + java(2) alive. [First 8-hourly entry; prior entries were 4-hourly.]
Frontier: CVX→2023-02, AMAT→2024-12, CRM→2017-09, PANW→2017-07, PM→2024-07. 32 names complete (PEP/TJX newly done). +8h is heavily suppressed: a ~3.9h ThetaData backend outage (18:35–22:27, post-maintenance) stalled the pull and tripped a false-completion (worker exited 0-done → supervisor set done-flags); recovered ~22:27, false flags cleared, relaunched 4-wide. Now full speed (~6/min) and pulling the previously-unstarted names — CRM/PANW landing, dirs climbing past 44.

| Name | Title | Expiration dates (min … max) | Chains | New (8h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | +3 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +93 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2026-06-05 | 524 | +26 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2024-12-27 | 469 | +123 |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2025-01-03 | 387 | +144 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2026-06-05 | 158 | +32 (complete) |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| PM    | PM option-history EOD    | 2016-01-08 … 2024-07-26 | 104 | +40 |
| CRM   | CRM option-history EOD   | 2016-01-08 … 2017-09-01 | 87  | +87 |
| PANW  | PANW option-history EOD  | 2016-01-08 … 2017-07-21 | 81  | +81 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-05-20 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-06 15:15 local · 16,537 chains · 44 names · +495 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 11816 + supervisor(3) + java(2) alive.
Frontier: CVX→2025-01, TJX→2025-01, AMAT→2024-07, PEP→2024-08, LLY→2025-12, PM→2023-05, RTX→2023-02. 29 names complete (no new this window). Throughput degraded (+495 vs +725 prior): two Theta read-timeout/429 storms (13:23, 14:01) each stalled ~22-33min then self-cleared without restart. CVX/AMAT/PEP filling (+123/+93/+89); new names TJX/PM/RTX started.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | +3 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2025-12-05 | 498 | +31 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2024-08-23 | 451 | +89 |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2024-07-05 | 346 | +93 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | 0 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2025-01-03 | 243 | +123 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| TJX   | TJX option-history EOD   | 2016-01-15 … 2025-01-03 | 84  | +84 |
| PM    | PM option-history EOD    | 2016-01-08 … 2023-05-12 | 47  | +47 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-05-20 | 20  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | 0 |
| RTX   | RTX option-history EOD   | 2020-04-17 … 2023-02-10 | 18  | +18 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-06 11:28 local · 16,072 chains · 41 names · +725 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 11816 + supervisor(3) + java(2) alive.
Frontier: AMAT→2021-01, PEP→2023-02, LLY→2025-05, CVX→2018-04. 29 names complete (PG newly done); AMAT/PEP filling fast (+210/+200), CVX/LLY climbing (+120/+80). Pull resumed 10:54 post-reboot (login-shell fix); end-date now 2026-06-06 (was 06-01).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | +2 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | +2 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-05 | 579 | +5 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | +2 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-05 | 578 | +1 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-05 | 577 | +2 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | +59 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-06-05 | 544 | +1 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-06-05 | 486 | +1 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2025-05-23 | 470 | +80 |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2023-02-03 | 370 | +200 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-06-05 | 295 | +1 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-06-05 | 267 | +1 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2021-01-08 | 262 | +210 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-05 | 252 | +2 (complete) |
| CVX   | CVX option-history EOD   | 2016-01-08 … 2018-04-20 | 120 | +120 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-06-24 | 25  | +4 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2017-08-18 | 20  | +10 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-05-20 | 20  | +3 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-06-17 | 12  | +4 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |

## 2026-06-06 07:28 local · 15,347 chains · 40 names · +873 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: AMAT→2016-12, LLY→2023-11, PG→2025-04, PEP→2019-04. 28 names complete (TXN/BA newly done); PG/LLY filling fast (+219/+187), PEP/AMAT just started (+170/+52).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | +79 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| BA    | BA option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | +166 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-05-29 | 485 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2025-04-17 | 485 | +219 |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2023-11-10 | 390 | +187 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-05-29 | 266 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| PEP   | PEP option-history EOD   | 2016-01-08 … 2019-04-05 | 170 | +170 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| AMAT  | AMAT option-history EOD  | 2016-01-08 … 2016-12-30 | 52  | +52 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-06 03:19 local · 14,433 chains · 38 names · +919 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: TXN→2024-09, LLY→2019-08, BA→2023-02, PG→2020-11. 26 names complete (JNJ/COST newly done); TXN/BA filling fast (+232/+178), PG/LLY just started (+255/+190).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | +62 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | +2 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-05-29 | 485 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2024-09-13 | 454 | +232 |
| BA    | BA option-history EOD    | 2016-01-08 … 2023-02-03 | 370 | +178 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-05-29 | 266 | 0 (complete) |
| PG    | PG option-history EOD    | 2016-01-08 … 2020-11-20 | 255 | +255 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| LLY   | LLY option-history EOD   | 2016-01-08 … 2019-08-23 | 190 | +190 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | 0 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-05 23:10 local · 13,484 chains · 36 names · +889 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: BA→2019-07, TXN→2020-02, JNJ→2025-01, COST→2026-04. 24 names complete (CSCO/LIN newly done); JNJ/COST filling fast (+258/+113), TXN/BA just started.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | +67 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2026-04-10 | 536 | +113 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-05-29 | 485 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2025-01-17 | 472 | +258 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-05-29 | 266 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| TXN   | TXN option-history EOD   | 2016-01-08 … 2020-02-07 | 214 | +214 |
| BA    | BA option-history EOD    | 2016-01-08 … 2019-07-12 | 184 | +184 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2026-05-15 | 114 | +53 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-05 19:32 local · 12,668 chains · 34 names · +739 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: LIN→2024-03 (new name), CSCO→2025-07, JNJ→2020-07. 22 names complete (QCOM/ADBE newly done); CSCO/COST/JNJ filling fast (+217/+116/+215), LIN just started.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | +110 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2025-07-18 | 498 | +217 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-05-29 | 485 | 0 (complete) |
| COST  | COST option-history EOD  | 2016-01-08 … 2024-04-19 | 433 | +116 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-05-29 | 266 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2020-07-02 | 235 | +215 |
| LIN   | LIN option-history EOD   | 2018-11-02 … 2024-03-28 | 81  | +81 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-05 15:23 local · 11,904 chains · 33 names · +1,012 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: JNJ→2016-04, COST→2022-01, CSCO→2021-03, QCOM→2024-03. 20 names complete (BAC newly done); ADBE topped up to frontier; QCOM/COST/CSCO filling fast (+203/+241/+273), JNJ just started.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | +261 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2026-05-29 | 485 | +21 |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2024-03-01 | 426 | +203 |
| COST  | COST option-history EOD  | 2016-01-08 … 2022-01-07 | 314 | +241 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| CSCO  | CSCO option-history EOD  | 2016-01-08 … 2021-03-26 | 273 | +273 |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-05-29 | 266 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| JNJ   | JNJ option-history EOD   | 2016-01-08 … 2016-04-01 | 13  | +13 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-05 11:14 local · 10,856 chains · 31 names · +870 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: QCOM→2020-01, BAC→2021-03, ADBE→2025-11, COST→2017-03. 19 names complete (XOM/UNH/COIN newly done); ADBE nearly done; BAC/QCOM/COST filling.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | +75 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | +106 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2025-11-28 | 459 | +119 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| BAC   | BAC option-history EOD   | 2016-01-08 … 2021-03-12 | 271 | +271 |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2026-05-29 | 266 | +23 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| QCOM  | QCOM option-history EOD  | 2016-01-08 … 2020-01-31 | 213 | +213 |
| COST  | COST option-history EOD  | 2016-01-08 … 2017-03-17 | 63  | +63 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-05 07:05 local · 9,955 chains · 28 names · +787 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: XOM→2024-09, UNH→2024-03, ADBE→2023-07, COIN→2025-11. 16 names complete; XOM/UNH/ADBE/COIN filling fast (+309/+216/+138/+124).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2024-09-13 | 454 | +309 |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2024-03-22 | 429 | +216 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2023-07-21 | 336 | +138 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2025-11-14 | 238 | +124 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-05 03:27 local · 9,242 chains · 28 names · +798 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: XOM→2019-05, UNH→2020-07, ADBE→2021-03, COIN→2023-08. 16 names complete (INTC/JPM/NFLX newly done); UNH/ADBE/XOM/COIN filling; ADBE being topped up from leftover.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | +8 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | +56 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | +20 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| UNH   | UNH option-history EOD   | 2016-01-08 … 2020-07-10 | 236 | +236 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2021-03-05 | 212 | +195 |
| XOM   | XOM option-history EOD   | 2016-01-08 … 2019-05-03 | 174 | +174 |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2023-08-25 | 122 | +109 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-04 23:18 local · 8,426 chains · 26 names · +598 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: INTC→2026-02, JPM→2025-03, NFLX→2026-01, COIN→2021-06. 13 names complete (V newly done); INTC/NFLX nearing their ends; COIN just started.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2026-05-29 | 543 | +152 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2026-02-13 | 528 | +226 |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2026-01-09 | 523 | +32 |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2025-03-28 | 482 | +181 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |
| COIN  | COIN option-history EOD  | 2021-04-30 … 2021-06-11 | 7   | +7 |

## 2026-06-04 19:09 local · 7,798 chains · 25 names · +766 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: V→2023-05, JPM→2021-07, INTC→2021-07, NFLX→2025-05. 12 names complete; NFLX/V/JPM/INTC filling fast (+223/+215/+267 this window).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2025-05-16 | 489 | +61 |
| V     | V option-history EOD     | 2016-01-08 … 2023-05-19 | 385 | +223 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2021-07-30 | 291 | +215 |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2021-07-23 | 290 | +267 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-04 16:02 local · 7,204 chains · 25 names · +592 in 4h · worker alive  *(lab-handoff snapshot)*
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: V→2020-03, JPM→2018-03, INTC→2017-08, NFLX→2024-06. 12 names complete (MSFT, GOOGL, GOOG, META, AAPL, AMZN, NVDA, TSLA, PLTR, WMT, AVGO, MU); V/JPM/INTC/NFLX filling.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | +32 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | +11 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | +55 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2024-06-21 | 442 | +75 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2020-03-27 | 221 | +221 |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2018-03-16 | 115 | +115 |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2017-08-04 | 83  | +83 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-04 15:28 local · 7,096 chains · 25 names · +555 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: V→2019-07, JPM→2017-09, INTC→2016-11, NFLX→2024-04. 12 names complete (MSFT, GOOGL, GOOG, META, AAPL, AMZN, NVDA, TSLA, PLTR, WMT, AVGO, MU); V/JPM/INTC just started.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | +46 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | +31 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2026-05-29 | 543 | +75 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2024-04-26 | 434 | +84 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| V     | V option-history EOD     | 2016-01-08 … 2019-07-12 | 184 | +184 |
| JPM   | JPM option-history EOD   | 2016-01-08 … 2017-09-22 | 90  | +90 |
| INTC  | INTC option-history EOD  | 2016-01-08 … 2016-11-11 | 45  | +45 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-04 11:10 local · 6,499 chains · 22 names · +450 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: AVGO→2026-02, TSLA→2026-01, NFLX→2022-07, MU→2024-09. 9 names complete; AVGO & TSLA nearing their ends; MU & NFLX filling.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2026-02-20 | 534 | +92 |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2026-01-16 | 524 | +70 |
| MU    | MU option-history EOD    | 2016-01-08 … 2024-09-20 | 455 | +188 |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2022-07-08 | 340 | +100 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-04 07:32 local · 6,096 chains · 22 names · +612 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: TSLA→2024-05, MU→2021-06, NFLX→2020-11, AVGO→2024-07. 9 names complete (MSFT, GOOGL, GOOG, META, AAPL, AMZN, PLTR, NVDA, WMT); MU & NFLX filling fast (+285 / +163).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2026-05-29 | 543 | +13 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2025-03-07 | 466 | +78 |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2024-07-19 | 446 | +73 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| MU    | MU option-history EOD    | 2016-01-08 … 2021-06-18 | 285 | +285 |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2020-11-06 | 253 | +163 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-04 03:23 local · 5,453 chains · 21 names · +562 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: NFLX→2017-07, WMT→2025-12, AVGO→2023-01, TSLA→2022-10. 8 names complete (MSFT, GOOGL, GOOG, META, AAPL, AMZN, PLTR, NVDA); WMT nearly done; NFLX just started.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2026-06-01 | 575 | +70 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2025-12-12 | 519 | +222 |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2025-03-07 | 383 | +45 |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2023-01-27 | 369 | +146 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| NFLX  | NFLX option-history EOD  | 2016-01-08 … 2017-07-07 | 79  | +79 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-03 23:15 local · 4,868 chains · 20 names · +658 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: NVDA→2025-08, WMT→2021-06, AVGO→2020-02, TSLA→2025-03. 7 names complete (MSFT, GOOGL, GOOG, META, AAPL, AMZN, PLTR); WMT & AVGO backfilling fast (+285 / +217 this window).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2026-06-01 | 574 | +56 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2025-08-15 | 501 | +49 |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2025-03-07 | 336 | +51 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-05-29 | 294 | 0 (complete) |
| WMT   | WMT option-history EOD   | 2016-01-08 … 2021-06-25 | 286 | +285 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| AVGO  | AVGO option-history EOD  | 2016-01-08 … 2020-02-28 | 217 | +217 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-03 19:07 local · 4,196 chains · 18 names · +428 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: AMZN→2025-11, PLTR→2026-04, NVDA→2024-03, TSLA→2025 backfill. 5 of Mag7 head complete (MSFT, GOOGL, GOOG, META, AAPL); AMZN & PLTR nearing their ends.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | 0 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2025-11-14 | 512 | +145 |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2025-01-17 | 450 | +72 |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2026-04-24 | 289 | +185 |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2025-03-07 | 285 | +26 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-03 15:30 local · 3,823 chains · 18 names · +437 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: PLTR→2023-05, AMZN→2023-03, NVDA→2023-02, TSLA→2025 backfill. AAPL now complete (5 of Mag7 head done: MSFT, GOOGL, GOOG, META, AAPL). PLTR is the first secondary name (backfilling from its 2020 IPO).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-06-01 | 577 | +97 (complete) |
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2025-01-17 | 390 | +87 |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2023-03-24 | 374 | +45 |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2025-03-07 | 263 | +72 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| PLTR  | PLTR option-history EOD  | 2020-10-16 … 2023-05-19 | 136 | +136 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-03 11:22 local · 3,378 chains · 17 names · +397 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: AAPL→2023-10, AMZN→2022-05, NVDA→2021-05, TSLA→2017-12. GOOGL & GOOG now complete (4 of Mag7 head done: MSFT, GOOGL, GOOG, META).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-06-01 | 575 | +30 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | 0 (complete) |
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-01-16 | 476 | +132 |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2022-05-06 | 329 | +27 |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2025-01-17 | 301 | +104 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| TSLA  | TSLA option-history EOD  | 2016-01-08 … 2025-03-07 | 189 | +104 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-03 07:13 local · 2,970 chains · 17 names · +477 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive. (Entry generated by the 30-min monitor loop; the 4h cron was retired — it kept missing its idle window.)
Frontier: GOOGL→2026-03, AAPL→2021-03, NVDA→2017-05, AMZN→2021-10. GOOG essentially complete (→2026-05-29).

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2026-05-29 | 543 | +97 |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2026-03-11 | 541 | +140 |
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-01-16 | 341 | +141 |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2021-10-29 | 302 | +27 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | 0 (complete) |
| NVDA  | NVDA option-history EOD  | 2016-01-08 … 2025-01-17 | 193 | +72 |
| TSLA  | TSLA option-history EOD  | 2018-01-05 … 2025-03-07 | 85  | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-03 03:32 local · 2,531 chains · 17 names · +265 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: GOOG→2024-11, GOOGL→2023-10, AMZN→2021-04, AAPL→2016 (backfill). META range now complete to 2026-06.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2024-11-15 | 463 | +105 |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2023-10-20 | 407 | +62 |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2021-04-30 | 276 | +19 |
| META  | META option-history EOD  | 2021-07-16 … 2026-06-01 | 250 | +48 (range complete) |
| AAPL  | AAPL option-history EOD  | 2016-01-08 … 2026-01-16 | 214 | +31 |
| NVDA  | NVDA option-history EOD  | 2018-01-05 … 2025-01-17 | 121 | 0 |
| TSLA  | TSLA option-history EOD  | 2018-01-05 … 2025-03-07 | 85  | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |

## 2026-06-02 23:09 local · 2,245 chains · 17 names · +167 in 4h · worker alive
Data title: EOD option-history chain (all-strikes, calls+puts, + open interest). Health: worker 42892 + supervisor(2) + java(2) alive.
Frontier: GOOG→2022-09, GOOGL→2022-07, AMZN→2020-11, META→2026-01.

| Name | Title | Expiration dates (min … max) | Chains | New (4h) |
|---|---|---|---:|---:|
| MSFT  | MSFT option-history EOD  | 2016-01-08 … 2026-06-01 | 576 | 0 (complete) |
| GOOG  | GOOG option-history EOD  | 2016-01-08 … 2022-09-09 | 349 | +45 |
| GOOGL | GOOGL option-history EOD | 2016-01-08 … 2022-07-22 | 342 | +43 |
| AMZN  | AMZN option-history EOD  | 2016-01-08 … 2020-11-13 | 254 | +27 |
| META  | META option-history EOD  | 2021-07-16 … 2026-01-02 | 196 | +52 |
| AAPL  | AAPL option-history EOD  | 2018-01-05 … 2026-01-16 | 183 | 0 |
| NVDA  | NVDA option-history EOD  | 2018-01-05 … 2025-01-17 | 121 | 0 |
| TSLA  | TSLA option-history EOD  | 2018-01-05 … 2025-03-07 | 85  | 0 |
| AMD   | AMD option-history EOD   | 2016-01-08 … 2016-07-22 | 29  | 0 |
| MMM   | MMM option-history EOD   | 2016-01-08 … 2016-05-27 | 21  | 0 |
| ADBE  | ADBE option-history EOD  | 2016-01-15 … 2017-05-19 | 17  | 0 |
| ABBV  | ABBV option-history EOD  | 2016-01-08 … 2016-04-29 | 17  | 0 |
| ACN   | ACN option-history EOD   | 2016-01-08 … 2016-04-08 | 14  | 0 |
| AES   | AES option-history EOD   | 2016-01-15 … 2016-12-16 | 12  | 0 |
| AFL   | AFL option-history EOD   | 2016-01-15 … 2016-11-18 | 11  | 0 |
| AOS   | AOS option-history EOD   | 2016-01-15 … 2016-10-21 | 10  | 0 |
| ABT   | ABT option-history EOD   | 2016-01-15 … 2016-05-20 | 8   | 0 |
