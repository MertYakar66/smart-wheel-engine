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

_Generated 2026-06-03. Total Theta footprint **3.4 GB** local-only (git-ignored, not on GitHub).
This is a static reference (distinct from the rolling 4-hourly log below). Use it tomorrow to see
which Theta datasets/names/dates you already hold so you can match/avoid re-pulling on Bloomberg._

**Shared universe:** unless noted, a dataset covers the **S&P 500 (~500 names)** = *Universe A* (listed
once at the bottom). Small/special universes are spelled out inline.

| # | Dataset (`data_processed/theta/…`) | Type / title | Symbols | Date coverage | Granularity | Files / size | Status |
|---|---|---|---|---|---|---|---|
| 1 | `option_history/` | **Full-depth EOD option chains** (ALL strikes, C+P) + open interest | 20 → 150 (Mag7-first) | exp **2016-01-08 … 2026-06-01** | one (ticker,expiration) = full chain EOD | 4,621 / 1.1 GB | **LIVE — in progress** (see rolling log) |
| 2 | `option_history_banded_backup_2026-06-01/` | **EOD option chains, Δ-BANDED strikes** (not full depth) + OI | **503** (Universe A) | exp **2016-01-15 … 2026-05-22** | one (ticker,expiration) | 51,729 / **2.2 GB** | COMPLETE (prior pull, backed up 06-01) |
| 3 | `chains/` | Full option-chain **snapshots** w/ greeks+IV+quotes+OI | 495 | snaps **2026-04-23, 05-24, 06-01** | point-in-time chain | 1,482 / 32 MB | COMPLETE (3 snapshots) |
| 4 | `iv_surface/` | Per-name **IV surface snapshots** (strike×right×δ×iv×mid×dte) | 502 (Univ A + 8 ETFs) | snaps **04-23, 05-24, 06-01** | point-in-time surface | 558 / 14 MB | COMPLETE |
| 5 | `iv_surface_history/` | **IV surface TIME-SERIES** (daily) | **4 only** (A, AAPL, ABBV, ABNB) | **2026-04-13 … 05-22** | daily, ticker/year/date | 107 / 1.5 MB | PILOT (4 names) |
| 6 | `iv_history/` | **ATM-IV TIME-SERIES** (daily `iv_atm`) | 497 (Universe A) | **2015-01-02 … 2026-03-20** (~11 yr) | daily per name | 497 / 25 MB | COMPLETE — long history |
| 7 | `index_options_chains/` | Index-option full-chain snapshots (greeks+IV+OI) | 8 indices* | snaps **04-23, 05-24, 06-01** | point-in-time | 21 / 964 KB | COMPLETE |
| 8 | `index_options_surfaces/` | Index-option IV surfaces | 8 indices* | snaps **04-23, 05-24, 06-01** | point-in-time | 21 / 1.6 MB | COMPLETE |
| 9 | `option_ohlc/` | **Per-contract daily OHLC + bid/ask** (selected contracts) | 502 | bars **~2026-05-07 … 05-22**; 4 expirations | per-contract daily | 1,507 / 15 MB | COMPLETE — narrow window |
| 10 | `stocks_eod/` | **Equity EOD OHLCV** (underlying stock bars) | 493 (Universe A) | **2024-04-23 … 2026-03-20** (~2 yr) | daily per name | 493 / 12 MB | COMPLETE |
| 11 | `vix_family/` | VIX term-structure index OHLC | VIX, VIX3M, VIX6M, VIX9D | **2023-04-24 … 2026-04-22** | daily | 1 / 52 KB | COMPLETE |
| 12 | `../vol_indices.parquet` (+ `_wide`) | **Vol-index time series** (long + wide) | VIX, VVIX, SKEW, MOVE, GVZ, OVX, VXN, VIX3M/6M/9D | daily (35,062 long / 3,783 wide rows) | daily | 548 KB | COMPLETE |

\* **Index universe (8):** SPX, SPXW, NDX, RUT, DJX, VIX, XSP, AAPL.  
**iv_surface ETF add-ons (8):** SPY, QQQ, DIA, IWM, XLE, XLF, XLK, XLV.  
**Empty / not pulled:** `data_processed/corporate_actions/` is an empty dir (Theta corp-actions never produced output).

### Field schemas (so you know which Bloomberg fields to match)

- **option_history & banded_backup** (22): `symbol, expiration, strike, right, created, last_trade, open, high, low, close, volume, count, bid_size, bid_exchange, bid, bid_condition, ask_size, ask_exchange, ask, ask_condition, open_interest, ticker` — EOD OHLC + quote (bid/ask w/ size, exchange, condition) + volume/count + OI. **No greeks/IV** in these.
- **chains** (23): `symbol, expiration, strike, right, delta, theta, vega, rho, epsilon, lambda, iv, iv_error, underlying_timestamp, underlying_price, bid, ask, bid_size, ask_size, open_interest, mid, ticker, snapshot_date` — greeks (Δ,Θ,V,ρ,ε,λ) + IV + quotes + OI. **Snapshot only** (3 dates).
- **iv_surface / index_options_surfaces** (10): `strike, right, delta, iv, mid, expiration, dte, ticker, snapshot_date`.
- **iv_surface_history** (8): `date, ticker, expiration, dte, strike, right, iv, mid`.
- **iv_history** (4): `iv_atm, ticker, source, date` — single ATM-IV per day, **11-yr history**.
- **option_ohlc** (12): `open, high, low, close, volume, bid, ask, ticker, expiration, strike, right, date`.
- **stocks_eod** (7): `open, high, low, close, volume, ticker, date`.
- **vix_family / vol_indices** (7): `open, high, low, close, symbol, source, date`; `vol_indices_wide` has per-index close columns (`vix_close, vvix_close, skew_close, move_close, gvz_close, ovx_close, vxn_close, vix3m_close, vix6m_close, vix9d_close`).

### Quick read for Bloomberg matching
- **Deepest historical asset:** the **Δ-banded option backup (#2)** — 503 names, 2016→2026, EOD OHLC+quotes+OI but **no greeks/IV**. The **live pull (#1)** is re-doing a 150-name subset at *full* strike depth (same fields).
- **Greeks/IV exist only as recent snapshots** (#3,#4,#7,#8 — three dates Apr–Jun 2026) plus the **11-yr ATM-IV series (#6)** and a **4-name daily IV-surface pilot (#5)**. If you need historical greeks/IV across the universe, Bloomberg is the gap-filler.
- **Equity OHLCV (#10)** only goes back ~2 yr (2024-04→2026-03); **ATM-IV (#6)** and **vol indices (#12)** go back to 2015.

### Universe A — the ~500 S&P names (datasets #2, #6, #10 and the base of #3,#4,#9)
`A, AAPL, ABBV, ABNB, ABT, ACGL, ACN, ADBE, ADI, ADM, ADP, ADSK, AEE, AEP, AES, AFL, AIG, AIZ, AJG, AKAM, ALB, ALGN, ALL, ALLE, AMAT, AMCR, AMD, AME, AMGN, AMP, AMT, AMZN, ANET, AON, AOS, APA, APD, APH, APO, APP, APTV, ARE, ATO, AVB, AVGO, AVY, AWK, AXON, AXP, AZO, BA, BAC, BALL, BAX, BBY, BDX, BEN, BG, BIIB, BK, BKNG, BKR, BLDR, BLK, BMY, BR, BRO, BSX, BX, BXP, C, CAG, CAH, CARR, CAT, CB, CBOE, CBRE, CCI, CCL, CDNS, CDW, CEG, CF, CFG, CHD, CHRW, CHTR, CI, CINF, CL, CLX, CMCSA, CME, CMG, CMI, CMS, CNC, CNP, COF, COIN, COO, COP, COR, COST, CPAY, CPB, CPRT, CPT, CRL, CRM, CRWD, CSCO, CSGP, CSX, CTAS, CTRA, CTSH, CTVA, CVS, CVX, D, DAL, DASH, DAY, DD, DDOG, DE, DECK, DELL, DG, DGX, DHI, DHR, DIS, DLR, DLTR, DOC, DOV, DOW, DPZ, DRI, DTE, DUK, DVA, DVN, DXCM, EA, EBAY, ECL, ED, EFX, EG, EIX, EL, ELV, EME, EMR, EOG, EPAM, EQIX, EQR, EQT, ERIE, ES, ESS, ETN, ETR, EVRG, EW, EXC, EXE, EXPD, EXPE, EXR, F, FANG, FAST, FCX, FDS, FDX, FE, FFIV, FICO, FIS, FISV, FITB, FOX, FOXA, FRT, FSLR, FTNT, FTV, GD, GDDY, GE, GEHC, GEN, GEV, GILD, GIS, GL, GLW, GM, GNRC, GOOG, GOOGL, GPC, GPN, GRMN, GS, GWW, HAL, HAS, HBAN, HCA, HD, HIG, HII, HLT, HOLX, HON, HOOD, HPE, HPQ, HRL, HSIC, HST, HSY, HUBB, HUM, HWM, IBKR, IBM, ICE, IDXX, IEX, IFF, INCY, INTC, INTU, INVH, IP, IPG, IQV, IR, IRM, ISRG, IT, ITW, IVZ, J, JBHT, JBL, JCI, JKHY, JNJ, JPM, K, KDP, KEY, KEYS, KHC, KIM, KKR, KLAC, KMB, KMI, KO, KR, KVUE, L, LDOS, LEN, LH, LHX, LII, LIN, LKQ, LLY, LMT, LNT, LOW, LRCX, LULU, LUV, LVS, LW, LYB, LYV, MA, MAA, MAR, MAS, MCD, MCHP, MCK, MCO, MDLZ, MDT, MET, META, MGM, MHK, MKC, MLM, MMC, MMM, MNST, MO, MOH, MOS, MPC, MPWR, MRK, MRNA, MS, MSCI, MSFT, MSI, MTB, MTCH, MTD, MU, NCLH, NDAQ, NDSN, NEE, NEM, NFLX, NI, NKE, NOC, NOW, NRG, NSC, NTAP, NTRS, NUE, NVDA, NVR, NWS, NWSA, NXPI, O, ODFL, OKE, OMC, ON, ORCL, ORLY, OTIS, OXY, PANW, PAYC, PAYX, PCAR, PCG, PEG, PEP, PFE, PFG, PG, PGR, PH, PHM, PKG, PLD, PLTR, PM, PNC, PNR, PNW, PODD, POOL, PPG, PPL, PRU, PSA, PSKY, PSX, PTC, PWR, PYPL, Q, QCOM, RCL, REG, REGN, RF, RJF, RL, RMD, ROK, ROL, ROP, ROST, RSG, RTX, RVTY, SBAC, SBUX, SCHW, SHW, SJM, SLB, SMCI, SNA, SNPS, SO, SOLS, SOLV, SPG, SPGI, SPY, SRE, STE, STLD, STT, STX, STZ, SW, SWK, SWKS, SYF, SYK, SYY, T, TAP, TDG, TDY, TECH, TEL, TER, TFC, TGT, TJX, TKO, TMO, TMUS, TPL, TPR, TRGP, TRMB, TROW, TRV, TSCO, TSLA, TSN, TT, TTD, TTWO, TXN, TXT, TYL, UAL, UBER, UDR, UHS, ULTA, UNH, UNP, UPS, URI, USB, V, VICI, VLO, VLTO, VMC, VRSK, VRSN, VRTX, VST, VTR, VTRS, VZ, WAB, WAT, WBD, WDAY, WDC, WEC, WELL, WFC, WM, WMB, WMT, WRB, WSM, WST, WTW, WY, WYNN, XEL, XOM, XYL, XYZ, YUM, ZBH, ZBRA, ZTS`
_(503 names in the banded backup; individual datasets differ by a handful as membership changed — e.g. stocks_eod has 493, iv_history 497, chains 495, iv_surface/option_ohlc 502 incl. ETFs.)_

---

<!-- NEW ENTRIES GO DIRECTLY BELOW THIS LINE -->

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
