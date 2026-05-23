# Smart Wheel Engine - Data Collection Report

**Report Date:** 2026-03-22
**Project:** Smart Wheel Trading System
**Status:** Data Collection Phase (In Progress)

---

## Executive Summary

The Smart Wheel Engine data collection effort has made significant progress with **core price and options data collected**, but several secondary datasets require additional Bloomberg terminal time to complete. The system architecture, extraction scripts, and import pipelines are fully built and ready to ingest data as it becomes available.

---

## 1. Data Successfully Collected

### 1.1 Bloomberg Terminal Exports (~250 MB, ~22 files)

Based on the user's lab session, the following datasets were pulled from Bloomberg:

| Dataset | File | Status | Size | Notes |
|---------|------|--------|------|-------|
| **OHLCV Price History** | `sp500_ohlcv.csv` | ✅ Collected | ~150 MB | Full S&P 500, 2018-2026 |
| **Options Flow** | `sp500_options_flow.csv` | ✅ Collected | ~40 MB | Call/put volume, OI |
| **Earnings History** | `sp500_earnings.csv` | ✅ Collected | ~5 MB | EPS, estimates, surprises |
| **Dividends** | `sp500_dividends.csv` | ✅ Collected | ~3 MB | Ex-dates, amounts |
| **Fundamentals** | `sp500_fundamentals.csv` | ✅ Collected | ~10 MB | Market cap, P/E, sectors |
| **Treasury Yields** | `treasury_yields.csv` | ✅ Collected | ~1 MB | 3M, 2Y, 10Y rates |

### 1.2 Reference Data

| Dataset | File | Status | Records |
|---------|------|--------|---------|
| **S&P 500 Constituents** | `sp500_constituents_current.csv` | ✅ Ready | 503 tickers |

### 1.3 Sample Data (Demo/Testing)

Located in `data_raw/`:

| Type | Files | Records | Purpose |
|------|-------|---------|---------|
| **OHLCV (yfinance)** | 5 CSVs | ~8,500 rows | Demo: MMM, AOS, ABT, ABBV, ACN |
| **Options Chains (yfinance)** | 5 CSVs | ~3,500 contracts | Demo snapshots |

---

## 2. Data Collection Failed / Empty

These files came back empty or malformed and need to be re-pulled:

| Dataset | File | Size | Issue | Action Required |
|---------|------|------|-------|-----------------|
| **Corporate Actions** | `sp500_corporate_actions.csv` | 2 bytes | Empty/failed | Re-pull with splits/spinoff formulas |
| **IV History** | `sp500_iv_history.csv` | 20 bytes | Empty/failed | Re-pull ATM IV time series |
| **IV History (dupe)** | `sp500_iv_history (1).csv` | 2 bytes | Empty duplicate | Delete |

### How to Fix: Corporate Actions
```excel
=BDP(A1&" Equity","EQY_SPLIT_DT")
=BDP(A1&" Equity","EQY_SPLIT_RATIO")
```
Pull for all tickers, save as `sp500_corporate_actions.csv`.

### How to Fix: IV History
```excel
=BDH("{TICKER} US Equity","30DAY_IMPVOL_100.0%MNY_DF,60DAY_IMPVOL_100.0%MNY_DF,20DAY_HV,60DAY_HV","20190101","20260320","Dir=V")
```
Pull for all tickers in separate sheets, combine into `sp500_iv_history.csv`.

---

## 3. Data Skipped / Deferred

These datasets are **not critical for MVP** and can be added later:

| Dataset | Priority | Why Skipped | Can Compute From |
|---------|----------|-------------|------------------|
| **Historical Returns** | Low | Compute from OHLCV | OHLCV close prices |
| **Realized Volatility** | Low | Compute from OHLCV | OHLCV high/low/close |
| **Correlations/Beta** | Low | Compute from OHLCV | Returns time series |
| **Factor Exposures** | Low | Compute from fundamentals | Fundamentals + returns |
| **News Sentiment** | Medium | Limited availability | N/A - optional enhancement |
| **ETF Flows** | Low | Not critical for wheel | N/A - optional |
| **Intraday 5-min** | Low | Storage intensive (~200GB) | N/A - future phase |

---

## 4. Data Requiring Cleanup

| File | Issue | Fix |
|------|-------|-----|
| `sp500_short_interest.csv.xlsx` | Wrong extension | Rename to `.xlsx` OR re-save as CSV |

---

## 5. Pipeline Readiness

### Import Scripts Available

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/pull_ohlcv.py` | Pull OHLCV via xbbg | ✅ Ready |
| `scripts/pull_historical_fundamentals.py` | Pull quarterly fundamentals | ✅ Ready |
| `scripts/pull_short_interest.py` | Pull short interest data | ✅ Ready |
| `scripts/pull_options_greeks.py` | Pull options IV surface | ✅ Ready |
| `scripts/pull_liquidity.py` | Pull bid/ask spreads | ✅ Ready |
| `data/bloomberg_import.py` | Load CSVs into system | ✅ Ready |
| `data/bloomberg_loader.py` | Data validation & transform | ✅ Ready |
| `scripts/process_bloomberg_exports.py` | Clean Excel exports | ✅ Ready |

### VBA Macros for Batch Export

| File | Purpose |
|------|---------|
| `scripts/bloomberg_excel_extractor.bas` | Extract data from Bloomberg Add-In |
| `scripts/export_sheets_to_csv.vba` | Export all sheets as CSV |

---

## 6. Data Inventory Summary

```
COLLECTED (Ready to Use)
========================
✅ OHLCV Price History        ~150 MB    Core pricing data
✅ Options Flow               ~40 MB     Volume, OI, put/call ratios
✅ Earnings History           ~5 MB      EPS, surprises, dates
✅ Dividends                  ~3 MB      Ex-dates, amounts, frequency
✅ Fundamentals               ~10 MB     Market cap, ratios, sectors
✅ Treasury Yields            ~1 MB      Risk-free rates
✅ S&P 500 Constituents       ~54 KB     Universe definition
                              --------
                              ~210 MB


FAILED (Need Re-Pull)
=====================
❌ Corporate Actions          Empty      Split dates & ratios
❌ IV History                 Empty      ATM IV time series


SKIPPED (Can Compute)
=====================
⏭️ Returns                    Compute from OHLCV
⏭️ Realized Vol               Compute from OHLCV
⏭️ Correlations               Compute from returns


DEFERRED (Future Phase)
=======================
📅 Short Interest             Needs file rename
📅 News Sentiment             Optional enhancement
📅 Intraday 5-min             Phase 2 (~200GB)
```

---

## 7. Next Steps

### Immediate (Before Next Lab Session)
1. **Delete empty files** - Remove the 3 failed/empty CSVs
2. **Fix short interest** - Rename `.csv.xlsx` to proper extension

### Next Lab Session (Priority Order)
1. **Corporate Actions** - Pull split dates/ratios for all tickers
2. **IV History** - Pull ATM IV time series (critical for IV rank)

### After Data Complete
1. Run `scripts/process_bloomberg_exports.py` to validate
2. Run `data/bloomberg_import.py` to load into system
3. Begin backtesting with `src/backtest/wheel_backtest.py`

---

## 8. Data Quality Notes

### Validation Checklist
- [ ] OHLCV: High ≥ Low, Volume ≥ 0, no gaps > 5 days
- [ ] Options: IV in valid range (0.01 to 3.0), DTE > 0
- [ ] Earnings: Dates are market dates, EPS is numeric
- [ ] Dividends: Ex-dates are valid, amounts > 0

### Known Data Quirks
- Bloomberg reports IV as percentage (35.42 = 35.42%), loader normalizes to decimal
- Some tickers may have different suffixes (GOOG vs GOOGL)
- Treasury yields are in percentage points, not decimal

---

## 9. Technical Specifications

### Target Data Ranges
- **Start Date:** 2018-01-01 (8+ years history)
- **End Date:** 2026-03-20 (current)
- **Universe:** S&P 500 (503 tickers)
- **Frequency:** Daily for most, Quarterly for fundamentals

### Storage Format
- Raw: CSV (from Bloomberg)
- Processed: Parquet (partitioned by ticker/date)
- Features: Parquet (in `data/features/`)

---

*Report generated by Smart Wheel Engine data collection system*
