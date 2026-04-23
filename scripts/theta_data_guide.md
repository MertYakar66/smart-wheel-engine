# Theta Data — what to pull and how

Single source of truth for running the Theta side of the data pipeline.

---

## 0. Prerequisites (one-time)

1. **Start Theta Terminal** (the tray app). It serves a local HTTP API at
   `http://127.0.0.1:25503`.
2. Confirm it's up:
   ```
   python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 25503)); print('UP')"
   ```
   If it errors, open the Terminal and try again.

---

## 1. See what your subscription unlocks

Run this FIRST. It probes every endpoint the engine could consume and
tells you which ones are available on your current tier.

```
python scripts/probe_theta_capabilities.py
```

Output:
- Console: grouped by category (stock / option / index / future), one line per endpoint, `✓ / × / ? / !` classification.
- File: `data_processed/theta_capabilities.json` — a machine-readable audit trail.

Classifications:
| Symbol | Meaning |
|---|---|
| `✓ OK` | 200 + non-empty body — you can pull this |
| `- EMPTY` | 200 + empty body — endpoint works, no test data |
| `× BLOCKED` | 403 — not on your subscription tier |
| `? MISSING` | 404 — endpoint doesn't exist / path wrong |
| `! ERROR` | other error — worth looking at |

Re-run any time you upgrade your Theta plan to see what opened up.

---

## 2. Core pulls (run in this order)

### 2a. Feature-pipeline building blocks

These don't need Theta at all — they use yfinance (free, no API key).

```
python scripts/pull_vol_indices.py --years 5                # 10 vol indices
python scripts/pull_treasury_yields_yf.py --incremental     # risk-free rate curve
python scripts/pull_fundamentals_yf.py --workers 4          # P/E, beta, sector per ticker
python scripts/pull_earnings_yf.py --workers 4              # earnings calendar
```

### 2b. Theta — indices (supersedes yfinance with authoritative CBOE data)

If your tier includes `/v3/index/history/*` (check `probe_theta_capabilities.py`):

```
python scripts/pull_theta_indices_history.py --years 5 --incremental
```

This overwrites the yfinance rows in `vol_indices.parquet` with Theta rows
(Theta rows win on duplicate dates — merge is automatic).

### 2c. Theta — VIX futures curve (UX1–UX8 equivalent)

If futures tier is available:

```
python scripts/pull_theta_vix_futures.py --years 5 --months 8 --incremental
```

Outputs:
- `data_processed/vix_futures.parquet` — long format, one row per (date, expiry)
- `data_processed/vix_futures_wide.parquet` — columns `ux1, ux2, …, ux8` for direct joins

### 2d. Theta — IV surface history per ticker

The highest-value pull for strategy logic. One-day snapshots of the full
chain across strikes and expiries.

```
# All 500 names, last 7 days (incremental). Use daily.
python scripts/pull_theta_iv_surface_history.py --universe sp500 --days 7 --workers 4

# First-time bulk load: last 2 years
python scripts/pull_theta_iv_surface_history.py --universe sp500 --start 2024-04-01 --workers 4
```

Output: `data_processed/theta/iv_surface_history/ticker=<X>/year=<Y>/date=<YYYY-MM-DD>.parquet`

### 2e. Theta — daily options flow

Per-ticker aggregates (put/call volume, OI change, unusual-volume flags).

```
python scripts/pull_theta_options_flow.py --universe sp500 --days 30 --workers 4
```

Output: `data_processed/theta/options_flow/<TICKER>.parquet`

### 2f. Theta — corporate actions

Splits + dividend history per ticker (fills the empty
`data/bloomberg/sp500_corporate_actions.csv`).

```
python scripts/pull_theta_corp_actions.py --universe sp500 --years 10 --workers 4
```

Outputs:
- `data_processed/corporate_actions/splits.parquet`
- `data_processed/corporate_actions/dividends.parquet`
- `data/bloomberg/sp500_dividends_theta.csv` (loader-compatible view)

### 2g. Feature-store backfill (after everything else)

Uses all of the data above. Takes ~15 min for all 500 tickers.

```
python scripts/backfill_features.py --workers 6 --force
```

---

## 3. One command does it all

The orchestrator runs every step in the right order, skipping Theta-dependent
ones automatically when the Terminal is down, and skipping the news step
when no API key is in env.

```
python scripts/pull_all.py               # live refresh
python scripts/pull_all.py --dry-run     # print the plan only
python scripts/pull_all.py --skip theta_corp_actions  # skip specific
python scripts/pull_all.py --only vol treasury        # only these
```

Safe to schedule daily via cron / Task Scheduler.

---

## 4. Heavy / optional pulls

These aren't part of the default refresh. Run manually when you want them.

### 4a. Intraday option tape

Massive data — one ticker × one expiry × 1 day can be 100k+ rows. Use
targeted queries only.

```
# 5 days of AAPL tape, ATM strike only, 35-DTE expiry
python scripts/pull_theta_option_tape.py --tickers AAPL --days 5 --atm-only
```

Output: `data_processed/theta/option_tape/ticker=<X>/date=<Y>/{trades,quotes}.parquet`

Use case: dealer-positioning refinement (classify prints as buy-initiated
vs sell-initiated from NBBO mid). Worth doing on the top ~20 wheel candidates.

---

## 5. Verify everything

Single command, comprehensive health check across all data sources:

```
python scripts/feature_smoke_test.py
```

Sections 15 (data_connectors), 22 (theta_history_pulls), and 26 (theta_outputs)
will flip from SKIP to PASS as Theta data lands on disk.

For just the Theta-related checks:
```
python scripts/feature_smoke_test.py --section theta --verbose
```

---

## 6. What's covered / not covered

| Feature | Source | Status |
|---|---|---|
| Stock OHLCV history | Theta (`stock/history/eod`) | ✓ script ready |
| IV surface (strike × expiry × date) | Theta (`option/history/greeks/iv`) | ✓ `pull_theta_iv_surface_history.py` |
| Options daily volume + OI | Theta (`option/history/volume,open_interest`) | ✓ `pull_theta_options_flow.py` |
| Full chain snapshot | Theta (`option/snapshot/greeks`) | ✓ existing connector |
| VIX family (index) history | Theta / Yahoo fallback | ✓ `pull_theta_indices_history.py` + `pull_vol_indices.py` |
| VIX futures UX1–UX8 | Theta (`future/history/eod`) | ✓ `pull_theta_vix_futures.py` |
| Stock splits / dividends | Theta (`stock/history/{split,dividend}`) | ✓ `pull_theta_corp_actions.py` |
| Intraday option tape (trades + quotes) | Theta (`option/history/{trade,quote}`) | ✓ `pull_theta_option_tape.py` |
| Treasury yields | yfinance fallback | ✓ `pull_treasury_yields_yf.py` |
| Fundamentals snapshot (P/E, beta, sector) | yfinance | ✓ `pull_fundamentals_yf.py` |
| Earnings calendar | yfinance | ✓ `pull_earnings_yf.py` |
| News sentiment | Polygon / Finnhub / Benzinga | ✓ `pull_news_sentiment.py` (needs API key) |
| Short interest / borrow fee | Bloomberg only | ○ no free alternative |
| Analyst revisions stream | Bloomberg only (yfinance has current snapshot) | ○ partial via yfinance |
| Macro calendar (FOMC, CPI) | Bloomberg only (Finnhub has free limited calendar) | ○ could add Finnhub adapter |
| Point-in-time index membership | Bloomberg (`sp500_index_membership.csv` already on disk) | ✓ wired into loader |

---

## 7. Daily production routine

```
# Every business day morning, in order:
python scripts/probe_theta_capabilities.py    # optional: only when Theta has updates
python scripts/pull_all.py                    # runs everything available
python scripts/feature_smoke_test.py --fast   # verify
```

Expect the orchestrator to take ~5-10 min end-to-end (network-bound).
The feature backfill step inside it adds ~15 min when Theta rows changed.
