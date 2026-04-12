# Smart Wheel Engine — Functional Test Prompt
# Paste this into Claude Web/Sonnet to test the running product.
# Prerequisites: python engine_api.py running on port 8787, dashboard npm run dev on port 3000

## Instructions for the tester

You are testing the Smart Wheel Engine — a quantitative options wheel strategy platform.
The system has two components running:
1. **Python API** at `http://localhost:8787` — serves all engine data
2. **Next.js Dashboard** at `http://localhost:3000/terminal` — terminal-style UI

Run the tests below **in order**. For each test, record PASS/FAIL/PARTIAL and any notes.
Be SPECIFIC about what you see — exact numbers, colors, labels.

---

## PHASE 1: Python API Direct Testing (curl or browser)

Test each endpoint by opening in browser or using curl. Verify JSON responses.

### 1.1 Core Endpoints
| # | URL | Expected | Check |
|---|-----|----------|-------|
| 1 | `http://localhost:8787/api/status` | `"status":"connected"`, `universe_size` > 400, `vix` > 0 | |
| 2 | `http://localhost:8787/api/candidates?limit=10&min_score=50` | 10 items, each has `strike` > 0, `premium` > 0, `delta` < 0, `probability` > 50 | |
| 3 | `http://localhost:8787/api/analyze/AAPL` | `spotPrice` > 0, `iv30d` between 10-80, `wheelScore` between 0-100, `sector` not empty | |
| 4 | `http://localhost:8787/api/analyze/ZZZZ` | HTTP 404, JSON has `"error"` field | |
| 5 | `http://localhost:8787/api/regime` | `regime` is one of: NEUTRAL/ELEVATED/HIGH_VOL/LOW_VOL, `vix` > 0 | |
| 6 | `http://localhost:8787/api/vix` | `vix` > 0, has `term_structure` field | |
| 7 | `http://localhost:8787/api/universe` | `count` > 400, `tickers` is array of strings | |
| 8 | `http://localhost:8787/api/fundamentals?ticker=MSFT` | Has `pe_ratio`, `beta`, `market_cap` fields, all > 0 | |
| 9 | `http://localhost:8787/api/calendar?days=90` | Has `events` array, at least one FOMC event | |

### 1.2 Chart Data Endpoints
| # | URL | Expected | Check |
|---|-----|----------|-------|
| 10 | `http://localhost:8787/api/chart/bollinger?ticker=AAPL&days=120` | `data` array of 120 items. Each has `date`, `close`, `open`, `high`, `low`, `upper`, `middle`, `lower`. Verify: `open <= high` and `low <= high` for ALL rows | |
| 11 | `http://localhost:8787/api/chart/rsi?ticker=AAPL&days=120` | 120 items, `rsi_14` between 0-100, `rsi_2` between 0-100 | |
| 12 | `http://localhost:8787/api/chart/atr?ticker=AAPL&days=120` | 120 items, `atr` > 0 for all rows | |
| 13 | `http://localhost:8787/api/chart/ohlcv?ticker=AAPL&days=120` | 120 items with `open`, `high`, `low`, `close`, `volume`, `sma20`, `sma50`. Verify `open <= high` | |
| 14 | `http://localhost:8787/api/chart/strangle?ticker=AAPL&days=120` | `data` array, each with `date` (YYYY-MM-DD format, NO time), `score` 0-100, `phase` | |

### 1.3 Options Analysis Endpoints
| # | URL | Expected | Check |
|---|-----|----------|-------|
| 15 | `http://localhost:8787/api/payoff?ticker=AAPL&strategy=csp&dte=45` | Has `strike`, `premium` > 0, `breakeven`, `data` array with `price` and `pnl` | |
| 16 | `http://localhost:8787/api/expected_move?ticker=AAPL&dte=45` | 3 `bands` (1σ, 1.5σ, 2σ), each with `upper` > spot and `lower` < spot, `period_vol` > 0 | |
| 17 | `http://localhost:8787/api/strikes?ticker=AAPL&strategy=csp&dte=45` | 5 recommendations, each `strike` < spot, `premium` > 0, `delta` < 0, `probabilityOtm` > 50, sorted by `score` descending | |
| 18 | `http://localhost:8787/api/strikes?ticker=AAPL&strategy=cc&dte=45` | 5 recommendations, most strikes > spot (OTM calls), `delta` > 0 | |
| 19 | `http://localhost:8787/api/strangle?ticker=AAPL` | `score` 0-100, `phase` is one of compression/expansion/post_expansion/trend, all 5 component keys present | |
| 20 | `http://localhost:8787/api/iv_history?ticker=AAPL&days=60` | 60 entries with `put_iv`, `call_iv`, `rv_30d` | |

### 1.4 Committee & AI Endpoints
| # | URL | Expected | Check |
|---|-----|----------|-------|
| 21 | `http://localhost:8787/api/committee?ticker=NVDA` | 4 advisors (Buffett, Munger, Simons, Taleb). Each has `keyReasons` (≥2 items), `criticalQuestions` (≥1 item), `hiddenRisks` (≥1 item). Taleb never `strong_approve`. | |
| 22 | `http://localhost:8787/api/committee?ticker=` | HTTP 400 error: "ticker parameter is required" | |
| 23 | `http://localhost:8787/api/memo?ticker=AAPL` | Has `memo` field with text content (either AI-generated or fallback). Has `analysis`, `committee` sub-objects. | |
| 24 | `http://localhost:8787/api/ollama_status` | Returns JSON with model availability info | |

### 1.5 OHLCV Data Integrity (Critical)
Run this validation on any 3 tickers. For EACH row in the response:
- `open` <= `high` (open must never exceed high)
- `low` <= `close` (low must never exceed close)  
- `low` <= `open` (low must never exceed open)
- `low` <= `high` (low is always the minimum)

Test: `http://localhost:8787/api/chart/ohlcv?ticker=MSFT&days=30`
Test: `http://localhost:8787/api/chart/ohlcv?ticker=JPM&days=30`
Test: `http://localhost:8787/api/chart/ohlcv?ticker=NVDA&days=30`

Report: X violations out of Y rows per ticker.

---

## PHASE 2: Dashboard Visual Testing

Open `http://localhost:3000/terminal` in Chrome/Edge. Test each UI component.

### 2.1 Initial Load
| # | Test | Expected | Check |
|---|------|----------|-------|
| 25 | Status bar loads | SPX, NDX, DJI prices visible with % changes | |
| 26 | Engine badge | Shows "CONNECTED" (green) within 15s. Should NOT show "PLACEHOLDER". May show "OFFLINE" (amber) briefly. | |
| 27 | VIX display | Shows real VIX value (>10), NOT "15.00". If engine still connecting, should show "0" or "---", NOT a fake value. | |
| 28 | Market regime | Shows ELEVATED or HIGH_VOL or NEUTRAL — matches the VIX level. NOT "---" after engine connects. | |
| 29 | Wheel scanner | Lists trade candidates with tickers, scores. STRIKE column should show real dollar values (NOT $0). PROB should show real % (NOT 0%). | |
| 30 | Greeks panel | If candidates loaded: IV should be a reasonable percentage (15-50%), NOT 3600%. Delta should be negative (around -0.20 to -0.40). | |

### 2.2 Chart Panel (type AAPL in the command line, press Enter)
| # | Test | Expected | Check |
|---|------|----------|-------|
| 31 | Analysis summary | Shows price (>$200), sector, IV%, RV%, VRP%, Beta, IV Rank (NOT 0), VIX level | |
| 32 | BB chart (default) | Bollinger bands render: red dashed upper, yellow middle, green dashed lower, white price line. Full chart visible with data across entire width. | |
| 33 | PRICE button | Click PRICE: candlestick/line chart with close, SMA20 (yellow dashed), SMA50 (purple dashed), volume bars. Chart should be FULL — not blank or 1-2 data points. | |
| 34 | RSI button | Click RSI: purple RSI(14) line + orange RSI(2) line. 70/30 reference lines. Chart should be FULL — not blank or just a few points. | |
| 35 | ATR button | Click ATR: cyan filled area chart. Values should be visible and reasonable (not blank). | |
| 36 | TIMING button | Strangle timing history. Green filled area. 80 and 60 threshold reference lines. X-axis shows dates in MM-DD format (NOT "2025-09-29 00:00:00"). | |
| 37 | PAYOFF button | CSP payoff curve renders within 10s. Shows breakeven, strike reference lines. Expected move bands below chart. Strike recommendations table below. | |
| 38 | AI MEMO button | Shows either AI-generated memo (if Ollama running) or data-only fallback template. IV RANK should show a real value (e.g., "28%"), NOT "0". | |
| 39 | 1M/2M/6M/1Y range | Click each range button — chart should update with different data density. | |
| 40 | Wheel score badge | Shows "WHEEL XX" badge with color: green (≥75), amber (55-74), red (<55). | |

### 2.3 Commands
| # | Command | Expected | Check |
|---|---------|----------|-------|
| 41 | Type `MSFT` | Opens MSFT chart view with correct MSFT data | |
| 42 | Type `BACK` | Returns to main dashboard view | |
| 43 | Type `HELP` | Shows available commands popup overlay (NOT navigation away) | |
| 44 | Type `ENGINE` | Refreshes engine data | |
| 45 | Type `WATCH TSLA` | Adds TSLA to watchlist. Price should eventually show (from engine fallback if no Finnhub key). | |

### 2.4 Calendar & News
| # | Test | Expected | Check |
|---|------|----------|-------|
| 46 | Macro Calendar — Upcoming | Should show at least one FOMC event from the engine calendar. NOT "No upcoming events". | |
| 47 | Key Indicators | Shows Fed Funds Rate, CPI, Unemployment, GDP. | |
| 48 | News panel | Stories load with tickers, timestamps. No React console errors about duplicate keys. | |
| 49 | News Refresh | Click REFRESH or type REFRESH — should attempt feed ingestion. | |

### 2.5 Cross-Ticker Validation
Open charts for these 3 tickers and verify each shows real data (not "No data available"):

| # | Ticker | Verify | Check |
|---|--------|--------|-------|
| 50 | AAPL | BB chart renders, price ~$248, wheel score ~59 | |
| 51 | MSFT | BB chart renders, price ~$387, wheel score ~74 | |
| 52 | JPM | BB chart renders, price ~$288, wheel score ~83 | |

For each ticker, also check RSI and PAYOFF tabs render correctly.

---

## PHASE 3: Data Cross-Validation

Compare API response values against dashboard display:

| # | Field | API Source | Dashboard Location | Match? |
|---|-------|-----------|-------------------|--------|
| 53 | AAPL price | `/api/analyze/AAPL` → `spotPrice` | Chart panel header price | |
| 54 | AAPL IV | `/api/analyze/AAPL` → `iv30d` | Chart panel IV% | |
| 55 | AAPL wheel score | `/api/analyze/AAPL` → `wheelScore` | WHEEL badge number | |
| 56 | VIX | `/api/vix` → `vix` | Status bar or regime section VIX | |
| 57 | Top candidate score | `/api/candidates` → first ticker score | Wheel scanner first row SCR | |

---

## PHASE 4: Investment Committee Deep Test

For each of these 3 tickers, call `/api/committee?ticker=XXX` and verify:

| # | Check | AAPL | MSFT | JPM |
|---|-------|------|------|-----|
| 58 | Returns 4 advisors (Buffett, Munger, Simons, Taleb) | | | |
| 59 | Each advisor has ≥2 `keyReasons` | | | |
| 60 | Each advisor has ≥1 `criticalQuestions` | | | |
| 61 | Each advisor has ≥1 `hiddenRisks` | | | |
| 62 | Taleb judgment is NOT `strong_approve` | | | |
| 63 | `report` field is a multi-line formatted text (>500 chars) | | | |
| 64 | `strike` is realistic (~8% below spot, NOT $480 hardcoded) | | | |

---

## PHASE 5: Edge Cases

| # | Test | Expected | Check |
|---|------|----------|-------|
| 65 | `/api/analyze/` (empty ticker) | Returns analysis for AAPL (default) — NOT a crash | |
| 66 | `/api/chart/bollinger?ticker=AAPL&days=5` | Returns exactly 5 entries, no crash | |
| 67 | `/api/chart/bollinger?ticker=AAPL&days=2000` | Returns historical data (may be less than 2000 if data starts later) | |
| 68 | `/api/chart/invalid_type?ticker=AAPL` | HTTP 400: "Unknown chart type" | |
| 69 | `/api/strikes?ticker=AAPL&strategy=butterfly` | Returns empty array `[]` | |
| 70 | Open 5 different tickers in rapid succession | Dashboard should not crash or show stale data | |

---

## Scoring

- **PASS**: Fully meets expected behavior
- **PARTIAL**: Works but with minor issues (note what)
- **FAIL**: Does not meet expected behavior (note what's wrong)

Report format:
```
Phase 1: XX/24 PASS, X PARTIAL, X FAIL
Phase 2: XX/26 PASS, X PARTIAL, X FAIL
Phase 3: X/5 PASS, X PARTIAL, X FAIL
Phase 4: X/7 PASS, X PARTIAL, X FAIL
Phase 5: X/6 PASS, X PARTIAL, X FAIL
TOTAL: XX/68 PASS
```

For any FAIL or PARTIAL, include:
1. What you expected
2. What you actually observed
3. Exact error message or screenshot description
