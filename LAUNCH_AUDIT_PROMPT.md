# Smart Wheel Engine — Launch Readiness Audit
# For ChatGPT Codex / Claude / Any AI Code Agent
# Run this against the live system: Python API on port 8787, Dashboard on port 3000

---

You are a senior quantitative finance engineer performing a launch-readiness audit on "Smart Wheel Engine" — an institutional-grade options wheel strategy platform. The system has two running components:

1. **Python Engine API** at `http://localhost:8787` — 24 REST endpoints serving quantitative data
2. **Next.js Terminal Dashboard** at `http://localhost:3000/terminal` — Bloomberg-style trading terminal

Your job is to run a comprehensive, adversarial quality audit across 6 domains. You are not just checking "does it return 200" — you are checking whether the **math is correct**, the **data makes financial sense**, the **UI is usable for a real trader**, and the **system behaves correctly under edge conditions**.

Be brutal. A real quant desk would reject anything with bad math or misleading displays.

---

## DOMAIN 1: OHLCV DATA INTEGRITY (15 tests)

The platform loads 5M+ rows of Bloomberg equity data. A previous audit found the open/high/low/close columns were systematically swapped. Verify the fix is correct.

### Tests

For EACH of these 5 tickers, fetch `/api/chart/ohlcv?ticker={TICKER}&days=30` and validate:

**Tickers:** AAPL, MSFT, NVDA, JPM, KO

For every row in every response:
1. `low <= open <= high` (open is between low and high)
2. `low <= close <= high` (close is between low and high)
3. `low <= high` (low is always the minimum)
4. `volume > 0` (no zero-volume trading days)
5. Dates are sequential business days (no gaps > 4 days, no weekends)

Then cross-validate: fetch `/api/analyze/{TICKER}` for each ticker. The `spotPrice` should match the LAST `close` value from the OHLCV data (within $0.50).

**Report format:**
```
AAPL: 30 rows, 0 constraint violations, spot cross-check: API=$247.64 vs OHLCV last close=$247.64 ✓
MSFT: 30 rows, ...
```

---

## DOMAIN 2: OPTIONS MATH VALIDATION (20 tests)

This is the core of the platform. Verify that Black-Scholes pricing, Greeks, expected moves, and strike recommendations are mathematically correct.

### 2A. Payoff Diagrams

Fetch `/api/payoff?ticker=AAPL&strategy=csp&dte=45`. Validate:

6. `premium > 0` (you can't sell a put for $0)
7. `strike < spotPrice` (CSP strikes should be OTM — below current price)
8. `breakeven = strike - premium` (verify the math exactly)
9. `maxProfit = premium * 100` (max profit is the premium received)
10. In the `data` array: for prices ABOVE strike, all `pnl` values should equal `maxProfit`
11. In the `data` array: for prices BELOW breakeven, all `pnl` values should be NEGATIVE
12. The `pnl` at `price = strike` should equal `premium * 100` (still max profit at strike)
13. The `pnl` at `price = breakeven` should be approximately $0 (within $5)

Now test covered call: `/api/payoff?ticker=AAPL&strategy=cc&dte=45`

14. For CC: profit is CAPPED above the strike (verify pnl flattens)
15. For CC: the breakeven should be below spot price

### 2B. Expected Move Bands

Fetch `/api/expected_move?ticker=AAPL&dte=45`. Validate:

16. 1σ band probability should be approximately 68.3% (within ±1%)
17. 1.5σ probability should be approximately 86.6% (within ±1%)
18. 2σ probability should be approximately 95.4% (within ±1%)
19. `period_vol` should equal `IV * sqrt(DTE/365)` (verify the math)
20. Band width: `upper - lower = 2 * spot * period_vol * sigma` for each band

### 2C. Strike Recommendations

Fetch `/api/strikes?ticker=AAPL&strategy=csp&dte=45`. Validate:

21. All strikes are BELOW spot (OTM puts)
22. All deltas are NEGATIVE (put deltas)
23. `probabilityOtm` is between 50-95% for all recommendations
24. Strikes are sorted by `score` descending
25. `breakeven = strike - premium` for each recommendation

Fetch `/api/strikes?ticker=AAPL&strategy=cc&dte=45`. Validate:

26. Most strikes are ABOVE spot (OTM calls)
27. All deltas are POSITIVE (call deltas)
28. Higher delta → higher premium (monotonic relationship)

### 2D. Greeks Sanity Checks

From `/api/candidates?limit=5&min_score=50`, for each candidate:

29. `delta` should be between -0.50 and -0.10 (OTM put delta range)
30. `premium > 0` (non-zero premium)
31. `strike > 0` and `strike < spot` (where spot comes from the `iv` or wheel score context)
32. `probability > 50` (OTM probability should be >50% for wheel candidates)
33. `expectedPnL > 0` (positive expected value — that's the whole point of the wheel)

---

## DOMAIN 3: INVESTMENT COMMITTEE QUALITY (15 tests)

The platform runs a 4-advisor investment committee (Buffett, Munger, Simons, Taleb) for each trade. Verify the output quality.

### Tests

Run `/api/committee?ticker=NVDA`, `/api/committee?ticker=KO`, `/api/committee?ticker=COIN` (if available, else use any volatile ticker).

For EACH ticker:

34. Exactly 4 advisors present: Buffett, Munger, Simons, Taleb
35. Each advisor has ≥2 `keyReasons` (not generic — should reference the ticker)
36. Each advisor has ≥1 `criticalQuestions` (should be probing, not boilerplate)
37. Each advisor has ≥1 `hiddenRisks`
38. Taleb's judgment is NEVER `strong_approve` (his philosophy prohibits it)
39. The `strike` in the trade is approximately 8% below spot (not a hardcoded value like $480)
40. The `report` field is >500 characters and reads like a professional committee memo

### Differentiation Tests

41. Buffett's analysis should mention business quality, moats, or long-term ownership
42. Munger's analysis should mention biases, inversion, or second-order thinking
43. Simons' analysis should mention probability, statistics, or quantitative metrics
44. Taleb's analysis should mention tail risk, fragility, or fat tails
45. For a HIGH-IV ticker: Simons should be more favorable than Buffett (premium sellers like high IV)
46. For a BLUE-CHIP ticker (KO): Buffett should be more favorable than for a speculative name
47. Across all tickers: no two advisors should have identical `keyReasons` (they must be independent)

### Validation

48. `/api/committee?ticker=` (empty) should return HTTP 400 with error message

---

## DOMAIN 4: STRANGLE TIMING ENGINE (10 tests)

The platform has a proprietary strangle entry timing system based on Bollinger Bands, ATR, RSI, trend, and range analysis.

### Tests

Fetch `/api/strangle?ticker=AAPL`. Validate:

49. `score` is between 0-100
50. `phase` is one of: `compression`, `expansion`, `post_expansion`, `trend`, `unknown`
51. `components` has exactly 5 sub-objects: `bollinger`, `atr`, `rsi`, `trend`, `range`
52. Each component has a `score` (0-100) and a `state` string
53. `metrics.rsi_14` is between 0-100
54. `metrics.bb_width_pctl` is between 0-100 (percentile)
55. `warnings.compression` is boolean
56. `warnings.expansion` is boolean

### Cross-Ticker Comparison

57. Fetch strangle for 3 tickers (AAPL, MSFT, JPM). All should have different scores (not hardcoded)
58. The ticker with the highest IV rank should generally have a higher strangle score (verify correlation)

---

## DOMAIN 5: DASHBOARD FUNCTIONAL TESTING (20 tests)

Open `http://localhost:3000/terminal` in a browser. Test the full user workflow.

### Initial State

59. On load: status bar shows market indices (SPX, NDX, DJI) with real prices
60. Engine badge shows "CONNECTED" (green) after connecting. NOT "PLACEHOLDER"
61. VIX display shows a value >10 (real data), not a stale "15.00"
62. Wheel Scanner shows candidates with REAL strikes (>$0), premiums (>$0), and probabilities (>0%)

### Chart Workflow

Type `AAPL` in the command line:

63. Analysis summary shows: price, sector, IV, RV, VRP, Beta, IV Rank (all non-zero real values)
64. BB chart: renders full Bollinger bands across entire chart width (not blank)
65. Click PRICE: full candlestick/line chart with SMA overlays (not blank, not 1-2 points)
66. Click RSI: full RSI(14) and RSI(2) lines with 70/30 reference lines
67. Click ATR: cyan filled area chart with reasonable ATR values
68. Click TIMING: strangle timing history with MM-DD date format on X-axis (no timestamps)
69. Click PAYOFF: loads CSP payoff diagram + expected move bands + strike recommendations table
70. Click AI MEMO: shows either Ollama-generated memo or data-only fallback with real IV Rank value

### Cross-Ticker Navigation (Critical)

71. Type `MSFT`: header says "MSFT CHART", price updates to ~$381 (NOT $247 AAPL data)
72. Type `JPM`: header says "JPM CHART", sector says "Financials" (NOT "Information Technology")
73. Type `BACK`: returns to main dashboard
74. Type `HELP`: shows command overlay popup (not page navigation)

### Calendar & Events

75. Macro Calendar shows at least one FOMC event with date and days-until countdown
76. Key Indicators section shows Fed Funds Rate, CPI, Unemployment, GDP

### Watchlist

77. Type `WATCH GOOGL`: GOOGL appears in watchlist
78. Watchlist shows price (from engine fallback if no Finnhub key) — not permanently "---"

---

## DOMAIN 6: SYSTEM ROBUSTNESS (10 tests)

### Error Handling

79. `/api/analyze/ZZZZ` (unknown ticker): returns HTTP 404 with descriptive error
80. `/api/analyze/` (no ticker): returns AAPL analysis (graceful default)
81. `/api/chart/invalid_type?ticker=AAPL`: returns HTTP 400 "Unknown chart type"
82. `/api/strikes?ticker=AAPL&strategy=butterfly`: returns empty recommendations (not crash)
83. `/api/committee?ticker=` (empty): returns HTTP 400 "ticker parameter is required"

### Boundary Conditions

84. `/api/chart/bollinger?ticker=AAPL&days=5`: returns exactly 5 data points
85. `/api/chart/bollinger?ticker=AAPL&days=2000`: returns all available history (no crash)
86. `/api/expected_move?ticker=AAPL&dte=1`: works for very short DTE
87. `/api/expected_move?ticker=AAPL&dte=365`: works for very long DTE
88. Rapidly open 5 tickers in the dashboard (AAPL→MSFT→NVDA→JPM→KO in <10s): no crashes, no stale data

### API Throughput

89. Call `/api/analyze/AAPL` three times in sequence. Response times should be <3s each (cached data)
90. Call `/api/committee?ticker=AAPL`. Response time should be <30s (committee is compute-heavy)

---

## SCORING RUBRIC

Rate each domain:

- **PRODUCTION READY** (90%+ pass): Ship it
- **NEAR READY** (75-89%): Minor fixes needed, 1-2 day remediation
- **NOT READY** (50-74%): Significant issues, 1-2 week remediation
- **CRITICAL FAILURES** (<50%): Fundamental problems, major rework needed

### Final Verdict Template

```
DOMAIN 1 — OHLCV Data Integrity:      X/15  [RATING]
DOMAIN 2 — Options Math:              X/20  [RATING]
DOMAIN 3 — Investment Committee:      X/15  [RATING]
DOMAIN 4 — Strangle Timing:           X/10  [RATING]
DOMAIN 5 — Dashboard Functional:      X/20  [RATING]
DOMAIN 6 — System Robustness:         X/10  [RATING]
═══════════════════════════════════════════════════
TOTAL:                                X/90  [OVERALL RATING]

LAUNCH RECOMMENDATION: [READY / NOT READY / CONDITIONAL]
```

For any test that FAILS, include:
1. What was expected (with the mathematical formula if applicable)
2. What was actually observed (exact values)
3. Severity: BLOCKER / MAJOR / MINOR
4. Suggested fix (one sentence)

---

## HOW TO RUN

1. Start the Python engine: `python engine_api.py` (from the repo root)
2. Start the dashboard: `cd dashboard && npm run dev`
3. Open `http://localhost:3000/terminal` in Chrome
4. Run all API tests via curl, browser, or Python requests
5. Run all dashboard tests manually in the browser
6. Compile the report using the template above
