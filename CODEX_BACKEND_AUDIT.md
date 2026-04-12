# Smart Wheel Engine — Backend Logic Audit for ChatGPT Codex
# Paste this entire prompt into Codex. It will write and execute Python tests.
# Prerequisite: `python engine_api.py` running on port 8787

---

You are a senior quantitative finance engineer auditing the backend of **Smart Wheel Engine** — an institutional-grade options wheel strategy platform. The Python API is running at `http://localhost:8787`.

Your job: write a single Python script (`audit.py`) and execute it. The script must hit every major endpoint, run mathematical assertions, and produce a structured PASS/FAIL report. Do not just describe what to test — actually write and run the code.

---

## SETUP

```python
import requests
import math
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import json

BASE = "http://localhost:8787"
RESULTS = []

def check(name, condition, expected, actual, severity="MAJOR"):
    status = "PASS" if condition else "FAIL"
    RESULTS.append({"test": name, "status": status, "expected": expected, "actual": actual, "severity": severity})
    symbol = "✓" if condition else "✗"
    print(f"  [{symbol}] {name}")
    if not condition:
        print(f"       Expected: {expected}")
        print(f"       Actual:   {actual}")
        print(f"       Severity: {severity}")
```

---

## DOMAIN 1 — OHLCV DATA INTEGRITY

For each ticker in `["AAPL", "MSFT", "NVDA", "JPM", "KO"]`:
1. GET `/api/chart/ohlcv?ticker={TICKER}&days=30`
2. For every row assert:
   - `low <= open <= high`
   - `low <= close <= high`
   - `low <= high`
   - `volume > 0`
3. Cross-check: GET `/api/analyze/{TICKER}` — the `spotPrice` must match the last `close` in OHLCV within $1.00
4. Report total violations per ticker (expect 0)

Also validate dates:
- Dates are strings in `YYYY-MM-DD` format (no time component)
- No gap between consecutive dates exceeds 5 calendar days

---

## DOMAIN 2 — OPTIONS MATH (verify with exact formulas)

### 2A. Black-Scholes CSP Payoff

GET `/api/payoff?ticker=AAPL&strategy=csp&dte=45`

Write this assertion block:
```python
r = requests.get(f"{BASE}/api/payoff?ticker=AAPL&strategy=csp&dte=45").json()
premium = r["premium"]
strike = r["strike"]
spot = r["spotPrice"]
breakeven = r["breakeven"]
max_profit = r["maxProfit"]

# Test 6: premium > 0
check("CSP premium > 0", premium > 0, "> 0", premium)

# Test 7: strike < spot (OTM put)
check("CSP strike < spot", strike < spot, f"< {spot}", strike)

# Test 8: breakeven math
expected_be = round(strike - premium, 2)
check("CSP breakeven = strike - premium", abs(breakeven - expected_be) < 0.02,
      expected_be, breakeven)

# Test 9: maxProfit = premium * 100
check("CSP maxProfit = premium * 100", abs(max_profit - premium * 100) < 0.01,
      premium * 100, max_profit)

# Test 10-13: payoff curve shape
data = r["data"]
prices = [d["price"] for d in data]
pnls = [d["pnl"] for d in data]

above_strike = [pnls[i] for i, p in enumerate(prices) if p > strike]
below_be = [pnls[i] for i, p in enumerate(prices) if p < breakeven - 1]
at_strike = [pnls[i] for i, p in enumerate(prices) if abs(p - strike) < 0.5]
at_be = [pnls[i] for i, p in enumerate(prices) if abs(p - breakeven) < 1.0]

check("CSP: pnl flat at maxProfit above strike",
      all(abs(v - max_profit) < 1.0 for v in above_strike),
      f"all ≈ {max_profit}", above_strike[:3])

check("CSP: pnl negative below breakeven",
      all(v < 0 for v in below_be),
      "all < 0", below_be[:3])

if at_strike:
    check("CSP: pnl ≈ maxProfit at strike",
          abs(at_strike[0] - max_profit) < 5,
          max_profit, at_strike[0])

if at_be:
    check("CSP: pnl ≈ $0 at breakeven",
          abs(at_be[0]) < 10,
          "≈ 0", at_be[0])
```

### 2B. Covered Call Payoff

GET `/api/payoff?ticker=AAPL&strategy=cc&dte=45`

```python
cc = requests.get(f"{BASE}/api/payoff?ticker=AAPL&strategy=cc&dte=45").json()
cc_data = cc["data"]
cc_prices = [d["price"] for d in cc_data]
cc_pnls = [d["pnl"] for d in cc_data]

# Above strike: PnL should be flat (capped)
above_cc_strike = [cc_pnls[i] for i, p in enumerate(cc_prices) if p > cc["strike"] * 1.02]
if len(above_cc_strike) >= 3:
    variance = max(above_cc_strike) - min(above_cc_strike)
    check("CC: profit capped above strike (flat pnl)",
          variance < 5.0,
          "variance < $5", variance)

# Breakeven should be below spot
check("CC: breakeven < spot",
      cc["breakeven"] < cc["spotPrice"],
      f"< {cc['spotPrice']}", cc["breakeven"])
```

### 2C. Expected Move Bands

GET `/api/expected_move?ticker=AAPL&dte=45`

```python
em = requests.get(f"{BASE}/api/expected_move?ticker=AAPL&dte=45").json()
spot = em["spot"]
iv = em["iv"]  # annualized, as decimal (e.g. 0.28 for 28%)
dte = 45
period_vol = em["period_vol"]

# Test 19: period_vol = IV * sqrt(DTE/365)
expected_pv = iv * math.sqrt(dte / 365)
check("period_vol = IV * sqrt(DTE/365)",
      abs(period_vol - expected_pv) < 0.001,
      round(expected_pv, 6), period_vol)

bands = {b["sigma"]: b for b in em["bands"]}

# Test 16-18: probability checks
check("1σ probability ≈ 68.3%",
      abs(bands[1]["probability"] - 68.27) < 1.0,
      "68.27 ± 1.0", bands[1]["probability"])

check("1.5σ probability ≈ 86.6%",
      abs(bands[1.5]["probability"] - 86.64) < 1.0,
      "86.64 ± 1.0", bands[1.5]["probability"])

check("2σ probability ≈ 95.4%",
      abs(bands[2]["probability"] - 95.45) < 1.0,
      "95.45 ± 1.0", bands[2]["probability"])

# Test 20: band width = 2 * spot * period_vol * sigma
for sigma, band in bands.items():
    expected_width = 2 * spot * period_vol * sigma
    actual_width = band["upper"] - band["lower"]
    check(f"{sigma}σ band width = 2*spot*pv*sigma",
          abs(actual_width - expected_width) / expected_width < 0.01,
          round(expected_width, 2), round(actual_width, 2))
```

### 2D. Strike Recommendations — CSP

GET `/api/strikes?ticker=AAPL&strategy=csp&dte=45`

```python
csp_strikes = requests.get(f"{BASE}/api/strikes?ticker=AAPL&strategy=csp&dte=45").json()
recs = csp_strikes["recommendations"]

spot_s = requests.get(f"{BASE}/api/analyze/AAPL").json()["spotPrice"]

for rec in recs:
    check(f"CSP strike {rec['strike']} < spot",
          rec["strike"] < spot_s, f"< {spot_s}", rec["strike"])
    check(f"CSP delta {rec['delta']} is negative",
          rec["delta"] < 0, "< 0", rec["delta"])
    check(f"CSP probOTM in [50,95]",
          50 <= rec["probabilityOtm"] <= 95,
          "[50, 95]", rec["probabilityOtm"])
    check(f"CSP breakeven = strike - premium",
          abs(rec["breakeven"] - (rec["strike"] - rec["premium"])) < 0.02,
          round(rec["strike"] - rec["premium"], 2), rec["breakeven"])

# Sorted by score descending
scores = [r["score"] for r in recs]
check("CSP recs sorted by score DESC",
      scores == sorted(scores, reverse=True),
      "descending", scores)
```

### 2E. Strike Recommendations — CC

GET `/api/strikes?ticker=AAPL&strategy=cc&dte=45`

```python
cc_strikes = requests.get(f"{BASE}/api/strikes?ticker=AAPL&strategy=cc&dte=45").json()
cc_recs = cc_strikes["recommendations"]

above_spot = sum(1 for r in cc_recs if r["strike"] > spot_s)
check("CC: most strikes above spot (OTM calls)",
      above_spot >= len(cc_recs) * 0.6,
      f"≥ {int(len(cc_recs)*0.6)} above spot", above_spot)

for rec in cc_recs:
    check(f"CC delta {rec['delta']} is positive",
          rec["delta"] > 0, "> 0", rec["delta"])

# Higher delta → higher premium (monotonic)
if len(cc_recs) >= 2:
    delta_prem = [(r["delta"], r["premium"]) for r in cc_recs]
    delta_prem.sort(key=lambda x: x[0])
    premiums_sorted = [p for _, p in delta_prem]
    is_monotonic = all(premiums_sorted[i] <= premiums_sorted[i+1]
                       for i in range(len(premiums_sorted)-1))
    check("CC: higher delta → higher premium",
          is_monotonic, "monotonic", premiums_sorted)
```

### 2F. Candidates Greeks Sanity

GET `/api/candidates?limit=5&min_score=50`

```python
cands = requests.get(f"{BASE}/api/candidates?limit=5&min_score=50").json()

for c in cands:
    ticker = c["ticker"]
    check(f"{ticker}: delta in [-0.50, -0.10]",
          -0.50 <= c["delta"] <= -0.10,
          "[-0.50, -0.10]", c["delta"])
    check(f"{ticker}: premium > 0",
          c["premium"] > 0, "> 0", c["premium"])
    check(f"{ticker}: strike > 0 and < spot",
          0 < c["strike"] < c["spot"],
          f"(0, {c['spot']})", c["strike"])
    check(f"{ticker}: probability > 50",
          c["probability"] > 50, "> 50", c["probability"])
    check(f"{ticker}: expectedPnL > 0",
          c["expectedPnL"] > 0, "> 0", c["expectedPnL"])
```

---

## DOMAIN 3 — INVESTMENT COMMITTEE QUALITY

```python
committee_tickers = ["NVDA", "KO", "AAPL"]

for ticker in committee_tickers:
    print(f"\n  Committee: {ticker}")
    r = requests.get(f"{BASE}/api/committee?ticker={ticker}").json()
    advisors = r["advisors"]
    names = [a["name"] for a in advisors]

    check(f"{ticker}: exactly 4 advisors",
          len(advisors) == 4, 4, len(advisors))
    check(f"{ticker}: all 4 present (Buffett/Munger/Simons/Taleb)",
          set(names) == {"Warren Buffett", "Charlie Munger", "Jim Simons", "Nassim Taleb"},
          "all 4", names)

    for advisor in advisors:
        name = advisor["name"]
        check(f"{ticker}/{name}: ≥2 keyReasons",
              len(advisor.get("keyReasons", [])) >= 2,
              "≥2", len(advisor.get("keyReasons", [])))
        check(f"{ticker}/{name}: ≥1 criticalQuestions",
              len(advisor.get("criticalQuestions", [])) >= 1,
              "≥1", len(advisor.get("criticalQuestions", [])))
        check(f"{ticker}/{name}: ≥1 hiddenRisks",
              len(advisor.get("hiddenRisks", [])) >= 1,
              "≥1", len(advisor.get("hiddenRisks", [])))

        if "Taleb" in name:
            check(f"{ticker}/Taleb: NOT strong_approve",
                  advisor.get("judgment") != "strong_approve",
                  "not strong_approve", advisor.get("judgment"))

    # Strike should be ~8% below spot
    spot_c = requests.get(f"{BASE}/api/analyze/{ticker}").json()["spotPrice"]
    trade_strike = r.get("trade", {}).get("strike", 0)
    pct_below = (spot_c - trade_strike) / spot_c if spot_c else 0
    check(f"{ticker}: strike ~8% below spot (4-15% range)",
          0.04 <= pct_below <= 0.15,
          "4-15% below spot", f"{pct_below:.1%}")

    # Report must be substantive
    report_len = len(r.get("report", ""))
    check(f"{ticker}: report > 500 chars",
          report_len > 500, "> 500", report_len)

# Differentiation: no two advisors share identical keyReasons
for ticker in committee_tickers:
    r = requests.get(f"{BASE}/api/committee?ticker={ticker}").json()
    all_reasons = []
    for a in r["advisors"]:
        all_reasons.extend(a.get("keyReasons", []))
    unique = len(set(all_reasons))
    total = len(all_reasons)
    check(f"{ticker}: all keyReasons unique across advisors",
          unique == total, f"all {total} unique", f"{unique}/{total} unique")

# Empty ticker → 400
r400 = requests.get(f"{BASE}/api/committee?ticker=")
check("Empty ticker → 400", r400.status_code == 400,
      "HTTP 400", r400.status_code, severity="BLOCKER")
```

---

## DOMAIN 4 — STRANGLE TIMING ENGINE

```python
r = requests.get(f"{BASE}/api/strangle?ticker=AAPL").json()

check("strangle score in [0,100]",
      0 <= r["score"] <= 100, "[0,100]", r["score"])
check("strangle phase is valid",
      r["phase"] in ["compression","expansion","post_expansion","trend","unknown"],
      "valid phase", r["phase"])

required_components = ["bollinger", "atr", "rsi", "trend", "range"]
comp_keys = list(r.get("components", {}).keys())
check("exactly 5 components",
      set(comp_keys) == set(required_components),
      required_components, comp_keys)

for comp_name in required_components:
    comp = r["components"][comp_name]
    check(f"{comp_name}: score in [0,100]",
          0 <= comp["score"] <= 100, "[0,100]", comp["score"])
    check(f"{comp_name}: state is string",
          isinstance(comp["state"], str), "str", type(comp["state"]).__name__)

metrics = r.get("metrics", {})
check("rsi_14 in [0,100]",
      0 <= metrics.get("rsi_14", -1) <= 100, "[0,100]", metrics.get("rsi_14"))
check("bb_width_pctl in [0,100]",
      0 <= metrics.get("bb_width_pctl", -1) <= 100, "[0,100]", metrics.get("bb_width_pctl"))

warnings = r.get("warnings", {})
check("warnings.compression is bool",
      isinstance(warnings.get("compression"), bool), "bool", type(warnings.get("compression")).__name__)

# Cross-ticker: scores should differ
scores_3 = {}
for t in ["AAPL", "MSFT", "JPM"]:
    scores_3[t] = requests.get(f"{BASE}/api/strangle?ticker={t}").json()["score"]
check("3 tickers have different strangle scores (not hardcoded)",
      len(set(scores_3.values())) > 1,
      "different values", scores_3)
```

---

## DOMAIN 5 — SYSTEM ROBUSTNESS

```python
import time

# Error handling
r404 = requests.get(f"{BASE}/api/analyze/ZZZZ")
check("ZZZZ → HTTP 404", r404.status_code == 404, 404, r404.status_code, "BLOCKER")
check("ZZZZ → has error field", "error" in r404.json(), "error field present", list(r404.json().keys()))

r_default = requests.get(f"{BASE}/api/analyze/")
check("/api/analyze/ → 200 (AAPL default)", r_default.status_code == 200, 200, r_default.status_code)

r_bad_chart = requests.get(f"{BASE}/api/chart/invalid_type?ticker=AAPL")
check("invalid chart type → 400", r_bad_chart.status_code == 400, 400, r_bad_chart.status_code)

r_butterfly = requests.get(f"{BASE}/api/strikes?ticker=AAPL&strategy=butterfly")
check("butterfly strategy → empty recs (not crash)",
      r_butterfly.status_code == 200, 200, r_butterfly.status_code)

# Boundary conditions
r5 = requests.get(f"{BASE}/api/chart/bollinger?ticker=AAPL&days=5")
data5 = r5.json().get("data", [])
check("days=5 returns exactly 5 rows", len(data5) == 5, 5, len(data5))

r2000 = requests.get(f"{BASE}/api/chart/bollinger?ticker=AAPL&days=2000")
check("days=2000 returns data (no crash)", r2000.status_code == 200, 200, r2000.status_code)

r_dte1 = requests.get(f"{BASE}/api/expected_move?ticker=AAPL&dte=1")
check("dte=1 expected move works", r_dte1.status_code == 200, 200, r_dte1.status_code)

r_dte365 = requests.get(f"{BASE}/api/expected_move?ticker=AAPL&dte=365")
check("dte=365 expected move works", r_dte365.status_code == 200, 200, r_dte365.status_code)

# Response time
times = []
for _ in range(3):
    t0 = time.time()
    requests.get(f"{BASE}/api/analyze/AAPL")
    times.append(time.time() - t0)
avg_ms = sum(times) / len(times) * 1000
check("analyze/AAPL avg response < 3s",
      avg_ms < 3000, "< 3000ms", f"{avg_ms:.0f}ms")
```

---

## FINAL REPORT

At the end of the script, print this:

```python
print("\n" + "="*60)
print("SMART WHEEL ENGINE — BACKEND AUDIT RESULTS")
print("="*60)

domains = {
    "D1 OHLCV Integrity":     [r for r in RESULTS if "OHLCV" in r["test"] or "ohlcv" in r["test"] or any(t in r["test"] for t in ["AAPL:","MSFT:","NVDA:","JPM:","KO:"])],
    "D2 Options Math":        [r for r in RESULTS if any(k in r["test"] for k in ["CSP","CC:","period_vol","sigma","band","delta","premium","breakeven","prob"])],
    "D3 Committee Quality":   [r for r in RESULTS if any(k in r["test"] for k in ["Committee","committee","advisor","Buffett","Taleb","Simons","Munger","keyReason","report","strike ~8"])],
    "D4 Strangle Timing":     [r for r in RESULTS if any(k in r["test"] for k in ["strangle","score in","phase","component","rsi_14","bb_width"])],
    "D5 Robustness":          [r for r in RESULTS if any(k in r["test"] for k in ["ZZZZ","analyze/","chart type","butterfly","days=","dte=","response"])],
}

total_pass = total_fail = 0
for domain, tests in domains.items():
    passed = sum(1 for t in tests if t["status"] == "PASS")
    failed = sum(1 for t in tests if t["status"] == "FAIL")
    total_pass += passed
    total_fail += failed
    rating = "PRODUCTION READY" if failed == 0 else ("NEAR READY" if failed <= 2 else "NOT READY")
    print(f"\n{domain}: {passed}/{passed+failed} [{rating}]")
    for t in tests:
        if t["status"] == "FAIL":
            print(f"  FAIL [{t['severity']}] {t['test']}")
            print(f"       Expected: {t['expected']}")
            print(f"       Actual:   {t['actual']}")

total = total_pass + total_fail
pct = total_pass / total * 100 if total else 0
verdict = "READY" if pct >= 90 else ("CONDITIONAL" if pct >= 75 else "NOT READY")

print(f"\n{'='*60}")
print(f"TOTAL: {total_pass}/{total} ({pct:.1f}%)")
print(f"LAUNCH RECOMMENDATION: {verdict}")
print("="*60)
```

---

## INSTRUCTIONS FOR CODEX

1. Create the file `audit.py` in the repo root with all code above assembled into one script
2. Run: `python audit.py`
3. If any test fails with severity BLOCKER — stop, show the error, and explain the root cause in the engine code
4. For MAJOR failures — show the file and line number where the bug lives
5. For MINOR failures — note them but continue
6. At the end, output the full report table

Do not guess or mock results. All assertions must run against the live API.
