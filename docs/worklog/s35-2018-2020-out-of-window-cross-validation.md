---
id: S35
title: 2018-2020 out-of-window cross-validation
kind: backtest
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Cross-validate the engine's predictive signal and dollar-
alpha against a **completely different 3-year window** from S22 / S27
/ S32 / S34 (all 2022-2024). Test the engine's behaviour through 2018
chop + Q4 selloff, 2019 strong bull, 2020 COVID crash + V-recovery.
Directly compare to S27 ($100k, same 24-ticker universe, post-fix
engine) — only the time window changes.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
2018-01-02 → 2020-12-31 (756 trading days), 24 SP500 tickers
identical to S22 / S27 / S32 / S34. **$100k starting capital**
(matches S27 for direct comparison). `require_ev_authority=False`.
Three parallel `WheelTracker` instances per friction level
(frictionless / bid_ask / full). Post-IV-PIT-fix engine on
`origin/main`.

**Status.** Done. **Verdict: signal generalizes (ρ = 0.50 in 2020,
DOUBLE S27's 0.22); dollar-alpha does NOT generalize (engine +3.57%
vs SPY ~+45% = −41pp).** Plus the discovery of a 504-day OHLCV
history gate that effectively makes this a 2020-only backtest.

**Findings:**

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | $104,473 | $103,594 | $103,566 |
| Return | +4.47% | +3.59% | **+3.57%** |
| Short puts opened | 19 | 19 | 19 |
| Spearman ρ (2020 only, n=1,946) | 0.5028 | 0.5003 | **0.4970** |
| Mean realized (all puts) | −$148.67 | −$165.78 | **−$169.21** |
| Mean realized (executed puts) | −$1,097.51 | −$1,118.20 | **−$1,125.25** |
| Executed hit-rate | 68.4% | 68.4% | 68.4% |
| Skipped (no BP) | 221 | 221 | 221 |
| SPY same window | — | — | ~+45% |

- **F1 — Signal generalizes.** ρ = 0.50 in 2020 is 2× higher than
  S27's 0.22 in 2022-2024. The engine's ranking quality is both
  scale-invariant (S32 confirms) AND window-invariant (S35 confirms).
- **F2 — Dollar-alpha does NOT generalize.** Engine +3.57% vs SPY
  ~+45% = **−41pp underperformance**. The "+27pp over SPY" headline
  from S22 / S27 is **window-specific**, not a robust engine
  property.
- **F3 — Engine wisely sat out COVID.** During 2020-02-15 →
  2020-05-15, the engine refused 99.8% of 482 candidate rows (took
  only 1 trade — which won +$231). Mean realized of all 482
  candidates was −$369.55 — engine refusals correct in aggregate.
- **F4 — Post-COVID positions performed poorly.** 18 trades taken
  after the COVID window averaged −$1,125 realized despite ρ=0.50.
  The engine's `prob_profit` was over-optimistic in the unusual
  post-pandemic vol environment.
- **F5 — 504-day OHLCV history gate is REAL.** Verified live: at
  as_of=2018-06-15, every ticker is dropped with
  `gate=history, reason="history 115d < required 504d"`. S35
  effectively becomes a 2020-only backtest (252 useful trading days)
  because OHLCV starts 2018-01-02. **New caveat for deployment:
  recently-listed names are unrankable until 2 years of OHLCV
  history accumulates.**
- **F6 — Q3 still beats Q0 but all quartiles negative.** Q3 realized
  mean −$76 vs Q0 −$316 (4× spread). The signal is clean. But every
  quartile has negative mean — the engine ranks correctly relative
  but loses absolutely in this regime.

**AI handoff:**

- The window-specific dollar-alpha is the most consequential finding
  for `docs/PRODUCTION_READINESS.md`'s deployment matrix. The
  "+27pp over SPY" must now be qualified BOTH "$100k-class" AND
  "2022-2024-class".
- 504-day history gate worth surfacing in onboarding docs.
- A multi-year backtest (e.g., 2020-2024 = 5 years post-gate)
  would average out window-specific noise. (S38 closed this.)

Full doc: `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md`.
