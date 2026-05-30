---
id: S38
title: Multi-window backtest at 100 tickers / $1M / 2020-2024
kind: backtest
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** S34 (Terminal C) found "+11.6pp over SPY at $1M with
100 tickers over 2022-2024." S35 (Terminal B) found −41pp over
2018-2020 at $100k / 24 tickers, showing window-sensitivity. S38
closes the gap by re-running S34's $1M / 100-ticker setup over a
**5-year multi-window (2020-01-02 → 2024-12-31)** that includes
COVID + 2021 bull + 2022 bear + 2023-2024 recovery. Question:
does S34's "engine beats SPY" generalize to a longer window, or
was it 2022-2024-window-favored?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Identical to S34 except the window. Same 100 tickers
(first alphanumeric SP500 names). $1M starting capital.
35-DTE / 25-delta puts → wheel into CC on assignment → hold to
expiry. Three parallel `WheelTracker` instances (frictionless /
bid_ask / full). `require_ev_authority=False`. Post-IV-PIT-fix
engine on `origin/main` (commit `2da76ff`). Driver under
`%TEMP%\s38_backtest\` (not committed; per Sn convention).
~3 hours runtime, 17,360 candidate-evaluations per friction
level (52,080 total ledger rows).

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py` (the §2 ranker route) — daily
re-rank, top-3 trade attempts per day, frictionless +
bid_ask + full-friction parallel trackers.

**Status.** Done. **Verdict: S34's +11.6pp result was
2022-2024-window-favored. Across the full 5-year window the
engine UNDERPERFORMS SPY by ~52pp.** The engine's dollar-alpha
varies from −52pp (5-year multi-window) to +27pp (S27 anomaly)
across the five completed $-scale backtests. There is no single
number that represents the engine's "edge."

**Findings:**

- **(F1 — window-sensitivity confirmed at scale, full
  friction)**. Final NAV $1,331,764 = +33.18% over 2020-2024.
  SPY over the same window returned ~+85% (price + dividends).
  **Engine UNDERPERFORMS SPY by ~52pp.** Headline numbers across
  all backtests:

  | Sn | Capital | Universe | Window | Engine | SPY | Delta |
  |---|---|---|---|---|---|---|
  | S27 | $100k | 24 | 2022-2024 | +51% | +24% | **+27pp** |
  | S32 | $1M | 24 | 2022-2024 | +1.8% | +24% | **−22pp** |
  | S34 | $1M | 100 | 2022-2024 | +35.6% | +24% | **+11.6pp** |
  | S35 | $100k | 24 | 2018-2020 | +3.6% | ~+45% | **−41pp** |
  | **S38** | **$1M** | **100** | **2020-2024** | **+33.2%** | **~+85%** | **−52pp** |

  Five configurations, five different deltas spanning −52pp to
  +27pp. **Dollar-alpha is a multi-dimensional function of
  (capital × universe × window).**

- **(F2 — signal generalizes at scale, ρ = 0.358)**.
  Statistically overwhelming across N=17,192 candidates.
  Slightly higher than S34's 0.327. Per-year: 2020 ρ=0.55,
  2021 ρ=0.21, 2022 ρ=0.37, 2023 ρ=0.31, 2024 ρ=0.31. **Never
  negative.** The ranker genuinely ranks better than random in
  every year, including the COVID crash year (highest ρ).

- **(F3 — realized executed P&L is NEGATIVE)**. The 305 executed
  put trades + 168 CCs over 5 years generated **−$28,647** in
  realized P&L. All NAV growth (+$331,764) came from equity-beta
  on assigned stock positions held through 2023-2024 bull
  (108.6% of NAV gain attributable to equity-beta residual).
  Same shape as S27 (NAV +$51,444 / realized −$3,421) and S35
  (NAV +$3,566 / realized −$48,326). **The engine's put-selection
  alpha is consistently negative on average across all multi-
  window backtests.**

- **(F4 — engine refusal during COVID was correct, again)**.
  847 candidates in 2020-02-15 → 2020-05-15; 19 executed
  (97.8% refusal rate). Mean realized of all 847 candidates =
  **−$254 per trade** if blindly executed (i.e., ~−$215k of
  losses avoided). **This is the engine's strongest defensible
  property** — the refusal mechanism is a real risk control.

- **(F5 — concentration risk amplified at scale)**. Top 5
  contributors to executed realized P&L:
  BKNG +$10,940 / BIIB +$4,046 / AZO +$3,234 / ADSK +$2,534 /
  CHTR +$2,373 (total +$23,127). Net of all 62 traded tickers
  = **−$28,647**. **The other 57 tickers collectively LOST
  ~$52k.** Aggregate realized P&L is negative despite BKNG-
  style outliers. Concentration risk is even more extreme than
  S34's single-window finding.

- **(F6 — quartile monotonicity at extremes, broken in middle)**.
  Q0 realized mean +$28; Q1 +$3; Q2 −$67; Q3 +$206. **Q3 beats
  Q0 by 7.4×**, but Q2 went negative. Signal at the tails is
  clean; middle of the distribution has noise. Same pattern as
  S22/S27/S32.

- **(F7 — capital deployment 22.6%)**. Average across the 5-year
  window. ~77% of $1M sits in cash earning nothing during a
  multi-year bull market. **This is the dominant explanation for
  the −52pp gap.** Even with universe expansion to 100 tickers,
  the engine cannot capture enough of SPY's bull-market return
  because it under-deploys. Strategy-stack additions (S37
  follow-on) become more important: the cash buffer is the
  capacity blocker, not the BP gate.

- **§2 verified.** Same `rank_candidates_by_ev` path as
  S22/S27/S32/S34/S35. No engine code touched. 0 rank failures
  over 52,080 evaluations. The §2 contract holds.

**Realism Check.**

| Aspect | Engine (S38) | External reference | Verdict |
|---|---|---|---|
| 2020-2024 SPY return | ~+85% (price + div) | Public data: SPY 320 → ~600 = +85% incl div | ✓ Reference correct |
| Engine NAV +33.18% over 5y = ~5.9% annualized | Comparable to long-dated treasuries + small premium | Conservative income strategy expectation | ✓ Internally consistent |
| Engine refuses 97.8% of COVID candidates | If blindly executed, mean P&L = −$254/trade | Defensive behavior expected during crisis | ✓ Confirms S35 finding |
| Engine ρ = 0.358 over 5y | Spearman ranks alpha is preserved out-of-sample | Statistical: N=17,192 with p << 1e-100 | ✓ Robust signal |
| Realized executed P&L = −$28,647 | The engine LOSES money on average per executed trade | Counterintuitive: NAV up while realized down | ✓ All NAV growth = equity-beta on assigned stocks |

**Verdict.**

- **The "engine beats SPY" framing is window-specific.** S34's
  +11.6pp at $1M / 100t / 2022-2024 was an outlier. The honest
  multi-window expectation is **engine underperforms SPY by
  20-50pp in bull-dominated 3-5 year windows**.
- **The engine's defensible value proposition is income +
  refusal, NOT dollar-alpha.** +33% over 5 years = ~5.9%
  annualized, which is a reasonable conservative-income return
  with strong crisis refusal. SPY-beating is not.
- **The realized executed P&L is negative across S27, S35, and
  S38.** All NAV growth in those backtests came from equity-
  beta on assigned positions. **The put-selection signal ranks
  correctly (ρ = 0.32–0.36) but loses money on average per
  trade.** F4 fix (B1) becomes more important; if the engine
  could correctly refuse the worst tail-risk candidates, the
  realized P&L might flip positive.
- **Autonomous deployment verdict remains NO.** F4 tail-risk
  gap + D17-live still apply. The multi-window result reinforces
  that forward dollar-alpha is not predictable from any single
  backtest. The deployment matrix in
  `docs/PRODUCTION_READINESS.md` should be revised to
  acknowledge the −52pp / +27pp window dependence.

**AI handoff.**

- **The deployment matrix needs amendment.** Update
  `docs/PRODUCTION_READINESS.md` §5 supervised-$1M case from
  "Conditional ✅" (citing S34's +11.6pp) to "Conditional ⚠ with
  explicit underperformance acknowledgment" citing S38's −52pp
  multi-window result.
- **Marketing / pitch material that cites "engine beats SPY"**
  should be qualified as 2022-2024-window-specific, not a
  forward estimate.
- **B3 capacity blocker is closed structurally but NOT in dollar
  terms.** Universe expansion enables more deployment (22.6% vs
  S32's 10.8%) but does not produce SPY-beating returns.
- **F4 fix (B1) is now more important.** If F4 could correctly
  refuse the worst tail-risk candidates, the −$28k realized P&L
  might flip positive. The Fix A attempt (lookback 5y→2y/3y)
  failed; new approaches need exploration.
- **Strategy-stack expansion (S37 follow-on) is now more
  important** as a way to deploy the idle cash buffer. The
  capacity blocker is the cash sitting in BP, not the BP gate
  itself.
- **Consider repositioning** the engine as a conservative income
  strategy (~5.9% annualized + crisis refusal) rather than an
  alpha strategy. That's a defensible value proposition the
  multi-window data supports.

**Methodology debt.**

- **Single multi-window only.** S38 uses one 5-year window. A
  rolling-5-year-window study (2015-2019, 2016-2020, 2017-2021,
  2018-2022, 2019-2023) would surface whether the −52pp result
  is itself a feature of the 2020-2024 specific window or a
  general property of the engine at scale.
- **Bloomberg-only.** Same S6 dependency as other backtests.
- **No SPY-included tracker.** S38 reports "engine vs SPY" using
  an external ~+85% reference. A direct SPY-in-tracker simulation
  with the same friction model would let us compare dollar-
  for-dollar on identical assumptions.
- **In-sample HMM/POT-GPD parameters.** The forward distribution
  parameters (lookback windows, threshold quantiles) were fit on
  data overlapping the backtest period. A true out-of-sample
  verification would re-fit parameters on pre-2020 data only.

---
