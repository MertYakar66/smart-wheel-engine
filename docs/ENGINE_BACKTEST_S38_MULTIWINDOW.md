# Engine backtest — S38: multi-window at 100 tickers + $1M (2026-05-26)

**Question:** *Does S34's "+11.6pp over SPY at $1M with 100 tickers"
result hold over a longer multi-window run? Or is it 2022-2024-window-
specific (as S35 suggested)?*

**Headline answer:** **It's window-specific.** Over the full
**2020-2024 (5-year)** window at $1M with the same 100-ticker
universe, the engine returns **+33.18% (full friction)** vs SPY's
**~+85%** = **~52pp UNDERPERFORMANCE.** S34's +11.6pp was
2022-2024-window-favored. The engine's value at scale is
**conservative income generation with strong refusal in crises**,
not SPY-beating dollar alpha.

**Setup:** identical to S34 except the time window:
- 2020-01-02 → 2024-12-31 (1,258 trading days)
- 100 first-alphanumeric SP500 tickers (same as S34)
- $1M starting capital
- 35-DTE / 25-delta short puts, wheel into CC, hold to expiry
- `require_ev_authority=False`
- Three parallel `WheelTracker` instances per friction level
- Post-IV-PIT-fix engine on `origin/main` (commit `2da76ff`)

---

## Per-friction-level results

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | $1,348,704 | $1,332,071 | **$1,331,764** |
| Return | +34.87% | +33.21% | **+33.18%** |
| Short puts opened | 305 | 305 | 305 |
| CCs opened | 168 | 168 | 168 |
| Put assignments | 69 | 69 | 69 |
| Hit-rate (executed) | 77.0% | 77.0% | 77.0% |
| Spearman ρ (N=17,192) | 0.3615 | 0.3594 | **0.3576** |
| Mean realized / executed put | $-47 | $-84 | **$-91** |
| Skipped insufficient_bp | **0** | **0** | **0** |
| Capital deployment (avg) | — | — | **22.6% of $1M** |
| Closed positions | 279 | 279 | 279 |

Same execution count across all 3 levels — BP never binding even over
5 years.

---

## Engine vs SPY — engine UNDERPERFORMS by ~52pp over 5 years

| | Engine (full friction) | SPY buy-and-hold |
|---|---|---|
| 2020-01-02 → 2024-12-31 return | **+33.18%** | ~+85% (price + dividends) |
| Final NAV on $1M start | $1,331,764 | ~$1,850,000 |
| **Delta** | **−52pp** | — |

**This is the most informative result of S38.** S34's +11.6pp result
at the same universe / capital / strategy was **window-favored**.
Across a longer 5-year window that includes:

- COVID crash + V-recovery (2020)
- Strong 2021 bull
- 2022 bear
- 2023-2024 recovery + bull

…the engine's wheel strategy generates only +33% while a passive SPY
hold delivers ~+85%. The gap is dominated by the engine's
**partial deployment** (22.6% average) — ~77% of NAV sits in cash
earning nothing during a multi-year bull market.

### Comparison across all backtests (full friction, post-fix engine)

| Sn | Capital | Universe | Window | Engine | SPY | Engine vs SPY | Driver |
|---|---|---|---|---|---|---|---|
| S27 | $100k | 24 | 2022-2024 | +51% | +24% | **+27pp** | BP-saturated; engine accidentally over-deployed |
| S32 | $1M | 24 | 2022-2024 | +1.8% | +24% | **−22pp** | Capacity-constrained; under-deployed |
| **S34** | **$1M** | **100** | **2022-2024** | **+35.6%** | **+24%** | **+11.6pp** | **Universe expansion + window-favored** |
| **S35** | **$100k** | **24** | **2018-2020** | **+3.6%** | **~+45%** | **−41pp** | **Window-adverse (504-day history gate)** |
| **S38** | **$1M** | **100** | **2020-2024** | **+33.2%** | **~+85%** | **−52pp** | **Long multi-window — capacity gap dominates over 5 years** |

**Five configurations, five different engine-vs-SPY deltas spanning
−52pp to +27pp.** The dollar-alpha is a **multi-dimensional function
of (capital × universe × window)**. There is no single number that
represents the engine's "edge."

---

## Per-year breakdown (full friction, puts only)

| Year | n | ρ | Mean realized | Executed | Notes |
|---|---|---|---|---|---|
| 2020 | 3,467 | **0.548** | −$16 | 78 | Engine wisely sat out COVID; ρ very high |
| 2021 | 3,410 | 0.211 | **+$149** | 63 | Bull market; lower ρ (less to discriminate) |
| 2022 | 3,390 | 0.370 | −$118 | 47 | Bear year; engine took losses despite ρ=0.37 |
| 2023 | 3,391 | 0.309 | +$93 | 71 | Recovery |
| 2024 | 3,534 | 0.312 | +$102 | 46 | Bull continuation |

**ρ ranges 0.21–0.55 across years; never negative.** The engine
genuinely ranks better than random in every year measured. The
variation reflects market predictability, not engine quality.

**But aggregate mean realized is NEGATIVE.** 2020 and 2022 bear years
contributed enough losses to drag the full-window realized P&L below
zero (see Alpha decomposition below).

---

## Alpha decomposition — same shape as S27/S34

**Full friction NAV gain: +$331,764**

| Source | Dollar contribution | Share |
|---|---|---|
| **Realized P&L from 305 executed put trades + 168 CCs** | **−$28,647 (NEGATIVE!)** | -8.6% |
| Equity-beta residual (open positions + assigned stock appreciation) | +$360,411 | **108.6%** |

**The engine LOST $28,647 on its 305 executed put trades over 5
years.** All NAV growth (+$331,764) came from **stock appreciation
on assigned positions** held through the 2023-2024 bull.

Same pattern as S27 (NAV +$51,444 with realized P&L −$3,421) and
S35 (NAV +$3,566 with realized P&L −$48,326). **The engine's
put-selection alpha is consistently NEGATIVE on average across
all multi-window backtests.** The wheel strategy's profitability
comes from being long the right stocks via assignments, not from
the put premiums per se.

---

## Concentration risk — BKNG still dominates

Top 5 contributors to executed realized P&L:

| Ticker | Total realized | Trades |
|---|---|---|
| BKNG | +$10,940 | (S38 includes 2020-2021 BKNG runs from $1,500 → $5,000+) |
| BIIB | +$4,046 | |
| AZO | +$3,234 | |
| ADSK | +$2,534 | |
| CHTR | +$2,373 | |
| **Top-5 total** | **+$23,127** | |
| Net total (all 62 traded tickers) | **−$28,647** | |
| Implied negative contributors | **−$51,774** | (the other 57 tickers net out to a $52k loss) |

**The top 5 tickers generated +$23k, but the remaining 57 tickers
collectively LOST $52k.** Concentration risk is even more extreme
than S34 — the engine's "winners" are concentrated in a handful of
high-priced ticker bull runs, while the broader strategy loses on
average.

**ρ is robust to BKNG removal** (consistent with S34 finding). The
ranking signal works; the dollar outcome at scale is concentration-
dependent and regime-dependent.

---

## Quartile means (full friction, puts only) — monotonicity broken in middle

| Quartile | n | EV mean | Realized mean | Hit-rate |
|---|---|---|---|---|
| Q0 (low) | 4,300 | −$92 | **+$28** | 73.9% |
| Q1 | 4,296 | −$4 | +$3 | 76.1% |
| Q2 | 4,298 | $25 | **−$67** | 76.8% |
| Q3 (high) | 4,298 | $184 | **+$206** | 80.2% |

**Q3 beats Q0 by 7.4× in realized mean** ($206 vs $28). Signal at
the extremes is clean. **But Q2 went negative** — middle of the
distribution has noise. This is consistent with S22/S27/S32 patterns:
the top quartile carries the alpha; mid quartiles are noisy.

---

## COVID period — engine wisely sat out (again)

| Metric | Value |
|---|---|
| Candidates in 2020-02-15 → 2020-05-15 | 847 |
| Executed | 19 (97.8% refusal rate) |
| Mean realized of all candidates | **−$254 per trade** if blindly executed |

Confirms S35's COVID finding: **the engine's refusal mechanism is
correct in aggregate**. If it had executed every COVID candidate,
it would have averaged −$254 per trade × 847 = ~−$215k of losses.
Instead it took 19 trades and broke roughly even.

This is **the engine's most defensible property** for any
production deployment: the refusal layer is a real risk control.

---

## Findings

- **F1 — Window-sensitivity confirmed at scale.** S34's +11.6pp at
  $1M / 100t / 2022-2024 does NOT generalize to 2020-2024. Over the
  full 5-year window the engine LOSES to SPY by ~52pp. The "engine
  beats SPY" framing is window-specific even at favorable universe
  and capital settings.
- **F2 — Signal generalizes (ρ = 0.358).** Statistically overwhelming
  across 17,192 candidates. Higher than S34's 0.327 — more
  candidates strengthen the signal. The ranker is real; the dollar
  outcome is not.
- **F3 — Realized executed P&L is NEGATIVE (−$28,647).** Same shape
  as S27 (−$3,421) and S35 (−$48,326). **The engine's put-selection
  alpha is consistently negative on average across multi-window
  backtests.** All NAV growth comes from equity beta on assignments.
- **F4 — Engine refusal during COVID was correct.** 97.8% refusal
  rate; mean realized of refused candidates was −$254 per trade.
  This is the engine's strongest defensible property.
- **F5 — Concentration risk even more extreme than S34.** Top 5
  tickers contributed +$23k; the other 57 tickers lost $52k net.
  Aggregate realized P&L is negative DESPITE BKNG-style outliers.
- **F6 — Quartile monotonicity holds at extremes but broken in
  middle.** Q3 beats Q0 by 7.4×; Q2 is negative. Ranking signal is
  real at the tails.

---

## Implications for `docs/PRODUCTION_READINESS.md`

S38 substantially **weakens the supervised-$1M case** that S34
appeared to support:

**Before S38 — §5 deployment matrix:**

> Supervised $500k–$1M, universe ≥ 100 tickers — **Conditional ✅**
> based on S34's +11.6pp over SPY at $1M / 100t / 2022-2024.

**After S38 — should be revised to:**

> Supervised $500k–$1M, universe ≥ 100 tickers — **Conditional ⚠
> with explicit underperformance acknowledgment.** S34's window
> showed +11.6pp; S38's longer 5-year window shows −52pp. The
> engine produces positive returns (+33% over 5 years) but
> materially underperforms SPY in bull-dominated multi-year
> windows.

The engine is **not a SPY-beating dollar-alpha generator at scale**.
It is a **conservative income generation strategy with strong
crisis refusal**. Real-money deployment should:

1. Acknowledge the −52pp / +27pp / +11.6pp dollar-alpha range
   across configurations.
2. NOT expect to beat SPY in sustained bull markets.
3. Lean on the refusal mechanism as the engine's core value.

The autonomous deployment verdict remains **NO** — F4 tail-risk +
D17-live still apply, and the multi-window result reinforces that
forward dollar alpha is not predictable from any single backtest.

---

## What this validates / invalidates

| Earlier claim | S38 verdict |
|---|---|
| Engine ranks better than random (ρ ≈ 0.22) | **CONFIRMED at ρ = 0.358 over 5y** |
| Engine beats SPY at $1M with 100t universe (S34: +11.6pp) | **REVISED — window-specific. Multi-window shows −52pp** |
| Engine sits out crises wisely (S35 COVID 99.8% refusal) | **CONFIRMED at 97.8% refusal in S38's COVID window** |
| Engine's NAV growth is mostly equity beta (S34: 92%) | **CONFIRMED at 108.6% in S38 (executed P&L is NEGATIVE)** |
| Concentration risk (BKNG-style outliers drive aggregate) | **CONFIRMED and AMPLIFIED — top 5 tickers earn +$23k while net is −$28k** |
| Universe expansion solves the capacity gap | **PARTIALLY — deployment 22.6% (vs S32's 10.8%) but engine still loses to SPY at this scale over multi-window** |
| In-sample HMM/POT-GPD parameters caveat | **STILL APPLIES** |

---

## AI handoff

- **The deployment matrix needs amendment.** S34's +11.6pp result
  was an outlier. The honest multi-window expectation is
  **engine underperforms SPY by 20-50pp in bull-dominated 3-5
  year windows**.
- **Marketing / pitch material that cites the "+11.6pp over SPY"
  from S34** should be qualified as a 2022-2024-window-specific
  result, not a forward estimate.
- **The B3 capacity blocker is closed structurally** (universe
  expansion does enable more deployment) but **NOT closed in the
  sense of producing SPY-beating returns**. Those are different
  questions.
- **F4 fix (B1) becomes even more important** given the negative
  realized P&L. If the engine could correctly refuse the worst
  trades (the F4 case), the realized P&L might flip positive
  instead of −$28k. Currently the executed trades on average
  LOSE money; F4 fix could change that.
- **Strategy-stack and multi-contract** (S37's follow-on
  recommendations) are now more important as ways to deploy the
  cash buffer that currently sits idle.
- **Consider documenting the engine as an income strategy, not
  an alpha strategy.** The +33% over 5 years = ~5.9% annualized
  is comparable to long-dated treasuries + a small premium. That's
  a defensible value proposition. SPY-beating is not.

---

## Method appendix

Harness: `%TEMP%\s38_backtest\run.py` (not committed; same throwaway
pattern as S22/S27/S32/S34/S35). Identical to S34 except:
- `START_DATE = 2020-01-02` / `END_DATE = 2024-12-31`
- `WORK_DIR = /tmp/s38_backtest`

Output: `%TEMP%\s38_backtest\rank_log.csv` (52,080 rows = 17,360 per
friction level × 3 levels). Runtime: ~3 hours wall-clock on the dev
box.
