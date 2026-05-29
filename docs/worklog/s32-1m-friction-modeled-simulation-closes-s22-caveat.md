---
id: S32
title: $1M friction-modeled simulation (closes S22 Caveat 3)
kind: backtest
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Close S22 / S27's acknowledged-but-never-measured Caveat 3
(frictionless P&L). Same window, same universe, same engine
(post-IV-PIT-fix), but scaled 10× to **$1M starting capital** with a
three-layer friction overlay (bid/ask + commission + assignment
slippage). Three parallel `WheelTracker` instances per friction level
to isolate friction-dollar cost from execution-divergence cost.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
2022-01-03 → 2024-12-31 (753 trading days), 24 SP500 tickers identical
to S22 / S27. `require_ev_authority=False` (matches S22 / S27 so the
only variable vs S27 is capital + friction). Friction model: half-
spread `max($0.05, 8% of premium)`, commission `$0.65/contract` for
open and close-on-assignment, assignment slippage `10bp × strike × 100`
on equity notional. Throwaway harness at `%TEMP%\s32_backtest\run.py`
(same pattern as S22 / S27).

**Status.** Done. **Verdict: signal preserved, friction drag is
~0.27% NAV (vastly smaller than S22's "2-5% per leg" worst case), but
the headline +27pp-over-SPY narrative INVERTS at $1M — engine
underperforms SPY by −22pp because capital deployment averages only
10.8% of NAV. The S22 / S27 "engine beats SPY" claim was a
$100k-specific artifact, not a scale-invariant property of the engine.**

**Findings:**

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | $1,021,626 | $1,018,601 | $1,018,514 |
| Return | +2.16% | +1.86% | +1.85% |
| Spearman ρ | 0.1940 | 0.1923 | 0.1918 |
| Executed puts | 95 | 95 | 95 |
| Mean realized / executed put | $199 | $174 | $170 |
| Skipped `insufficient_bp` | **0** | **0** | **0** |
| SPY same-window | — | — | ~+24% |

- **F1 — Signal is scale-invariant.** ρ = 0.19 at $1M matches S27's
  ρ = 0.22 at $100k within noise. Capital scale does not affect
  prediction quality.
- **F2 — Friction drag is 0.27% NAV / 14% of frictionless alpha.**
  Closes Caveat 3. S22's "2-5% per leg" worst case was over-
  conservative for liquid SP500 25-delta options today.
- **F3 — Engine UNDERPERFORMS SPY at $1M by −22pp.** Engine +1.85%,
  SPY ~+24%. **The +27pp-over-SPY headline from S22 / S27 was a
  $100k-capital artifact — at $1M the engine cannot deploy enough
  capital to produce competitive percentage returns.**
- **F4 — Capital deployment averages 10.8% of $1M NAV.** Constrained
  by universe size (24 names), one-position-per-name rule, hold-to-
  expiry pattern, and `top_n=10` × `MAX_NEW_PER_DAY=3` daily limits.
  ~$890k of NAV sat idle for the duration.
- **F5 — 2022 bear ρ = 0.37 with mean realized −$18.** Re-surfaces
  the F4 tail-risk gap (from S22 / S27) under friction conditions:
  engine ranks bear-market trades correctly, but tail-risk machinery
  still doesn't widen `prob_profit` enough on the worst losers. PR
  #196 is the regression watch.
- **F6 — Caveat 2 (in-sample HMM / POT-GPD parameters) explicitly
  restated** in this Sn's doc, closing the silent omission flagged
  by PR #197's P9 finding.

**AI handoff:**

- **Highest-leverage next backtest:** expand universe from 24 to 100+
  SP500 names and re-run S32. Hypothesis: average deployment rises to
  ~40-60%, absolute return rises proportionally, the −22pp SPY gap
  closes substantially.
- **Production-pricing implication:** the 24-ticker wheel + 35-DTE +
  25-delta + hold-to-expiry strategy is a **$100k-class strategy**.
  At $1M without structural capacity changes (larger universe,
  multi-contract per name, or strategy stack), it underperforms a
  passive SPY hold.

Full doc: `docs/ENGINE_BACKTEST_S32_FRICTION.md`.
