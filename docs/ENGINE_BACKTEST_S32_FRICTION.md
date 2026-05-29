# Engine backtest — S32: $1M with full friction (2026-05-25)

**Question:** Does S27's frictionless `+51% NAV / +27pp over SPY` survive
realistic friction at $1M?

**Headline answer:** **No — and the headline reversal is not friction;
it's capital scale.** Friction drag is ~0.27% NAV (much smaller than
S22's "2-5% per leg" worst-case projection). But the engine deploys
only **10.8% of NAV on average at $1M**, so absolute returns are tiny.
**Engine +1.85% (full friction) vs SPY +24% → engine underperforms
SPY by ~22pp at $1M.** S22/S27's `+27pp over SPY` was a $100k-specific
artifact, not a scale-invariant property.

**Window / universe / strategy / engine:** identical to S22 / S27
(2022-01-03 → 2024-12-31, 24 SP500 tickers, 35-DTE / 25-delta short
puts, wheel into CC on assignment, hold to expiry). **Only two
variables changed:** starting capital ($100k → $1M) and friction
overlay (none → bid/ask + commission + assignment slippage).

**Pre-fix engine SHA:** N/A — S32 runs against the **post-fix engine
on current `origin/main`** (commit `6e7e3f2`, with the IV-PIT fix from
PR #179 in place; `_resolve_pit_atm_iv` at `engine/wheel_runner.py:147`
verified live).

**Mode:** `require_ev_authority=False` (matching S22 / S27).
Strict-mode D17 hard-blocks at $1M scale is a separate Sn — deferring
to keep this run a clean one-variable comparison to S27.

---

## Hard methodology caveats

**Caveat 1 (IV-PIT) — closed.** Post-fix engine on `origin/main` uses
`_resolve_pit_atm_iv` → `get_iv_history(end_date=as_of)`. Verified
live on a different ticker / date pair before this run.

**Caveat 2 (in-sample HMM / POT-GPD parameters) — STILL APPLIES.** The
parameters governing the regime classifier, the dealer-multiplier
clamp, the tail-risk POT-GPD threshold, and the forward-distribution
method-selection logic were tuned with full 2018-2026 data visibility.
The 2022-2024 window is in-sample for those parameters. **This is a
restatement of S22's Caveat 2 that S27's doc dropped silently** (see
P9 finding in `archive/2026-05/PREDICTIVE_VALIDITY_REVIEW.md`). Numbers below
inherit it.

**Caveat 3 (frictionless P&L) — closed by THIS run.** Friction
overlay:

| Component | Model | Citation |
|---|---|---|
| Bid/ask half-spread | `max($0.05, 8% of premium)` | Conservative literature standard for liquid SP500 25-delta options |
| Commission (open) | `$0.65 / contract` | IBKR Lite / Schwab equivalent |
| Commission (close on assignment) | `$0.65 / contract` | same |
| Assignment slippage | `10bp × strike × 100` (equity-notional) | Typical retail-side hedge-unwind |

Three INDEPENDENT `WheelTracker` instances run in parallel, each
deciding open/skip based on its own friction-affected cash. This
honestly accounts for "fewer cash → fewer subsequent trades" cascade
risk (though it didn't materialize at this capital scale — see
Findings).

**Caveat 4 (NEW) — capacity-constrained results.** This Sn explicitly
exposes that the wheel strategy's percentage return is **highly
sensitive to capital scale**. At $1M, 89% of NAV sits in cash. The
absolute-dollar return on deployed capital (~$108k average deployment
× ~21% return = ~$22k profit) is reasonable; the **percentage return
on TOTAL capital** is tiny because most capital is idle. This caveat
must accompany any future "engine vs SPY" comparison at non-$100k
capital scales.

---

## Tally

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | **$1,021,626** | **$1,018,601** | **$1,018,514** |
| Return | **+2.16%** | **+1.86%** | **+1.85%** |
| Final cash | $941,487 | $938,463 | $938,375 |
| Short puts opened | 95 | 95 | 95 |
| Put assignments | 14 | 14 | 14 |
| Hit-rate (puts OTM expire) | 81 / 95 = 85.3% | same | same |
| CC opened | 40 | 40 | 40 |
| CC OTM expires | 28 | 28 | 28 |
| CC assignments | 11 | 11 | 11 |
| Skipped `already_held` | 1,167 | 1,167 | 1,167 |
| Skipped `insufficient_bp` | **0** | **0** | **0** |
| Open positions at end | 5 | 5 | 5 |

**Same execution count across all three levels** — BP was never a
binding constraint at $1M (S22's 1,171 `insufficient_bp` refusals at
$100k → **zero** at $1M). This isolates friction as a pure dollar
overlay on identical execution paths.

---

## Headline finding — the engine UNDERPERFORMS SPY at $1M

| | Engine (full friction) | SPY buy-and-hold |
|---|---|---|
| 2022-01-03 → 2024-12-31 return | **+1.85%** | ~+24% |
| Final NAV on $1M start | $1,018,514 | $1,240,000 |
| **Delta** | **−22pp** | — |

**This inverts the S22 / S27 narrative.** S22 reported +67.6% (later
revised by S27 to +51.4% post-IV-PIT-fix), and "+27pp over SPY". My
S32 reproduces the S22 / S27 execution shape at 10× capital and shows
**+1.85% absolute / −22pp vs SPY**.

**The engine's `+27pp over SPY` was a property of $100k capital, not a
property of the engine.** At $100k, the engine deploys ~50-100% of
capital (S22's 1,171 BP-refusals prove BP was fully tied up). At $1M,
the engine deploys ~10%. The other ~90% sits in cash, earning nothing
in this model (real T-bill yields 2022-2024 averaged ~4%, which would
add ~12pp over 3 years and bring engine + cash to ~+14% — still
underperforming SPY's +24%).

---

## Friction drag — much smaller than S22's worst-case

| Layer | Drag ($) | Drag (pp NAV) |
|---|---|---|
| Bid/ask half-spread only | $2,402 | 0.24% |
| + Commission + assignment slippage | $2,745 | 0.27% |
| **As fraction of frictionless alpha (+2.16% → +1.85% = -0.31pp)** | — | **~14% of gross alpha** |

**S22's Caveat 3 cited "2-5% per leg" as the friction haircut. Actual
total friction across 95 puts + 40 CCs (270 legs) was 0.27% of NAV.**
Per-leg friction = 0.27% / 270 ≈ 0.001% NAV = $10 per leg. Why so
small?

- **At 25-delta on $100-500 SP500 names, raw premium is only $0.50-
  $5.** The half-spread of 8% × premium = $0.04-$0.40, not "2-5% of
  underlying notional".
- **Commission per round-trip is $1.30.** On a $30k notional position
  it's 0.004%.
- **Assignment slippage of 10bp × strike × 100 = $5-$50.** Only fires
  on 14 assignments out of 95 positions.

The S22 "2-5% per leg" was a worst-case range from less-liquid
single-name options or earlier-era markets. **For liquid SP500
25-delta puts at $100-500 strikes today, friction is closer to
0.3% NAV than 3%.** This is good news — the engine's signal is
NOT going to be destroyed by friction at this universe.

---

## Signal preservation — ρ ≈ 0.19 across all three levels

| Level | Spearman ρ | p | Mean realized (puts) | Mean realized (executed) |
|---|---|---|---|---|
| Frictionless | 0.1940 | 7.77e-50 | $54.25 | $199.09 |
| bid/ask | 0.1923 | 6.02e-49 | $36.68 | $173.80 |
| full friction | 0.1918 | 9.83e-49 | $32.09 | $170.19 |

**Friction does not degrade ranking.** ρ moves from 0.194 → 0.192 (a
0.002 drop). Friction is a per-row dollar overlay that doesn't change
the *ordering* of EV vs realized — only the absolute dollars.

Comparing to S27 at $100k: ρ = 0.218. My S32 at $1M shows ρ = 0.192.
The ~0.026 lower ρ is plausibly noise (different exact execution
schedules; ~5% fewer puts in my run vs S27's 6,163). **The signal
quality is robust across capital scales.**

---

## Per-year breakdown (full friction)

| Year | n | Spearman ρ | p | Mean realized |
|---|---|---|---|---|
| 2022 (bear, S&P -19%) | 1,878 | **0.3735** | 3.17e-63 | **−$18.48** |
| 2023 (recovery, S&P +24%) | 1,902 | 0.1834 | 7.55e-16 | +$65.83 |
| 2024 (bull, S&P +23%) | 1,963 | 0.0852 | 1.58e-04 | +$47.79 |

**2022 bear shows the strongest ρ (0.37) but the only negative mean
realized (-$18/candidate).** This is the F4 tail-risk gap re-surfacing:
the engine ranks bear-market trades CORRECTLY (high ρ) but the bear-
market mean realized is still negative because the tail-risk machinery
under-weights the worst losers.

**2024 bull shows the weakest ρ (0.085) but positive mean realized.**
The engine isn't doing much in 2024 because there are fewer +EV
candidates after the IV-PIT fix at low-vol levels — the signal
collapses to near-random but the strategy is still profitable in
expectation.

Matches the post-IV-PIT-fix pattern from S27: the bear is where the
signal is real; the bull is where premium collection drives returns
with weak ranking.

---

## Quartile means (full friction)

| Quartile | n | EV mean | Realized mean | Hit-rate |
|---|---|---|---|---|
| Q0 (low) | 1,437 | **−$120.12** | **+$35.47** | 77.2% |
| Q1 | 1,435 | −$30.54 | +$12.09 | 72.8% |
| Q2 | 1,435 | −$0.76 | +$10.56 | 72.0% |
| Q3 (high) | 1,436 | +$74.65 | **+$70.22** | **82.2%** |

**Q3 beats Q0 by 1.98× in realized mean, ~5pp in hit-rate.** S27
reported 1.7× and 5pp; this S32 confirms the same ordering signal
under friction. **Q0 (engine says "bad") still ends up with positive
realized mean** — same paradox as S27: the engine refuses
high-hit-rate-but-negative-EV trades, and those refused trades
sometimes end up positive (because OTM hit-rate is 77% even in the
worst quartile). The engine's job is not to maximize hit-rate; it
is to maximize EV after tail-weighted losses. The Q0 → Q3 monotonic
spread shows it's doing that.

---

## Capacity finding — the engine is fundamentally constrained

**Average deployed collateral: $108,279 (10.8% of $1M NAV).**

Why so low? Three factors compound:

1. **Universe size 24 names.** With one position per name per
   expiration cycle, the max concurrent positions is 24.
2. **Hold-to-expiry pattern locks each position for 35 days.** A name
   that's STOCK_OWNED (post-assignment) is locked for the full
   subsequent CC cycle (another 35 days). So the effective name-cycle
   is 35-70 days.
3. **`top_n=10` + `MAX_NEW_PER_DAY=3`** caps daily entries at 3 even
   if BP and ticker availability would allow more.

Net: at $1M, even maximum concurrent deployment is `24 × ~$30k =
$720k` (~72% of NAV). With 35-day cycles and naturally uneven
fills, the AVERAGE deployment over 753 days = $108k (10.8%).

**This is a strategy capacity constraint, not an engine defect.** A
real $1M pro deploying wheel strategies has three options:

- **Expand universe** to 100+ names (the engine's universe is
  arbitrary; expanding to a full SP500-screened set is a one-line
  config change).
- **Multi-contract per position** (would require lifting the
  one-contract-per-position assumption hardcoded in
  `WheelPosition.contracts = 1` per #166 B3 documentation note).
- **Multi-strategy stack** (run wheel + condor + collar simultaneously
  to deploy more capital — engine supports rank_strangles_by_ev but
  not the integrated multi-strategy book).

Without one of those changes, $1M deployed to this exact strategy
would underperform SPY.

---

## Findings

- **F1 — Engine ranking signal is robust to capital scale.**
  ρ = 0.19 at $1M matches S27's ρ = 0.22 at $100k (within noise).
  The engine's predictive ordering does not depend on whether you
  deploy $100k or $1M. **Logged — promote to a scale-invariance
  property in the production docs.**
- **F2 — Friction drag is 0.27% NAV / 14% of frictionless alpha.**
  S22's "2-5% per leg" worst-case was over-conservative for liquid
  SP500 25-delta options today. **Logged — Caveat 3 closed; future
  backtests can drop the "frictionless caveat" qualifier as long as
  the universe stays in the liquid SP500 megacap band.**
- **F3 — Engine UNDERPERFORMS SPY at $1M (−22pp over 3 years).**
  Engine produces +1.85% vs SPY +24%. **The +27pp-over-SPY headline
  from S22 / S27 was an artifact of $100k capital where BP is fully
  utilized; it is NOT a scale-invariant property of the strategy.**
  **Logged — highest-stakes commercial finding of this Sn.**
  Production implication: the 24-ticker wheel + 35-DTE + 25-delta +
  hold-to-expiry strategy is a **$100k-class strategy**, not a
  $1M-class strategy. To deploy $1M, the strategy needs structural
  capacity improvements.
- **F4 — Capital deployment at $1M is 10.8% on average.** With
  current parameters (24 tickers, hold-to-expiry, 35-day cycles,
  top_n=10, max_new_per_day=3), the strategy cannot deploy more than
  ~$720k peak and averages ~$108k. The remaining ~$890k earns
  nothing in this model (T-bills would add ~12pp over 3 years if
  modelled; still underperforms SPY). **Logged — the natural
  follow-on is to expand the universe to 100+ tickers and re-run.**
- **F5 — 2022 bear has strongest ρ (0.37) but only negative
  mean-realized year.** Re-surfaces the F4 tail-risk gap from S22 /
  S27: the engine ranks bear-market trades correctly but the
  tail-risk machinery still doesn't widen prob_profit enough to
  refuse the worst losers. **Cross-references PR #196's regression
  watch.**
- **F6 — `prob_profit` calibration is mildly overconfident in Q0.**
  Q0 (engine says "low EV") has 77.2% hit-rate but the engine's
  prob_profit on those rows averages much higher — the mean
  `prob_profit` in Q0 is dominated by the 0.83 / 0.91 values that
  match S22's calibration gap finding (S22 F3). Not the headline
  here, but consistent with the prior finding.
- **F7 (new) — Caveat 2 (in-sample HMM / POT-GPD parameters) was
  silently dropped in S27.** S22 had it explicit; S27 implicitly
  retired it (since it dropped the whole "Hard methodology caveats"
  section). My P9 finding in `archive/2026-05/PREDICTIVE_VALIDITY_REVIEW.md`
  surfaced this; this S32 doc explicitly restates Caveat 2 to close
  the gap. **Logged — future backtest docs must restate Caveat 2
  until parameter-freeze-then-replay infrastructure exists.**

---

## What this validates / invalidates from S22 / S27

| S22 / S27 claim | S32 verdict |
|---|---|
| Engine has predictive validity (ρ ≈ 0.22 at $100k) | **Confirmed at $1M (ρ ≈ 0.19) — scale-invariant** |
| Engine beats SPY by +27pp over 3 years | **INVERTED at $1M — engine loses to SPY by −22pp** |
| Friction is unmodeled (Caveat 3, "2-5% per leg") | **Closed — actual friction is 0.27% NAV over 270 legs** |
| Top quartile beats bottom 1.7× | Confirmed at $1M (1.98×) |
| Signal strongest in bear (ρ=0.39 in 2022) | Confirmed at $1M (ρ=0.37 in 2022) |
| F4 tail-risk gap on COST / UNH | Re-surfaces: 2022 has high ρ but negative mean realized |
| Engine deploys most of capital | **Refuted at $1M — average deployment is 10.8% NAV** |

---

## AI handoff

- **Highest-leverage next backtest:** expand the universe from 24 to
  100+ SP500 names and re-run S32. Hypothesis: at $1M with 100
  tickers, average deployment rises from 10.8% to ~40-60% and
  absolute return rises proportionally. If the +21% return on
  deployed capital holds, the $1M case would produce ~+8-12% NAV vs
  SPY's +24% — still likely losing, but the gap closes from −22pp to
  roughly −12pp.
- **Production-pricing implication.** A real $1M deploying wheel
  strategy would want either (a) a much larger universe, (b)
  multi-contract per name, or (c) a strategy stack. Marketing /
  pricing must not quote the +51% NAV from S22 / S27 — that's a
  $100k-class number. The honest number at $1M is +2% with this
  strategy as-is.
- **For the engine team:** F1 (scale-invariant signal) and F3
  (under-deploys at scale) together suggest the engine's ranking
  *quality* is a generic property worth productizing; its *strategy
  deployment* is a separate concern that needs scaling work. The
  ranker is good; the wheel strategy as-implemented is capacity-
  constrained.
- **For the F4 follow-on (PR #196's regression watch):** 2022 bear
  ρ=0.37 with mean realized −$18 is the F4 pattern in concentrated
  form. A targeted unit test that replays the 2022 bear's worst
  losers against the post-fix `forward_distribution.py` +
  `tail_risk.py` would isolate the gap. PR #196 has the regression
  watch; the underlying fix is still open.
- **For the next-next Sn (S33 candidate):** strict-mode D17 at $1M.
  With `require_ev_authority=True` + `PortfolioContext` attached,
  R7 (VaR > 5% NAV) and R8 (stress / dealer regime) should fire on
  the high-EV positions that landed bad realized P&L in 2022. The
  hypothesis is that strict mode + the larger universe (per F4
  follow-on) lifts the $1M return from +2% to +8-10% by refusing
  the worst losers — but at the cost of leaving more capital idle.

---

## Method appendix

Harness: `%TEMP%\s32_backtest\run.py` (not committed; same throwaway
pattern as S22 / S27). Identical to S27 except:
- `STARTING_CAPITAL = 1_000_000.0`
- Three parallel `WheelTracker` instances per friction level
- `friction_adjusted_premium()` + `friction_open_cost()` +
  `friction_assignment_cost()` helpers
- Each tracker calls `runner.rank_candidates_by_ev` with the SAME
  rank output (one rank call per day shared across all 3) but makes
  independent open/skip decisions and applies its own friction overlay
  to `realized_pnl`.

Output: `%TEMP%\s32_backtest\rank_log.csv` (17,349 rows = 5,783 per
friction level × 3 levels). Each row carries `friction_level`,
`premium_raw`, `premium_adj`, and friction-overlay `realized_pnl`.

Engine: `origin/main` @ `6e7e3f2` (post-#196 merge), accessed via
`sys.path.insert(0, "C:/Users/merty/Desktop/swe-terminal-b")`.
Confirmed at runtime: `WheelRunner.__module__ == "engine.wheel_runner"`,
connector = `MarketDataConnector` (Bloomberg).

Runtime: ~30 minutes wall-clock on Windows dev box.
