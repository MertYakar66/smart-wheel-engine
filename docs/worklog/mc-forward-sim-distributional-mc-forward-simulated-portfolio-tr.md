---
id: mc-forward-sim
title: Distributional MC forward simulated-portfolio track
kind: feature
status: complete
terminal: builder
pr:
decisions: []
date: 2026-07-05
headline: Wired the dormant Monte-Carlo + copula machinery into a live distributional forward simulated-portfolio track — equity fan, terminal/drawdown distributions, and a correlation-to-1 tail — reconciled against the deterministic backtest NAV. Reporting-only, off the §2 decision path.
surface:
  - engine/sim_portfolio.py
  - scripts/run_forward_sim.py
  - tests/test_sim_portfolio.py
---

## Goal

We have one deterministic backtest path per window. The engine already ships
tested Monte-Carlo (`engine/monte_carlo.BlockBootstrap`), copula
(`engine/portfolio_copula.portfolio_cvar_copula`), and performance-metric
machinery — all **off the decision path and unwired**. Turn that into a live,
persisted **distributional** view of a simulated wheel book: the full outcome
distribution (equity fan, terminal-return tail, drawdown, correlation-to-1
stress) instead of a single path. ~70% wiring + validation, not a from-scratch
build.

Hard constraints: never bypass `EVEngine.evaluate`; never modify the
decision-layer trio (`ev_engine.py` / `wheel_runner.py` /
`candidate_dossier.py`); the copula is a **reporting/stress overlay only** and
must never feed `ev_dollars`, a verdict, or the R7/R8 gate thresholds; write
all simulated output to a namespace separate from real IBKR data.

## What we tried

1. **Confirmed the surface before building.** Read the four MC/copula/perf
   modules and `backtests/regression/_common.run_backtest` — signatures matched
   the starting map (`BlockBootstrap.simulate`, `portfolio_cvar_copula`,
   `WheelTracker.equity_curve` rows `{date, portfolio_value, cash,
   num_positions}`). Provider logged: **`MarketDataConnector`** (bloomberg).
   Smoke test healthy at an in-window `as_of` (5 rows, zero nulls).
2. **New standalone module** `engine/sim_portfolio.py` (NOT the trio): derive a
   strategy-return series from the tracker equity curve → `BlockBootstrap` →
   p5/p25/p50/p75/p95 equity fan + terminal-return + drawdown distributions +
   `cvar_5`/`var_5`/`prob_loss`/`prob_severe_loss`; a `reconcile_with_backtest`
   check; and a `portfolio_correlation_tail` copula overlay (empirical +
   correlation-to-1 stress spoke).
3. **Driver** `scripts/run_forward_sim.py` reads a persisted backtest dir (or
   runs one with `--run`), builds the report, and persists to the SIM namespace.

## What worked

- **Reconciliation math.** Setting the MC horizon `n_days == len(returns)`
  makes the bootstrap median terminal track the realized final NAV
  (E[log-terminal] over the empirical return distribution equals the realized
  log-terminal at equal step counts). Primary 2024 run reconciled at a **1.92%
  median gap**, realized path inside the p5–p95 band.
- **Correlation-to-1 as the headline.** At near-perfect correlation the
  t-vs-Gaussian `tail_amplification` *collapses toward 1* (Gaussian already
  models a fully-coordinated crash), so the honest corr-to-1 number is the
  **CVaR-magnitude multiple** vs the book's realized correlation, not the
  t/Gaussian ratio. Added `stress_vs_empirical.t_cvar_multiple`.
- **Determinism** under seed 42; full report is JSON-safe; every field is
  labelled `model` (MC/copula) vs `engine-measured`, and the copula block is
  explicitly `reporting_only=True` / `feeds_ev=False`.

## What didn't

- **Bare `pytest tests/` runs the multi-hour backtest_regression tests.** The
  `backtest_regression` marker does NOT auto-skip; only `-m "not
  backtest_regression"` deselects it (that is exactly what CI's Test Suite job
  does, `ci.yml:131`). First full-suite run stalled entering S34; killed the
  pytest process and re-ran with the marker exclusion.
- **Suspiciously low `mean|corr|=0.18`** looked like a date-misalignment bug.
  It was not — direct check confirmed AAPL/MSFT daily corr **0.47** over 2024;
  the 0.18 aggregate is genuine 2024 sector dispersion (UNH crash, energy vs
  tech). `get_ohlcv` returns a `DatetimeIndex` (not a `date` column), so
  `pd.DataFrame(series)` was already aligning on dates correctly. Hardened the
  driver to inner-join returns (`dropna(how="any")`) so the copula can never
  see tail-misaligned series regardless.

## How we fixed it

Shipped three files, trio untouched:

- `engine/sim_portfolio.py` — pure library (arrays in, JSON-safe report out):
  `strategy_returns_from_equity_curve`, `monte_carlo_bands`,
  `reconcile_with_backtest`, `portfolio_correlation_tail`, `build_sim_report`.
  Imports only `monte_carlo` + `portfolio_copula` (asserted by a §2 AST test).
- `scripts/run_forward_sim.py` — orchestrator: reads `tracker_state.json` +
  `rank_log.csv`, derives per-name notional weights (median rank-log strike ×
  100 × position count — closed records lack `put_strike`), fetches per-name
  returns from the connector, assembles the report, persists
  `mc_forward_sim.json` / `sim_portfolio_history.json` / `mc_equity_bands.json`
  to `$SWE_SIM_DATA_DIR` or the gitignored `data_processed/sim/`.
- `tests/test_sim_portfolio.py` — 16 tests: reconciliation, band monotonicity,
  determinism, short-series guard, corr-to-1 worsening, single-name skip,
  report shape/labels, and the §2 no-trio-import guard.

## Evidence

Provider + smoke (in-window `as_of`, since data frontier is 2026-07-02):

```
PROVIDER_CONNECTOR_CLASS: MarketDataConnector
5-ticker smoke @ 2026-06-02: 5 rows, NULL ev_dollars/iv/premium = 0/0/0
```

Primary forward book — 12 diversified names, full 2024, $200k, friction=full:

```
backtest: final_nav=210,508 (+5.25%)  trades=24  assigns=3  equity_marks=244
run_forward_sim.py --output-dir data_processed/sim/primary_2024:
  engine-measured realized final NAV: $210,508 (+5.25%)
  model MC median terminal return:   +7.27%  (p5 -5.44% / p95 +20.13%)
  RECONCILIATION: in_band=True median_close=True gap=1.87% -> reconciled=True
  copula EMPIRICAL:  t_cvar=0.0183  tail_amplification(t/gauss)=1.045  verdict=negligible_tail_dependence
  copula CORR->1 STRESS: t_cvar worsens x1.85 vs realized corr (mean|corr|=0.18)
```

Interpretation: the book's realized cross-name correlation is low (0.18) so the
per-name-independent EV path's blind spot looks benign — but the copula shows
that if correlations spike to 1 (a crisis), the book's tail CVaR **nearly
doubles (×1.85)**. That coordinated-crash tail is precisely what the EV path
cannot see, and surfacing it is the point of this overlay.

Tests + gates:

```
pytest tests/test_sim_portfolio.py -q  -> 16 passed
ruff check / format --check             -> clean
check_manifest_coverage.py              -> OK (0 uncovered)
git status engine/ev_engine.py engine/wheel_runner.py engine/candidate_dossier.py -> untouched
```

**Phase 4 — out-of-window validation (2022 bear, same 12-name universe):**

```
backtest: final_nav=204,509 (+2.25%)  traded_names=11
  engine-measured realized final NAV: $204,509 (+2.25%)
  model MC median terminal return:   +2.89%  (p5 -15.30% / p95 +24.58%)
  RECONCILIATION: in_band=True median_close=True gap=0.61% -> reconciled=True
  copula EMPIRICAL:  t_cvar=0.0285  tail_amplification(t/gauss)=1.057
  copula CORR->1 STRESS: t_cvar worsens x1.32 vs realized corr (mean|corr|=0.42)
```

Three things this out-of-window run confirms (honest framing: 2022 is
out-of-*window*, not out-of-*parameter* — HMM/POT-GPD/dealer params were tuned
on full history incl. 2022, per **E5**):

1. **Reconciliation holds across regimes** — the MC median tracks the
   deterministic NAV even tighter in 2022 (0.62% gap vs 1.92% in 2024).
2. **The fan reflects the regime** — 2022's left tail is far fatter (p5
   **-15.30%** vs -5.44% in 2024) and the empirical tail CVaR is bigger (0.0285
   vs 0.0183). The distribution is not a fixed shape; it widens in the bear.
3. **The corr-to-1 stress is regime-aware and most useful when the book looks
   safe** — in the 2022 bear, realized cross-name corr is already high (0.42),
   so forcing it to 1 only worsens the tail ×1.32; in calmer 2024, realized
   corr is low (0.18), so the corr-to-1 jump is a larger ×1.85. The overlay's
   incremental warning is biggest precisely when a book *looks* diversified but
   has not been crisis-tested — exactly the blind spot the per-name-independent
   EV path has.


## Caveats that still stand (preserved, not papered over)

- **E1/E3** ~92% of backtest NAV gain was equity-beta on assigned stock (not
  put alpha); P&L historically single-name-dominated. The aggregate fan hides
  single-name concentration.
- **E5** engine params (HMM/POT-GPD/dealer clamp) are parameter-IN-sample; a
  different-window run is out-of-*window*, not out-of-*parameter*.
- **D19/D21** `ev_dollars` nets only entry-leg cost; forward-dist samplers mix
  trading-day bars with calendar DTE (~46% horizon over-dispersion). The
  realized P&L this fan is built on inherits both.
- **Bootstrap** resamples the book's *deployed-capital* returns i.i.d.-in-blocks
  and cannot invent regimes absent from the sampled window (survivorship:
  universe = current members).
- **Copula** is a reporting/stress overlay only (`feeds_ev=False`); it never
  touches `ev_dollars`, a verdict, or the R7/R8 thresholds (§2 invariant 5).

## Adversarial review + hardening

Ran a 5-dimension adversarial review workflow (§2-safety / numerics /
reconciliation / caveat-honesty / driver-edges) with a per-finding refute pass:
**no §2 violation** (module confirmed off the decision path), 10 confirmed
findings (2 medium, 8 low). Fixes applied before the PR:

- **[medium] Reconciliation base.** The gap was measured against nominal
  `initial_capital`, but the return series compounds off the **first equity
  mark** — so the gap conflated the nominal→first-mark drift with bootstrap
  error (the tests masked it by prepending a mark == capital). Rebased the fan +
  reconciliation on the first mark; `engine_measured` now reports both
  `initial_capital` and `book_base_nav`. Regression-pinned with a
  first-mark≠capital curve (180k vs 200k) that would false-fail under the old
  base. Reconciled gaps barely moved on the real books (first mark ≈ capital
  there) but now measure pure bootstrap fidelity.
- **[medium] Copula alignment.** `portfolio_correlation_tail` filtered
  non-finite values *per name*, which shifts one column's dates on an interior
  NaN and understates correlation. Switched to **listwise** (joint-row)
  deletion; pinned with an interior-NaN test asserting the correlation matches
  the drop-that-row-across-all-names reference (and a no-overlap skip).
- **[low]** Honest docstrings (median vs mean / Jensen, not an exact identity;
  removed a stale `generated_kind` key from the docstring); graceful
  `insufficient_history` report instead of an uncaught `ValueError` on an
  empty/short book; consistent notional weight scale (book-median-strike
  fallback, not a raw count that the sum-of-abs normalization would near-zero).

## Unresolved / handoff

- **Dashboard view (Phase 3, P1) not wired.** A `/api/portfolio/montecarlo`
  (+`/stress`) additive endpoint + a Recharts fan panel would surface this on
  the operator dashboard — deferred to coordinate with the Dashboard terminal
  (§6 owns `dashboard/` + the `/api/portfolio` pipeline). The persisted
  `mc_equity_bands.json` is already shaped for such a panel; label MC panels
  "model" and calibration panels "engine-measured".
- **SIM artifacts are gitignored** (`data_processed/sim/`), so they are
  regenerated on demand via `run_forward_sim.py`, not committed.
