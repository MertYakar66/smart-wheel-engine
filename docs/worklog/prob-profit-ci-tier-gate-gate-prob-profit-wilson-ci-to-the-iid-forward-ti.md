---
id: prob-profit-ci-tier-gate
title: Gate prob_profit Wilson CI to the IID forward tier (RA-4)
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-01
headline: prob_profit's Wilson CI is now emitted only on the IID empirical_non_overlapping forward tier; suppressed (null) on the overlapping/bootstrap/har_rv/lognormal tiers where N is not an independent-trial count and the interval would be false precision.
surface:
  - engine/forward_distribution.py
  - engine/wheel_runner.py
  - tests/test_prob_profit_ci.py
  - tests/test_covered_call_ranker.py
  - tests/test_strangle_ev_ranker.py
---

## Goal
Close **RA-4** from the 2026-06-01 reliability/realism audit (RA-N are the
audit's internal finding numbers, namespaced to avoid collision with the
`DECISIONS.md` D-number series, which is live at D23). The prob_profit
Wilson 95% CI (shipped on `claude/prob-profit-ci-honesty` + propagated to the
CC/strangle rankers, `engine_api`, and the Ollama memo on
`claude/prob-profit-ci-propagate`) was computed over `n_scenarios = len(pnls)`
for **whatever** forward tier produced the draws. A Wilson binomial interval is
only an honest *sampling* spread when N is a count of INDEPENDENT trials. That
holds for exactly one tier — `empirical_non_overlapping` (~30–35 disjoint
windows). On the other tiers the reported N is autocorrelated or synthetic
(`empirical_overlapping`, `block_bootstrap`/`har_rv` at n≈5000,
`lognormal_fallback` at n≈20000), so the CI renders deceptively **tight** —
false precision, the exact opposite of the honesty goal. The dashboard consumer
half (`claude/dashboard-prob-profit-ci`) already guarded this with
`samplingCiHonest`; this is the engine-half root-cause fix, which also makes the
`/api/candidates` JSON and the trade memo honest.

## What we tried
1. **Gate inside `ev_engine.evaluate` (rejected).** The Wilson CI is computed at
   a single site there — but `evaluate` does **not** know the fine cascade tier.
   It only stashes a coarse `self._last_distribution_source`
   (`empirical` / `price_scenarios` / `lognormal_fallback`); it cannot tell
   `empirical_non_overlapping` from `empirical_overlapping`/`block_bootstrap`/
   `har_rv` (all arrive tagged `"empirical"`). Threading the tier into `evaluate`
   would change a §2 API signature and the default-None behavior for every
   caller — larger trio surface, more risk, for no extra correctness.
2. **Gate at the 3 ranker row-dict sites (shipped).** The fine tier label
   (`method`) and `res.prob_profit_ci_*` are BOTH in scope where each ranker
   builds its row dict — the natural, minimal gate point. `EVResult` and
   `evaluate`'s signature stay byte-untouched.

## What worked
- A single predicate `is_iid_forward_source(source)` in
  `engine/forward_distribution.py` (the module that *owns* the tier labels;
  non-trio). Mirrors the dashboard's `samplingCiHonest` — one Python source of
  truth.
- `ci_ok = is_iid_forward_source(method)` per candidate, gating the
  `n_scenarios` + CI columns in all three rankers (put / CC / strangle two
  legs). Because every CI consumer (`engine_api`, `trade_memo`, dashboard) reads
  the **ranker columns** — not `EVResult` directly — this one gate fixes every
  trader-facing surface at once.
- The CI bundle (`n_scenarios` + `ci_low/high`) is suppressed together, matching
  the dashboard's group-gate so N is never shown as standalone false confidence.

## What didn't
- The naïve assumption "the live path is always non-overlapping so the gate is a
  no-op". TRUE for the **put** ranker (fixed ~35 DTE → all
  `empirical_non_overlapping`, N=35), but FALSE for the **CC/strangle** rankers:
  their longer-DTE grid cells have too few non-overlapping windows and fall to
  `empirical_overlapping` → the propagate branch was emitting a false-precision
  CI on **real** candidates (incl. the top-EV AAPL strangle row at
  as_of=2026-03-20). The gate is a genuine fix there, not a theoretical guard.
  This also forced 3 existing propagate tests to be rewritten from "every
  survivor has a CI" to the correct tier-conditional contract.

## How we fixed it
- `engine/forward_distribution.py`: `_IID_FORWARD_SOURCES = {"empirical_non_overlapping"}`
  + `is_iid_forward_source()`.
- `engine/wheel_runner.py` (TRIO): import the predicate in all three rankers;
  `ci_ok` local before each row dict; `... if ci_ok and not np.isnan(...) else None`
  on the n_scenarios + CI columns. Strangle uses one `ci_ok` for the shared N +
  both legs (both legs walk the same forward path → one `method`).
- Tests: new `TestIsIidForwardSource` (predicate) + `TestRankerGatesCiOffIidTier`
  (relabel-monkeypatch proves the put ranker suppresses the CI on a forced
  `block_bootstrap` tier while prob_profit is unchanged; IID control). Rewrote
  the CC/strangle survivor + real-survivor CI tests to the tier-conditional
  contract (CI present + bracketing on IID rows; null on non-IID rows).

## Evidence
- Live tier probe (10 tickers, put ranker, as_of=2026-03-20): all
  `empirical_non_overlapping`, n_scenarios=35 → gate is a no-op for the put path.
- `pytest tests/test_prob_profit_ci.py tests/test_covered_call_ranker.py
  tests/test_strangle_ev_ranker.py tests/test_tv_api.py tests/test_trade_memo_ci.py`
  → 120 passed (before the gate-aware test rewrite: 3 failed exactly on the
  non-IID CC/strangle cells, confirming the gate fires on real output).
- `ruff format --check` + `ruff check` on all 5 changed files: clean.
- §2: diff does NOT touch `engine/ev_engine.py`; `prob_profit` / `ev_raw` /
  `ev_dollars` / verdict / dealer clamp / EV-authority token unchanged — the gate
  only nulls observability columns. Independent §2 second-read run as a 3-lens
  adversarial panel.

## Unresolved / handoff
- **Merge order:** this branch is stacked on `claude/prob-profit-ci-propagate`
  (which includes `claude/prob-profit-ci-honesty`). Merge that chain (or this
  branch, which contains it) before/with the dashboard consumer branch
  `claude/dashboard-prob-profit-ci`.
- The dashboard's `samplingCiHonest` guard is now belt-and-suspenders (the API
  will already send null on non-IID tiers) — harmless, keep it.
- Optional future: emit an *effective* sample size on the non-IID tiers instead
  of plain suppression (more informative than null), and the RA-2 earnings-gate
  PIT diagnostic + the GATED calibration band remain open from the audit.
