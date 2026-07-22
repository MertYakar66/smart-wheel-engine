---
id: validation-v4-capacity-curve
title: V4 capacity curve — capital-equivalent contract ladder, dormant impact term armed via swept ADV proxy
kind: verification
status: complete
terminal:
pr:
decisions: []
date: 2026-07-13
headline: The last desk-practice gap (execution/capacity realism beyond one contract at mid) gets its harness — a shared-rank contract ladder that sidesteps the tracker's contract-blindness exactly, arms the engine's dormant Almgren-Chriss term at true order size, and reports the capacity knee as a function of the undocumented option-ADV proxy rather than pretending a point estimate.
surface: [backtests/capacity_curve.py, scripts/run_capacity_curve.py, tests/test_capacity_curve.py, docs/VALIDATION_PHASE_PLAN.md]
---

## Goal

V4 of the validation phase (plan doc section 7, pre-registered before any
code): where is the knee of edge vs deployed dollars? Measurement-only,
loose mode, zero engine changes (the `r10_strict_driver.py`
copy-the-driver pattern).

## What we tried

- Passing `contracts=N` to the existing `run_backtest`: rejected — recon
  established the tracker is contract-blind in every cash flow (premium
  credit, BP reserve, assignment shares, scored P&L), so `contracts>1`
  is internally inconsistent and has never been exercised in any locked
  study. Recorded as a discovered-constraint finding (plan 7.0(1)); no
  tracker fix ships from a validation workstream.
- Extending the tracker to carry a real contract count: rejected for V4 —
  that is a D-series engine decision, not a measurement harness's call.
- A static x N replay over one rank_log (the s45 counterfactual
  pattern): rejected as the primary method — it cannot carry the
  path-dependence of BP saturation; kept as the fallback for scales
  where the ladder is too expensive.
- Strict-mode ladder: excluded — HT-D's F6 shows the portfolio-delta cap
  is structurally miscalibrated for a wheel book and would swamp the
  capacity signal (plan 7.0(3)).

## What worked

Capital-equivalent scaling: ladder point N runs a plain 1-contract
tracker at `capital = BASE/N` — every cash flow is exactly 1/N of the
N-contract world under the linear cost model, so the simulation is exact
without touching the tracker; the size-dependence enters ONLY through
the overlay, which prices each fill with the engine's own dormant
Almgren-Chriss sqrt term (`calculate_slippage` at `bid_ask_spread=0`,
true order size N, shipped k=0.10) on top of `full` friction, plus a 10%
participation cap on proxied option ADV (`r x avg_vol_30d`, r swept over
{1e-5, 1e-4, 1e-3} — worklist A9 sanctions the stock-ADV proxy but no
ratio number exists anywhere, so the knee is reported per-r). One shared
daily rank + a per-(ticker, day) covered-call cache serve all 16 grid
points at ~1x the rank bill. 16 fast tests.

## What didn't

- The original 7.2(1) linearity expectation ("identical returns at every
  N") was WRONG — it contradicted 7.2(5): per-position economics are
  scale-invariant but portfolio deployment scales with N, so the honest
  A/A is `return_pct` PROPORTIONAL to N until the first BP refusal.
  Caught while implementing the check, corrected in the plan doc with a
  pre-run marker before any run; expectations otherwise untouched.

## How we fixed it

Corrected 7.2(1) to the proportionality form; `linearity_check` compares
`return_pct / N` across unthrottled control points and fails only on
deviation WITHOUT a BP refusal.

## Evidence

- `python3 -m pytest tests/test_capacity_curve.py -q` -> 16 passed.
- Pre-registration: plan doc section 7 (`0d4cee6`), committed before any
  V4 code; three-reader recon (tracker mechanics, S34/R10 scale
  conventions, sizing-data landscape) recorded in 7.0 before design.
- **V4-pilot DONE (2026-07-13, 68 min, 16 grid points)**: linearity
  A/A PASS (every proportionality deviation coincides with BP refusals);
  expectation 5 FALSIFIED in direction — BP refusals begin at N=5 (363),
  the linear segment at $1M/24t ends before N=5; knee shifts with the
  proxy ratio (expectation 3 confirmed); and the emergent headline:
  beyond the BP knee at 24 names the curve is a CONCENTRATION LOTTERY
  (control ladder non-monotone +7.0 -> +16.0 -> +11.9 -> +35.7; an
  impact arm can even beat its control by luck-of-the-draw
  redirection) — high-N knee cells are noise at this universe size;
  the capacity question proper needs 100-name breadth. Full numbers:
  plan section 7.4.

## Unresolved / handoff

- **V4 CLOSED 2026-07-13** (both 100t NAV arms, 133 min each): $1M is
  BP-bound at EVERY rung (refusals from N=1; breadth accelerates
  saturation — question (a) answered opposite); the $10M arm confirms
  expectation 6 on all three prongs with a SUBSTANTIVE A/A (return/N
  agreeing to 2e-12 on the unthrottled segment) and delivers the first
  unconfounded impact knee (r=1e-5 column, zero BP refusals, interior
  N*=10). Structural aliasing (($10M, N, r) == ($1M, N/10, r/10))
  recorded + doubles as a cross-run determinism check. Full numbers:
  plan sections 7.4-7.5.
- The tracker contract-blindness constraint (7.0(1)) queued for the
  findings record; a real multi-contract tracker is a D-series decision.
- The data-grounded knee stays deferred to the Theta option-volume pull
  (acquisition plan E-13); B5 owns the k calibration.
