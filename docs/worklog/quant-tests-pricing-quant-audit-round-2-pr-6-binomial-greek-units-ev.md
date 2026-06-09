---
id: quant-tests-pricing
title: "Quant audit round 2 PR-6: binomial Greek units + evaluate degenerate-dte (W63-W64)"
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Cross-checks the binomial tree's vega/theta/rho UNITS vs analytic BSM on q=0 calls (American==European) to <5% — a 100x unit slip would otherwise be invisible; pins EVEngine.evaluate finiteness at degenerate dte (<=0); gamma left loose (tree-noise)
surface: [tests, quant]
---

## Goal
Quant audit round 2, PR-6 — pricing core + evaluate gates. Pin the binomial Greek
UNITS (only delta/gamma were validated; vega/theta/rho units unasserted) and
evaluate's degenerate-dte finiteness. Tests-only.

## What we tried
Probed binomial_american_full vs black_scholes_all_greeks + evaluate at dte<=0 on
`a2eea4e` before writing.

## What worked
- **W63 binomial vega/theta/rho units (HIGH):** on q=0 CALLS (American tree price ==
  European BSM, no early-exercise premium) the binomial vega/theta/rho match BSM to
  <5% (probed: vega rel 1e-4, theta 8e-4, rho 8e-3) — confirming the documented "per
  1 vol point / annual / per 1% rate" units. A unit slip (per 1.0 vs per 0.01) would
  be ~100x; 5% comfortably distinguishes. delta matches to <0.02.
- **W64 evaluate degenerate dte:** dte in {0,-5,1} -> finite ev_dollars + ev_per_day;
  ev_per_day == ev_dollars at dte<=1 (the max(dte,1) divisor floor — no div-by-zero);
  dte=35 -> ev_per_day < ev_dollars (real holding period divides).

## What didn't
- **EVResult has no `ev_raw`** — the probe corrected my field assumption (it's
  internal); the public fields are ev_dollars / ev_per_day / prob_* / cvar_* / etc.
- **Binomial GAMMA not tightly pinned:** ~25% off BSM at 800 steps (finite-difference
  tree noise, not a unit issue) — asserted same-sign + within 3x only; already
  validated elsewhere.
- **Put cross-check NOT used:** American put != European put (early-exercise premium),
  so the BSM cross-check only holds for q=0 calls.

## How we fixed it
`tests/test_pricing_evaluate_invariants.py` (new): 7 tests, all passing. ruff-clean.

## Evidence
- Probe (py -3.12) on `a2eea4e`: greek cross-check table (vega/theta/rho/delta <1%,
  gamma 25%); degenerate-dte finiteness + floor. Touches: test file + fragment +
  manifest row. No trio/data edits.

## Unresolved / handoff
- Deferred from this surface (LOWer / scattered): assignment-fee-on-ITM, payoff
  anchor-node guarantee, omega_ratio 1000 cap, prob_touch 2x reflection, the
  deterministic ATM delta, and gate-ordering (event-lockout-before-BSM — the lockout
  FIRING is already covered by W16/W30). Fold into a pricing/evaluate follow-up if wanted.
- Last planned round-2 PR: PR-7 decision reviewers (R3/R5 exact boundaries, EventGate
  earliest-hit, cap nav=0) + the deferred vol-surface (SVI/Spline + (E) Spline-0.20).
