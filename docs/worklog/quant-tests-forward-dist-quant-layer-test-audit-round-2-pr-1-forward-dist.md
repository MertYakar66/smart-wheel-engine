---
id: quant-tests-forward-dist
title: Quant-layer test audit round 2 PR-1: forward-distribution cascade + realized-vol invariants (W38-W43)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Behaviour-pins the EV integrand's cascade tier-SELECTION (not membership), sampler determinism, empirical boundary, and realized-vol guards; probe exposed an incomplete _log guard (+inf leak) tracked as (E) #382 via xfail(strict)
surface: [tests, quant]
---

## Goal
Round 2 of the test-coverage audit. The 2026-06-09 data-test audit (W14-W37)
covered DATA accessors -> ranker OUTPUT; this round pins the **math in between** —
starting (PR-1) with the forward-distribution cascade (`engine/forward_distribution.py`),
the integrand `EVEngine.evaluate` integrates over, plus the realized-vol estimators
(`engine/realized_vol.py`). Tests-only, no §2 surface touched; any trio/engine-fix
gap is tracked as (E), not grabbed.

## What we tried
A 7-agent parallel coverage-recon workflow over the quant + decision modules
surfaced 48 candidate gaps. **Every load-bearing claim was re-verified against
source** (subagents hallucinate) before writing a line — and an empirical probe
on `a2eea4e` confirmed the exact tier-selection inputs, determinism, and the
realized-vol edge behaviour.

## What worked
- **W38 cascade tier-selection (HIGH):** the only prior test
  (`test_cascading_fallback_picks_best`) asserts `method in (4-tuple)` — a
  false-green that passes for ANY tier. Pinned the EXACT tier at four controlled
  history depths (600/20->NOS, 200/20->overlapping, 112/55->block_bootstrap,
  70/20->har_rv) + empty->none, verified by probe.
- **W39 sampler determinism (HIGH):** `block_bootstrap`/`har_rv` same-seed ==,
  different-seed != (pins seed plumbing, not a frozen constant).
- **W41 empirical boundary:** exact `min_samples`, `n<=horizon`, and the
  `isfinite` filter dropping inf-poisoned returns.
- **W42 variance floor:** GK/RS `max(var,0)` yields finite 0.0 on degenerate bars.

## What didn't (the verify-against-source payoff)
- **W40 was over-claimed by the recon.** The subagent said "non-positive prices ->
  NaN for any estimator." The probe disproved it: a zero in a *denominator*
  (`low=0`) makes parkinson/garman_klass leak **+inf** (the `_log` guard sits on the
  post-division ratio), and an all-negative bar is **silently swallowed** to a
  normal-looking 0.076. Only `close_to_close` and a fully-zero bar give clean NaN.
- So W40 was SPLIT: the clean guards are pinned as passing (T) tests; the +inf leak
  is pinned with `xfail(strict=True)` asserting the DESIRED NaN (flips when fixed),
  and the incomplete guard is filed as **(E) #382** — NOT asserted as correct
  (the #366 anti-pattern). Severity LOW: connector OHLCV-positivity tests block the
  trigger upstream and these estimators feed VRP/MODEL_CARDS, not the EV integrand.

## How we fixed it
`tests/test_forward_distribution_invariants.py` (new): 19 tests (17 pass, 2 xfail).
ruff-clean; degenerate-input numpy RuntimeWarnings suppressed per-class.

## Evidence
- Probe (py -3.12) confirmed all four tier labels + none; bb/hr same-seed== /
  seed-diff!=; estimator edge table (all-zero->NaN; all-neg->0.076 swallow; low=0->
  +inf leak); GK/RS floor->0.0.
- Full fast-lane suite on `a2eea4e`: 3019 passed, 3 failed (all `test_theta_connector`
  — known Windows-only Theta/F3 flakes per `windows-local-vs-ubuntu-ci`; ubuntu CI green).
- Touches: the new test file + this fragment + FILE_MANIFEST row. No engine/data/trio edits.

## Unresolved / handoff
- **(E) #382** — harden `_log`/estimator guard (clamp raw prices, no +inf, no
  swallow); the xfail flips green when it lands. Needs §2-style review (engine math).
- Remaining quant-coverage PRs (round-2 plan): PR-2 tail/copula/stress, PR-3 HMM
  regime, PR-4 skew/surface, PR-5 dealer clamp, PR-6 pricing/Greeks/gates, PR-7
  reviewers — each one surface, held for review. Plus the 2nd (E) from recon
  (`SplineVolSurface` silent-0.20 vs D9; t-copula df bounds) to file when those PRs land.
