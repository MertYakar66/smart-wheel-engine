---
id: quant-tests-skew-dynamics
title: "Quant audit round 2 PR-4: skew-dynamics invariants (W56-W59)"
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Pins the standalone skew-math (NS fail-fast + degenerate fit, skew_momentum degenerate history, ivs_dislocation [-1,1] bound); the live skew_mult clamp is dormant-on-Bloomberg + trio so not re-pinned, and skew-boost-cant-rescue is already covered generically
surface: [tests, quant]
---

## Goal
Quant audit round 2, PR-4 — the skew surface. Pin the standalone skew-math
primitives in `engine/skew_dynamics.py`. Tests-only.

## What we tried
Probed every behaviour on `a2eea4e` before writing.

## What worked
- **W56** NelsonSiegelTermStructure fail-fast: `iv_at`/`factor_loadings` raise
  RuntimeError before fit.
- **W57** NS degenerate-fit: n==1 -> beta0 = the single IV, converged False,
  n_points 1, iv_at == beta0; n==0 (all points masked by finite & T>0 & y>0) ->
  the 0.20 sentinel level.
- **W58** skew_momentum degenerate: empty -> NaN momentum + steepening False; a
  history <= short_window -> momentum 0 (both rolling slices collapse) + no signal.
- **W59** ivs_dislocation composite_score stays in the documented [-1, 1] band on
  normal + extreme-outlier term structures.

## What didn't (scope corrections vs the recon)
- **skew_mult clamp NOT re-pinned.** The recon flagged the `clip(1.0-0.5*slope,
  0.85, 1.08)` mapping (W) as unpinned — but it lives in `wheel_runner.py:1516`
  (the decision TRIO) AND is dormant on the Bloomberg path (no put/call skew ->
  slope==0 -> mult==1.0; memory bloomberg-iv-no-skew). Not unit-testable without
  trio access or a skew-bearing data path; left as a dormant-path note.
- **skew-boost-cant-rescue NOT re-pinned.** The §2 invariant (a regime/skew boost
  can't flip -EV to +EV) is already covered by the generic regime-multiplier test;
  no skew-specific duplicate.
- **ivs_dislocation clip can't be forced to engage** — least-squares residuals with
  an intercept are mean-zero, so the composite is structurally ~0; W59 pins the
  documented [-1,1] BOUND instead of a clip-engagement.

## How we fixed it
`tests/test_skew_dynamics_invariants.py` (new): 7 tests, all passing. ruff-clean.

## Evidence
- Probe (py -3.12) on `a2eea4e`: NS before-fit RuntimeError x2; n=1 beta0=0.30
  converged False; n=0 beta0=0.20; skew_momentum empty NaN / n<=5 momentum 0;
  composite ~1.7e-16 (in band). Touches: test file + fragment + manifest row. No trio/data edits.

## Unresolved / handoff
- **vol_surface (volatility_surface.py) deferred**: SVI butterfly-no-arb +
  implied_vol nonneg (dormant D9, documented math) and the (E) `SplineVolSurface`
  silent-0.20 fallback (contradicts the D9 fail-loud contract) — bundle into a
  dedicated vol-surface pass (low priority; dormant).
- Remaining higher-value round-2 PRs next: PR-5 dealer `[0.70,1.05]` clamp (HIGH §2),
  PR-6 pricing/Greeks/gates (binomial Greek units HIGH), PR-7 reviewers.
