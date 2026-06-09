---
id: quant-tests-hmm-regime
title: "Quant audit round 2 PR-3: HMM regime-multiplier invariants (W50-W55)"
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Pins the HMM multiplier envelope sup/inf (0.2..1.25), same-seed determinism, the unfit-guard RuntimeError, and the RegimeDetector degenerate fallback; fit's missing non-finite guard (NaN multiplier) tracked as (E) #386 via xfail; the crisis over-firing left UNPINNED as a known limitation
surface: [tests, quant]
---

## Goal
Quant audit round 2, PR-3 — the 4-state Gaussian HMM (`engine/regime_hmm.py`)
that produces the regime multiplier scaling `ev_dollars`, plus the heuristic
`RegimeDetector`. Pin the GAPS only (the `T<K*3` / near-constant fit guards are
already covered by test_quant_upgrades). Tests-only; (E) tracked, not grabbed.

## What we tried
Probed every invariant on `a2eea4e` before writing.

## What worked
- **W50 envelope sup/inf:** one-hot on each label returns exactly its weight
  (crisis 0.2 … bull_quiet 1.25), so the sup is 1.25 and the inf is 0.2; 200
  Dirichlet draws all land in [0.2, 1.25]. The existing test only checked a band
  on one diffuse posterior (passes for any weights <=~1.25).
- **W51 same-seed determinism:** identical means / labels / multiplier across two
  same-seed fits (the cache + backtest fingerprint depend on this).
- **W53 unfit guards:** predict_proba / viterbi / position_multiplier raise
  RuntimeError before fit (the guard the ranker's neutral-1.0 fallback relies on).
- **W54 RegimeDetector:** `_calculate_realized_vol` returns the 0.20 default on
  <2 returns / empty.

## What didn't (honest non-pins)
- **NaN/inf input is NOT guarded (W55).** Probe: `fit` does not raise on a NaN/inf
  observation (`np.nanstd` ignores NaNs); it returns NaN means -> a NaN multiplier.
  The ranker try/except only catches a RAISE, so only the downstream ev_engine
  non-finite clamp saves EV. Pinned with `xfail(strict)` asserting the DESIRED raise
  + filed (E) #386 — not asserted as correct. LOW: connector OHLCV-positivity blocks
  the upstream trigger and the engine clamp is the net.
- **Crisis over-firing left UNPINNED.** A benign low-vol uptrend's last bar gets
  mult 0.388 (the documented F4 over-firing — "crisis fires on 98% of dates").
  I did NOT add a "benign -> not down-weighted" test: it would either lock in the
  over-firing or speculate a fix. It is a known calibration limitation, not a clean
  pinnable invariant — recorded here, not in a test.
- **diff-seed label invariance not pinned** (potential different local optima);
  only same-seed determinism is rock-solid, so that is what's pinned.

## How we fixed it
`tests/test_regime_hmm_invariants.py` (new): 7 tests (5 pass, 2 xfail). ruff-clean.

## Evidence
- Probe (py -3.12) on `a2eea4e`: one-hot mults 0.2/0.5/1.0/1.25; 200 simplex in
  [0.342, 1.150]; same-seed means==, mult==; NaN/inf -> mult=nan; before-fit
  RuntimeError x3; benign uptrend mult 0.388.
- Touches: the new test file + this fragment + FILE_MANIFEST row. No engine/data/trio edits.

## Unresolved / handoff
- **(E) #386** — add a non-finite guard to `GaussianHMM.fit` (raise like the other
  degenerate guards) so the neutral-1.0 fallback engages cleanly; the xfail flips green.
- Remaining round-2 PRs: PR-4 skew/surface, PR-5 dealer `[0.70,1.05]` clamp, PR-6
  pricing/Greeks/gates, PR-7 reviewers — one surface each, held for review.
