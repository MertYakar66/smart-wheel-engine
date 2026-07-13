---
id: validation-v3-param-plateau
title: V3 parameter-plateau sweep — F4/R11 plateau-or-peak verdicts + activation gates for the dormant surfaces
kind: verification
status: complete
terminal:
pr:
decisions: []
date: 2026-07-12
headline: The hand-set constants get perturbed for the first time — F4 threshold/cap swept with engine passes under the sanctioned patch lever (PLATEAU/CLIFF/PEAK/DOMINATED verdicts, rank-quality guard), R11 cutoffs swept offline on the V1 captures with both the breach and the D23 over-confidence lenses, and the POT/heavy-tail/bootstrap constants activation-gated instead of vacuously passed.
surface: [backtests/param_plateau.py, scripts/run_param_plateau.py, tests/test_param_plateau.py, docs/VALIDATION_PHASE_PLAN.md]
---

## Goal

V3 of the validation phase (plan doc section 6, pre-registered before any
code): the Phase-0 static constants have never been perturbed. Sweep what
actually acts on the ranked path (F4 widening, R11 cutoffs), require
plateaus rather than peaks, and honestly gate the rest on measured
activation instead of sweeping dormant machinery. Measurement-only; trio
untouched.

## What we tried

- Sweeping every Phase-0 constant uniformly: rejected — the regime overlay
  is already out-of-parameter tested (parameter_oos), the clamps are rails,
  and the GPD/heavy-tail/bootstrap constants mostly never run on Bloomberg
  (the GPD needs >= 200 scenarios; ~99% of rows ride the N~35
  non-overlapping tier). Disposition table in plan section 6.0.
- A single breach-rate lens for R11: insufficient — R11's D23 rationale is
  top-bin WIN-RATE over-confidence after elevated-vol readings, not cvar
  breaches; the sweep carries both lenses (breach lift AND the
  forecast-minus-realized over-confidence gap, flagged vs unflagged).

## What worked

`backtests/param_plateau.py` — `f4_patched` context manager (pins
threshold/slope/max_widening on both forward_distribution functions;
loud on unknown kwargs), `build_f4_sweep_table` (V1 TAIL_TABLE schema so
V1 statistics run unchanged; pre-registered grids enforced),
`activation_from_frames` (POWERED / NOT_POWERED / NO_DIAGNOSTIC_COLUMNS
at the 2% floor), `r11_sweep` (both lenses, date-clustered CIs),
`plateau_verdict` (PLATEAU / CLIFF / PEAK / DOMINATED at 2-sigma combined
SE, guard-aware), `f4_axis_report` (primary = elevated+crisis breach
rate; guard = top-15 per-date rho >= shipped's block-CI lower bound).
Driver `scripts/run_param_plateau.py` (activation / r11 / f4-build with
resume / f4-analyze). 18 fast tests.

## What didn't

- First synthetic F4-patch test drew its "vol spike" with raw normals —
  at n=30 the sample-std noise (~13%) drowned the premise (measured ratio
  0.96 vs intended ~1.2). Fixed by std-normalizing the blocks exactly.
- `activation_from_frames` crashed on frames missing the diagnostic
  columns (pd.to_numeric on a scalar); fixed + pinned.

## How we fixed it

See above. Both fixes were harness-side, caught by the fast tests before
any run.

## Evidence

- `python3 -m pytest tests/test_param_plateau.py -q` -> 18 passed.
- Pre-registration: plan doc section 6 (`f67d669`), committed before any
  V3 code; 6.3/6.4 criteria frozen.
- **Activation diagnostic (2026-07-12, 8 spot dates x 24 names, 87
  rows):** gpd_fit_rate 2.30% (2 rows — exactly the overlapping-tier
  rows), heavy_tail fires 0.00%, n_scenarios>=200 0.00%. Read together
  with the full captures (V1-a mix: 0.73% overlapping at 24t; V1-b:
  11.4% at 100t): activation is SCALE-DEPENDENT — effectively NOT
  POWERED at 24t, potentially powered at 100t; and the ξ-gate/penalty
  pair is unpowered everywhere observed (the flag never fired). POT
  sweep deferred accordingly; 100t activation spot-check added to the
  terminal card.
- **R11 sweep, 24t (2026-07-12, n=2,735):** the shipped cell (25.0,
  0.90) sits on a SHELF, not a spike — but the shelf inverts the
  mechanism on this window: flagged rows breach 0.0% vs 7.1% for the
  calm unflagged top-bin (lift 0.0; lift <= 1 across the entire grid),
  flagged mean realized BEATS unflagged by +$368, and the D23
  over-confidence gap is SMALLER in the flagged region (+0.067 vs
  +0.108). On a window with no crisis onset (24t starts 2022-01), the
  danger R11 guards against lives BELOW its VIX threshold — the F-V1-1
  onset-blindness seen from the reviewer's side. NOT a verdict on R11
  yet: D23/i11 validated it on onset windows; the 100t sweep (2020-02
  start) arbitrates.
- **F4 sweeps DONE (2026-07-13, 9 passes x 1,971 rows): PLATEAU on both
  axes** — and barely load-bearing: threshold moves the fire rate 5x
  (28.8% -> 6.0%) with ZERO change in elev+crisis breach (1.63% flat
  for 1.1-1.4); the F4-OFF control differs from shipped by 2 breaches
  in 1,596 rows; cap effect saturates at 1.15. Expectation 1 confirmed
  (not a fitted artifact), expectation 2 falsified in substance (OFF ~
  shipped). Third independent arrival at "the trailing-vol reaction
  channel is not where tail honesty comes from" (with V2-c and F-V1-1).
  Full tables: plan doc section 6.6.

## Unresolved / handoff

- **V3 CLOSED 2026-07-13** (overnight 100t bundle, 4h40m, all clean):
  (a) R11 arbitration — 2020 in-sample only partially restores the
  premise; shipped (25.0, 0.90) shows lift 0.96 and a still-inverted
  over-confidence gap (+0.075 vs +0.111); lift crosses 1 only above
  VIX 27.5 -> F-V3-1 to the re-baseline queue (mis-aimed cutoffs; the
  calm-entry over-confidence is unreachable by any VIX rule).
  (b) Activation POWERED at 100t (gpd 4.12%) for the POT diagnostic;
  heavy_tail 0.00% everywhere -> the EV-path half stays NOT POWERED;
  POT sweep deferred with low-decision-relevance rationale.
  (c) F4 PLATEAU on both axes at 100t; OFF-control worse by only
  0.064pp (~0.10 SE) — barely load-bearing, replicated.
  Full numbers: plan doc sections 6.6-6.7 + the executor's
  v3_overnight_report.md.
- No PEAK verdict anywhere: no swept constant is a fitted artifact.
