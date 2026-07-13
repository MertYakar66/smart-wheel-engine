---
id: validation-v5-reverse-stress
title: V5 reverse stress — worst-admissible-book adversary + margin procyclicality resolved honestly
kind: verification
status: in_progress
terminal:
pr:
decisions: []
date: 2026-07-13
headline: The measured blind spots of V1-V4 compose into an explicit cheapest-path-to-ruin search — a hindsight adversary bounded by the entry-time gate stack (R10 contracts, R9 sectors, cash-secured budget) over the V1-b capture, plus the finding-shaped fact that the CSP mandate makes the classical margin spiral structurally absent, with the assignment-wave and Reg-T-levered counterfactuals quantifying what remains.
surface: [backtests/reverse_stress.py, scripts/run_reverse_stress.py, tests/test_reverse_stress.py, docs/VALIDATION_PHASE_PLAN.md]
---

## Goal

V5 of the validation phase (plan doc section 8, pre-registered before any
code): what breaks the book that no gate catches? Measurement-only, pure
offline computation; damage bounds, never sizing advice.

## What we tried

- Simulating a classical margin spiral on the tracker book: rejected on
  recon — the tracker's BP reserve is FULL collateral (cash-secured by
  construction; its admission check is Reg-T but the reserve is 100%),
  so the spiral is structurally absent under the CSP mandate. Recorded
  as a design fact; V5-b decomposes the blind spot into the
  assignment-wave stress and the LEVERED counterfactual instead.
- An exact knapsack for the adversary: rejected — greedy
  loss-per-collateral-dollar is a documented lower bound on the true
  optimum (conservative), simpler, and pre-registerable.

## What worked

`backtests/reverse_stress.py` — `worst_admissible_book` (entry-time-honest
admissibility: ev>0, R10 contract cap, R9 sector budget from the
fundamentals snapshot with unknown-sector exemption counted loudly,
cash-secured budget; hindsight loser-only greedy), `reverse_stress_search`
(per-date distribution, ruin list at the pre-registered 25% NAV
threshold, entry-VIX strata), `build_saturated_book` (engine-chosen,
EV-ranked, the V4 saturation fact), `replay_assignment_wave`
(intrinsic-only daily marking = lower bound on trough damage;
trough-liquidation counterfactual; Reg-T levered counterfactual under the
stressed-margin sweep). Driver `scripts/run_reverse_stress.py`
(search / margin). 12 fast tests on stub data.

## What didn't

- (runs pending — the two terminal cards)

## How we fixed it

- (pending)

## Evidence

- `python3 -m pytest tests/test_reverse_stress.py -q` -> 12 passed.
- Pre-registration: plan doc section 8, committed before any V5 code;
  8.3 expectations frozen.
- V5-a / V5-b terminal runs: pending (cards issued 2026-07-13).

## Unresolved / handoff

- V5-a on the Windows terminal (search over tail_table_100t.csv);
  V5-b on the Mac terminal (margin phase). Both minutes-scale, offline.
- Closure after both reports: ruin-date list + gate-permission
  statement; any "gates insufficient" finding to the re-baseline queue.
