---
id: validation-v5-reverse-stress
title: V5 reverse stress — worst-admissible-book adversary + margin procyclicality resolved honestly
kind: verification
status: complete
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

- Expectation 4's >= 20% trough clause is NOT met on the measured book
  (10.03% NAV) — but the number is biased low on two stacked axes
  (intrinsic-only marking; the saturated book exhausted the 2-bday
  capture menu at $321k of $1M). Disposition: not established on a
  by-construction-conservative book; full-ranking time-value replay is
  the optional follow-up.
- The x1.25/x1.5 levered day-1 calls in ALL windows are
  near-tautological (the twin is capitalized at exactly initial margin,
  zero buffer) — recorded with the caveat so they are not over-read;
  x1.0 is the regime-discriminating cell.

## How we fixed it

Nothing to fix — both harness runs were clean on first attempt; the
caveats are dispositions, recorded in plan sections 8.5-8.6.

## Evidence

- `python3 -m pytest tests/test_reverse_stress.py -q` -> 12 passed.
- Pre-registration: plan doc section 8, committed before any V5 code;
  8.3 expectations frozen.
- **V5-a DONE** (Windows, ~1 min): ruin-class IS gate-admissible — 8
  COVID-onset ruin dates, deepest 36.5%/36.4% NAV from CALM entries
  (VIX 13.7/14.2); realized 3.18-7.07x modeled book CVaR on all 8;
  top-bin-only bounded at 19.2% (never ruin). Expectations 1-3
  confirmed; falsifier rejected.
- **V5-b DONE** (Mac, ~2 s): COVID assignment wave 100% (27/27 ITM);
  trough 10.03%/terminal 7.54% NAV on a doubly-conservative measured
  book; Reg-T twin called bday 13 at x1.0 (COVID only) — expectation 5
  confirmed both clauses; expectation 4 split (see What didn't).

## Unresolved / handoff

- **V5 CLOSED 2026-07-13**: F-V5-1 (gate stack admits ruin-class
  calm-onset composition) -> the re-baseline queue alongside
  F-V1-1/2/4 + F-V3-1; the CSP mandate recorded as the load-bearing
  protection (structural absence of the margin spiral; un-forceable
  holding quantified by the levered twin's day-13 call). Full numbers:
  plan sections 8.5-8.6.
- Optional follow-up (non-blocking): full-ranking, time-value-marked
  assignment-wave replay to establish/retire the >= 20% trough clause.
- Next and last workstream: V6 — the single pre-registered lockbox
  spend (deep history, 2008/1998, delisted names; refusal mechanism).
