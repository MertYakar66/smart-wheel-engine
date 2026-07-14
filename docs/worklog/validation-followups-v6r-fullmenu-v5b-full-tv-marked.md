---
id: validation-followups
title: Post-phase follow-ups — V6-r1 full-menu H1 un-censoring + V5-b-full TV-marked wave
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-07-14
headline: The two caveats the closed phase itself recorded, resolved by pre-registered runs (plan §10, committed before code) — §10.1 re-reads the spent 2007-2009 slice at top_n=100 to decide whether H1's FAIL was censored by top-of-book logging (CAVEAT_RETIRED vs CENSORING_LOAD_BEARING at the 0.5 cut, counted+discounted ledger read), and §10.2 replays the crisis-eve assignment wave on a FULL-ranking saturated book with entry-IV BSM time-value marking across a {1.0, 1.5, 2.0} IV bracket to dispose of V5-b's >= 20% trough clause (ESTABLISHED / RETIRED_PRACTICAL / OPEN, frozen rule). Both runs on operator terminals overnight.
surface: [backtests/survivorship.py, backtests/reverse_stress.py, scripts/run_v6r_fullmenu.py, scripts/run_reverse_stress.py, tests/test_v6r_fullmenu.py, tests/test_reverse_stress.py, docs/VALIDATION_PHASE_PLAN.md]
---

## Goal

The phase closed with two self-recorded caveats that are cheap to
resolve and matter for how far F-V6-1 and F-V5-1's clauses extend:
(1) H1's FAIL was measured on a top_n=15 rank log — saturated at 1.0
whenever >= 15 of ~100 names were EV-positive; (2) V5-b's trough number
rode a capture-limited book with intrinsic-only marks. Plan §10
pre-registers both resolutions (specs + falsifiable expectations +
frozen disposition rules) before any code, per the phase discipline.

## What we tried

- Marking the TV replay with per-day as-of IV from the connector:
  rejected for the first pass — entry-IV-constant marking with an
  explicit IV-multiple bracket {1.0, 1.5, 2.0} is simpler, honestly
  labeled as a bracket, and strictly brackets the vol blowout without a
  claim to measuring it; real option marks stay with E-13/Theta.
- Extending the frozen `run_v6_lockbox.py`: rejected — the lockbox
  driver stays byte-frozen as the §9 record; V6-r1 gets its own driver
  with its own frozen SPEC (`top_n=100`) and its own ledger row.

## What worked

- `backtests/survivorship.py`: additive diagnostic carry — the modeled
  tail block (`cvar_5`, `pnl_p25/p50/p75`, `n_scenarios`,
  `distribution_source`) rides onto the rank log when the frame has it,
  so future offline analysis of a captured slice needs no further read.
- `scripts/run_v6r_fullmenu.py`: frozen SPEC (§9.0 params except
  top_n=100), FM1/FM2 verdicts in code, raw capture written before
  verdicts, per-verdict exception capture (the V6 attempt-1/2 lesson),
  deep-panel precondition, no CLI overrides.
- `backtests/reverse_stress.py`: `marking="time_value"` mode on
  `replay_assignment_wave` (`_put_mark`: BSM from entry iv x iv_mult,
  dte counting down 7/5 per bday, never below intrinsic; intrinsic
  fallback counted in `n_no_iv`), `iv` carried on the saturated book.
- `scripts/run_reverse_stress.py margin-full`: fresh full-menu rank at
  each §8 crisis eve (top_n=100, min_ev -1e9), intrinsic A/A control +
  the three TV legs, expectation-1 sanity check enforced in-driver
  (rc=1 + loud print on violation), frozen §10.2 disposition rule
  computed in code.
- 9 new fast tests (TV-marking monotonicity/fallback/rejection, book iv
  carry, FM1 boundary at the 0.5 cut, FM2 appetite lines): 26 passing
  across the touched suites.

## What didn't

*(slot — filled from the executor reports)*

## How we fixed it

*(slot — filled from the executor reports)*

## Result

*(PENDING — two terminal cards issued: V6-r1 on the deep-data machine
(~75 min), V5-b-full on either machine (minutes). §10.1 verdict +
ledger completion and the §10.2 disposition land here and in the plan
on report.)*
