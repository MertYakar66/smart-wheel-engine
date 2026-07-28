---
id: validation-followups
title: Post-phase follow-ups — V6-r1 full-menu H1 un-censoring + V5-b-full TV-marked wave
kind: verification
status: completed
terminal: V6-r1 on the Windows deep-data machine (62.4 min, 36,755 rows); V5-b-full on the MacBook (minutes)
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

- V5-b-full expectation 2 (>= 80% budget saturation) was FALSIFIED — and
  on all four crisis eves, not just the two thin ones: observed 32%
  (COVID) / 13% / 31% / 46%. Not a harness bug (expectation 1's
  mechanical sanity held on all 16 legs, `sanity_violations: []`): the
  fresh full menu ranks 66 names at COVID but only 27 clear the engine's
  own `ev_dollars > 0` gate, so `build_saturated_book` stops at $321k
  because the positive-EV menu is exhausted, not because the $1M budget
  binds. The falsification strengthens the retire rather than weakening
  it — recorded as-is, in the engine's favor.

## How we fixed it

- Nothing to fix (measurement-only; no engine change ships). The
  falsified saturation expectation was interpreted, not patched: it
  reframes §8.6's "capture-limited caveat" as MOOT — the full-menu books
  came back byte-identical to V5-b's capture-limited books
  (27/$321,050, 10/$133,450, 3/$313,950, 8/$463,600), so the tail-table
  capture had already captured the entire positive-EV menu. The real
  limiter was always the EV gate, never the data capture.
- One executor-side false alarm, no run impact: a git-bash `kill -0`
  watcher false-negatived on the native Windows PID and reported the
  V6-r1 run "exited" at ~1.5 min; the run was healthy and never
  restarted (re-verified via PowerShell, re-watched off log markers).

## Result

Both runs completed clean overnight; both caveats resolved in the
engine's favor. Recorded in plan §10.3, the §2 ledger row, and findings
§5/§6.

- **§10.1 V6-r1** — one clean 62.4-min pass, 36,755 rows (~3.8× V6's
  9,750). **FM1 = `CAVEAT_RETIRED`** (baseline 0.5040, grind 0.9828,
  ratio 1.95 > 0.5): un-censored at `top_n=100` the grind EV-positive
  rate is still 0.98 — top-15 logging hid no refusal; **H1's FAIL is
  unconditional, F-V6-1 stands as written.** FM2 report-only: 64/650
  dates below the saturation guard of 15 (all calm-ramp / 2009 tail,
  none in the grind), 0 below the opens appetite of 3. Diagnostic tail
  block carried onto the retained `v6r_rank_log.csv.gz` — no further deep
  read needed for this slice. Counted + discounted SECOND ledger read;
  1998/LTCM untouched.
- **§10.2 V5-b-full** — rc=0, `sanity_violations: []`, `n_no_iv = 0`
  everywhere (genuine BSM TV marks). **Trough clause = `RETIRED_PRACTICAL`**:
  COVID troughs at 10.03% NAV on every leg including tv_x2 (< 20%). The
  engine's own `ev_dollars > 0` gate caps crisis-eve deployment at ~32%
  of NAV — the 20% trough is unreachable through the engine's choices.
  Expectations 1 and 4 held; 2 falsified (above); 3 disposed
  `RETIRED_PRACTICAL`. No deep read, no ledger row (modern CSVs only).

Standing bound after both runs: premium coupling (§9.3 note 2, §2.4) on
F-V6-1, which waits on real option marks (E-13 / Theta). Nothing
committed by the executors; the brain recorded and pushed.
