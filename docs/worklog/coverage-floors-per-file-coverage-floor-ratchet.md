---
id: coverage-floors
title: "Per-file coverage-floor ratchet for the decision trio + engine core"
kind: fix
status: shipped
terminal: X
pr:
decisions: []
date: 2026-07-03
headline: "The aggregate --cov-fail-under=80 spans ~15,320 statements, so wheel_runner could shed 20pp of coverage without tripping it; a stdlib script now floors 7 engine-core files at measured-on-main minus 2pp — green on day one, red only on decay, missing-file fails loud"
surface:
  - .github/workflows/ci.yml
  - scripts/check_coverage_floors.py
  - tests/test_coverage_floors_script.py
---

## Goal

Adversarial review 2026-07-01: "no per-file coverage floor
(wheel_runner 75.3% under aggregate-80 green)". The rescued 2026-06-15
branch (`c1fdbd4`) carried the mechanism's origin design.

## What we tried / worked

Recon measured CURRENT per-file coverage from the CI coverage-json
artifacts (run 28638294523 @ `4c5a1a4`): both matrix legs (3.11/3.12)
identical to 4 decimal places across all 83 files — so enforcing on
both legs carries zero cross-version flake risk. wheel_runner is at
79.35 today (the review's 75.3 was vs d487b17; #458/#460-era tests
raised it).

## What didn't

- **The rescued thresholds**: wheel_runner 54 (from the 2026-06-15
  measurement 57 − 3) would permit ~25pp of decay from today's 79.35.
  Mechanism credited, thresholds discarded.
- **The rescued mechanism** (three bare `coverage report --fail-under`
  lines): pyproject's `fail_under = 80` is inherited by any bare
  `coverage report` that forgets the flag (silently mis-gates a single
  file at 80); `--include` matching zero files is a soft error. The
  json+script path compares full-precision floats, keeps one greppable
  floors dict, and FAILS LOUD on a missing file (the xfail-false-green
  lesson).
- **Flooring more files without measuring**: other files among the 83
  sit much lower; the floored set is exactly the trio + engine core
  the review flagged (pinned by a census test).

## How we fixed it

`scripts/check_coverage_floors.py` (stdlib-only; floors = int(measured)
− 2pp: ev_engine 93, wheel_runner 77, candidate_dossier 89,
data_connector 88, event_gate 95, wheel_tracker 81,
portfolio_risk_gates 96) + a Test Suite step after the coverage.json
artifact upload, both matrix legs, default `success()` condition
(deliberately NOT `if: always()` — a floor report on a partial run is
noise on an already-red job). Aggregate `--cov-fail-under=80` left
untouched (script prints the 84.79 aggregate informationally).

## Evidence

- 7 script pins green (pass / at-floor / below-floor names the file /
  missing-file loud / Windows path keys / exit codes / floored-set
  census).
- Live run against the REAL main artifact: all 7 floors ok, exit 0 —
  the day-one-green property proven on actual data, not synthetic.
- The PR's own CI run exercises the new step on both legs.

## Unresolved / handoff

- Floors are selection-dependent (branch-inclusive, `-m "not
  backtest_regression"`): recalibrate in the same PR that changes the
  Test Suite selection or omit list.
- Ratchet floors UP whenever a coverage-raising PR lands (deliberate,
  never silent).
- If `--cov` is ever added to the Integration lane, its coverage.json
  is NOT floor-checked (the step lives in Test Suite only).
