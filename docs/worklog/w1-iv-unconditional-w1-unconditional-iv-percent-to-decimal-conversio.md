---
id: w1-iv-unconditional
title: W1 unconditional IV percent-to-decimal conversion
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-08
headline: Delete the `if iv>3.0` IV magnitude-sniffer in wheel_runner (4 sites) → unconditional /100 (D20 shape); snapshot-neutral by enumeration; behaviour-preserving ×100 migration of 72 decimal-IV test stubs + 2 contract rewrites
surface: [engine/wheel_runner.py]
---

## Goal

Close audit finding W1 / issue #356: the ranker normalised IV with a
*conditional* `if iv > 3.0: iv = iv / 100.0` magnitude-sniffer at
`wheel_runner.py:198/1101/2418/2980`. vol_iv and fundamentals IV are
authoritatively PERCENT, and the >3.0 threshold left a genuine sub-3%
reading (e.g. 2.5 = 2.5%) undivided — the `(0,5]` guard then accepted it as
250%. Same class as the D20 treasury fix. Decision-trio change → full
ceremony (lane claim #113, markers, operator §2 second-read, sign-off).

## What we tried / what worked

Replaced all four sites with an unconditional `iv = iv / 100.0` (kept the
sane-band guard). All four are percent-sourced when reached:
`_resolve_pit_atm_iv` reads vol_iv (percent); the three ranker fallbacks
(sites 2/3/4) run only inside `if iv is None:` on the fundamentals percent
value. No double-conversion is possible.

## What didn't / the surprise

The sniffer was **load-bearing across the test suite**: 72 tests in 11 files
seeded IV as a *decimal* (0.22–0.30) relying on the heuristic's
decimal-tolerance. Deleting it dropped those candidates (0.28 → 0.0028 →
gated) and collapsed survivor-dependent assertions (test_ranker_transparency
21, test_strangle_ev_ranker 19, test_covered_call_ranker 16, …). Surfaced
the blast radius to the operator before churning; chose the wheel_runner-lane
full migration.

## How we fixed it

Behaviour-preserving migration: multiplied every decimal-IV stub by 100
(`0.25 → 25.0`), so post-fix the candidate sees the *same* decimal it did
before → every downstream assertion holds. Only the **2 tests that asserted
the removed behaviour** were rewritten to pin the new percent-only contract:
`test_ranker_iv_pit::test_sub_three_percent_iv_divided_unconditionally`
(2.5 → 0.025, unit-level) and
`test_audit_viii_unit_invariants::test_fundamentals_iv_percent_normalised_unconditionally`.

## Evidence

- Snapshot-neutral by **complete enumeration**: only 17 sub-3% vol_iv rows
  exist anywhere; just BG/CEG are in a snapshot universe (UNIVERSE_100), both
  on 2025-11-14 — outside every snapshot window (S27/S32/S34
  end 2024-12-31; S35 ends 2020-12-31). Fundamentals-fallback IV floor 3.4%.
  → no in-window as_of consumes a changed IV. Backtest markers re-run to
  CONFIRM byte-identical (see report).
- 11 breaker files: 329 passed after migration. Full fast suite + markers in
  the PR report. Trio diff = wheel_runner.py only (17/-15).

## Unresolved / handoff

Twin heuristic in `engine/wheel_tracker.py:1545` (`_connector_atm_iv`, MTM
path) filed as #360 — out of this PR's claimed `wheel_runner` lane, its own
follow-up. Held for operator's independent §2 second-read + sign-off; no
auto-merge.
