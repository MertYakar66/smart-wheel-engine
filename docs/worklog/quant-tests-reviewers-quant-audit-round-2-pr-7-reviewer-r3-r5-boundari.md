---
id: quant-tests-reviewers
title: "Quant audit round 2 PR-7: reviewer R3/R5 boundaries + EventGate earliest-hit (W65-W67)"
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Pins the EXACT §2 downgrade-only boundaries — R5 inclusive ev>=threshold (==→proceed), R3 strict diff>tol (==→not-skip) + engine_spot>0 guard, and EventGate earliest-in-window hit across mixed wildcard/ticker events; closes the planned 7-PR round
surface: [tests, quant]
---

## Goal
Quant audit round 2, PR-7 (last planned) — the §2 downgrade-only reviewer rules
(`candidate_dossier.py` EnginePhaseReviewer) + the event lockout (`event_gate.py`).
Pin the EXACT decision boundaries (a stealth `>=`↔`>` flip would silently re-route
candidates). Tests-only; §2 asserted, never weakened.

## What we tried
Probed R3/R5 via a directly-constructed CandidateDossier + EnginePhaseReviewer
(explicit min_proceed_ev for a deterministic boundary) and EventGate.is_blocked, on
`a2eea4e`, before writing.

## What worked
- **W65 R5 inclusive threshold:** ev == min_proceed_ev -> proceed; ev just below ->
  review. A flip to strict `>` would demote at-threshold candidates silently.
- **W66 R3 strictness + guard:** diff > tol -> skip/spot_price_mismatch; diff == tol
  -> NOT skipped (strict `>`); engine_spot <= 0 -> the `engine_spot>0` guard skips
  the spot check entirely (candidate continues to R5).
- **W67 EventGate earliest-hit:** a wildcard macro (earlier) beats a ticker-specific
  earnings (later) — `is_blocked` sorts hits by date and reports the earliest,
  regardless of wildcard vs specific; empty window -> not blocked.

## What didn't
- All clean (the recon was accurate here). The cp1252 console mangles the ± in the
  reason string, so the test asserts on substrings ("fomc", the ISO date), not ±.

## How we fixed it
`tests/test_reviewer_eventgate_invariants.py` (new): 7 tests, all passing. ruff-clean.

## Evidence
- Probe (py -3.12) on `a2eea4e`: R5 10.0->proceed / 9.99->review; R3 5%->skip,
  2%(==tol)->proceed, spot0->proceed; EventGate -> fomc@2026-06-12 (earliest).
  Touches: test file + fragment + manifest row. No trio/data edits.

## Unresolved / handoff
- **Round-2 (T) plan COMPLETE** (PR-1..7, W38-W67). Deferred lower-value items: R6
  at-the-put-wall boundary + sector-cap nav=0 + R9/R10 asymmetry (reviewer setup,
  LOW/INFO); the dormant vol-surface (SVI butterfly-no-arb + implied_vol nonneg,
  D9) + the 4th (E) `SplineVolSurface` silent-0.20 vs D9; the deferred pricing items
  (assignment-fee, payoff anchor, omega cap, prob_touch). Optional follow-up pass.
- **3 (E) issues open** (#382 realized_vol _log, #384 t-copula df, #386 HMM
  non-finite) — each with a flip-on-fix xfail, awaiting the §2 ceremony.
