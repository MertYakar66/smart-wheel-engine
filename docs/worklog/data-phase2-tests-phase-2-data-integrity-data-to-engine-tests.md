---
id: data-phase2-tests
title: Phase 2 data integrity + data-to-engine tests
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-07
headline: Phase-2 of the data audit — 2 real-CSV test suites (integrity + data→engine) turning the Phase-1 findings into strong assertions; confirmed defects tracked as xfail(strict)+issue; trio byte-identical
surface: [tests/test_data_integrity_bloomberg.py, tests/test_data_to_engine.py]
---

## Goal

Phase 2 of the data+engine audit ([[data-engine-audit-phase1-2026-06-07]] /
PR #353): convert the ranked weaknesses into STRONG tests on the real
`data/bloomberg/` CSVs. Additive only; no decision-trio edits. A test that
fails on genuinely-bad data is a SUCCESS — confirmed defects are
`xfail(strict=True)` + a linked issue (CI green, defect tracked, xfail flips
red the day it's fixed).

## What we tried / what worked

Two suites, both reusing the `HAS_BLOOMBERG_DATA` skipif from
`test_data_integration.py` (NOT duplicating the synthetic-fixture
`test_data_connector.py`):

- **A — `tests/test_data_integrity_bloomberg.py`** (data contract): schema,
  OHLC positivity + the load-bearing column-rename invariant, vol_iv
  unit-consistency + sane band + zero-skew, dividends `>= -1e-9`, date
  hygiene, cross-file referential integrity with the 2026-03-23 seam
  membership split encoded structurally, seam continuity + BK→BNY, treasury
  band (allows the brief real negative T-bill prints), and the fast-CI
  fingerprint-completeness guard (durable W3 replacement).
- **B — `tests/test_data_to_engine.py`** (data→output): everything routed
  through `WheelRunner.rank_candidates_by_ev` / `rank_covered_calls_by_ev`
  (no §2 bypass), pinned to frontier 2026-06-04: output sanity (R1a finite,
  banded), cascade-tier correctness, no-silent-drops, graceful degradation
  of thin/garbage, determinism, the dividends→CC ex-div mechanism (DIS), a
  corrupt/truncated negative control via an injected fixture connector, and
  the 5-ticker smoke. Full-universe produced/dropped pin behind `slow`.

## What didn't / corrections applied (verified vs main 12645d4)

- **W3 dropped** — main's `backtests/regression/_common.py` already defines
  `connector_data_sha256()` (L177) pinning all 9 `_FILES`, recorded in both
  snapshot fingerprints (L636/L987) + drift-checked by
  `test_snapshot_data_fingerprint_matches_current`. My Phase-1 read was the
  stale primary working tree. Corrected #353 (W3→INFO) and replaced it with
  the fast-CI completeness guard here. Residual: the drift COMPARE runs on
  the slow lane, not fast CI.
- **W2 confirmed**: `get_fundamentals`/`get_credit_risk` take no `as_of`
  (data_connector.py:828/868) — xfail(strict), issue #354.
- **W6 split**: 11 long-history blue-chips truncated (xfail backfill, #355)
  vs 6 genuinely-recent names (graceful-degradation pin).
- The "delta ∈ [-1,0]" check became strike<spot + prob_assignment band —
  the ranker output carries no `delta` column.

## How we fixed it

Made every defect assertion honest (xfail(strict) for confirmed defects,
band/tolerance pins for accepted-real data like negative T-bills and the
dividend float-noise). Negative control injects a fixture connector via
`WheelRunner(data_dir=tmp_path)` so a corrupt/truncated CSV is rejected by
the REAL ranker, not a mock.

## Evidence

- A: 30 passed / 1 xfailed (the 4 NaN-price OHLCV rows, #357).
- B: 8 passed / 12 xfailed (W2 + 11 W6) fast; slow full-universe pin passes
  (511→480 produced / 31 dropped / 0 vanished, 45s).
- Trio `git diff origin/main` empty. ruff clean.

## Unresolved / handoff

Issues filed: #354 (W2 lookahead), #355 (W6 backfill — 11 names), #356 (W1
IV-heuristic, **trio lane-claimed PR — not this lane**), #357 (W10/W11 + 4
NaN rows, low-pri). The W6 xfails flip green per name as OHLCV is backfilled
(CASY via `docs/CASY_BACKFILL_SPEC.md` / #339). W2's real fix is PIT
fundamentals (data layer + callers).
