---
id: ci-integration-honest
title: "Vacuous CI Integration Tests job made real (item 5b)"
kind: fix
status: shipped
terminal: X
pr:
decisions: []
date: 2026-07-02
headline: "The Integration Tests CI job ran ZERO tests behind continue-on-error since its creation (exit 5 masked, reported green); it now runs 26 real cross-boundary tests with --strict-markers and fails on exit 5, so the lane can never silently regress to vacuous"
surface:
  - .github/workflows/ci.yml
  - tests/test_portfolio_api_endpoints.py
  - tests/test_ev_engine_upgrades.py
---

## Goal

Campaign item 5b (adversarial review 2026-07-01: "CI 'Integration
Tests' job vacuous — zero `integration`-marked tests + continue-on-error").
Live-log proven: the job collected 3,459 items, deselected all,
pytest exited 5, `continue-on-error` masked it, job reported green.

## What we tried / worked

Recon censused every custom marker in tests/ (only `backtest_regression`
and `slow` were ever applied — `integration` was registered in conftest
but used nowhere) and every integration-SHAPED test. Two genuinely
cross-boundary, headless-CI-green candidates existed with zero new
infrastructure:

- `tests/test_portfolio_api_endpoints.py` (25 tests) — spins the real
  `engine_api` stdlib handler on an ephemeral loopback port and
  exercises it over actual HTTP (fixture-backed, no external services).
- `tests/test_ev_engine_upgrades.py::test_fallback_distribution_seed_is_process_independent`
  — real `subprocess.run` × 2 proving cross-process EV determinism.

## What didn't

- Marking the Theta connector tests `integration`: they auto-skip in
  CI (no Theta Terminal), so including them would let the lane go
  green-on-all-skips if the real-socket mark were ever removed. Left
  unmarked — the lane contains only tests that actually execute in CI.
- Excluding the marked tests from the Test Suite job: would remove 25
  real-socket tests from the coverage-gated matrix and risk a
  coverage-gate flip. The double-run (both lanes) is intentional and
  costs seconds.

## How we fixed it

- `.github/workflows/ci.yml`: dropped `continue-on-error`, added
  `--strict-markers`; exit 5 (zero selected) now FAILS the job — the
  anti-vacuity ratchet.
- `pytestmark = pytest.mark.integration` on the real-socket file (with
  a comment explaining the mark is load-bearing);
  `@pytest.mark.integration` on the subprocess-determinism test.
- TESTING.md CI section: the "and integration tests" claim now
  describes the real lane + the intentional double-run.

## Evidence

- Local: `pytest tests/ -m integration --strict-markers` → **26
  passed, 3,433 deselected** in 4.3 s; both files still green in the
  normal lane (36 passed); ruff clean; yml parses.
- Vacuity ratchet logic: on pre-fix main the same yml would fail
  (0 selected → exit 5 → no continue-on-error) — the job can only be
  green with a populated lane.
- CI run on the PR is the positive proof (job must select 26).

## Unresolved / handoff

- Per-file coverage floors (the review's "wheel_runner 75.3% under
  aggregate-80 green") remain open — the rescued
  `claude/rescue-2026-06-15-fixes` branch has a ci.yml design; separate
  concern from this lane.
