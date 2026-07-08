---
id: audit-test-hygiene
title: 2026-07 repo audit — Batch C test-suite hygiene
kind: refactor
status: in-flight
terminal:
pr:
decisions: []
date: 2026-07-08
headline: Retired test_new_modules.py (folded unique coverage, dropped duplicates), removed two bare-pass no-ops, added make_gbm_ohlcv conftest helper, dropped two zero-consumer fixtures. Suite green.
surface: [tests/test_advisors.py, tests/test_new_modules.py, conftest.py, tests/test_adversarial_news.py]
---

## Goal
Batch C of the 2026-07 repo audit (D28): test-suite hygiene — remove no-op and
duplicated tests, consolidate coverage, and add a reusable OHLCV helper — with
zero net coverage loss.

## What worked
- **tests/test_adversarial_news.py** — deleted the two bare-pass no-op tests
  (`test_slow_provider_timeout`, `test_rate_limited_provider`). File retained; no
  taxonomy/manifest change.
- **Retired tests/test_new_modules.py** — folded its *unique* coverage into
  `tests/test_advisors.py` (the `TestTalebAdvisor` class + the committee
  `review_portfolio` / `post_mortem` modes) and dropped the duplicated cases.
  Deleted the file + its `TESTING.md` and `FILE_MANIFEST.md` rows in the same
  commit (both CI gates stay green).
- **conftest.py** — added a parametric `make_gbm_ohlcv(n, seed, start,
  close_only=False)` **helper function** (not a fixture; self-contained
  `default_rng`, never mutates global numpy RNG). Dropped the two zero-consumer
  fixtures `sample_iv_series` + `sample_greeks`. Left the session-autouse
  option-premium rail-neutralizer untouched.

## What didn't (two premise corrections, evidence-based)
- The plan said "drop the duplicated Taleb assertions (covered in
  test_advisors.py)". **Taleb was NOT covered there** — `test_advisors.py` had
  TestBuffett/Munger/Simons but no `TestTalebAdvisor`, and
  `git grep -l "TalebAdvisor" tests/*.py` → **only test_new_modules.py**. So
  dropping would have lost the only direct Taleb unit coverage → **folded** it
  instead.
- The plan said "fold the get_current_risk_free_rate NaN-fallback test into the
  data_integration test file". It is **already there** —
  `tests/test_data_integration.py:145-146` has the byte-identical
  `pd.isna(...)` + `fallback=0.05` assertions → folding would duplicate → **dropped**.

## How we fixed it
Verified every duplication/coverage claim with a caller grep before acting
(`normalize_ticker` → test_data_connector.py ×9; `WheelRunner`/`TickerAnalysis`
→ test_wheel_runner_coverage.py:22; risk-free fallback → test_data_integration.py).
Only removed a test where the coverage provably lived elsewhere.

## Evidence
- `pytest tests/test_advisors.py tests/test_adversarial_news.py` → 82 passed.
- `check_manifest_coverage.py` → OK; `test_testing_md_taxonomy.py` → 2 passed.
- `ruff check` on the CI-scoped changed files (tests/) → clean. (conftest.py is
  outside the CI ruff path list — its pre-existing I001 is unchanged.)
- Full `pytest -m "not backtest_regression"` → **3458 passed, 0 failed** (391.8s).

## Unresolved / handoff
- `docs/TESTED_SURFACE_MAP.md` (generated) still lists the retired
  `test_new_modules.py`; regenerate once after the D28 wave lands — do not
  hand-edit a generated file.
