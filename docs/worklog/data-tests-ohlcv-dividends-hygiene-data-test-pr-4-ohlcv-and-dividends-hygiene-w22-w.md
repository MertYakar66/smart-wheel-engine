---
id: data-tests-ohlcv-dividends-hygiene
title: Data-test PR-4 OHLCV and dividends hygiene (W22/W23/W25 + rate_1m)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Phase-2 PR-4 of the data-layer test audit. Adds W22 (per-name OHLCV ≥504 depth invariant — a new <504 name outside the known thin set fails, vs the per-name W6 xfail that only covers 11 and skips off-frontier), W23 (two-sided NaN-price pin catching the count GROWING, complementing the one-directional xfail), W25 (dividend epsilon-negatives stay immaterial through get_next_dividend), and the W10 rate_1m residual (coverage gap vs rate_3m + as_of-before-coverage→NaN). Test-only; trio/data untouched.
surface: [tests/test_data_integrity_bloomberg.py, tests/test_data_to_engine.py]
---

## Goal
<!-- What we set out to do, and why. -->

Fourth Phase-2 PR. PR-4 = OHLCV/dividends hygiene — the lower-severity (T) items
that harden the integrity suite's one-directional / slack pins, plus the stranded
W10 rate_1m residual. Held for review before PR-5 (credit).

## What we tried
<!-- Approaches, in the order we tried them. -->

- `test_data_integrity_bloomberg.py`:
  - **W22** `test_ohlcv_per_name_depth_invariant` — {names <504 bars} ⊆ a pinned
    KNOWN_THIN set (17 names). A NEW silent truncation fails.
  - **W23** `test_ohlcv_nan_price_rows_are_the_known_four` — pins the exact 4
    NaN-price (ticker,date) keys; catches the count GROWING (the xfail only flips
    on the fix → 0).
  - **W10** `test_treasury_rate_1m_coverage_gap_and_nan_before` — rate_1m starts
    later than rate_3m (the documented gap) + an as_of before coverage → NaN (the
    safety property), band-checked.
- `test_data_to_engine.py`:
  - **W25** `test_dividend_epsilon_negative_is_immaterial` — synthetic -2.4e-14
    dividend → `get_next_dividend` returns it ≥ -1e-9 (immaterial, no spurious
    negative carry). Producer clamp = (D) #357.

## What worked

All 4 new tests pass; suites stay green (55 passed).

## What didn't
<!-- The dead ends + WHY. -->

Self-review caught a brittleness in the first rate_1m cut: `assert first >= 2001-01-01`
asserts an ABSENCE (rate_1m has no pre-2001 data) — a legitimate future backfill of
rate_1m to pre-2001 would FALSE-FAIL it. Reworked to pin the GAP relative to rate_3m
(`first_1m > first_3m`) and derive the "before coverage" as_of from the actual start
(`first_1m - 30d`), so the safety property (NaN-not-0) is pinned, not a hard date.

## How we fixed it
<!-- The approach that shipped. -->

Test-only. W22/W23 are two-sided pins (catch growth/new defects, complement the
existing one-directional W6/NaN xfails). W25 characterizes the epsilon is immaterial
through the connector (the producer clamp is (D) #357, not grabbed). rate_1m pins the
safety property robustly.

## Evidence
<!-- Exact commands run, numbers. -->

Worktree off `origin/main aafde55` (post PR-3 merge), provider `MarketDataConnector`.

- 17 names <504 bars (== KNOWN_THIN); NaN-price rows = BIIB 2020-11-06/2023-06-09 +
  TPL 2019-05-16/2019-07-09; get_next_dividend(-2.4e-14) → -2.4e-14 (≥ -1e-9);
  rate_1m first non-null 2001-07-31 (vs rate_3m 1994), band [-0.173, 5.627],
  as_of=1999 → NaN.
- `py -3.12 -m pytest tests/test_data_integrity_bloomberg.py tests/test_data_to_engine.py -m "not slow" -q`
  → **55 passed, 13 xfailed, 1 deselected**. `ruff` clean.

## Unresolved / handoff
<!-- What's still open. -->

- **HOLD for review before PR-5** (credit, off-EV: W24 sp_rating ladder suffix-strip +
  Altman-Z plausibility). After PR-5 the register W14–W27 (T) items are all landed;
  W28 (D) stays tracked.
- Producer clamp for dividend epsilon-negatives = (D) #357 (not grabbed).
