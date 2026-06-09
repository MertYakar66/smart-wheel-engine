---
id: data-tests-iv-surface
title: Data-test PR-1 IV surface — W14/W18/W21/W26 + W27 characterization
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Phase-2 PR-1 of the data-layer test audit (docs/DATA_TEST_AUDIT_2026-06-09.md). Adds 7 real-data IV-surface assertions — the #363 served-IV band gate on the bundled CONNECTOR read (W14, the marquee gap, previously synthetic-only), the ranker iv == as-of PIT IV from the real connector (W18), realized-vol positivity/finiteness (W21), the authoritative IV-band constants (W26), and a passing characterization that #363 does NOT clean the fundamentals-fallback IV path (W27, connector fix tracked as (E) in #369). Test-only; trio/data untouched.
surface: [tests/test_data_integrity_bloomberg.py, tests/test_data_to_engine.py]
---

## Goal
<!-- What we set out to do, and why. -->

First Phase-2 PR after the 2026-06-09 register was greenlit. PR-1 = the IV surface
(highest blast radius — IV feeds every premium/EV — and it tests our own #363 gate).
Lock the IV gaps the register surfaced, all as real-data assertions through the
connector. Held for review before PR-2.

## What we tried
<!-- Approaches, in the order we tried them. -->

Added 7 tests across the two existing real-CSV suites (no new files):

- `test_data_integrity_bloomberg.py` (real-data integrity):
  - **W14** `test_served_vol_iv_band_via_connector` — `MarketDataConnector()._load('vol_iv')`
    has every non-null IV in `(3.0, 10000]` (the served band, post-#363).
  - **W14** `test_served_vol_iv_gate_removes_raw_sub3` — the GATE, not the file, removes
    the sub-3.0 garbage (raw still carries it; served has 0). Robust to a clean refresh.
  - **W21** `test_vol_iv_realized_vol_columns_positive_finite` — `volatility_30d/60d/90d/260d`
    positive + finite on the real file (they feed F4 RV-widening).
  - **W26** `test_connector_iv_band_constants_are_authoritative` — pins
    `_IV_LOW_FLOOR==3.0` / `_DEEP_IV_SENTINEL_FLOOR==10000.0` as the authoritative
    Bloomberg-served rule (distinct from the Theta-only `utils.data_validation`).
- `test_data_to_engine.py` (data→engine, no §2 bypass):
  - **W18** `test_ranker_iv_equals_real_pit_iv` — `rank(...).iv` for AAPL == the as-of PIT
    IV computed independently from the bundled file (`mean(put,call)` of the last
    `get_iv_history(end_date=FRONTIER)` row, `/100`). Tied to FRONTIER.
  - **W27a** `test_fundamentals_fallback_iv_input_is_percent` — the fallback IV input
    (`get_fundamentals['implied_vol_atm']`) is percent on the real file.
  - **W27b** `test_363_gate_does_not_clean_fundamentals_iv` — the same 2.0 the vol_iv gate
    NULLs survives uncleaned through `get_fundamentals` (the inline heuristic is the only
    normaliser on that path).

## What worked

All 7 pass on the bundled data; the existing suites stay green.

## What didn't
<!-- The dead ends + WHY. -->

n/a — straight implementation off the reviewed register.

## How we fixed it
<!-- The approach that shipped. -->

Test-only. W14 deliberately reads through the **connector**, not the raw CSV (the
reviewer's explicit ask) — the raw-file band test already exists and tolerates the
sub-3.0 rows; the gap was that nothing asserted the SERVED read. W27 is a **passing
characterization** of current behaviour (the #363 gate has a fundamentals-fallback
hole); the connector-side fix is an (E) engine change tracked in **#369**, behind the
§2 ceremony — not grabbed here.

## Evidence
<!-- Exact commands run, numbers. -->

Worktree `swe-dt-pr1` off `origin/main d0cdcde`, provider `MarketDataConnector`.

- `py -3.12 -m pytest tests/test_data_integrity_bloomberg.py tests/test_data_to_engine.py -m "not slow" -q`
  → **45 passed, 13 xfailed, 1 deselected** (the 13 xfails = pre-existing W2 PIT + 11 W6
  backfill + NaN-price; the deselect = the slow full-universe split).
- `ruff check` + `ruff format --check` on both files → clean.
- Probe basis (`scripts/audit_data_tests.py`): served put IV min 3.127, 0 cells ≤3.0,
  17 raw sub-3.0 rows NULLed; realized-vol columns all positive; AAPL PIT IV resolves.

## Unresolved / handoff
<!-- What's still open. -->

- **HOLD for review before PR-2.** PR-2 = W15 (real +EV / −EV sign controls tied to the
  FRONTIER fixture) + W16 (real earnings→event-lockout wire-in). Then PR-3 (fundamentals/
  sector, incl. W17 DEFAULT_SECTOR_MAP coverage — characterize before any R9 rewire),
  PR-4 (OHLCV/dividends hygiene + the rate_1m residual), PR-5 (credit, off-EV).
- **#369** (E): connector-side clean of the fundamentals-fallback IV path. Do NOT grab
  in a test PR — needs the §2 lane-claim ceremony.
