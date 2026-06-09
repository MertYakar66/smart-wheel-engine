---
id: data-tests-fundamentals-sector
title: Data-test PR-3 fundamentals and sector (W19/W20/W17 + issue 372)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Phase-2 PR-3 of the data-layer test audit. Adds W19 (eqy_dvd_yld_12m band + GICS-11 set on the real file), W20 (real dividend_yield flows into the BSM carry q — controlled synthetic + real percent assertion), and W17 (characterization that R9's DEFAULT_SECTOR_MAP ignores the pulled GICS — 379/511 names collapse to 'Unknown'). Opened issue #372 (E) for the R9→GICS rewire; W17 is a passing characterization that flips when #372 lands. Test-only; trio/data untouched.
surface: [tests/test_data_integrity_bloomberg.py, tests/test_data_to_engine.py]
---

## Goal
<!-- What we set out to do, and why. -->

Third Phase-2 PR after PR-2 (#371) merged. PR-3 = the fundamentals/sector surface:
yield/GICS content validity (W19), the dividend_yield→BSM carry wire on real data
(W20), and the R9 sector-map coverage gap (W17). Characterize W17 before any R9
rewire (per review). Held for review before PR-4.

## What we tried
<!-- Approaches, in the order we tried them. -->

- `test_data_integrity_bloomberg.py`:
  - **W19** `test_fundamentals_dividend_yield_in_band` — eqy_dvd_yld_12m non-negative,
    ≤30% on the real file (NaN allowed).
  - **W19** `test_fundamentals_gics_sector_is_canonical_11` — gics_sector_name ⊆ the
    canonical GICS 11.
  - **W17** `test_r9_sector_map_ignores_pulled_gics_characterization` — many real
    names with a GICS sector get `'Unknown'` from `SectorExposureManager.get_sector`
    (>50). Passing characterization; flips when #372 wires GICS into R9.
- `test_data_to_engine.py`:
  - **W20** `test_dividend_yield_reaches_bsm_carry` — (a) real `get_fundamentals('CAG')`
    yield in percent; (b) controlled synthetic, identical except dividend_yield 0% vs
    8%, asserts the carry lowers the 25-delta short-put strike (`strike_high <
    strike_zero`). Routes through `rank_candidates_by_ev` (no §2 bypass).

## What worked

All 5 new tests pass on the bundled data; the suites stay green (51 passed).

## What didn't
<!-- The dead ends + WHY. -->

W17's coverage probe showed the gap is **material** (132/511 mapped → 379 collapse to
`'Unknown'`), so per the reviewer's pre-authorization it warranted an **(E) issue
(#372)**, not just a doc note. W20's carry DIRECTION was verified empirically first
(yld 8% → strike 80.5 < 81.0 at yld 0%) before pinning — the sane 80.5 strike also
proves the `/100` percent→decimal fired (8.0-as-800% would collapse the forward).

## How we fixed it
<!-- The approach that shipped. -->

Test-only. W17 is a passing characterization referencing #372 (the R9→GICS rewire is
(E), behind the §2 ceremony — NOT grabbed). W20 uses a controlled synthetic for the
carry logic + one real-data percent assertion; direction pinned, not magnitude.

## Evidence
<!-- Exact commands run, numbers. -->

Worktree off `origin/main 5d79ac7` (post PR-2 merge), provider `MarketDataConnector`.

- `DEFAULT_SECTOR_MAP` = 132 entries; `get_universe()` = 511 → **379 unmapped → 'Unknown'**.
- Real fundamentals: eqy_dvd_yld_12m [0.057%, 10.77%], 95 NaN, 0 neg, 0 >30;
  gics_sector_name = exactly the 11 canonical sectors (0 outside).
- W20 carry probe: yld=0 → strike 81.00 / ev 22.30; yld=8 → strike 80.50 / ev 32.24.
- `py -3.12 -m pytest tests/test_data_integrity_bloomberg.py tests/test_data_to_engine.py -m "not slow" -q`
  → **51 passed, 13 xfailed, 1 deselected**. `ruff` clean.

## Unresolved / handoff
<!-- What's still open. -->

- **HOLD for review before PR-4.** PR-4 = OHLCV/dividends hygiene (W22 depth invariant,
  W23 NaN-price two-sided, W25 epsilon-clamp, + the rate_1m pre-2001 NaN residual).
  Then PR-5 (credit, off-EV: W24).
- **#372** (E): wire `gics_sector_name` into R9 (R9 currently ignores the pulled GICS).
  Behind the §2 ceremony; W17 flips when it lands.
