---
id: MP-A
title: Close S42 dossier defensive guards (Findings #1-4)
kind: fix
status: in-flight
terminal: A
pr: 275
decisions: []
date: 2026-05-30
headline: Close S42 Findings #1-4 with defensive guards in the R7-R10 path (skip malformed rows; honour explicit contracts=0); §2 invariant preserved — reviewers stay downgrade-only.
surface:
  - engine/candidate_dossier.py
  - engine/portfolio_risk_gates.py
  - engine/risk_manager.py
  - tests/test_dossier_r9_r10_audit.py
  - tests/test_portfolio_risk_gates.py
---

## Goal

S42 (PR #265) pinned four low-severity sharp edges in the dossier
reviewer's R7-R10 path with `pytest.raises(...)` — by design as a
forcing function. This card closes those forcing-function pins so
they become assertions of the *new* graceful behaviour, with the
source-edits and test-updates bundled in the same diff (the
audit-to-fix-to-updated-audit pattern).

All four findings required upstream data corruption to trigger
(missing/zero/negative strikes, explicit `contracts=0`). None are
live production bugs today; the hardening prevents silent damage if
an upstream layer ever produces malformed data.

§2 invariant preserved by construction. Fixes are defensive (skip
malformed rows; honour explicit `0`). Reviewers remain
downgrade-only. No `ev_dollars` / `ev_raw` / multiplier code edited.
CLAUDE.md §2 ("No tradeable candidate bypasses `EVEngine.evaluate`")
holds.

## What we tried

Approached as four separate defensive edits paired with assertion
updates to the existing S42 audit tests:

1. **Finding #1 (R9 sector-exposure path):** `SectorExposureManager.calculate_sector_exposures`
   in `engine/risk_manager.py` raised `KeyError` on rows missing
   `symbol` / `strike` / `contracts`.
2. **Findings #2 + #4 (R8 stress + R7 VaR paths):** rows with
   `strike <= 0` or `contracts <= 0` reached the BSM pricer via
   `check_var` / `check_stress_scenario` and pre-empted R9/R10's
   own defensive guards, making R10's try/except unreachable for
   malformed rows.
3. **Finding #3 (R9/R10 truthy fallback):** `contracts or 1` in
   `engine/candidate_dossier.py` (lines 432, 467) silently
   substituted `1` when the caller passed an explicit `contracts=0`.

## What worked

- **Defensive `.get()` calls in `SectorExposureManager`** (Finding
  #1): skip malformed rows rather than raising — matches the
  existing `.get()` + try/except pattern already in
  `check_single_name_cap`.
- **Pure-function filter `_filter_bsm_safe_positions` in
  `engine/portfolio_risk_gates.py`** (Findings #2 + #4): drops rows
  that would crash BSM (`strike <= 0`, `contracts <= 0`, missing
  `symbol`); wired into both `check_var` and `check_stress_scenario`
  before `rm.calculate_var` / `tester.run_scenario`. R8 no longer
  pre-empts R9/R10 for malformed rows; R10's defensive try/except is
  now reachable.
- **Drop `or 1` truthy fallback in the R9/R10 dossier paths**
  (Finding #3): explicit `contracts=0` now produces
  `proposed_notional=0`, caught by the existing
  `if nav > 0 and proposed_notional > 0` guard.
- **Renamed five audit tests** (F4.3 / F4.4 / F6.3 / F6.4 / F6.5) to
  reflect post-fix semantics. Same audit value; they now pin the
  graceful behaviour rather than the sharp-edge behaviour.
- **Direct unit tests for `_filter_bsm_safe_positions`** added in
  `tests/test_portfolio_risk_gates.py::TestFilterBSMSafePositions`
  (15 tests covering empty input, all-valid passthrough,
  zero/negative/None/non-numeric strike filtering, zero/negative
  contracts filtering, missing/empty-string symbol filtering,
  identity preservation, no input mutation, plus two
  boundary-of-contract pins — float contracts silently
  int-truncated, NaN strike currently passes the `> 0` check; both
  pinned as the current contract so a future tightening trips the
  test loudly).

## What didn't

- **Adding a `proposed_notional > 0` guard directly to the R7/R8
  blocks in the dossier (mirroring R9/R10's existing guard).** Less
  defensive than filtering at the *gate* boundary —
  `_filter_bsm_safe_positions` protects ALL callers of `check_var` /
  `check_stress_scenario`, not just the dossier path. Also shorter:
  one helper + two one-line calls vs four block edits.
- **Hardening `engine.option_pricer.black_scholes_price` to return
  a sentinel on `K <= 0` instead of raising.** Would change the
  global pricer contract, which downstream consumers (`StressTester`,
  `RiskManager`) currently depend on for fail-loud semantics.
  Filtering at the gate boundary preserves the loud contract for
  direct pricer users while making the dossier reviewer defensive
  against malformed inputs.
- **Marking the closed findings as `@pytest.mark.xfail` to preserve
  the pre-fix expectation.** Risks "xfail forever" rot; cleanly
  renaming + re-asserting is more honest.

## How we fixed it

Five edits in one commit (`d69ae82` pre-rebase → `8a1fce3`
post-rebase), then a second commit (`420cc71` → `a41c975`) adding
direct unit tests for the new helper:

- `engine/candidate_dossier.py` — drop `or 1` truthy fallback on
  `contracts` in the R9 and R10 paths (lines 432, 467). Explicit
  `contracts=0` now produces `proposed_notional=0`, caught by the
  existing `if nav > 0 and proposed_notional > 0` guard.
- `engine/portfolio_risk_gates.py` — add pure-function
  `_filter_bsm_safe_positions` (drops rows with `strike <= 0`,
  `contracts <= 0`, or missing `symbol`); wire into `check_var` and
  `check_stress_scenario`.
- `engine/risk_manager.py` —
  `SectorExposureManager.calculate_sector_exposures` uses defensive
  `.get()` calls; skips malformed rows rather than raising
  `KeyError`. Matches the existing pattern in `check_single_name_cap`.
- `tests/test_dossier_r9_r10_audit.py` — F4.3 / F4.4 / F6.3 / F6.4 /
  F6.5 updated from `pytest.raises(...)` pins to assert the new
  graceful verdicts. Test names refactored to reflect post-fix
  semantics.
- `tests/test_portfolio_risk_gates.py` —
  `TestFilterBSMSafePositions`: 15 direct unit tests for the new
  helper (see "What worked" above).

### S42 Findings #1-4 RESOLVED — cross-reference

| Finding | Closed by | Renamed test |
|---|---|---|
| **#1** (R9 sector path raises `KeyError` on missing keys) | defensive `.get()` calls in `engine/risk_manager.py` `SectorExposureManager.calculate_sector_exposures` | F4.3 → `test_r9_skips_held_position_missing_strike_gracefully` |
| **#2** (R8 stress crashes on `strike <= 0`) | `_filter_bsm_safe_positions` in `engine/portfolio_risk_gates.py` wired into `check_stress_scenario` | F6.3 → `test_r8_stress_filters_candidate_strike_zero_gracefully` |
| **#3** (R9/R10 silently substitute 1 for explicit `contracts=0`) | drop `or 1` truthy fallback in `engine/candidate_dossier.py` lines 432, 467 | F6.4 → `test_r9_r10_path_honors_explicit_zero_contracts_after_or_fix` |
| **#4** (R8 stress crashes on `contracts <= 0`, hiding R10) | same `_filter_bsm_safe_positions` filter, also wired into `check_var` | F6.5 → `test_r8_stress_filters_held_position_negative_strike_gracefully` |

The audit's `pytest.raises(...)` pins acted as the forcing function
they were designed to be — the hardening source edits tripped them,
and the same commit updated the assertions to pin the new graceful
behaviour. The renamed tests still pin behaviour; just the
now-correct behaviour.

## Evidence

```text
pytest tests/test_dossier_r9_r10_audit.py -v          → 32 passed in 0.70s
pytest tests/test_portfolio_risk_gates.py::TestFilterBSMSafePositions
                                                       → 15 passed in 0.63s
pytest tests/test_dossier_r9_r10_audit.py \
       tests/test_dossier_invariant.py \
       tests/test_portfolio_risk_gates.py \
       tests/test_risk_manager.py \
       tests/test_audit_invariants.py \
       tests/test_authority_hardening.py \
       tests/test_audit_viii_*.py \
       tests/test_launch_blockers.py                   → 245 passed in 27.8s
pytest tests/                                          → 2487 passed, 3
                                                         pre-existing
                                                         Windows-local
                                                         test_theta_connector
                                                         failures
ruff format --check + ruff check                       → clean on edits
```

5-ticker EV smoke (sanity check; engine math unchanged):
XOM $137.57 / JPM $124.90 / MSFT $90.97 / UNH $62.62 / AAPL $20.45 —
identical to Terminal B's verified baseline at PR #244.

### Major-Session-cycle housekeeping

- **Rebase target:** `origin/main` @ `482bc79` (post-#286).
- **Frozen-file cleanup:** `docs/USAGE_TEST_LEDGER.md` edits from the
  original commit dropped during the rebase — per
  `docs/PARALLEL_SESSIONS.md` §10 and the
  [MP cycle GO comment](https://github.com/MertYakar66/smart-wheel-engine/issues/113#issuecomment-4581380777),
  the ledger is frozen and the S42-Findings-#1-4-RESOLVED note moves
  here.
- **`CHANGELOG.md`:** retained from the original commit; Major
  Session reconciles at cycle close per the GO comment.

## Unresolved / handoff

- **None for Findings #1-4.** All four closed.
- **NaN-strike defence** is left as a separate concern. The filter
  intentionally does not handle NaN; `test_nan_strike_currently_passes`
  pins the current contract so a future hardening trips loudly.
  NaN-strike defence probably belongs in the data layer upstream of
  the gates, not inside this defensive filter.
- **Float-contracts validation** similarly deferred — contracts in
  production are always `int` (1 per `WheelTracker` emission); float
  contracts only appear via direct API calls.
- **Pattern handoff:** if a future PR adds R11 + surface findings,
  use the same audit-to-fix-to-updated-audit cycle — pin current
  behaviour with `pytest.raises(...)` first, ship the audit, then
  close the findings in a follow-up PR with the assertion updates
  bundled (PR #265 → this PR).
- **Filter scope handoff:** current filters check `strike <= 0` and
  `contracts <= 0` only. If a future PR routes a different class of
  malformed data through (NaN strike, non-numeric contracts), the
  gate filters need extending.
