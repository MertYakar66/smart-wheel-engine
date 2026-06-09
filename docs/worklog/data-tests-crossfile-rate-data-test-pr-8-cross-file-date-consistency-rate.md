---
id: data-tests-crossfile-rate
title: Data-test PR-8 cross-file date consistency + rate-accessor divergence (W36/W37)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Phase-2 round-2 PR-8 (final) of the data-layer test audit. Adds W36 (cross-file vol_iv↔ohlcv last-date consistency — the un-gated _resolve_pit_atm_iv would price BSM against stale IV; data-side pin, engine gate = (E) #378) and W37 (the EV-path rate accessor data_integration.get_current_risk_free_rate returns silent 0.05 before treasury coverage vs the connector's NaN — pins the divergence, engine alignment = (E) #378). Both latent today / live under deep_history. Test-only; trio/data untouched.
surface: [tests/test_data_integrity_bloomberg.py]
---

## Goal
<!-- What we set out to do, and why. -->

Round-2 completeness critic found two latent robustness gaps that only fire under
deep_history / a staggered refresh. PR-8 lands the DATA-side pins; the engine-side
fixes are tracked as (E) #378 (not grabbed).

## What we tried
<!-- Approaches, in the order we tried them. -->

`tests/test_data_integrity_bloomberg.py`:
- **W36** `test_vol_iv_ohlcv_last_date_consistency` — for every name in both ohlcv +
  vol_iv (non-null IV), |last_ohlcv_date − last_iv_date| ≤ 5 days. Catches a refresh
  shipping IV staler than OHLCV (which `_resolve_pit_atm_iv` — no staleness gate, unlike
  the spot path — would silently price against).
- **W37** `test_data_integration_rate_before_coverage_divergence` — pins that
  `engine.data_integration.get_current_risk_free_rate` (the ranker-path rate accessor)
  returns 0.05 before treasury coverage while the connector returns NaN. The 'before'
  as_of is derived from the actual coverage start (refresh-robust).

## What worked

Both pass; integrity suite 46 passed / 1 xfailed.

## What didn't
<!-- The dead ends + WHY. -->

Initial recon said "0 names with any IV↔OHLCV gap"; the precise probe found max gap =
1 day (a name whose IV wasn't computed on the very last OHLCV bar), so the pin is ≤5
(comfortable margin), not ==0.

## How we fixed it
<!-- The approach that shipped. -->

Test-only, data-side pins. The engine-side fixes (gate IV staleness in
_resolve_pit_atm_iv like the spot path; align the rate fallback to NaN) are (E) #378 —
both latent on the current monolith (max IV gap 1d; 504-bar gate + 1994 treasury
preclude a pre-coverage tradeable), live under deep_history. Not grabbed.

## Evidence
<!-- Exact commands run, numbers. -->

Worktree off `origin/main 1ee614a` (post PR-7 merge), provider `MarketDataConnector`.

- W36: 509 common names, max |last_oh − last_iv| = **1 day**, 0 names > 5d.
- W37: `get_current_risk_free_rate(before-coverage)` = **0.05** vs connector **NaN**.
- `py -3.12 -m pytest tests/test_data_integrity_bloomberg.py -q` → **46 passed,
  1 xfailed**. `ruff` clean.

## Unresolved / handoff
<!-- What's still open. -->

- **HOLD for review.** This completes the round-2 register (W29–W37). All (T) items
  W14–W37 are now landed via PR-1..PR-8.
- **(E) #378** (IV-staleness gate + rate-fallback divergence) tracked, not grabbed.
- CAPSTONE next: land the audit doc branch (`claude/data-test-audit`: the register +
  reproducible `scripts/audit_data_tests.py`) updated to the final state — W14–W37
  landed, capability corrections C1/C2/C3 (incl. vix→R11), the (E) issues
  #369/#372/#378, the (D)/xfail trackers, and the clean realism-check result.
