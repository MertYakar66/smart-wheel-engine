---
id: safety-perimeter-audit-fixes
title: Non-finite EV authority guard + dossier held-book symbol schema (audit C-2/C-3)
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-07-13
headline: Closed two safety-perimeter holes surfaced by the 2026-07-13 engine audit — a NaN/inf EV that could open a real position, and a ticker-vs-symbol key mismatch that made the dossier D17 gates silently drop the operator's held book.
surface:
  - engine/wheel_tracker.py
  - engine_api.py
---

## Goal
Close the two "Finding 1" safety holes from `docs/audits/ENGINE_AUDIT_2026-07-13.md`
(the Fable-5 audit), both in code paths where the docs claimed closure:

- **C-2 — non-finite EV opens a position.** The production tracker's
  EV-authority gate (`WheelTracker.issue_ev_authority_token` /
  `_consume_ev_authority_token`) refuses non-positive EV with a bare
  `ev_dollars <= 0`. `NaN` and `+inf` both pass that test (`nan <= 0` and
  `inf <= 0` are False), so a garbage EV row could be issued a valid
  authority token and open a real position. The dossier reviewer already
  had this guard (R1a, `math.isfinite`); the tracker's own gate never got it.
- **C-3 — dossier held book silently dropped.** `_build_portfolio_context_from_params`
  (the `/api/tv/dossier` D17 wire) emitted held-put dicts keyed `"ticker"`,
  but every consumer (`engine/portfolio_risk_gates.py`,
  `engine/stress_testing.py`, `engine/risk_manager.py`) reads `pos["symbol"]`
  (and `pos["is_short"]`, defaulting to False for the single-name gate). So
  R7/R8/R9/R10 never saw the operator's held positions — the soft-warns
  could not fire.

## What worked
Independent code-level reproduction confirmed both findings before touching
anything: the two `<= 0` sites at `wheel_tracker.py:448/:594`, and the
producer/consumer key split (producer `"ticker"` at `engine_api.py`, every
consumer `pos["symbol"]`). The two existing D17-wire tests never crossed the
producer→gate seam (one checks the producer dict shape, the other hand-builds
a `PortfolioContext` with the correct `symbol` key), which is exactly why the
bug survived a green suite.

## What didn't
A bare `ticker`→`symbol` rename is insufficient: the greeks-based gates
(R7 VaR, R8 stress, R9 sector) also hard-require `pos["dte"]` and `pos["iv"]`,
and R10's single-name gate skips any position whose `is_short` is not truthy
(its default is `False`). So the honest fix is to emit the full canonical
held-position schema (`symbol, option_type, strike, dte, iv, contracts,
is_short`), mirroring `engine/wheel_tracker.py` `take_snapshot`.

## How we fixed it
- `engine/wheel_tracker.py`: added `import math`; inserted an
  R1a-equivalent `math.isfinite` refusal (distinct reason `ev_non_finite`)
  BEFORE the `<= 0` check at BOTH the issue and consume gates. Tightens the
  perimeter only — cannot rescue any trade.
- `engine_api.py`: `_build_portfolio_context_from_params` now emits the
  canonical schema — `symbol` (not `ticker`), derived `dte`, `is_short=True`,
  and a documented placeholder `iv` (`_HELD_PUT_PLACEHOLDER_IV = 0.30`,
  since the `puts_held` CSV can't carry per-name IV). The placeholder is safe
  here because R7-R10 are downgrade-only soft-warns: an approximate IV can
  only add caution, never rescue a bad trade. **Judgment call flagged for
  trio/API sign-off** per the audit.
- Tests: added `TestTrackerEVAuthorityNonFinite` (issue + consume reject
  NaN/±inf, distinct reason, finite-positive still works) and
  `TestHeldBookIsConsumableByRiskGates` (crosses the producer→gate seam:
  feeds the API-produced positions to R10 and asserts the $36k held book is
  seen). Updated the two prior D17-wire tests that were asserting the buggy
  `"ticker"` schema.

## Evidence
- Targeted: `pytest tests/test_ev_non_finite_defense.py tests/test_tv_dossier_d17_wire.py tests/test_portfolio_risk_gates.py -q` → **116 passed**.
- Full suite: `pytest tests/ -q --continue-on-collection-errors` → see commit
  (7 pre-existing collection errors are missing-dep only: `hypothesis`,
  `yfinance`, theta/external-network connectors; none in the 4 changed files).

## Unresolved / handoff
- The placeholder held-put IV (0.30) is a stopgap; a richer operator
  integration should supply real per-name IV to the dossier endpoint.
- This closes Finding 1 (C-2/C-3) only. The audit's larger items —
  survivorship-universe backtests, the EV-assembly optimisms (Finding 2),
  and the deployment-doc recompute (Finding 4) — remain open. See
  `docs/audits/ENGINE_AUDIT_2026-07-13.md` §5 for the recommended order.
