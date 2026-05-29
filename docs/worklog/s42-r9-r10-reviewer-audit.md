---
id: S42
title: R9 + R10 reviewer audit
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** PR #255 (R9 sector_cap) and PR #262 (R10 single-name
exposure cap) expanded the dossier downgrade ruleset from R1-R8 to
R1-R10. S42 audits the two new rules systematically — beyond the
seven happy-path spot checks already in
`tests/test_dossier_invariant.py` — for the four properties an
external operator needs to trust them in production: behavioural
correctness, downgrade-only invariant, fail-closed-on-missing-data
semantics, and clean interaction with R1 / R7 / R8.

**Setup.** `SWE_DATA_PROVIDER=bloomberg` (Cowork default), read-only
against `engine/candidate_dossier.py` and
`engine/portfolio_risk_gates.py`. New file
`tests/test_dossier_r9_r10_audit.py` (32 tests organised in six
families):

1. **Family 1 — R9 fires correctly** (5 tests). Pins the D17
   locked default `_DEFAULT_MAX_SECTOR_PCT = 0.25`, multi-position
   same-sector aggregation, unknown-sector fallback.
2. **Family 2 — R10 fires correctly** (5 tests). Pins
   `_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10`, multi-holding aggregation
   on the same ticker, long-position skip (`is_short=True` filter),
   put+call short aggregation, different-ticker non-aggregation.
3. **Family 3 — Downgrade-only invariant** (6 tests). R1 negative
   EV and R1a non-finite EV both short-circuit R9 and R10; R7 and
   R8 also short-circuit subsequent rules. No soft-warn can rescue
   a blocked verdict.
4. **Family 4 — Fail-closed on missing context** (4 tests). No
   `PortfolioContext`, empty default `PortfolioContext()`, and the
   KeyError-on-missing-strike paths (findings, below).
5. **Family 5 — Cross-rule interaction** (5 tests). Pins the rule
   ordering R7 -> R8(stress) -> R8(dealer) -> R9 -> R10 and the
   short-circuit-via-`return` contract.
6. **Family 6 — Edge cases** (5 tests). Strict-`>` boundaries at
   25% sector and 10% single-name (exact cap passes), plus three
   pinned findings (below). Pinned by separate
   `test_min_proceed_ev_dollars_default_pinned` and
   `test_fixture_math_for_r9_boundary_is_exact` module-level tests.

**Status.** Done. 32/32 audit tests passing in 0.61s. Launch-blocker
subset 103/103 passing (count grown from the documented 93 since
#255/#262 added R9/R10 tests). 5-ticker EV smoke unchanged
(XOM $137.57 / JPM $124.90 / MSFT $90.97 / UNH $62.62 / AAPL $20.45).
Ruff format + check clean on the new file.

**Behaviour confirmed (no defects on the documented surface):**

- R9 strict-`>` boundary at exactly 25% sector exposure passes; only
  strictly above fires. Pinned by F6.1.
- R10 strict-`>` boundary at exactly 10% single-name exposure
  passes; only strictly above fires. Pinned by F6.2.
- R9 and R10 are downgrade-only: cannot rescue R1's `blocked` (both
  `negative_ev` and `ev_non_finite` paths). Pinned by F3.1-3.3.
- R7 / R8 short-circuit silences R9 and R10 notes — only the first
  firing rule's note appears in `review_notes`. Pinned by F3.4-3.6
  and F5.
- When all four soft-warns (R7 / R8 stress / R8 dealer / R9 / R10)
  would simultaneously fire, R7 wins by code order. The verdict is
  `review`, the reason is `portfolio_var_breach`, and only the R7
  note is recorded. Pinned by F5.5.
- Missing `PortfolioContext` (None) and default-constructed
  `PortfolioContext()` (nav=0, no holdings): R9 and R10 silent. No
  exceptions. Pinned by F4.1-4.2.
- Locked defaults `_DEFAULT_MAX_SECTOR_PCT = 0.25` and
  `_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10` are honoured (no override
  surface in the dossier path). Pinned by F1.3 and F2.1.

**Findings (sharp edges surfaced during the audit — all four
low-severity; not live production bugs since their trigger
conditions require upstream data corruption):**

1. **Finding #1 — R9 path raises `KeyError` on a held position dict
   missing `strike`.** `SectorExposureManager.calculate_sector_exposures`
   (in `engine/risk_manager.py`) does `pos["strike"]` directly with
   no `.get()` fallback. A held position dict missing the `strike`
   key crashes the dossier reviewer at R9 before R10 even runs.
   Pinned by F4.3 with `pytest.raises(KeyError)`. Production reach:
   low — `take_snapshot()` always constructs proper dicts from
   `WheelPosition`, so this only matters if a future caller
   constructs `held_option_positions` by hand.

2. **Finding #2 — R8 stress crashes first when candidate strike is
   `0`.** The dossier code path runs R8 stress BEFORE R9/R10. R8
   stress -> `check_stress_scenario` -> `StressTester.run_scenario`
   -> `black_scholes_price` validates `K > 0` and raises
   `ValueError` for strike <= 0. R9/R10's `if nav > 0 and
   proposed_notional > 0` guard is structurally unreachable when
   the candidate has strike=0. Pinned by F6.3 with
   `pytest.raises(ValueError)`. Production reach: low — option
   chains never carry zero strikes.

3. **Finding #3 — `contracts=0` is silently coerced to 1 in the
   R9/R10 path via `or 1` truthy fallback.**
   `engine/candidate_dossier.py` lines 432 and 467 read::

       contracts = int(ev_row.get("contracts", 1) or 1)

   The `or 1` was intended to handle missing keys but also coerces
   an explicit `contracts=0` to `1` because `0 or 1 == 1`. A
   degenerate `contracts=0` candidate is therefore sized as 1
   contract for the cap check. Pinned by F6.4. Production reach:
   low — `WheelTracker` emits `contracts=1`. Hardening: replace
   with `int(c) if (c := ev_row.get("contracts")) is not None else 1`.

4. **Finding #4 — R10's defensive try/except is structurally
   unreachable via the dossier path when R8 stress runs first on
   the same malformed held position.** `check_single_name_cap`
   has a defensive `try ... except (TypeError, ValueError):
   continue` for malformed rows (e.g. negative strikes from
   corrupted data). But on the dossier path R8 stress runs first
   and crashes the same row via the BSM pricing call (same root as
   Finding #2 but on the held-position side). R10's defensive code
   is exercised only by direct `check_single_name_cap` unit tests.
   Pinned by F6.5 with `pytest.raises(ValueError)`. Production
   reach: low — same reasoning as Finding #2.

**Follow-ups (none filed as separate PRs — see rationale).**
All four findings are low-severity defensive issues, not live
production bugs. Their trigger conditions require upstream data
corruption (missing/zero/negative strikes, contracts=0) that
`WheelTracker` and `MarketDataConnector` do not produce today.
Hardening them would be a small follow-up PR that:

- Adds a `proposed_notional > 0` guard to R8 stress + R7 VaR (the
  same guard R9 + R10 already have).
- Changes `pos["strike"]` -> `pos.get("strike", 0.0)` in
  `SectorExposureManager.calculate_sector_exposures`.
- Replaces `or 1` truthy fallback with explicit None handling in
  the R9/R10 path.
- Updates the F4.3 / F6.3 / F6.5 audit tests from
  `pytest.raises(...)` to the new graceful behaviour.

None of these are decision-layer changes (§2 untouched). Not
queued — surface area is small enough that an audit-prompted
hardening PR is the natural next move if the user wants it. The
audit tests will trip when the behaviour changes, so they form a
forcing function rather than a passive note.

**No engine math changed.** No `ev_dollars` / `ev_raw` /
multiplier code edited. The audit is read-only against §2.

PR: pending push on `claude/usage-test-s42-r9-r10-audit`.

---
