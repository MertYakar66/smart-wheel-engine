---
id: Phase-1-spec
title: Phase 1 (E)-trio execution spec — turnkey file:line plan for #372/#369/#378
kind: feature
status: held
terminal: autonomous session 2026-06-23
pr:
decisions: []
date: 2026-06-23
headline: docs-only turnkey spec for the supervised (E) trio (#372 R9→GICS, #369 IV-fallback clean, #378 IV-staleness+rate) verified vs origin/main; no engine code
surface: [docs/PHASE1_E_TRIO_EXECUTION_SPEC.md]
---

## Goal

Make the supervised, EV-moving Phase 1 of the wiring campaign (`docs/WIRING_CAMPAIGN.md`)
fast and low-risk by producing a **turnkey, file:line-accurate** implementation spec for the
three (E) trio fixes — **#372** (R9 sector cap → real GICS), **#369** (extend the #363 IV gate
to the fundamentals-fallback path), **#378** (IV-staleness gate + rate-fallback divergence).
Docs only; the actual fixes are decision-layer / risk-gate changes that land under §2
supervision, not here.

## What we tried

- **Verified every line against `origin/main` @ 83eacdd, not the 2026-06-09 audit.** The audit's
  numbers had drifted: `get_current_risk_free_rate` `:323→:300`; the ranker `sector` column
  `:1777→:1822/2751/3379`; `_resolve_pit_atm_iv` still `:153`. First-hand reads + a 4-agent
  read-only verification workflow (one per fix + a test-inventory agent) cross-checked the
  surfaces and the existing characterization tests.
- **Adjudicated, did not relay.** Spot-checked the agents' new claims (the W37 test name, the
  three fallback sites, the band constants) against the bytes before citing them.

## What worked

- **#372 is lower-risk than the audit implied:** the real GICS is *already in scope* —
  `get_fundamentals` exposes `gics_sector_name` as `["sector"]` (`data_connector.py:897`) and
  `wheel_runner` already loads it (`:504 analysis.sector`); the ranker rows just ignore it and
  use `DEFAULT_SECTOR_MAP.get(...)` at `:1822/2751/3379`. And `check_sector_cap` *already*
  accepts a `sector_map=` param (`portfolio_risk_gates.py:350`) — so the fix can thread a
  GICS-built map through existing seams (minimal trio surface).
- **Pinned the exact characterization tests to flip** per fix (e.g. #372 →
  `test_r9_sector_map_ignores_pulled_gics_characterization`,
  `test_ranker_transparency.py::test_sector_column_uses_default_sector_map`; #369 →
  `test_363_gate_does_not_clean_fundamentals_iv`; #378 → W37
  `test_data_integration_rate_before_coverage_divergence`), plus the new engine-side tests #378
  needs (none exist today).

## What didn't

- The audit line numbers were stale — confirmed why the spec verifies against current `main`.
- Two #369 decision points are left to the §2 panel (NULL-band only vs also percent→decimal at
  the connector; whether to collapse the 4 duplicated `if iv>3.0: iv/=100` heuristics) — they
  change which of the two W27 tests flip, so the spec documents both rather than guessing.

## How we fixed it

Shipped `docs/PHASE1_E_TRIO_EXECUTION_SPEC.md` (per-fix finding/fix/§2-role/CEREMONY, tests to
flip + add, the §2-panel checklist, and the **#378-before-0A** ordering) + a FILE_MANIFEST row +
this worklog. Held for review; no engine code.

## Evidence

- All anchors re-verified on `origin/main` (grep + read): `risk_manager.py:1579/1742/1758`,
  `portfolio_risk_gates.py:343/372/384`, `candidate_dossier.py:483/497`,
  `wheel_runner.py:153/198/475/533-545/588-592/1822/2751/3379/1127/2462/3035`,
  `data_connector.py:125/140/227/295/317/897`, `data_integration.py:300/323/330/338/342`.
- Tests confirmed present: `test_data_integrity_bloomberg.py:494/733/758`,
  `test_data_to_engine.py:471/482`, `test_ranker_transparency.py:449`,
  `test_portfolio_risk_gates.py:187/228`.

## Unresolved / handoff

- Execution is the supervised session: land `#372 → #369 → #378`, each lane-claimed + §2-panel +
  held, **before** Phase 0A (the IV↔spot staleness coupling) and the single Phase-R re-baseline.
- The Phase 0B `xfail(strict)` scaffold `test_broad_pull_wiring_xfail.py::test_r9_sector_grouping_uses_real_gics`
  flips green when #372 lands (drop the marker).
