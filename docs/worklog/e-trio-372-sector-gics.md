---
id: e-trio-372
title: "#372 — R9 sector cap + ranker sector → real GICS, not DEFAULT_SECTOR_MAP"
kind: fix
status: held
terminal: supervised session 2026-06-23
pr:
decisions: []
date: 2026-06-23
headline: R9 sector cap + ranker sector column route off the static DEFAULT_SECTOR_MAP onto the connector's real gics_sector_name via a counted resolver threaded through the existing sector_map= param; EV-moving, held for §2 review
surface: [engine/risk_manager.py, engine/wheel_runner.py, engine/candidate_dossier.py, engine/wheel_tracker.py, engine/ibkr_portfolio_adapter.py, engine/portfolio_risk_gates.py]
---

## Goal

Land the first (E)-trio fix (`docs/PHASE1_E_TRIO_EXECUTION_SPEC.md` §1; audit
W17 / C2): the R9 sector cap — hard gate, soft-warn, **and** the ranker `sector`
column — was driven end-to-end by the static `DEFAULT_SECTOR_MAP` (132 names),
silently bucketing every off-list name into `"Unknown"`. Route it onto the real
`gics_sector_name` the connector already serves, never silently.

## What we tried

- **Verified the call graph against `origin/main` @ `8a5252b` first-hand + a
  read-only recon agent.** `check_sector_cap` has four runtime callers (dossier
  R9 `candidate_dossier.py:483`, tracker hard gate `wheel_tracker.py:2069`, IBKR
  adapter `ibkr_portfolio_adapter.py:913`) plus the three ranker rows
  (`wheel_runner.py:1822/2751/3379`) that bypass the gate but whose label the
  code comment **requires** to match the gate's bucket (the F6 fix).
- **Minimal-surface threading (spec-prescribed):** no gate-signature change —
  thread a per-run GICS map via the existing `sector_map=` param + a new
  optional `PortfolioContext.sector_map`.

## What worked

- **Vocabulary is already aligned:** `DEFAULT_SECTOR_MAP`'s values are canonical
  GICS-11 names, so GICS-primary + DEFAULT-fallback never desync. `resolve_sector`
  = GICS-11 primary → `DEFAULT_SECTOR_MAP` → counted `"Unknown"`;
  `build_gics_sector_map` logs the Unknown count.
- **Per-caller GICS source:** the tracker builds the map from `self.connector`;
  the adapter (connector-free, file-based) builds it from the holdings' own
  display sector — which **unifies** its display and gate sectors (removing the
  prior DEFAULT-only re-derivation); the dossier reads `ctx.sector_map` + the
  candidate's own GICS from its (now GICS-resolved) ranker row.

## What didn't

- The spec assumed flipping the W17 / acceptance tests that probe the **bare**
  `SectorExposureManager()`. The minimal-surface fix leaves the bare manager on
  `DEFAULT_SECTOR_MAP` (documented legacy fallback) and routes GICS at the call
  site — exactly the path the xfail scaffold's own docstring told us to query on
  removal. So those tests were rewritten to assert the **resolver / call-site**
  behaviour, not the bare manager; the bare-manager characterization is retired
  in favour of `test_r9_resolver_groups_by_real_gics`.

## How we fixed it

`risk_manager.py` (GICS_11 + resolve_sector + build_gics_sector_map) →
ranker rows → dossier R9 → tracker (ctx builder + hard gate) → IBKR adapter, all
via `sector_map=` / `PortfolioContext.sector_map`. Tests: dropped the #372
`xfail`, flipped the W17 + ranker pins to GICS-primary, added resolver units.

## Evidence

- `ruff check` + `ruff format --check` clean on all 10 changed files.
- Targeted suite: 264 passed / 15 xfailed (the acceptance scaffold now lives;
  the #354/#355 xfails untouched). Collateral sweep (adapter, tracker, dossier
  invariants, engine-api concentration, authority hardening): 212 passed.

## Unresolved / handoff

- Held for the §2 panel (lane-claim block names `wheel_runner.py` +
  `candidate_dossier.py`). EV-moving → its `ev_mean` shift is absorbed by the
  **single** Phase R re-baseline alongside #369 / #378 — do not re-pin here.
- Next in the trio: **#369** (extend the #363 IV gate to the fundamentals
  fallback), then **#378** (IV-staleness gate + rate divergence).
