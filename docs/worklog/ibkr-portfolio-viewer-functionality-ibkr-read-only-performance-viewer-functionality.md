---
id: ibkr-portfolio-viewer-functionality
title: IBKR read-only performance viewer — functionality round (D24 adapter + D26 endpoints + dashboard wiring)
kind: feature
status: in-flight
terminal:
pr:
decisions: [D24, D26]
date: 2026-06-06
headline: Wired /portfolio to real data via a point-in-time IBKR snapshot → ibkr_portfolio_adapter (outside the trio) → six read-only /api/portfolio/* endpoints → the existing Next.js viewer; observational only, mock kept as typed fallback.
surface:
  - engine/ibkr_portfolio_adapter.py
  - engine_api.py
  - dashboard/src/app/(terminal)/portfolio/page.tsx
  - dashboard/src/components/portfolio/use-portfolio-data.ts
  - dashboard/src/app/api/portfolio/[sub]/route.ts
  - tests/test_ibkr_portfolio_adapter.py
  - tests/test_portfolio_api_endpoints.py
  - tests/fixtures/ibkr/portfolio_snapshot.json
---

## Goal

Turn the approved `/portfolio` aesthetics page (mock-only) into a live read-only
performance viewer fed by real engine data, per design-doc D26 — without
touching the CI-gated decision trio and without ever producing a tradeable
verdict (CLAUDE.md §2/§3; finding I1: `ev_dollars` ≠ realized P&L).

## What we tried

Mapped the four engine analytics modules + `engine_api` routing + the dashboard
consumption (parallel readers) to get exact signatures, then built the pipeline:

```
data_processed/ibkr/{portfolio_snapshot,portfolio_history,wheel_ledger}.json
  → engine/ibkr_portfolio_adapter.py  (outside trio; reuses portfolio_tracker /
       wheel_tracker / performance_metrics / portfolio_risk_gates)
  → engine_api.py  GET /api/portfolio/{summary,positions,returns,income,risk,history}
  → dashboard /api/portfolio/[sub] proxy → usePortfolioData hook → existing components
```

Three on-disk artifacts (design-doc §6.5): the §2.2 point-in-time snapshot
(account + per-instrument positions), an accumulated monthly equity series, and a
closed-trade ledger. Numbers reproduce design-doc Appendix A + `mock.ts` so the
live view matches the approved aesthetics.

## What worked

- **Adapter as the whole observational layer.** It builds the engine
  `PortfolioContext` (gate-shape `held_option_positions` with `symbol`+`is_short`,
  not the `ticker`-keyed shape the dossier param-builder uses) and every viewer
  payload, reusing the existing analytics rather than reimplementing them.
- **Genuine reuse, validated to match the mock:** `portfolio_tracker.get_returns`
  reproduces 1M/3M/YTD/1Y to <0.0004; `wheel_tracker.get_performance_summary`
  yields win-rate 0.68; `performance_metrics` gives drawdown/Sharpe; the
  `portfolio_risk_gates` R9/R10 fire on the adapter-built context (the D24 payoff).
- **Universe discipline + FX:** CNQ/ENB (TSX) are exposure-only (counted in
  denominators, never rankable); CAD normalized to USD via the snapshot `fx_rates`.
- Dashboard components made prop-driven with `mock.ts` as a typed per-slice
  fallback + a live/mock indicator; look preserved.

## What didn't

- `portfolio_tracker.get_returns` "all_time" anchors to the curve start, not
  inception — so All-time return came out as 0.129 (= 1Y) instead of 0.445.
  Fixed by anchoring All to `inception_capital` and 1D/1W to the account deltas
  (snapshot-delta method, design-doc §6.4); 1M/3M/YTD/1Y still come from TWR.
- `next build` (eslint-config-next 16.1.6) errored on the inherited aesthetics
  code: a render-time `Th` component in `holdings-table` (hoisted to module
  scope) and `set-state-in-effect` on the hydration-safe date format (one-shot,
  justified suppression).
- **A real IBKR pull (handed over mid-task) exposed three gaps the demo fixture
  hid:** (1) the live snapshot carries JSON `null` on MCP-non-derivable fields
  (day/week deltas, realized-YTD) → `float(None)` crashed `/summary`; (2) a
  snapshot-only drop (no ledger/history yet) made `/summary` 503 instead of
  showing the real balance sheet; (3) CLS/AMD short puts had been **assigned to
  stock** (CLS 500 sh ≈ 131% NAV on margin) and the CSP-only breach logic left
  the book's biggest concentration un-flagged.

## How we fixed it

Shipped the adapter + endpoints + wiring above. `data_processed/ibkr/` is the
gitignored runtime location; a frozen demo copy lives in `tests/fixtures/ibkr/`
and the directory is overridable via `SWE_IBKR_DATA_DIR` (tests + fresh-clone
demo point there). FILE_MANIFEST + DECISIONS (D24/D25/D26) updated.

Live-book robustness (none of these change the demo's rendered numbers):
null-safe numeric reads (`_num` / `_opt_num`) so null fields surface as `null`
("—" in the UI), not a misleading 0 or a crash; `/summary` loads the ledger
optionally so a snapshot-only drop still serves the real balance sheet;
`breach` now flags an **assigned** name over the single-name cap (covered-call
stock still excluded), and such names enter the single-name meter — so the
assigned-CLS 131% concentration shows. Pinned by two new tests
(`test_assigned_stock_breaches_and_surfaces`, `test_null_kpis_*`).

## Evidence

- `pytest tests/test_ibkr_portfolio_adapter.py tests/test_portfolio_api_endpoints.py` → 39 passed.
- `npx tsc --noEmit` clean; `npm run lint` 0 errors; `npm run build` exit 0
  (`/portfolio` + `/api/portfolio/[sub]` in the route map).
- `scripts/check_manifest_coverage.py` → OK; `ruff check` / `ruff format --check` clean.
- Live HTTP smoke (engine on a temp port): all six endpoints 200 with the
  approved numbers (netLiq 144507, win-rate 0.68, premium30d 12400, YTD −8593,
  All +44507, Semis sector 312% / R9 breach, CLS/MU/AMD/MRVL R10 breach).

## Unresolved / handoff

- D24 Track A (thread the snapshot `PortfolioContext` into
  `build_candidate_dossiers` so R7–R11 arm on every live scan) is deferred — it
  touches the dossier path and lands separately.
- Conversational ask-bar (D26 §6.2 Phase B) is still a visual affordance only.
- Since-inception TWR awaits the IBKR Flex `CashTransactions` ingest (§6.4).
