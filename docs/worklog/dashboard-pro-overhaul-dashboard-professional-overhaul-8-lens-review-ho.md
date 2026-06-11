---
id: dashboard-pro-overhaul
title: Dashboard professional overhaul - 8-lens review, honesty fixes, pro tracker surfaces
kind: feature
status: in-flight
terminal: Dashboard
pr:
decisions: []
date: 2026-06-10
headline: Period returns were misstated ~10pp (YTD -12.19% shown vs -2.17% true); every fabricated display killed; engine's served-but-dropped risk verdicts now render
surface: [dashboard, engine_api.py, engine/ibkr_portfolio_adapter.py]
---

## Goal
Make the dashboard a professional daily driver: review every surface, fix what
is wrong, surface what the engine already serves, expand cockpit/terminal/news,
and verify the wiring to the engine — without touching the decision trio or
weakening any §2 invariant.

## What we tried
1. 8-lens parallel review workflow (wiring, cockpit, terminal, news, portfolio,
   quality, §2-safety, live-runtime probes) over the swe-dash worktree + the
   live rig (:3000/:8787) → 75+ findings, heavily cross-confirmed.
2. Wave 1 (orchestrator): engine_api.py + /api/engine proxy honesty fixes.
3. Wave 2: four parallel implementation agents on disjoint file-ownership sets
   (portfolio / cockpit / terminal / news), each required to re-verify every
   finding against source before editing.
4. Wave 3 (orchestrator): manifest + runbook sync, full verification gauntlet.

## What worked
- The disjoint-ownership parallel implementation: 52 modified + 9 new files,
  zero cross-agent conflicts, tsc/eslint/pytest clean on first integration.
- Requiring implementers to re-verify findings: all 12 portfolio findings were
  re-confirmed against source/live payloads before any edit (0 hallucinated).
- The findings-as-files handoff (per-area JSON dumped to temp, read by each
  implementer) kept prompts small and evidence verbatim.

## What didn't
- First implementation run died wholesale on the session token limit after
  ~3h: 4 agents x 147 tool calls, ZERO file edits landed (all were still in
  their read/verify phase). Lesson: the read-first discipline made the retry
  free (clean tree), but parallel heavy agents burn limit fast — relaunch via
  the persisted workflow scriptPath worked immediately after reset.
- The review's suggested fix location for the TWR bug (portfolio_tracker.py
  _calculate_twr) was outside the viewer's ownership; the fix landed in the
  adapter's returns_view instead, taking the tracker replay off the viewer
  path entirely.

## How we fixed it
Six commits on claude/dashboard-pro-overhaul (40b0a83..bec65db):
1. wave-1 wiring honesty (status data_frontier/provider; regime real term
   structure, fabricated confidence/trendScore removed; candidates
   targetDelta + modeled expiration; concentration + dealer_positioning proxy
   actions; URL-encoding; ungated /api/execute + fabricated impact route
   DELETED).
2. portfolio: date-anchored period returns (YTD -12.19% -> -2.17%; pct/usd
   share one anchor; 1D/1W null-honest), engine R7-R10 gates board, real
   Income view from the never-fetched /income, Sharpe/Sortino/MaxDD +
   drawdown tab, margin panel, leg strike/expiry/DTE/moneyness, frozen-SPY
   benchmark cut at last real move, premium month dedup.
3. cockpit: as_of from data_frontier (was hardcoded 2026-03-20 labeled
   "freshest"), authoritative dossier verdicts with live nav/holdings
   attached, CVaR5 sign truth, live-NAV concentration + armed-caps check,
   VIX term-structure banner + unknown-state honesty, sort/filter/column
   chooser, AbortController race fix, copy-trade-plan ticket.
4. terminal: ALL fabricated data removed (placeholder index/futures tape,
   fake macro stats, fake AI-agent panel, Delta 0.000, Conf 70%); live-book
   panel with margin warning, dealer-positioning + ticker workbench panels,
   honest EOD tape + FEED badge, per-held-name earnings, poll discipline,
   un-latching error boundaries, truthful command palette, watchlist route
   2->1 engine calls/ticker.
5. news: ingestion actually scheduled (instrumentation.ts + node-cron,
   verified live - stories land), universe-validated entities, clustering
   orphan fix, calendar/feed/top honesty, engine EOD price fallback,
   (main) nav -> decision surfaces.
6. docs: FILE_MANIFEST rows + DASHBOARD_TERMINAL.md §9 truth-sync.

## Evidence
- pytest: 70/70 (adapter/import/portfolio-API/launch-blockers); ruff clean.
- tsc --noEmit: 0 errors; eslint 0 errors (12 pre-existing warnings in
  untouched files); next build: passes, deleted routes gone from route table.
- Live adapter payloads post-restart: YTD {pct:-0.0217, usd:-3383}; 1M
  {+0.0158, +2372} sign-consistent; 1D/1W {null, null}; /risk gates + 5-row
  all-exposure concentration (CLS 123% visible); legs carry dte/moneyness.
- Headless-Chrome DOM dumps (cockpit/terminal/portfolio): 0 hits for
  '5,234.18'/'Conf: 70'/'Trend +0.00'/'Delta 0.000'/VRAM; 0 $NaN/undefined%;
  new surfaces present (Risk gates, Income, Sharpe, Target Δ, frontier date).
- check_manifest_coverage: OK (715 entries, 0 uncovered, 0 dangling).

## Unresolved / handoff
- ~~scripts/dashboard_refresh.py _sync_curve SPY carry-forward~~ **CLOSED
  2026-06-11** (follow-up PR): _sync_curve now derives each point from real
  SPY closes (Gateway Mode 2, or agent-time connector closes via
  `spy-repair --closes` Mode 1) and heals frozen tails; canonical history
  repaired (3 points re-derived; benchmark had hidden a ~4% SPY decline).
- ~~Funnel "ranked == shown by construction"~~ **CLOSED 2026-06-11**
  (follow-up PR): /api/candidates serializes frame.attrs["drops_summary"];
  the cockpit funnel renders scanned -> passed gates -> top-N with the
  per-gate breakdown.
- Engine earnings CSV frontier is 2026-01-29 -> held-name earnings panels are
  honest but empty until the Terminal-gated data pull lands (batched into the
  data-queue Terminal session; NOT closable from the Dashboard session).
- Dealer panels render their empty state on Bloomberg (no option chains) -
  they light up when a chain-bearing provider lands.
- 3 test pins updated because they froze the defective returns behavior
  (test_period_returns, test_returns_shape, test_adapter_returns_view_null_safe)
  - flagged for reviewer eyes in the portfolio commit body.
