---
id: wire-r9-r10-concentration-preview
title: Wire R9/R10 concentration caps onto a live operator surface (/api/concentration_preview)
kind: feature
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-07
headline: New GET /api/concentration_preview makes the armed R9/R10 caps fire on an operator path — closes the "zero live callers" dormancy gap.
surface: [engine_api.py]
---

## Goal
The D17 concentration caps — **R9 sector (25% NAV)** and **R10 single-name
(10% NAV)** — are implemented, decoupled into `enforce_*` flags, armed by
`engine.wheel_runner.make_live_book_tracker`, and exercised by
`consume_into_live_book` (the §2-safe one-call production wire shipped in #343).
But a survey + grep confirmed those caps had **zero callers outside `tests/`**:
no `engine_api` / dashboard / scripts path invoked the armed wire, so the caps
protected nothing in production ("documented protection ≠ active protection",
`PROJECT_STATE.md` §3; the I3-A dormancy trap). Goal: give the armed caps a
real operator surface so they fire on a live path, without touching the
decision trio or the §2 invariant.

## What we tried
Considered three surfaces for the wire:
1. **Dashboard API route** (`dashboard/src/app/api/...`) — rejected: the
   dashboard/portfolio layer is owned by another terminal this cycle.
2. **A `scripts/` CLI** — viable and lowest-blast-radius, but not reachable by
   the dashboard and not "live" (operator must run it manually).
3. **An `engine_api.py` GET endpoint** — chosen: it is the canonical operator
   HTTP surface (`:8787`), mirrors the existing `/api/candidates` ranker path,
   and is reachable by any operator tooling.

Naming: started as `/api/live_book` (matching `consume_into_live_book`) but
renamed to `/api/concentration_preview` — in this codebase "live book" already
connotes the **IBKR real portfolio** (`docs/IBKR_LIVE_BOOK_INTEGRATION.md`,
`/api/portfolio/*`); naming by *purpose* avoids that conflation and "preview"
honestly signals the ephemeral / no-routing nature (reinforces §3).

## What worked
A thin, additive exposure of the already-§2-safe wire:
- Module-level pure function `build_concentration_preview(runner, ...)` (kept
  out of the handler class so it is unit-testable without booting the server —
  same pattern as `_resolve_port`). It calls `runner.consume_into_live_book`,
  then serialises per-candidate admit/refuse outcomes + the structured
  cap-breach audit (`_ev_authority_log` reject entries) + the armed cap %s.
- `_handle_concentration` GET handler parses query params, resolves the
  universe scope, and returns the payload via `_send_json`.
- Route `GET /api/concentration_preview`.

## What didn't
No dead ends. One design check: the endpoint is a **GET** even though "consume
into book" sounds mutating — justified because it is safe + idempotent (the
tracker is built per request and discarded; nothing persists), so GET is the
semantically correct, §3-clear choice and matches `/api/candidates`.

## How we fixed it
The wire adds **no new EV path**. Candidates come only from
`rank_candidates_by_ev` (→ `EVEngine.evaluate`); the armed caps **refuse**
over-concentration and never touch `ev_raw` / `ev_dollars` / `prob_profit` /
the dealer multiplier; the D16 launch gate inside `consume_ranker_row` still
refuses `ev_dollars <= 0` at token issuance (no negative-EV rescue); the book
is ephemeral (no persistence, no order routing — §3). The decision trio
(`ev_engine.py` / `wheel_runner.py` / `candidate_dossier.py`) is **untouched**.

## Evidence
- `tests/test_engine_api_concentration.py` (4 tests) drives a synthetic
  concentrated batch through the REAL `consume_into_live_book` /
  `make_live_book_tracker` / `portfolio_risk_gates` (ranker faked, cap math
  unmocked):
  - `test_caps_fire_on_the_live_path` — 3 IT names @ 8% open; a 20%-NAV name →
    `single_name_breach` (R10); a 4th IT name (sector → 32%) → `sector_cap_breach`
    (R9).
  - `test_negative_ev_is_refused_not_rescued` — `ev_dollars=-5` → outcome
    `ev_authority_refused`, position not opened.
  - `test_metadata_and_caps_surfaced` — caps `{sector: 0.25, single_name: 0.10}`,
    `authority="ev_ranked_concentration_gated"`, note states "no orders are routed".
  - `test_diversified_book_all_open_positive_control` — small diversified book
    all opens.
- Local run: `py -3.12 -m pytest tests/test_engine_api_concentration.py
  tests/test_production_tracker_caps.py -q` → **10 passed in 0.72s**.

## Unresolved / handoff
- §2 sign-off is required before merge (no auto-merge). Lane claimed on #113.
- Follow-ups (not in this PR): a `consume_covered_call_row` mirror for the
  call leg; an optional dashboard panel that calls the endpoint; the IBKR
  real-book path (D24/D26) is the *other* way to close the dormancy gap and is
  tracked separately.
