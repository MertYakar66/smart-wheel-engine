---
id: wallclock-staleness
title: "Universe wall-clock frontier staleness: connector warn + ranker attrs + opt-in refuse (D1-2/D3-2)"
kind: fix
status: shipped
terminal: X
pr: 470
decisions: []
date: 2026-07-03
headline: "A 27-day-stale OHLCV frontier was runtime-invisible (the only gate was frontier-relative, reading 0 when the whole universe is stale, while the event gate used the real wall clock); now: once-per-connector warn at >7d, structured attrs['staleness'] on all three rankers + API surfacing, and an opt-in universe-wide refusal — default byte-identical"
surface:
  - engine/data_connector.py
  - engine/wheel_runner.py
  - engine_api.py
  - tests/test_wallclock_staleness.py
---

## Goal

Adversarial review 2026-07-01, CRITICAL-2 compound (b) — D1-2/D3-2:
"27-day frontier staleness invisible at runtime; no wall-clock check
anywhere incl. dashboard frontier chip." #463's residuals explicitly
queued this with the rescued `claude/rescue-2026-06-15-fixes` branch
as design input. Mixed clocks made it worse than invisible: at
`as_of=None` spots/IV price off the frontier bar while the event gate
uses REAL today (`date.today()`), so lockout windows are current
against month-old prices.

## What we tried

Recon extracted the rescued branch's design (`refuse_stale_live`
param, warn-else-drop, per-ticker) and mapped today's surface: rail
7d wall-clock bound (#463, done), earnings overlay 45d warn (#464,
done — the 3-layer template), OHLCV frontier NOTHING (the gap),
treasury/VIX latest-row with no bound (documented residual).

## What worked

The #464 template, adapted: deterministic pin already exists
(`EXPECTED_FRONTIER`, catches stale TREES); add runtime layers for a
CURRENT tree with old data.

## What didn't

- **The rescued design's threshold** (reusing
  `max_as_of_staleness_days` = 30): would NOT have fired on the
  motivating 27-day case. Threshold is a separate connector constant
  `_OHLCV_FRONTIER_STALE_DAYS = 7` — parity with the rail's 7d
  wall-clock bound (both answer "is this the current market state?");
  max legitimate market-closed gap ≈ 4-5 days, so 7 has zero false
  positives on a maintained box.
- **The rescued design's per-ticker drops** when armed: N identical
  entries for a universe-wide condition. The refusal here
  short-circuits ONCE before the loop with a single `ticker: "*"`
  drop.
- **A row column for staleness**: changes ranker output schema
  (diagnostic-column pins, EV-authority hashing). `df.attrs` is the
  §2-safe channel (drops precedent — survivor rows untouched).
- **Any default refuse**: blanks every `as_of=None` book on a stale
  box (the #462 lesson). Default = warn-and-rank, fail-open loudly.

## How we fixed it

- **Layer 1** (`engine/data_connector.py`): `get_data_frontier` logs
  once per connector when the frontier is > 7d behind today
  (`_warned_stale_frontier` flag, the `_warned_stale_earnings_snapshot`
  pattern). WARN-only — the return value never changes; only reached
  at `as_of=None` (pinned by `test_asof_none_staleness`), so dated
  backtests never hit it.
- **Layer 2** (`engine/wheel_runner.py`, trio, lane-claimed):
  `_frontier_staleness_info` builds a structured dict (frontier /
  wall-clock date / age / threshold / stale) at each ranker's
  staleness_ref resolution; `_attach_drops_summary` gains an optional
  `staleness` arg and attaches `attrs["staleness"]`. Dated queries get
  the explicit `{"checked": False}` sentinel — wall-clock reads gate
  on `as_of is None`, byte-identity preserved. `engine_api` surfaces
  the attrs at both candidates payloads and `/api/status` gains
  `frontier_age_days` (per-request — the frontier string is
  process-cached but the clock moves).
- **Layer 3**: `refuse_stale_live: bool | None = None` on all three
  rankers; `None` → `SWE_REFUSE_STALE_LIVE` env arm (live deployments
  arm without code changes). Armed + `as_of=None` + stale → ONE
  universe-wide drop (`gate="data"`, reason distinct from both
  pre-existing staleness strings), empty frame, before the per-ticker
  loop. Drop-only: a refusal can never rescue (§2).

## Evidence

- 18 new tests (`tests/test_wallclock_staleness.py`): warn-once /
  fresh-quiet (real connector on tmp CSVs), attrs shape at stale /
  fresh / dated-sentinel, survivor-rows byte-identity, param + env
  arming, param-False-overrides-env, default fail-open,
  fresh-not-refused, dated-never-refused-even-armed, CC/strangle
  parity, helper unit pins. Weekend trap caught in review of my own
  test: `bdate_range(end=Saturday)` snaps to Friday — expectations
  anchored to the actual written max.
- Mutation: the test file cannot even IMPORT on main (the helpers are
  new) — the entire capability is absent there.
- Neighbor pins green pre-suite: `test_asof_none_staleness` +
  `test_pit_leaks` + `test_tv_api` + `test_engine_api_hardening` +
  `test_portfolio_api_endpoints` + `test_launch_blockers` = 116
  passed. Full fast suite + 3-refuter panel in the PR body.

## Unresolved / handoff

- **Dashboard chip**: engine_api now serves `frontier_age_days` +
  `staleness` — the chip's severity rendering is the Dashboard
  terminal's lane (CLAUDE.md §6; deliberately no `dashboard/src`
  edits here).
- treasury `get_risk_free_rate` and VIX (feeds R11 `vix_level`) still
  serve latest-row with NO wall-clock bound at `as_of=None` —
  documented residual, own lane (EV-adjacent thresholds need their own
  justification).
- `select_book` (multi-grid aggregator) does not attach staleness —
  its component ranks do; noted, not wired.
- Arming `SWE_REFUSE_STALE_LIVE` in the operator's live shell is an
  OPERATOR decision (it hard-refuses live ranks on any >7d-stale box).
