---
id: m3-asof-none-staleness-gate
title: as_of=None staleness gate (M3 brain-audit)
kind: fix
status: shipped
terminal:
pr:
decisions: []
date: 2026-06-11
headline: "fix(wheel_runner): resolve as_of=None to the universe data frontier so the staleness gate engages"
surface: [engine/wheel_runner.py, engine/data_connector.py, tests/test_asof_none_staleness.py]
---

## Goal

Brain-audit finding M3: `rank_candidates_by_ev` (and its CC/strangle siblings) had no staleness
check at `as_of=None`. With `None`, every ticker independently resolved to its own latest OHLCV
bar. Index leavers from the 2026-03-23 reconstitution seam (CTRA, BK, EPAM, HOLX, LW, MOH, MTCH,
PAYC — all gap-76d behind the universe frontier 2026-06-04) ranked at stale spot prices. CTRA
returned +$40.43 EV at as_of=None while an explicit `as_of=2026-06-11` correctly dropped it.

## What we tried

The spec considered (and rejected) several alternatives: global as_of=None -> frontier rewrite
(would shift the event-gate reference back ~1 week, potentially un-locking an earnings lockout =
forbidden rescue); per-scan-set frontier (CTRA-alone scan has gap 0 = does not drop); connector-side
refusal (get_ohlcv must serve stale data for legitimate historical backtests).

## What worked

Narrow surgical fix: resolve a `staleness_ref` once before the per-ticker loop, then swap the gate
condition `if as_of is not None:` -> `if staleness_ref is not None:`. The staleness_ref is:
- `as_of` when explicit (preserves the existing S32 F3 gate byte-identical)
- `conn.get_data_frontier()` when as_of=None AND the connector has the method (new M3 path)
- `None` when the connector lacks the method (legacy/Theta behavior via hasattr guard)

Added `MarketDataConnector.get_data_frontier(dataset='ohlcv')` to data_connector.py: returns
`_load('ohlcv')['date'].max()` clamped to today (so a corrupt future-dated row cannot inflate the
frontier and black out the universe). Reconciled with `engine_api._data_frontier` in the docstring.

`cutoff` is (re)assigned inside the gate block — the PIT-truncation block is skipped at as_of=None
so it never binds `cutoff` (UnboundLocalError trap from the spec).

Reason strings are distinct: None-path uses "behind universe data frontier"; explicit path keeps
the byte-identical "beyond latest data" string pinned by test_pit_leaks.py.

## What didn't

Global as_of resolution would have re-truncated IV/rates at the frontier date and shifted the
event-gate reference — both forbidden by CLAUDE.md §2 (no rescue, no EV drift on survivors).

## How we fixed it

1. `engine/data_connector.py`: added `get_data_frontier(dataset='ohlcv')` after `get_universe`.
2. `engine/wheel_runner.py` (trio): three gate sites updated — `rank_candidates_by_ev` (pre-loop
   staleness_ref + condition swap + reason branch), `rank_covered_calls_by_ev` (inline resolution
   before gate, condition swap, reason branch, `return _empty()`), `rank_strangles_by_ev` (same).
   PIT truncation, spot, IV, risk-free rate, event `today_date` (line: `date.fromisoformat(as_of)
   if as_of else date.today()`), forward distribution — all UNTOUCHED.
3. `tests/test_asof_none_staleness.py`: 12 tests covering stale drop, fresh survival, drop-only
   invariant, byte-identical survivors, explicit path untouched, custom threshold override, legacy
   connector degradation, corrupt-future-row clamp, CC + strangle siblings.
4. `tests/test_data_connector.py`: `TestDataFrontier` class (7 tests) pinning the new helper.
5. `TESTING.md` + `FILE_MANIFEST.md`: new file rows added.

## Evidence

Spec validator: APPROVE_WITH_CHANGES. All required_changes implemented:
- Frontier helper reconciled with engine_api._data_frontier in docstring + today-clamp applied.
- Sibling rankers corrected to single-ticker / `return _empty()` pattern (not `continue`).
- `cutoff` (re)assigned inside gate block.
- STALE stubs carry >504 rows; per-ticker-keyed stubs in all multi-ticker tests.
- Negative/clamp test: `test_future_dated_corrupt_row_does_not_blackout_universe` + connector test.
- None-path reason string wires `cutoff.date().isoformat()` (frontier), not as_of (which is None).

Smoke tickers AAPL/MSFT/JPM/XOM/UNH are all gap-0 from the frontier — output unchanged.
S27/S32/S34/S35 markers pass explicit `as_of=today.isoformat()` — explicit branch preserved
verbatim, no re-baseline event.

## Unresolved / handoff

- Issue #378(a): stale IV for stale explicit-as_of tickers — NOT fixed here (get_data_frontier
  is dataset-parameterized for that session to reuse). The M3 fix masks the OHLCV-stale leaver
  instance of #378(a) in the None path but does not resolve #378.
- BK drop attribution shifts event->data gate at as_of=None (observability-only; BK was already
  excluded, just at a different gate).
- ThetaConnector keeps the as_of=None bypass until it implements get_data_frontier (flagged).
