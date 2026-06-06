---
id: fix-credit-rating-deadread
title: "R0a: credit_rating dead-read fix re-shipped with lane ceremony + unit test"
kind: fix
status: complete
terminal: desktop
pr:
decisions: []
date: 2026-06-06
headline: Re-shipped the R0a credit-rating dead-read one-liner as its own PR (split out of the docs-only #332) with the decision-layer lane ceremony + a regression unit test pinning that analyze_ticker reads the get_credit_risk "sp_rating" key, not the raw Bloomberg field name.
surface: [engine/wheel_runner.py, tests/test_credit_rating_population.py]
---

## Goal
PR #332 carried a trio edit (`engine/wheel_runner.py`) bundled with the data-layer
plan docs, which tripped the Decision-Layer Lane Claim CI gate. Split it: make
#332 docs-only, and re-ship the one-liner as its own PR with the lane ceremony +
a unit test.

## What we tried / how we fixed it
- `analyze_ticker` read `credit.get("rtg_sp_lt_lc_issuer_credit", "")`, but
  `data_connector.get_credit_risk()` returns that field under the friendly key
  **`sp_rating`** (it maps raw -> friendly). So `credit_rating` was silently `""`
  for every ticker. Fixed to `credit.get("sp_rating", "")`.
- Added `tests/test_credit_rating_population.py`: a `WheelRunner` with a minimal
  stub connector (`analyze_ticker` wraps every other connector call in
  try/except, so only `get_fundamentals`/`get_credit_risk`/`get_ohlcv` must
  exist; inject via the lazy `_connector` attr). Three cases: populated from
  `sp_rating`; **regression guard** that reading the raw field name yields `""`
  (the connector never returns that key); and `""` when no credit data.
- Lane ceremony: claimed `engine/wheel_runner.py` on board #113 and added the
  `<!-- lane-claim files: engine/wheel_runner.py board: <#113 comment> -->` block
  to the PR body so `check_lane_claim.py` passes.

## §2 / re-baseline
Safe, no re-baseline. `credit_rating` is OFF the EV-authoritative path: it feeds
only the legacy heuristic `_compute_wheel_score()` (`fund_score += 10` for A/B,
the sole computational read at `wheel_runner.py:667`) via `screen_candidates()`,
plus the memo (`trade_memo.py`) and API (`engine_api.py`) display — never
`rank_candidates_by_ev` / `EVEngine.evaluate`. The four regression snapshots are
driven by the EV path, so they don't move. Verified by the prior 7-agent pass +
the full suite (2804 passed on the byte-identical edit).

## Evidence
- New test: `pytest tests/test_credit_rating_population.py` -> 3 passed.
- ruff clean; `check_manifest_coverage.py` OK (manifest row added for the test).
- Branch `claude/fix-credit-rating-deadread` off `origin/main`.

## Unresolved / handoff
None. The companion docs-only #332 records the broader data-layer plan; this PR is
just the R0a code change. Both left OPEN for review; nothing merged.
