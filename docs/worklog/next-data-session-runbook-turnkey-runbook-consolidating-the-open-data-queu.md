---
id: next-data-session-runbook
title: Turnkey runbook consolidating the open data queue (#339/#355/#354/#357)
kind: docs
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-08
headline: One ordered, doc-only execution plan (docs/NEXT_DATA_SESSION_RUNBOOK.md) so a single logged-in Bloomberg Terminal session clears the whole open data queue in one pass — draws the Bloomberg-gated (CASY + 10 blue-chip backfills) vs. git-reconstructable (BK↔BNY collapse, dividends union, UNIVERSE_100 re-derive, 4-snapshot re-baseline) line explicitly
surface: [docs/NEXT_DATA_SESSION_RUNBOOK.md]
---

## Goal

After the IV gate (#363) + preflight guard (#364) landed, the remaining open
queue is all **data work**: #339 (BK↔BNY continuity + CASY backfill + dividends
re-pull + 4-snapshot re-baseline), #355 (11 blue-chip OHLCV truncations), #354
(dateless-fundamentals lookahead), #357 (low-pri hygiene). None of it is doable
additively in the Cowork sandbox (no Bloomberg Terminal; 6 of 9 connector CSVs
have no in-repo producer). Operator chose: make the next Terminal session
**turnkey** — consolidate the scattered steps (the #339 body, the #355 table,
`CASY_BACKFILL_SPEC.md`, `DATA_POLICY §5`, `TESTING.md` re-baseline workflow)
into one ordered runbook. Doc-only; no data touched; no decision-trio touch.

## What tried / what worked

Read the four issues + the canonical sources from a worktree off the **new
`main` (67b57fc)** — deliberately NOT the stale primary clone (the exact
stale-tree trap the #364 preflight guard now catches). Verified each load-bearing
fact against source before writing it into the runbook:
- pull-script frontier edits: `scripts/pull_ohlcv.py:19`, `scripts/pull_liquidity.py:26` (`end_date="2026-03-20"`).
- re-baseline command + snapshot ids: `python -m backtests.regression.<id> --update-snapshot` for `s27_ivpit_24t_100k` / `s32_friction_24t_1m` / `s34_universe_100t_1m` / `s35_oos_24t_100k` (`tests/test_backtest_regression.py` `_BACKTESTS`; `TESTING.md` §re-baseline).
- xfail flips: `tests/test_data_to_engine.py::test_blue_chip_history_is_complete[<ticker>]` (#355), `::test_fundamentals_credit_are_point_in_time` (#354).
- `UNIVERSE_100` derivation = `MarketDataConnector().get_universe()[:100]`, enforced by `test_universes_match_connector` (`backtests/regression/universes.py`).
- drift guard = `test_snapshot_data_fingerprint_matches_current` (#340) goes red on any connector-data change → forces re-baseline.

## What didn't / the key correction

My first pass framed *all* of #339 as Terminal-blocked. `CASY_BACKFILL_SPEC.md`
(on-the-bytes audit) corrects that: **only CASY's 4 files truly need Bloomberg**
(plus the other 10 blue-chips' OHLCV, and #354's dated panel). BK↔BNY collapse,
the `CTRA/LW/MTCH/PAYC` dividends union, `UNIVERSE_100` re-derive, and the
re-baseline are **git-reconstructable / local**. The runbook leads with that
distinction (a Bloomberg-gated vs. git-reconstructable table) — it's the fact
that decides how much can proceed without a Terminal.

## How we fixed it

`docs/NEXT_DATA_SESSION_RUNBOOK.md`: Phase A (Bloomberg pulls — CASY via the
existing spec + 10 blue-chip backfills + optional frontier refresh + #354/#357
optional), Phase B (git-reconstructable integration), Phase C (re-baseline all 4
+ marker re-run + `EXPECTED_FRONTIER` bump + xfail flips + clear S34 provisional
+ §2 panel on the universe change). Plus FILE_MANIFEST row + this worklog.

## Evidence

Doc-only diff (no `.py`, no data, no trio). `gen_worklog_index.py --check` OK;
`check_manifest_coverage.py` OK; ruff N/A (no Python changed). Held for review.

## Unresolved / handoff

The runbook is the plan, not the execution — the actual pulls + integration +
re-baseline are the next Terminal session's work. Note: Phase B (BK↔BNY collapse
+ dividends union) is pure git and could be done *now* without a Terminal, but
the spec batches it after CASY to avoid a double `UNIVERSE_100` re-derive +
re-baseline. #354's code half is a separate **trio PR** (§2 review).
