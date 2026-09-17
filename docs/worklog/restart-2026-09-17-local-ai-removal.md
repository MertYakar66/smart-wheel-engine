---
id: restart-2026-09-17
title: Local-AI integrations removed (browser agent, Ollama memo and chat); Operator answers recorded
kind: refactor
status: completed
terminal: remote-sandbox
pr:
decisions: [D29]
date: 2026-09-17
headline: Removed local_agent/ (29 files, 8,273 lines), engine/trade_memo.py + /api/memo, /api/summary, /api/ollama_status, and the dashboard's Ollama chat panel, AI status indicator and chat tables (build-verified); recorded the Operator's answers of 2026-09-17 (subscriptions deferred, delta deferred, exit evaluator advisory, API prose) in D29; re-ordered the plan (Track F structure pass first, Track A parked) with a candidate table for the next deletions.
surface: [local_agent/, engine/trade_memo.py, engine_api.py, dashboard/, DECISIONS.md, docs/RESTART_PLAN_2026-09-16.md, ROADMAP.md, PROJECT_STATE.md]
---

## Goal

Execute the Operator's ruling R12 ("delete lines, files and pages related to the
local AI agent; a clean and direct engine"), record the four answers given the
same day, and turn "we focus on efficiency and structure now" into a concrete,
Operator-ruled candidate list rather than an open-ended cleanup.

## What we tried

1. Inventory by grep before deleting: `local_agent/` had no importer outside
   itself (only registry rows); `engine/trade_memo.py` had three API endpoints
   and one CI test as consumers, and imported `advisors` (not the reverse); the
   dashboard's Ollama use was the chat route/panel, the `ollama_status` engine
   proxy action and the status-bar "AI" indicator.
2. Python side by one exact-string patch script (asserted anchors); dashboard
   side by a subagent constrained to `dashboard/` with `npx next build` and
   `npx eslint src` as the acceptance check, re-run by the orchestrator.
3. Sizes of every remaining off-path candidate measured after the deletions and
   written into the plan (§7a) so the Operator can rule row by row.

## What worked

- Exact-string anchors again caught a drifted docstring (the deep-read design
  doc cites a `trade_memo.py` line that no longer exists) before the doc went stale silently.
- Keeping the trio untouched: the memo module read the ranker, not the reverse,
  so no decision-layer edit was needed this time.

## What didn't

- Nothing failed. One judgement call to ratify: the terminal grid lost its chat
  cell; the subagent kept the 3×2 grid with row 1 unchanged (Market/Vol | Options
  Engine | Live Book) and row 2 = Watchlist | Events spanning two columns (the
  Events panel is the one that truncates, so it takes the freed width); the symbol
  workbench's right column became a single full-height Options Engine cell. `zod`
  left the top-level dependencies (no importer; it survives as a transitive of the
  eslint config). Rendering was verified by build and lint only, not in a browser.

## How we fixed it

See CHANGELOG 2026-09-17. Registries (FILE_MANIFEST, MODULE_INDEX, TESTING,
REPO_MAP, READMEs, OPERATING_MODEL §9.1 one phrase) updated in the same commits.

## Evidence

- Fast lane after the removal (`python -m pytest tests/ -m "not backtest_regression" -q -p no:cacheprovider --ignore=tests/test_backtest_regression.py`):
  `3385 passed, 28 skipped, 4 deselected, 20 xfailed, 30 warnings in 945.15s`, exit 0
  (8 fewer passes than after the news removal, matching the 8 tests of the deleted `tests/test_trade_memo_ci.py`).
- Guards: `check_manifest_coverage.py` OK; `gen_worklog_index.py --check` OK; `check_doc_currency.py` OK; ruff clean on touched files.
- Dashboard: `npx next build` exit 0 and `npx eslint src` exit 0 after the removal (subagent report + orchestrator re-run).

## Unresolved / handoff

- Operator: rule row by row on the Track F candidate table (plan §7a); the
  suggested defaults are the Strategist's, not decisions.
- Track A stays parked until a data subscription is chosen; a brief (Track D)
  over the 77-day-old frontier would be fiction, so D waits for A.
- Delta target for short expiries: deferred, keep 0.25, ask again before Track B run 3.
