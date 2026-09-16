---
id: restart-2026-09-16
title: Operating Model v3 (one protocol), news-stack removal, restart rulings recorded
kind: refactor
status: completed
terminal: remote-sandbox
pr:
decisions: [D29]
date: 2026-09-16
headline: Recorded the Operator's restart rulings as D29; wrote OPERATING_MODEL v3 + docs/PROMPTING_STANDARD.md + docs/RESTART_PLAN_2026-09-16.md (tracks A–E as Execution Prompts); removed every news implementation (Python, API, tests, configs, registries; dashboard in the same PR series); retired the board/cards/July-channel machinery from hooks, template and archive; declined the history rewrite pending explicit confirmation.
surface: [OPERATING_MODEL.md, docs/PROMPTING_STANDARD.md, docs/RESTART_PLAN_2026-09-16.md, DECISIONS.md, ROADMAP.md, engine/wheel_runner.py, engine_api.py, scripts/orchestrate.py, scripts/pull_all.py, dashboard/]
---

## Goal

Turn the Operator's 2026-09-16 answers and six corrections into durable state:
one protocol (with prompting guidelines), no news code, the product direction
(7–28 DTE, event-aware, exit evaluator, strategist commentary) recorded and
planned, and the Bloomberg loss acknowledged in the data plan.

## What we tried

1. Inventoried every coupling before deleting: `grep` over engine/, scripts/,
   tests/, configs, registries and the dashboard for news identifiers; found the
   trio coupling (`use_news_sentiment` → `news_mult` in `combined_regime_mult`),
   the API buffer and two endpoints, nine news test files plus news-pipeline
   tests inside `test_infrastructure.py`, `pyproject`/`ci.yml` scopes, and the
   dashboard's `(main)` news route group.
2. One patch script with exact-string asserts for the Python/config/registry
   edits; `git rm -r` for the packages; ast-based removal of the news-only
   argparse flags and of `test_infrastructure.py` blocks importing `news_pipeline`.
3. Wrote OPERATING_MODEL v3 keeping v2's section numbering (§4.4, §7, §9.x
   references across the repo stay valid) and replacing only §2.4/§9.5's
   machinery; added §3.1 Operator-away mode from the July channels' surviving rules.
4. Dashboard news removal delegated to a subagent constrained to `dashboard/`
   with `npx next build` as the acceptance check (baseline build verified green first).

## What worked

- Keeping the section numbers: zero cross-reference churn outside README.
- The trio edit is comment-plus-plumbing only; `combined_regime_mult` semantics
  unchanged for every path (news multiplier was a constant 1.0 since D18).
- Exact-string asserts caught two drifted anchors before they silently no-op'd.

## What didn't

- `ast.get_source_segment` on a multi-line `add_argument` call left a dangling
  string fragment in `scripts/orchestrate.py`; restored from HEAD and removed
  whole statements by line range instead.
- `python scripts/check_manifest_coverage.py` reports a manifest row for an
  untracked new file as "non-existent path" — `git add` first.
- The Operator's "delete from history too" was not executed (see D29 rejected
  alternatives); tree deletion + issue closure instead.

## How we fixed it

See CHANGELOG 2026-09-16. Trio touched: `engine/wheel_runner.py` (news
plumbing removed), `engine/ev_engine.py` (one comment) — the PR carries the
lane-claim block.

## Evidence

- Targeted tests: `python -m pytest tests/test_infrastructure.py tests/test_pit_leaks.py tests/test_audit_viii_unit_invariants.py tests/test_engine_api_hardening.py tests/test_check_lane_claim.py tests/test_testing_md_taxonomy.py tests/test_ranker_transparency.py tests/test_wheel_runner_select_book.py tests/test_asof_none_staleness.py -q` → `189 passed in 32.87s`.
- Fast lane after the removal: `python -m pytest tests/ -m "not backtest_regression" -q -x --ignore=tests/test_backtest_regression.py` → `3393 passed, 28 skipped, 4 deselected, 20 xfailed, 30 warnings in 735.60s (0:12:15)`, exit 0.
- Guards: `check_manifest_coverage.py` OK; `gen_worklog_index.py --check` OK;
  `check_doc_currency.py` OK; `ruff check` / `ruff format --check` clean on touched files.
- Dashboard (subagent constrained to `dashboard/`, then re-verified by the orchestrator): 36 files deleted, 18 modified; `npm uninstall rss-parser node-cron @types/node-cron` → removed 7 packages; `rm -rf .next && npx next build` → "Compiled successfully", TypeScript pass, exit 0, route table `/`, `/_not-found`, `/api/chat`, `/api/engine`, `/api/events`, `/api/market`, `/api/portfolio/[sub]`, `/api/watchlist`, `/cockpit`, `/portfolio`, `/terminal`; `npx eslint src` → exit 0, 0 problems (baseline had 12 warnings, all in deleted files). Residual grep for news terms: 127 hits, every one accounted for (46 "history", 45 terminal watchlist, 22 engine calendar, 11 data-feed wording, 3 legacy DB-file comments/path, 1 chat persona sentence, 1 Finnhub User-Agent string).

## Unresolved / handoff

- Operator: turn on branch protection for `main`; merge this PR then #523;
  confirm or decline the git-history rewrite as a separate action.
- Dashboard judgement calls to ratify: `src/db/**`, `drizzle-orm`, `better-sqlite3`, `drizzle-kit` and the `db:*` scripts were TRIMMED, not deleted, because the terminal's watchlist, events, quote cache and chat persistence use them; `/api/watchlist` no longer returns or accepts `alertThresholdPct`; the local DB file name `data/finance-news.db` and the package name `finance-news` are unchanged so an existing local watchlist/chat history is still found; `ollama-ai-provider` was already unused before this run. Runtime was not exercised in the sandbox (no engine, Ollama, or local DB): evidence is build + lint.
- Track A's first question (which data subscriptions exist) gates everything downstream.
- Docs currency pass (stale worklog statuses; audit register / worklist; PRODUCTION_READINESS) is Track E.
