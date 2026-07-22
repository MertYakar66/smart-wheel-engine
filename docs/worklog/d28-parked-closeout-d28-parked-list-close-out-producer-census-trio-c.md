---
id: d28-parked-closeout
title: "D28 parked-list close-out: producer census, trio comment, src stubs, dep prune"
kind: docs
status: shipped
terminal: session-x-advisor
pr: TBD
decisions: [D28]
date: 2026-07-08
headline: "Closed 5 of D28's 14 parked items + resolved 2 keep-decisions; 7 remain deferred with reasons"
surface: [docs/DATA_POLICY.md, PROJECT_STATE.md, engine/ev_engine.py, src/, pyproject.toml, requirements.txt, DECISIONS.md, MODULE_INDEX.md, FILE_MANIFEST.md, docs/TESTED_SURFACE_MAP.md]
---

## Goal

Work through the D28 parked list (DECISIONS.md) with delegated operator
authority: close every item that is provably safe from static evidence,
resolve the pure-judgment calls, and leave anything that needs the
operator's environment (Bloomberg Terminal, local cron knowledge,
§6 Dashboard-terminal ownership) untouched — "do not remove anything
useful and important."

## What we tried

Triaged all 14 parked items into execute / defer:

**Executed (5):**
1. `docs/DATA_POLICY.md` §5 producer census — rewritten from
   "3 of 10 / 7 no producer" to the post-#477 truth: **9 of 10** with
   in-repo producers, only `sp500_earnings.csv` deferred. Mirrored fix
   in the PROJECT_STATE §1 data-currency blockquote (same stale claim).
2. `engine/ev_engine.py:130` comment (trio touch, lane-claimed) — the
   `regime_multiplier` field no longer claims `engine.regime_detector`
   as its source; now names the HMM-via-wheel_runner as the live
   caller-supplied source, regime_detector as the dormant alternative.
3. `src/{execution,models,risk}/__init__.py` empty stubs — removed
   (zero references repo-wide across code, pyproject, coverage config).
   PROJECT_STATE §4 + MODULE_INDEX src/ row + FILE_MANIFEST rows updated.
4. pyproject↔requirements divergence — **partial, evidence-based**:
   pruned six zero-import deps from `[project].dependencies` (polars,
   duckdb, optuna, plotly, rich, python-dotenv); added role headers to
   both files (pyproject = CI/packaging canonical; requirements.txt =
   laptop/hook runtime set).
5. `docs/TESTED_SURFACE_MAP.md` regenerated from a fresh full-suite
   coverage.json on the post-#490 tree (D28 explicitly queued this).

**Resolved as keep-decisions (2):** the ~8 one-off scripts stay in place
(doc-mapped reproducers, manifest-covered); `data/features/` stays
tracked (documented in-git AAPL sample of the feature-store layout).

**Deferred with reasons (7):** financial_news legacy dedup (cannot
verify the operator's local cron/Task Scheduler from this session —
`sentiment.sqlite` production may depend on that stack); local_agent
keep-vs-extract (side-project disposal, operator's call); staging
only-copy carriers (Bloomberg laptop + data-gate); news_pipeline
publisher/slo/robustness (wiring = live-pipeline behaviour change;
retiring removes useful scaffolding); backtests simulator/walk_forward
(simulator's PLACEHOLDER header names a planned upgrade path);
dashboard rails + package.json rename (§6 Dashboard-terminal territory).

## What worked

The salvage verification: `git grep` of each connector CSV name across
`scripts/pull_*.py` + the #477 commit stat + the runbook's #487 currency
note gave three independent attestations of the new census before any
doc was rewritten.

## What didn't

The originally-parked direction "consolidate to pyproject
[project].dependencies" is NOT safe as stated: CI installs only
`-e ".[dev]"`, so moving the laptop runtime deps (yfinance, arch,
matplotlib, …) into pyproject would newly activate currently-skipped
tests in CI — a CI-behaviour change, not a hygiene edit. Rejected;
recorded inline in DECISIONS.md D28 and in pyproject's NOTE block.

## How we fixed it

Smallest safe in-place edits per item; delete only the three stub files
with zero references; prune only deps with zero imports repo-wide
(verified including tests/, dashboard py, local_agent — the first
anchored grep missed loguru's 10 importers and typer's 10 CI-run
backtest reproducers; both kept).

## Evidence

- Census: `grep -rln <csv> scripts/pull_*.py` per file;
  `git show f0a6595 --stat`; runbook 2026-07-04 currency note.
- Stubs: `grep -rnE "src\.(execution|models|risk)"` over `*.py`,
  `*.toml`, `*.cfg` → empty; each stub was a 1-line docstring.
- Dep prune: per-dep `grep -rlE "^\s*(import X|from X)"` over all
  tracked `*.py` → 0 files for each of the six pruned; loguru=10,
  typer=10 (backtests/regression CLIs), aiohttp=5, lightgbm=1 → kept.
- Gates: `python scripts/check_manifest_coverage.py` → 0 uncovered /
  0 orphans; `ruff check` + `ruff format --check` clean on touched .py.
- Full suite (trio-touch rule, CLAUDE.md §4): see PR body for the
  run output on this tree.

## Unresolved / handoff

- The 7 deferred items above remain in D28's parked list (annotated
  in place with close-out status per item).
- If a future PR moves the laptop deps into pyproject, audit which
  skipped tests activate in CI first (`pytest -m "not
  backtest_regression" --collect-only` diff with/without the deps).
- `sp500_earnings.csv` producer remains the one census gap
  (`docs/NEXT_DATA_SESSION_RUNBOOK.md` queue).
