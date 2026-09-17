---
id: restart-2026-09-17-track-f
title: Track F structure cuts — advisors, ml/models, studies, src/, dormant modules, TradingView MCP + R2 note, docs archive
kind: refactor
status: completed
terminal: remote-sandbox
pr: 524
decisions: [D29, D30]
date: 2026-09-17
headline: Executed every Track F row except the wheel_runner unification under the ruling "all the cuts except wheel_runner; no protection for main" — eight commits on PR #524; src/ collapsed into engine/features + data/; R2 (chart context) is a note, not a stop (D30); main stays unprotected (D29 ruling 13); EV smoke byte-identical to the 2026-09-11 run.
surface: [advisors/, ml/, models/, studies/, src/, engine/features/, data/schemas.py, engine/candidate_dossier.py, engine/tradingview_bridge.py, engine/__init__.py, tradingview/, archive/2026-09/, OPERATING_MODEL.md, DECISIONS.md]
---

## Goal

Turn the Operator's ruling of 2026-09-17 ("Open the PR and go ahead with all
the cuts except wheel_runner. we dont need the protection for the main") into
one reviewable PR: the restart run so far (brief, protocol v3, news and
local-AI removals) plus every row of the Track F candidate table
(`docs/RESTART_PLAN_2026-09-16.md` §7a) except the `wheel_runner`
ranker/ladder unification, one commit per row, registries and docs updated in
the same commit as each cut, and the branch-protection statements dropped.

## What we tried

1. PR #524 opened first, with the lane-claim block naming all three trio
   files (the R2 change was known to be coming) and a
   `<request-as-sharpened>` block.
2. Cuts in dependency order: docs archive (#523 merged with two mechanical
   conflicts: the manifest rows and the worklog-index count), advisors,
   ml/models, studies, src/, dormant modules, TradingView MCP + R2, protection
   statements. Each cut: importer inventory by grep → exact-string patch script
   with asserted anchors → ruff → import smoke → the affected test files →
   manifest / index / currency guards → commit.
3. `src/`: the row said "move the two live modules, delete the rest". The
   seven research feature modules turned out to be the only inputs of
   `data/feature_pipeline.py` (the research feature store, documented in README
   and LAPTOP_SETUP), so they were moved with `technical.py` into
   `engine/features/` instead of deleted, and the feature-pipeline question was
   surfaced as a separate ruling (D2 update, plan §7a). Only the heuristic
   `src/backtest` (test-only, self-declared §2-non-compliant) was deleted.
4. Dormant modules: `model_validation.py` was in the row but is not dormant —
   `tests/test_binomial_tree.py` runs eight pricer cross-model tests through
   it. Kept, stated in the commit.
5. R2: the early `return "review", "chart_context_missing"` became a note;
   R3/R4 are skipped without a chart; R5–R11 run. The downgrade-only property
   test's non-overlay set is now empty; two reviewer tests were rewritten to
   pin the note and a third pins that a sub-threshold EV still lands in review
   (no upgrade path added). D30 written; D12/D13 marked superseded.

## What worked

- One patch script per cut with `assert count == 1` anchors caught every
  drifted line before it could be silently skipped (three anchor misses, all
  fixed on the spot).
- The 5-ticker EV smoke after all cuts reproduces the 2026-09-11 numbers
  exactly (MSFT 110.13 / XOM 80.14 / AAPL −12.61 / UNH −20.69; JPM dropped by
  the event gate), i.e. nothing on the EV path moved.
- Merging #523 into the branch instead of rebasing it kept its 20 renames as
  renames and its author's history intact.

## What didn't

- The lane-claim block had to name `engine/candidate_dossier.py` before that
  commit existed; the checker only tests that touched trio files are claimed,
  so an early claim is harmless, but it is a claim made ahead of the edit.
- Coverage: removing well-tested packages (advisors, ml) shifts the CI-scope
  percentage; the fast lane was re-run with `--cov=engine --cov=data
  --cov-fail-under=80` to prove the floor still holds (numbers below).
- `docs/TESTED_SURFACE_MAP.md` is a generated snapshot (2026-07-08) that still
  listed the deleted modules; regenerated from the coverage run's
  `coverage.json` rather than hand-edited.

## How we fixed it

Eight cut commits + this records commit on `claude/project-restart-ai-agents-kot5jr`
(PR #524). `OPERATING_MODEL.md` edits (§2/§5/§7 no protection; §9.1/§9.2
removals and the R2 line) are flagged in the commit messages as
Operator-governed edits made under the ruling.

## Evidence

- EV smoke (§9.4): `connector: MarketDataConnector`; 4 rows + `drops_summary
  {'total_dropped': 1, 'by_gate': {'event': 1}}`; 6.3 s.
- Dashboard: `npx next build` exit 0; `npx eslint src` exit 0.
- Guards: `check_manifest_coverage.py` OK (0 uncovered / 0 orphans);
  `gen_worklog_index.py --check` OK; `check_doc_currency.py` OK;
  `check_lane_claim.py --base origin/main` reports the three trio files touched
  (claimed in the PR body); `ruff check` + `ruff format --check` clean on the
  full CI scope; `mypy engine/features/ data/schemas.py` 553 pre-existing
  strict-mode errors (CI step is continue-on-error, unchanged in kind).
- Per-cut targeted suites (in the commit messages): 77 / 50 / 2 / 235 / 157 /
  200 passed.
- Fast lane, CI form (`python -m pytest tests/ -m "not backtest_regression" --cov=engine
  --cov=data --cov-fail-under=80`): `3155 passed, 28 skipped, 4 deselected, 20 xfailed,
  30 warnings in 1326.29s (0:22:06)`; `Required test coverage of 80% reached. Total
  coverage: 85.01%` (12,203 statements). 230 fewer passes than the 3385 of the
  local-AI run: the deleted advisors / ml / studies / src-backtest / dormant-module /
  MCP test files and the removed test classes.
- `docs/TESTED_SURFACE_MAP.md` regenerated from that run's `coverage.json`
  (`scripts/generate_tested_surface_map.py`).
- Regression lane (S27/S32/S34/S35 snapshot byte-identity): running at commit time;
  its result is recorded in the follow-up commit and in the PR's Run Summary.

## Unresolved / handoff

- Operator rulings still open, surfaced by the cuts: (a) the seven research
  feature modules + `data/feature_pipeline.py` + the committed `data/features/`
  sample shards — keep or delete as one decision; (b) `engine/policy_config.py`
  advisor/signal knob sections with no consumer left; (c) the remaining §7a rows
  (`utils/`, `staging/`, `backtests/` drivers, `scripts/`, `engine_api.py` split).
- The `wheel_runner` ranker/ladder unification stays open as its own campaign
  (excluded by the ruling).
- Track E first Operator step is now simply: merge #524.
