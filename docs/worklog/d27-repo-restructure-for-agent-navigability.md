---
id: d27-repo-restructure
title: Repo restructure for zero-memory agent navigability (D27)
kind: refactor
status: shipped
terminal: X
pr:
decisions: [D27]
date: 2026-06-09
headline: Staged structural pass — Stage 1 reconciled the root (index-doc drift to 2026-06-09, truthful .env.example, audit.py → scripts/audit_api_smoke.py); Stage 2 deep-dives each folder.
surface: [CHANGELOG.md, PROJECT_STATE.md, MODULE_INDEX.md, README.md, ROADMAP.md, .env.example, scripts/audit_api_smoke.py, DECISIONS.md]
---

## Goal
Operator mandate: restructure the repository so a zero-memory agent can
orient, find current state, and start contributing in the minimum number
of file reads. Two stages: (1) make the root a clean, minimal, accurate
entry point; (2) deep-dive every folder (docs/ → tests/ → scripts/ →
engine/ → periphery), eliminating redundancy and stale content. Hard
constraints: structure-only (no behaviour/import changes without shims),
`git mv` for every move, references updated in the same commit, decision
recorded as D27.

## What we tried
Stage 1 began with the full read_first sweep (CLAUDE/AGENTS/
PROJECT_STATE/MODULE_INDEX/DECISIONS/COMMIT_GUIDE/TESTING +
docs/SESSION_HANDOFF + the archived optionsengine audit) and a per-file
evaluation of all 26 root files against four questions (current? earns
root? duplicated? referenced?).

## What worked
- **Root relocation was already done (D14)** — the real Stage-1 gap was
  *content drift*: the index docs were last reconciled at `ec17e1d`
  (2026-06-03) with 58 commits since. Fixed additively: CHANGELOG gained
  the "2026-06 (early)" section (PRs #317–#394), PROJECT_STATE a dated §3
  wave summary + a PR-#343 supersession note on the 2026-06-01 "zero
  non-test callers" finding, MODULE_INDEX the missing
  `portfolio_risk_gates.py` / `ibkr_portfolio_adapter.py` / `studies/`
  rows, ROADMAP an "Open work" router + done-track compression per its
  own contract.
- **`.env.example` rewritten truthfully** — every variable now has a
  named reader in tracked code; phantom broker/notification vars dropped
  (grep-verified zero readers).
- **`audit.py` → `scripts/audit_api_smoke.py`** (git mv, 100% rename) —
  the D14 objection (CI ruff scope, lint then red) dissolved when Track F
  closed; verified ruff-clean before moving; 9 inbound refs updated in
  the same commit; D14 annotated SUPERSEDED-by-D27 on that bullet.

## What didn't
- Initial inbound-ref grep for `audit.py` filtered lines containing
  `test_audit` and silently hid TESTING.md:206 (a line naming both
  `test_audit_viii_e2e.py` and `python audit.py`). Caught by a second,
  positive-pattern grep. Lesson: when sweeping references, run one
  broad pass *without* noise filters before trusting a filtered count.

## How we fixed it
Three clean commits on `claude/sleepy-noether-l9jlfa`:
1. `b8bb78a` docs(root) — index-doc reconciliation (5 files).
2. `2176421` chore(env) — truthful .env.example.
3. `891bac0` refactor(scripts) — audit.py move + 9 ref updates.

## Evidence
- Baseline: 959 tracked files; HEAD `1985547`; highest decision D26.
- `python3 scripts/check_manifest_coverage.py` → OK (959 tracked /
  0 uncovered) after the move.
- `ruff check scripts/audit_api_smoke.py` → All checks passed (0.15.8).
- Env-var truth: `grep -rhoE 'SWE_[A-Z_]+'` → 11 real vars vs 0
  documented in the old template; `MAX_POSITION_SIZE`/TDA/Alpaca → 0
  readers.

## Stage 2 — docs/ (completed 2026-06-09)
83 top-level docs read line-by-line via four parallel sub-audits.
Verdict: the D14-extension architecture works — defects were narrow.
- Archived 3 (`f5fdfb1`): SESSION_HANDOFF (superseded banner since
  05-22), Claude_Prompting_Master_Guide (generic, zero refs),
  DATA_SPECIFICATION (self-declared aspirational; DATA_POLICY header
  repointed to DATA_INVENTORY).
- Truth-synced 4 banners + 2 routers + 3 reading-order headers
  (`eeb0666`): the IBKR design doc claimed "none adopted" while
  D24/D26 are IMPLEMENTED — the worst find of the audit.
- Deliberately NOT done: REVERIFICATION_REPORT_2026-05-26 archive
  (it is a generated legacy row in worklog/INDEX.md — generator
  churn for no gain); TESTED_SURFACE_MAP archive (live generator
  targets that path); doc merges (every suspected cluster — Theta
  quartet, setup duo, data roadmaps, IBKR trio — is a deliberate
  audience partition); VERIFICATION_INDEX engine-vs-passive span
  consolidation table (content authoring needing number re-verify
  across 5 reports — flagged as a follow-up, not structural work).

## Stage 2 — tests/ (completed 2026-06-09)
All 144 test files read (three parallel sub-audits) + green baseline
pinned: 3086 passed / 0 failed / 32 skipped / 19 xfailed in 6:49
(`python3 -m pytest tests/ -m "not backtest_regression"`).
- TESTING.md taxonomy completed 55 → 144 files (`0b1ef38`): 90 rows
  across existing tables + five new sections (Ranker/EV-path, W-series
  W38-W67, External-data adapters, Theta pullers, IBKR live book).
- New gate `tests/test_testing_md_taxonomy.py` (two-way: suite ⊆
  taxonomy, literal taxonomy paths ⊆ suite) — the manifest/worklog
  CI-gate pattern applied to the test map; it caught a 90th missing
  row pre-commit.
- README launch-blocker command gained the missing
  test_r11_elevated_vol.py (gate-drift vs TESTING/REPO_MAP/skill).
- Deliberately NOT done: renames/renumbering (filenames are decision
  anchors — DECISIONS.md pins ~50; REPO_MAP INVARIANT-PIN set forbids
  moves without §2 sign-off); merges of near-neighbour files
  (deliberate unit/integration/property layers; fresh W-series PRs
  mapped to the DATA_TEST_AUDIT register).

## Stage 2 — scripts/, engine/, periphery (completed 2026-06-09)
- scripts/ (53 .py + assets read): healthy; only action was the four
  populated-dir `.gitkeep` removals (`24c1d9a`, closes ROADMAP C3).
  An auditor's dead-code flags on five xbbg pullers were REFUTED —
  they are the documented Bloomberg producers
  (NEXT_DATA_SESSION_RUNBOOK:129-130; bloomberg_refresh_runbook rows
  #1/#9; tracked outputs).
- engine/ (52 modules read, claims re-verified): zero code changes;
  MODULE_INDEX truth restored (`55dabdb`) — stale A3 re-export
  section, four dormancy reclassifications (signals, signal_context,
  portfolio_intelligence, dependency_check); PROJECT_STATE's pinned
  stale-comment line number un-pinned.
- Periphery (src/, ml/, backtests/, config/, utils/, studies/,
  advisors/, both news trees, local_agent/, tradingview/, data/,
  dashboard/): structure sound everywhere (`ba8fe70`);
  config/settings.py + five utils modules status-noted; REPO_MAP
  src/ table precision fix. A second auditor's "FILE_MANIFEST lacks
  news-tree sections" claim was refuted (rows exist; CI gate
  enforces coverage).

## Unresolved / handoff
- Use `python3 -m pytest` in this sandbox (bare `pytest` resolves to
  an interpreter without the installed deps).
- Deferred with reasons recorded in D27: test renumbering, doc
  merges, src/ stub deletion, the VERIFICATION_INDEX
  engine-vs-passive span table, the trio-file stale comment
  (operator-greenlit touch), and the requirements.txt/pyproject
  dependency divergence (behavioral; flagged only).
- Follow-up idea from the docs audit: one engine-vs-passive span
  table in VERIFICATION_INDEX_2026-05-28.md consolidating the
  S32/S38/S40/S43/S44 numbers (currently stated across 5 reports).
- D27 entry to be appended to DECISIONS.md at campaign close (number
  re-checked at merge per PARALLEL_SESSIONS rule 9).
- Dependency divergence flagged, deliberately untouched: pyproject deps
  ≠ requirements.txt; `hmmlearn` absent from both yet needed at runtime.
- `pull_branch.bat`'s default BRANCH_NAME points at a dead branch —
  edit-per-use by design; left as-is.
