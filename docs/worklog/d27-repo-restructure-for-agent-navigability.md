---
id: d27-repo-restructure
title: Repo restructure for zero-memory agent navigability (D27)
kind: refactor
status: in-flight
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

## Unresolved / handoff
- Stage 2 pending: docs/ (largest redundancy surface), tests/ (naming/
  coverage map), scripts/, engine/, then periphery (src/, ml/,
  local_agent/, financial_news/, news_pipeline/, notebooks/, config/,
  utils/, studies/, backtests/, tradingview/, archive/).
- D27 entry to be appended to DECISIONS.md at campaign close (number
  re-checked at merge per PARALLEL_SESSIONS rule 9).
- Dependency divergence flagged, deliberately untouched: pyproject deps
  ≠ requirements.txt; `hmmlearn` absent from both yet needed at runtime.
- `pull_branch.bat`'s default BRANCH_NAME points at a dead branch —
  edit-per-use by design; left as-is.
