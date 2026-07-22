---
id: audit-dead-code-d28
title: 2026-07 repo audit — Batch D D28 dead-code retirement
kind: refactor
status: in-flight
terminal:
pr:
decisions: [D28]
date: 2026-07-08
headline: Nine verified-dead retirements (each with its manifest/index/taxonomy rows) + the D28 DECISIONS record and PROJECT_STATE/CHANGELOG sync. engine imports clean; suite green.
surface: [DECISIONS.md, engine/observability.py, engine/earnings_drift.py, src/data/__init__.py, docs/LAUNCH_READINESS.md]
---

## Goal
Batch D of the 2026-07 repo audit: retire the nine files the audit verified as
dead-undefended, land the **D28** DECISIONS record (scope/method + the full
carried-forward parked list), and keep every descriptive doc in sync.

## What worked
Nine retirements, each with co-edits so the CI gates stay green:
1. `engine/observability.py` + `tests/test_observability.py` (test-only; not
   re-exported by `engine/__init__.py`) + TESTING + FILE_MANIFEST + MODULE_INDEX rows.
2. `engine/earnings_drift.py` + `tests/test_earnings_drift.py` (test-only) + rows.
3. `data/feature_provenance.py` (zero importers).
4. `dashboard/web_vitals.py` + the surgical removal of
   `tests/test_infrastructure.py::TestWebVitals` (kept the rest of the file) +
   FILE_MANIFEST row. (§6 Dashboard-terminal *scaffold* file — not the portfolio pipeline.)
5. `data/bloomberg/sp500_iv_history.csv` (20-byte empty stub; covered by the
   `data/bloomberg/*.csv` glob so no manifest row; updated DATA_INVENTORY §1 +
   the larder annotation 7→6).
6. Five `dashboard/public/*.svg` Next.js scaffold icons (the app
   `dashboard/src/app/favicon.ico` kept) → the `*.svg` glob row removed.
7. `news_pipeline/browser_agents/grok_agent.py` (zero refs incl. tests; no
   `browser_agents/__init__` re-export to trim; the `GROK` enum in `types.py`
   left as the smallest cut) + FILE_MANIFEST row.
8. `local_agent/utils/efficiency.py` (dead intra-package) + FILE_MANIFEST row.
9. `src/data/validators.py` + its two `src/data/__init__.py` export lines
   (`schemas.py` in the same package stays LIVE via `data/quality.py` ←
   `wheel_runner.py:2043`) + FILE_MANIFEST row.

Plus: the **D28** DECISIONS.md entry (Decision / Why / Rejected alternatives /
parked list / Pinned by), and PROJECT_STATE §3 + CHANGELOG synced (AGENTS
keep-in-sync).

## What didn't (in-sync fix beyond the named co-edits)
- `docs/LAUNCH_READINESS.md` §9 was built around `engine/observability.py`
  (TraceContext/DecisionJournal/AuditLogger) — deleting the module would dangle
  the section. Rewrote §9 to keep the LIVE `WheelTracker._ev_authority_log`
  audit surface and note the scaffold's D28 retirement.
- `ROADMAP.md:101` also names `earnings_drift.py`, but as a **past-tense closed
  Track-F record** (a bug fixed there in 2026-05) — left intact (history, not a
  live dependency).

## Evidence
- `check_manifest_coverage.py` → 0 uncovered / 0 orphan; `test_testing_md_taxonomy.py` → 2 passed.
- `python -c "import engine; import engine.wheel_runner; import src.data"` → OK.
- `git grep` for every deleted module across live code/tests → **no residual refs**.
- `ruff check` on changed CI-scoped `.py` → clean.
- Full `pytest -m "not backtest_regression"` → green (see PR).

## Unresolved / handoff
- The full carried-forward parked list lives in `DECISIONS.md` D28 (financial_news
  legacy dedup, local_agent keep-vs-extract, staging carriers, DATA_POLICY §5,
  pyproject↔requirements, TESTED_SURFACE_MAP regen, …).
