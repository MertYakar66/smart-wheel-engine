---
id: audit-doc-truth-pass
title: 2026-07 repo audit — Batch A doc/comment truth-pass
kind: docs
status: in-flight
terminal:
pr:
decisions: []
date: 2026-07-07
headline: Truth-pass fixing 5 MODULE_INDEX drift entries, 6 stale docs, GOVERNANCE solo-trim, and a THETA_PULL_SESSION_NOTES→THETA_USAGE merge; docs/comment-only, suite green.
surface: [MODULE_INDEX.md, docs/THETA_USAGE.md, docs/PARALLEL_SESSIONS.md, data/broad_pull_loaders.py, engine/earnings_drift.py]
---

## Goal
Batch A of the 2026-07 repo audit (D-number assigned at merge): correct
documentation/comment drift discovered during the folder-by-folder read, using
the smallest safe in-place fix ("observe accurately, dispose conservatively").
No behaviour change — docs, docstrings, and one dead-doc merge only.

## What we tried
Every claim was verified against source in a clean `origin/main` worktree
before editing (dual-form caller greps, `git ls-files` existence checks). Where
a supplied premise did not hold on the bytes, the item was reframed to the
verified truth rather than written as given.

## What worked
- **MODULE_INDEX.md truth fixes** (all verified): `regime_detector` → **Dormant**,
  superseded by `regime_hmm` on the live path (`wheel_runner.py:1975`);
  `observability` → **test-only** (only `tests/test_observability.py`; not
  re-exported by `engine/__init__.py`; live twin is `data/observability.py`
  `:109`); `earnings_drift` → **test-only** (not EV-wired); `portfolio_copula`
  → **Dormant** (smoke + coverage tests only); removed the stale FinanceNews
  note (README was repaired, PROJECT_STATE §5) and fixed the `Next.js v15`→`v16`
  heading (`dashboard/package.json:21` = `16.1.6`).
- **6 stale docs refreshed in place** (dated docs kept, banners added, bodies
  preserved): DATA_LAYER_ACTIVATION_ROADMAP (connector deep-read shipped, gated
  `SWE_DEEP_HISTORY` default OFF); PHASE1_E_TRIO_EXECUTION_SPEC (all three (E)
  fixes #372/#369/#378 landed); PARALLEL_SESSIONS + MAJOR_SESSION_PROMPT
  (4-terminal rig retired 2026-06-01 — **§5 lane-claim CI contract explicitly
  preserved**); bloomberg_refresh_runbook (producer gap closed by the xbbg
  salvage).
- **GOVERNANCE.md**: practice-note banner + §9 rewrite — phantom Model
  Committee / Quant Team / Risk Team / on-call trimmed to the single-operator
  reality (operator wears every hat; CI is the automated gate).
- **THETA_PULL_SESSION_NOTES.md → THETA_USAGE.md §20** merge + delete; inbound
  refs repointed (AGENTS:50, DATA_POLICY:130, PROJECT_STATE:461); FILE_MANIFEST
  row removed; manifest coverage re-verified clean.
- **Code-comment/docstring truth**: `data/broad_pull_loaders.py` "Nothing
  consumes it" corrected (BroadPullLoader consumed at `engine/data_connector.py`
  :894/:1063/:1582); `engine/earnings_drift.py` softened to analytics-only;
  `scripts/bloomberg_export.vba` superseded header.
- **DATA_INVENTORY.md**: annotated the 7 non-canonical larder assets (verified:
  exactly 7 CSVs read by neither `_FILES` nor `consolidated_loader.py`).
- **TESTED_SURFACE_MAP.md**: regenerated from a fresh full-suite `coverage.json`
  (83 modules, suite 84.8%).

## What didn't
- The runbook item was supplied as "producer-gap claim now false; defer to
  DATA_POLICY §5" — but **DATA_POLICY §5 (≈ lines 169-189) itself still asserts
  the gap** ("7 have no runnable producer"). The salvaged pullers
  (`pull_vol_iv`/`pull_dividends`/`pull_vix_term_structure`/`pull_corporate_actions`)
  demonstrably write the connector CSVs, so §5 is stale too. DATA_POLICY §5 was
  **not** in the authorised edit scope, so the runbook currency-note flags the
  §5 staleness and it is **parked for the operator** rather than silently edited.

## How we fixed it
Smallest-in-place edits only, each backed by a verified file:line. Dated docs
(index-in-place per `docs/worklog/README.md`) got additive status banners, not
body rewrites. The one delete (THETA_PULL_SESSION_NOTES) was a full content
merge with all gates cleared first.

## Evidence
- `python scripts/check_manifest_coverage.py` → OK, 0 uncovered / 0 orphan
  after the THETA delete + row removal.
- `ruff format --check` + `ruff check` on the 2 changed `.py` → clean.
- `pytest -m "not backtest_regression" --cov` → **3468 passed, 28 skipped,
  4 deselected, 16 xfailed, 0 failed** (535.9s). coverage.json fed the
  TESTED_SURFACE_MAP regen.

## Unresolved / handoff
- **Operator decision:** `docs/DATA_POLICY.md` §5 (~169-189) producer census is
  stale post-salvage (canonical doc, out of this PR's authorised scope).
- Phase-2 finding (separate): `engine/ev_engine.py:123` code comment still names
  `engine.regime_detector` for the `regime_multiplier` field — the source is the
  HMM; left for the engine/ Phase-2 block.
