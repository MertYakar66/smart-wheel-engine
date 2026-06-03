---
id: docs-freshness-rcount
title: Docs-freshness sweep: reviewer rule count R1-R10 to R1-R11 plus count/link drift
kind: docs
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-02
headline: Canonical orientation docs drifted behind the code (reviewer count stuck at R1-R10 / older R1-R6/R1-R8; engine_api 32 vs 34 endpoints; 25 vs 22 Bloomberg CSVs; 127 vs 108 smoke checks). Verified each against origin/main and corrected the live docs only.
surface: []
---

## Goal
Bring the canonical orientation docs back in sync with the code. The prior
`docs/REPO_EFFICIENCY_AUDIT.md` fixed an `R1-R6 → R1-R10` reviewer-count drift;
the code has since shipped **R11** (elevated-vol top-bin size-down, `DECISIONS.md`
D23) and the same doc surface drifted one rule behind again. Docs-only,
truth-neutral, single-concern (no engine/test changes); separate from the
in-flight re-baseline (`claude/rebaseline-backtest-snapshots`) and IBKR
(`claude/ibkr-portfolio-tracker`) branches.

## What we tried
A 7-agent read-only audit workflow over the full doc/README surface, baselined
in an isolated worktree at `origin/main` (`0a9c17c`), each finding self-verified
against the code/tree (grep the source, check file existence) and deduped by a
synthesis agent — 21 raw findings → 10 fixes. Load-bearing findings re-confirmed
by hand against `engine/candidate_dossier.py` before editing.

## What worked
- **Reviewer rule count R1-R10 → R1-R11** across the live canonical docs:
  `CLAUDE.md` (§1 header + a new §2 R11 bullet authored from the source
  docstring + D23), `README.md`, `MODULE_INDEX.md`, `PROJECT_STATE.md`,
  `docs/REPO_MAP.md` (count + rule enumeration), `FILE_MANIFEST.md`, and
  `docs/LAUNCH_READINESS.md` (which was the most stale — §2 row at R1-R6, §3
  header/table at R1-R8 with no R9/R10/R11 rows; authored the three missing
  table rows from `CLAUDE.md` §2 + D23).
- **Numeric drift:** engine_api endpoint count `32 → 34` (`README.md`,
  `MODULE_INDEX.md`, `dashboard/README.md`); Bloomberg CSV count `25 → 22`
  and smoke-test checks `127 → 108` (`docs/LAPTOP_SETUP.md`, `MODULE_INDEX.md`).
- **`docs/DATA_SPECIFICATION.md`:** added a STATUS banner (forward-looking
  Parquet design spec, not on-disk reality — the committed layer is 22 flat
  CSVs) rather than a misleading mechanical `.parquet → .csv` swap (the
  filenames also differ).

## What didn't
- **The synth's mechanical R-count bump conflated two different claims.** It
  proposed bumping `TESTING.md` L53 (`test_dossier_invariant` rules R1-R10 →
  R1-R11), but that line describes *what that test covers*, not the canonical
  count. `tests/test_dossier_invariant.py` has R7/R8/R9/R10 tests and **no R11**
  (grep: no `R11`/`elevated_vol`/`vix_level`). R11 is pinned by
  `tests/test_r11_elevated_vol.py`. Reverted L53 to R1-R10; added the R11 test
  to the two LAUNCH_READINESS citations that assert R1-R11 coverage. Left
  `REPO_MAP.md:48` `test_dossier_invariant (R1–R10)` unchanged — it is correct.
- **Did NOT touch** `archive/**`, dated point-in-time reports
  (`REALISM_VERIFICATION_2026-05-28`, `CODE_REVIEW_2026-05-30`,
  `VERIFICATION_INDEX_2026-05-28`, backtest Sn reports), or `DECISIONS.md`
  append-only entries — their `R1-R6/R1-R8/R1-R10` refer to their own moment;
  editing them would falsify the record.

## How we fixed it
Targeted `Edit`s in the live docs only, each anchored on a unique string;
load-bearing R11 wording mirrored from `engine/candidate_dossier.py` and
`DECISIONS.md` D23. New R11 bullet (CLAUDE §2) and R9/R10/R11 rows
(LAUNCH_READINESS §3) authored to match the existing R7-R10 style.

## Evidence
- R11 live: `engine/candidate_dossier.py` L64-65 (`R11_TOP_BIN_PROB=0.90`,
  `R11_VIX_THRESHOLD=25.0`), L553-561 (`pp > R11_TOP_BIN_PROB and vix_f >
  R11_VIX_THRESHOLD` → review); wired in `engine/wheel_runner.py`; D23 is the
  highest D-number; pinned by `tests/test_r11_elevated_vol.py`.
- Endpoint count: `engine_api.py` header enumerates 34 routes (24 main GET + 7
  TV GET + 3 POST). CSV count: `data/bloomberg/*.csv` = 22. Smoke checks:
  `grep -c 'h\.run(' scripts/feature_smoke_test.py` = 108.
- Post-edit grep: zero residual `R1-R10/R1-R6/R1-R8` / `32 endpoint` / `25
  Bloomberg` / `127 checks` in the live canonical docs.

## Unresolved / handoff
- **`tradingview/OVERVIEW.md`** — deferred (NEEDS_REVIEW). Describes a Mac /
  Cowork / `~/Desktop/TradingView/` / `.sh` launcher / port-9222 workflow that
  conflicts with the canonical repo-local `tradingview/{research,models,pine}/`
  paths and the Windows MSIX setup; could be an intentional external workspace.
  Needs the owner's call (also the stale `Last updated` date).
- **`docs/PRODUCTION_READINESS.md`** — broadly stale (says `R1-R8` and "all 32
  Sn usage tests (S1–S32)" while the ledger is well past S32). Left untouched: a
  one-line R-count fix would imply the rest is current. Needs a dedicated
  refresh or an archival decision.
- **`docs/DATA_SPECIFICATION.md`** — banner added; the body still describes the
  aspirational Parquet schema. A full reconcile-or-archive is a separate call.
