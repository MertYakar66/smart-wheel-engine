---
id: onboarding-launch-clarity
title: Onboarding + launch-doc clarity (R11 merge-gate, AGENTS, PROJECT_STATE, data-doc honesty)
kind: docs
status: complete
terminal: onboard
pr:
decisions: []
date: 2026-06-02
headline: Docs-only onboarding/launch-doc clarity pass. Added the R11 test (test_r11_elevated_vol.py) to the launch-blocker pytest subset everywhere it's documented (the §2 merge gate had been pinning only R1-R10 via test_dossier_invariant), surfaced R11 in AGENTS.md and the REPO_MAP pin list, made the data docs honest (DATA_SPECIFICATION is aspirational; 6 of 9 connector CSVs have no in-repo producer; *_yf.csv files are unconsumed), refreshed tradingview/OVERVIEW.md to Windows-primary, and de-staled PROJECT_STATE + PRODUCTION_READINESS Sn high-water. Baselined against e1d7453 (post-#323); items already fixed by #323 were verified and skipped.
surface: []
---

## Goal

Make a fresh AI agent's onboarding accurate and close the one §2-relevant
documentation gap: R11 (the elevated-vol top-bin size-down, `DECISIONS.md`
D23) was absent from the documented launch-blocker test subset, so the
merge gate as written pinned only R1-R10 (`test_dossier_invariant.py`). R11
is pinned separately by `test_r11_elevated_vol.py`, which no launch-blocker
command listed. Plus a batch of onboarding/data-doc truth fixes.

## What worked

Baselined every item against the current worktree at `e1d7453` (PR #323
had already fixed much of the R-count / endpoint drift; the audit that
produced the backlog was baselined one commit earlier at `0a9c17c`).
Verified each stale claim still existed before editing; skipped the ones
#323 already closed.

Applied the conflation-trap rule rigorously: per-test coverage
annotations ("`test_dossier_invariant` (R1-R10)") were left as R1-R10
(correct for that test) and `test_r11_elevated_vol (R11)` was *added*
alongside, rather than bumping the per-test line to R1-R11.

## How we fixed it

R11 launch-blocker subset (the §2-critical change) — added
`tests/test_r11_elevated_vol.py` to the pytest command in:
- `docs/LAUNCH_READINESS.md` §4
- `TESTING.md` (running-tests subset + a launch-blocker taxonomy row +
  the `candidate_dossier.py` per-module command)
- `.claude/commands/launch-blockers.md` (the skill)
- `docs/REPO_MAP.md` (the embedded "single source" command + the
  INVARIANT-PIN list)

Other items:
- `AGENTS.md` — added the canonical R1-R11 / R11=D23 statement + the
  `test_r11_elevated_vol.py` pin to the hard-rule section.
- `docs/REPO_MAP.md` — `D1…D21` → drift-proof `D1…` (DECISIONS now D23).
- `docs/DATA_POLICY.md` — softened the DATA_SPECIFICATION pointer to
  "aspirational"; §5 note that `*_yf.csv` are unconsumed and that 6 of 9
  connector CSVs (incl. `sp500_vol_iv_full.csv`) have no in-repo producer,
  cross-linked to `bloomberg_refresh_runbook.md`.
- `scripts/pull_fundamentals_yf.py` — docstring corrected (connector reads
  `sp500_fundamentals.csv`, not `_yf`; `_yf` is unconsumed), mirroring the
  honest note in `pull_earnings_yf.py`. Docstring-only.
- `tradingview/OVERVIEW.md` — Windows-primary refresh (repo-local
  `tradingview/` workspace, `launch-tradingview-cdp.ps1`, dropped the
  Mac/Cowork framing, pointer to `docs/TRADINGVIEW_INTEGRATION.md`, date
  bump 2026-06-02).
- `docs/PRODUCTION_READINESS.md` §8 — Sn high-water "S1-S32" replaced with
  a reference to the `docs/worklog/INDEX.md` ledger as source of truth; and
  the LAUNCH_READINESS gate-ref count drift "R1-R8" → "R1-R11" corrected
  (commit e2c6384).
- `PROJECT_STATE.md` — S46 status corrected (completed, was "in flight");
  added S47 high-water row; added a point-in-time data-currency note
  (CSVs as-of 2026-03-20, refresh partially blocked).

Skipped (already fixed on `e1d7453` by #323): LAUNCH_READINESS §3 reviewer
table already had R9/R10/R11 rows; DATA_SPECIFICATION already had the
NOT-IMPLEMENTED STATUS banner; PROJECT_STATE §1 already listed R1-R11.

## Evidence

- Highest `DECISIONS.md` entry: D23 (R11). Worklog high-water: S47.
- `engine/data_connector.py` reads `sp500_fundamentals.csv` /
  `sp500_earnings.csv`; `_yf` variants have no reader (per
  `bloomberg_refresh_runbook.md`).
- `python scripts/check_manifest_coverage.py` and
  `python scripts/gen_worklog_index.py --check` both pass.
- `python -m ruff check scripts/pull_fundamentals_yf.py` +
  `ruff format --check` clean (only docstring changed).

## Unresolved / handoff

- `docs/DATA_SPECIFICATION.md` Overview prose is still present-tense
  ("All data is stored in Parquet"); the top STATUS banner already
  overrides it. A deeper prose rewrite was out of scope.
