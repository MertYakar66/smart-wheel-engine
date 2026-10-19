---
id: staging-untrack
title: Untrack the 20 staging data CSVs (D31's last gap)
kind: refactor
status: in-flight
terminal: desktop
pr:
decisions: [D31]
date: 2026-10-19
headline: git no longer tracks the 20 data CSVs under staging/ (7,152,880 B); each has a byte-identical copy at the same path in the data root (20 of 20 by git blob id) and so in swe-data/; .gitignore ignores staging/**/*.csv; the 25 scripts and notes under staging/ stay tracked; nothing on disk changed and no history was rewritten.
surface: [.gitignore, staging/blue_chips/, staging/casy/, staging/fundamentals_pit/]
---

the 20 staging CSVs are untracked; their copies stay in the data root and swe-data

## Goal

Close D31's last gap (`PROJECT_STATE.md` §0 B item 4). Git still tracked 20 data CSVs under `staging/`
(15 in `staging/blue_chips/`, 4 in `staging/casy/`, 1 in `staging/fundamentals_pit/`; 7,152,880 B),
although the desktop data root and `swe-data/` on Drive hold each of them byte for byte. On 2026-10-19
the Operator answered "Yes, untrack them (Recommended)". This run takes the 20 out of the index with
`git rm --cached`, which leaves every file on disk, and adds `staging/**/*.csv` to `.gitignore` so they
cannot come back. The scripts and notes beside them stay tracked. No history is rewritten: git history
keeps the old copies.

## What we tried

The card's steps, in order, in the worktree `.claude/worktrees/staging-untrack` on the new branch
`claude/staging-untrack` from `origin/main` (`428f6a9`). Every step ran once.

- **Step 0.** The worktree and branch, `at main's tip`, the mark (`behind/ahead of main 0/0 · run mode
  change`), and `0` changes.
- **Step 1.** `csv files tracked under staging: 20`, then `with a byte-identical copy at the same path in
  the root: 20 of 20`. Each file's git blob id was compared with `git hash-object --no-filters` of the
  root's copy. Last, `other tracked files under staging: 25`.
- **Preflight beyond the card.** The 20 files total 7,152,880 B on disk, as `PROJECT_STATE.md` says.
  These paths name neither the three folders nor any of the 20 file names: `engine/`, `data/`, `tests/`,
  `scripts/`, `utils/`, `dashboard/`, `backtests/`, `.github/`, `config/`, `conftest.py`,
  `pyproject.toml` and `engine_api.py`. Outside `staging/`, the names appear only in prose:
  - `FILE_MANIFEST.md`, in the description of the `staging/fundamentals_pit/` folder row;
  - `docs/CASY_BACKFILL_SPEC.md`;
  - two worklog fragments;
  - one archived review.

  Each of the three folders keeps tracked scripts and notes.
- **Step 2.** `git rm --cached` of the 20, and the ignore rule. `git status` shows the 20 `D` lines and
  ` M .gitignore`. Then `staged deletions: 20`, `csv still tracked under staging: 0`, `csv on disk under
  staging, now ignored: 20` and `other tracked files under staging: 25`.
- **Step 3.** The checks (Evidence).
- **Step 4.** This fragment, the CHANGELOG entry and the regenerated index, then the checks again on the
  tree to be committed.

## What worked

All of it, as the card expected. The files stay on disk in this worktree, where git now ignores them,
and in the data root and `swe-data/`.

## What didn't

Nothing failed in this run. The fast lane shows exactly the two failures the card names as local to this
desktop, and both pass in CI:
- `test_a_mark_missing_from_claude_md_fails` fails on a cp1252 encoding issue on Windows;
- `test_distill_real_aapl_partition` needs a 2023 AAPL partition the desktop does not have.

One trap in the card's own filter: `grep -E "^FAILED|passed|failed"` also matched the test id
`tests/test_forward_distribution_invariants.py::TestEmpiricalBoundary::test_nonfinite_returns_are_filtered_not_passed_to_ev`,
because its name contains `passed`. It comes from the warnings summary, not from a failure. Run alone,
it reads `1 passed, 1 warning`.

## How we fixed it

Nothing needed fixing.

## Evidence

Step 3, on the worktree after step 2 (the fast lane, then ruff on CI's lint scope, then the structure,
manifest-coverage and doc-currency checks):

```
tests/test_forward_distribution_invariants.py::TestEmpiricalBoundary::test_nonfinite_returns_are_filtered_not_passed_to_ev
FAILED tests/test_check_working_structure.py::test_a_mark_missing_from_claude_md_fails
FAILED tests/test_option_premium_accessor.py::test_distill_real_aapl_partition
= 2 failed, 3706 passed, 24 skipped, 8 deselected, 20 xfailed, 171 warnings in 1031.01s (0:17:11) =
All checks passed!
367 files already formatted
working structure: 6 of 6 checks pass
manifest paths matching nothing : 0
OK: FILE_MANIFEST.md accounts for every tracked file.
doc-currency: OK
```

The 20 untracked files:
- `staging/blue_chips/`: `COHR_ohlcv.csv`, `COHR_vol_iv.csv`, `CPB_ohlcv.csv`, `DPZ_ohlcv.csv`,
  `KMB_ohlcv.csv`, `LITE_ohlcv.csv`, `LITE_vol_iv.csv`, `PLTR_ohlcv.csv`, `SATS_ohlcv.csv`,
  `SATS_vol_iv.csv`, `VEEV_ohlcv.csv`, `VEEV_vol_iv.csv`, `VRT_ohlcv.csv`, `VRT_vol_iv.csv`,
  `WMT_ohlcv.csv`;
- `staging/casy/`: `casy_earnings.csv`, `casy_liquidity.csv`, `casy_ohlcv.csv`, `casy_vol_iv.csv`;
- `staging/fundamentals_pit/`: `sp500_fundamentals_pit.csv`.

The step outputs and the Run Summary are on the pull request.

## Unresolved / handoff

- **Other checkouts drop their copies on the new `main`.** Once this merges, any checkout that moves to
  it, the pen's included, loses its own working-tree copies of the 20 CSVs, because git deletes a file
  that a commit untracks. The copy that stays is the data root's, and so `swe-data/`'s, byte for byte
  the same (step 1). Nothing in the engine, the tests or CI reads them.
- **`staging/integrate_phase1b.py` now runs from the data root.** It reads the fragments by
  repo-relative paths (`staging/casy/`, `staging/blue_chips/`) and the monoliths under
  `data/bloomberg/`. From now on it runs with the data root as its working folder; the monoliths have
  lived only there since D31. It writes the monoliths, so running it stays operator-gated, as
  `FILE_MANIFEST.md` says.
- **`FILE_MANIFEST.md`'s three folder rows still match**, since each folder keeps tracked scripts and
  notes, and the coverage check passes. Their descriptions still speak of the CSVs as being in the
  folder. Refreshing that wording is the pen's call: this card left `FILE_MANIFEST.md` alone.
- **`PROJECT_STATE.md` §0 B item 4** still reads "Untracking them remains proposed". The pen updates it,
  and this fragment's `status` and `pr`, at the close after the merge.
