---
id: data-home-untrack-2026-09-23
title: Untrack the market data from git (D31 step 5)
kind: refactor
status: completed
terminal: sandbox
pr: 530
decisions: [D31]
date: 2026-09-23
headline: the 87 tracked data files left the index (no history rewrite); CI runs without data — floors recalibrated to the no-data measurement, three tests moved off checkout paths, a guard fails any data committed again
surface: [.gitignore, FILE_MANIFEST.md, scripts/check_coverage_floors.py, tests/test_data_manifest.py, tests/test_data_connector.py, tests/test_option_premium_accessor.py, tests/test_paper_book.py, docs/DATA_INVENTORY.md, docs/DATA_POLICY.md, PROJECT_STATE.md, DECISIONS.md]
---

## Goal

D31 step 5: git stops holding market data, without rewriting history. Gate:
the desktop's `checked 144 manifest files: 144 ok, 0 missing, 0 mismatched`
(reported 2026-09-23 on `claude/data-home-desktop-check` @ `fc266bb`, after
#528). CI must stay green with no data in the checkout, and the desktop,
which reads the data root, must keep its full lane.

## What we tried

1. The untrack set, derived two ways and compared: the data-shaped files
   `git ls-files` lists under `data/`, `data_raw/`, `data_processed/` (the
   manifest tool's own definition of "not data": `*.py`, `*.md`, `.gitkeep`,
   the manifest) against the manifest rows whose `git_source` is `git:main`.
   Identical: 87 = 87.
2. `git rm --cached` of those 87, then the CI form run on a **clone of the
   commit**. The 2026-09-18 no-data proof hid the whole data folders; the
   real layout keeps `data/bloomberg/` (for `EXTRACTION_GUIDE.md`), and that
   difference mattered (below).

## What worked

- 87 staged removals, 87 files still on disk. What git tracks under the data
  trees is now the 13 `data/*.py` modules, `data/DATA_MANIFEST.json`,
  `data/bloomberg/EXTRACTION_GUIDE.md` and `data_processed/.gitkeep`.
- `.gitignore`: one D31 block. `data/bloomberg/**` with docs re-included,
  `data/features/`, `data_raw/`, `data_processed/`, `data_archive/`; the AAPL
  feature-sample exception is gone. Checked with `git check-ignore --no-index`
  on sample paths: every data path ignored; code, the manifest and `*.md`
  trackable.
- `FILE_MANIFEST.md`: the nine rows that listed tracked data are gone; the
  guard passes (`uncovered tracked files: 0`, `manifest paths matching
  nothing: 0`).
- The CI form without data, first pass: `1 failed, 2961 passed, 270 skipped`;
  coverage 83.56% (gate 80); the Quantitative Validation job's files
  `181 passed`; the Integration job `26 passed, 3220 deselected`. The two
  floors that only held with data failed as measured on 2026-09-18
  (data_connector 82.46 vs 88, wheel_runner 76.84 vs 77); every other floored
  file kept at least 1.2pp.

## What didn't

- **`TestDataFrontier.test_real_frontier_ge_expected` failed without data.**
  It skipped only when the `data/bloomberg/` folder was missing, but git keeps
  that folder for its guide, so the test ran on an empty folder and got no
  frontier. The 2026-09-18 proof hid whole folders and could not see this.
- **Two tests pointed at the checkout, not the data root.** The option-premium
  accessor's Theta larder would skip forever on the desktop once Theta sits in
  the root. The paper book's "never touches the real IBKR dir" test would
  fingerprint an empty checkout folder there, proving nothing. Neither fails
  anywhere today; both went wrong the moment the data left the checkout.
- **Thirty-one operator scripts and one regression harness still read or wrote data inside the checkout**
  (Codex review on this PR, P1, verified): with the data untracked and
  `SWE_DATA_ROOT` pointing at the desktop root, the Theta pulls would have
  exited ("constituents not found") and written into the checkout,
  `run_pipeline` would have fallen back to MAG7 silently, `ibkr_import` would
  have joined against an empty universe, and the yfinance refreshes, the
  validation runners and the feature backfill would have read or written
  the checkout. The 2026-09-18 pass re-rooted the engine, the data layer and
  a first set of scripts; a full sweep found the rest.
- A quick three-file run without `-m "not backtest_regression"` showed
  `test_end_to_end_seed_and_forward_integration` failing without data. That is
  the slow lane, which CI deselects and the desktop runs with its data; CI's
  selection gives `108 passed, 2 skipped, 1 deselected` on those files.

## How we fixed it

- The frontier test reads `paths.bloomberg_dir()` and carries
  `requires_data`. The Theta larder is `paths.theta_dir() / "option_history"`
  and the real IBKR dir is `paths.ibkr_dir()`.
- `scripts/check_coverage_floors.py`: `engine/data_connector.py` 88 → 80 and
  `engine/wheel_runner.py` 77 → 74, the CI-form measurement without data
  minus 2pp, with the reason in the file. The script says to recalibrate "in
  the same PR that legitimately shifts coverage"; taking the data out of CI is
  that PR.
- Every script that reads or writes data now resolves its paths through
  `engine.paths`, keeping today's behaviour when `SWE_DATA_ROOT` is unset:
  repository-anchored paths become `(paths.data_root() or <repo>) / …`,
  CWD-relative strings become `paths.resolve("…")`, Theta and processed outputs
  use `paths.theta_dir()` / `paths.processed_dir()` / `paths.option_premium_dir()`,
  and relative `--out` / `--out-dir` / `--universe` / `--log-csv` values are
  resolved after parsing. Scripts that had no `sys.path` bootstrap got one.
  Proof: all 32 load; with a root set, every module-level data path lands
  under it (the eight that need `yfinance` / `xbbg` loaded with stand-in
  modules); the three sites Codex named read the root's constituents
  (`ibkr_import.load_universe`, `run_pipeline.get_universe('sp500')`,
  `pull_theta_option_history._load_universe()` → `['AAPL', 'MSFT']` from a
  scratch root); with the variable unset the legacy paths are identical.
  Three of them (`download_*.py`) were committed with CRLF; `.gitattributes`
  (`eol=lf`) normalises them on this edit.
- `tests/test_data_manifest.py::test_git_tracks_no_market_data` replaces the
  #528 "every tracked data file has a row" guard: no data file may be tracked
  under the data trees. The last Windows CRLF write in that file is bytes (the
  desktop found it; #528's fixture change had exposed it).

## Evidence

```
tracked data files: 87 | manifest git:main rows: 87 | identical sets: True
staged removals: 87 | still on disk: 87 of 87

# CI form, clone of the untracking commit, no data, SWE_DATA_ROOT unset
= 1 failed, 2961 passed, 270 skipped, 8 deselected, 6 xfailed, 28 warnings in 388.84s =
FAILED tests/test_data_connector.py::TestDataFrontier::test_real_frontier_ge_expected
Required test coverage of 80% reached. Total coverage: 83.56%
FAIL engine/data_connector.py  82.46% (floor 88%)
FAIL engine/wheel_runner.py    76.84% (floor 77%)
quant job files: 181 passed · integration: 26 passed, 3220 deselected

# after the fixes, the same clone at the final commit
= 2961 passed, 271 skipped, 8 deselected, 6 xfailed, 28 warnings in 357.56s =
Required test coverage of 80% reached. Total coverage: 83.56%
ok engine/data_connector.py 82.46% (floor 80%) · ok engine/wheel_runner.py 76.84% (floor 74%)
ok candidate_dossier 90.24 · ev_engine 95.55 · event_gate 97.33 · portfolio_risk_gates 98.16 · wheel_tracker 83.62
requires_data skips: 54 (the 53 of 2026-09-18 plus the frontier test)

# floors against the with-data coverage.json of this morning's lane (3190 passed)
ok engine/data_connector.py 90.13% (floor 80%) · ok engine/wheel_runner.py 79.73% (floor 74%) · aggregate 85.09%
```

## Unresolved / handoff

1. **Step 6** (delete the four non-`main` branches, close #507) waits for the
   desktop's round 3: a verified full-history bundle (its heads equal to
   `git ls-remote`) and `data_archive/` plus the bundle on Drive, `rclone
   check` clean.
2. **Step 2b**: the data laptop is gone, so Drive `swe-local-only` is the only
   other copy of the Theta corpus and the feature shards; round 3 pulls them
   to the desktop root.
3. **The slow lane without data**: `test_parameter_oos.py`,
   `test_parameter_oos_100t.py`, `test_freeze_replay.py` (partly guarded) and
   `test_paper_book.py`'s end-to-end test raise `FileNotFoundError` instead of
   skipping when there is no root. They run by hand on the desktop, with data;
   a `requires_data` mark would make a sandbox run skip visibly instead.
4. **Whether the Tier-C stores join the manifest** (the Theta corpus alone is
   ~133k files) is open; today `check` covers the 144 git-derived rows and
   `rclone check` against Drive covers Tier C.
