---
id: data-home-2026-09-18
title: Data home — desktop data root (SWE_DATA_ROOT), checksum manifest, materialize-from-git (D31 steps 1–4)
kind: feature
status: in-flight
terminal: remote-sandbox
pr: 526
decisions: [D31]
date: 2026-09-18
headline: The Operator ruled that every dataset lives on the main desktop and git stores none of it; this run built the verified path there — a 99-file sha256 manifest generated from the git objects, `engine/paths.py` + `SWE_DATA_ROOT` so the engine reads a root outside the checkout (EV smoke byte-identical with the in-repo data hidden), a `materialize` command that fills a root from git without ever overwriting, and the `requires_data` posture for CI and sandboxes. Untracking (step 5) is held until the desktop `check` passes.
surface: [engine/paths.py, scripts/data_manifest.py, data/DATA_MANIFEST.json, conftest.py, engine/data_connector.py, data/consolidated_loader.py, data/broad_pull_loaders.py, data/feature_store.py, data/pipeline.py, data/bloomberg_loader.py, data/bloomberg.py, engine/data_integration.py, engine/ibkr_portfolio_adapter.py, engine/paper_book.py, backtests/regression/_common.py, docs/DATA_POLICY.md, docs/DATA_INVENTORY.md, DECISIONS.md, OPERATING_MODEL.md, README.md, TESTING.md]
---

## Goal

Execute the Operator's data-home verdict of 2026-09-18 without risking a byte:
"all data without any exception must be accessible in this very main Windows
machine … not depend on GitHub for storing data, for both previously collected
and for the future collections … Google Drive is still there but delayed."
The agreed safe sequence, each step verified before the next: (1) an
authoritative manifest of everything GitHub holds, (2) bring the git-only
datasets onto the desktop, (3) verify by checksum, (4) a CI/sandbox posture
that needs no data, (5) only then untrack the data from git, (6) delete the
data branches after verification; a history purge stays a separate decision.

## What we tried

1. **Manifest from the git objects, not from a working tree.** The 99 datasets
   live in three places: 71 files tracked on `main` (`69bf3b9`), the 13 deep
   slices on `deep-history/bloomberg-raw` (`68a48b2`), the 15 day-bot tick
   files on `claude/daybot-bloomberg-pull` (`2abf850`). Each was streamed out
   of its git blob, hashed (sha256) and sized — 1,431 MB in 28 s — and
   recorded with its `git_source`; the three commits are pinned in
   `git_sources`. The Drive folder ids from the migration draft were carried
   into a `drive` map.
2. **One resolver, no trio edits.** `engine/paths.py` re-roots a *relative*
   path whose first component is `data`, `data_raw` or `data_processed` under
   `SWE_DATA_ROOT`; unset, it returns the path unchanged — the legacy
   CWD-relative behaviour. The connector re-roots the `data_dir` it is handed,
   so `WheelRunner`'s `"data/bloomberg"` default is untouched (no lane claim).
   The narrower overrides (`SWE_DATA_PROCESSED_DIR`, `SWE_IBKR_DATA_DIR`,
   `SWE_OPTION_PREMIUM_DIR`, `SWE_SIM_DATA_DIR`) keep winning and are
   themselves resolved. Sixteen test modules and every loader / rail /
   regression-harness path literal now go through the resolver.
3. **Proof with the data hidden.** A complete external root was built in the
   sandbox scratchpad (`check` → 99 ok), `data/bloomberg`, `data/features`,
   `data_raw`, `data_processed` were moved aside, and the §9.4 EV smoke plus
   the fast lane were run with `SWE_DATA_ROOT` pointing at the root, then
   without it.
4. **`materialize`.** Fills a root from the git objects the manifest names:
   creates only what is missing, verifies every byte against the manifest,
   never overwrites (a differing file is reported as MISMATCH and kept), names
   the branch to fetch when an object is absent, supports `--dry-run`.
   Proven in the sandbox: all 99 resolve; the `raw` group written and
   re-checked byte-exact; a second run is a no-op; a tampered file is kept.
5. **`requires_data`.** A registered marker plus a `conftest.py` hook that
   skips marked tests, with a visible reason, when
   `data/bloomberg/sp500_ohlcv.csv` is absent under the root. Most data-backed
   suites already had module-level `skipif` guards that now resolve through
   `paths`; the marker covers the rest.

## What worked

- The EV smoke with the in-repo data hidden and `SWE_DATA_ROOT` set reproduced
  the §9.4 numbers exactly (MSFT 110.13 / XOM 80.14 / AAPL −12.61 / UNH −20.69,
  JPM event-dropped).
- `check` on the external root: `checked 99 manifest files: 99 ok, 0 missing,
  0 mismatched`.
- Every guard green after the doc rewrite: manifest coverage, worklog index,
  doc currency; ruff clean on the CI scope.

## What didn't

- The first proof run died at collection: my mechanical patch of
  `tests/test_corp_action_gate.py` produced
  `paths.bloomberg_dir() / "x.csv".exists()` (precedence — `.exists()` bound
  to the string). Fixed with parentheses; a sweep found no second instance.
- `tests/test_broad_pull_loaders.py` and `tests/test_data_integration.py` had
  the `from engine import paths` line inserted inside a parenthesised import;
  moved after the closing parenthesis.
- The Tier-C Drive backup record on `backup/drive-tier-c-2026-07-22` says the
  `theta` upload (≈11 GB) was still in progress on 2026-07-22 and never
  re-verified; the record is salvaged into `docs/DATA_INVENTORY.md` §C.1 with
  that caveat, so the branch can go.

## How we fixed it

Steps 1–4 ship in one PR (code + docs + tests). Step 5 (untracking, no history
rewrite) is a separate PR held until the Operator reports the desktop
`check` line; step 6 (branch deletion, #507 closure) follows step 5 and a
Drive copy of the ticks, which today exist only on their branch.

## Evidence

Sandbox, 4 CPUs, `SWE_DATA_PROVIDER=bloomberg`; the external root `R` is a
scratchpad copy of the 99 manifest files (`check --root R` → `99 ok, 0 missing,
0 mismatched`); "hidden" = `data/bloomberg`, `data/features`, `data_raw`,
`data_processed` moved aside for the run and restored (0 deletions after).

| Run | Posture | Fast lane (`-m "not backtest_regression"`) |
|---|---|---|
| [0] | in-repo data, no root (legacy) | `3174 passed, 28 skipped, 4 deselected, 20 xfailed` in 611 s |
| [A] | data hidden, `SWE_DATA_ROOT=R` | `2 failed, 3175 passed, 29 skipped` — both failures were resolver bypasses / a shared-dir assumption, fixed in `a6ef421` and `305313d` |
| [B] | data hidden, no root, CI coverage form | `58 failed, 2954 passed, 205 skipped, 3 errors` in 264 s; `Total coverage: 83.58%` (floor 80) — the 61 ids are the `requires_data` set |
| marked | `SWE_DATA_ROOT=<empty dir>`, the 18 files of [B]'s list | `154 passed, 72 skipped (51 with the D31 reason), 2 xfailed, 0 failed` |

EV smoke with the data hidden and `SWE_DATA_ROOT=R`: `MSFT 110.13 / XOM 80.14 /
AAPL −12.61 / UNH −20.69`, `drops: {'total_dropped': 1, 'by_gate': {'event': 1}}`.
`tests/test_deep_read_connector.py` with `SWE_DATA_ROOT=R`: `10 passed` — the six
deep-history assembly tests ran for the first time anywhere.
`materialize --dry-run` into an empty root: `would write 99, 0 unavailable`;
`--group raw` written and re-checked: `11 ok`; second run `11 already present,
wrote 0`; a tampered file: `MISMATCH … (kept, not overwritten)`.
Final no-data proof in the exact CI form (`pytest tests/ -m "not
backtest_regression" --cov=engine --cov=data --cov-fail-under=80`, the four data
trees hidden, no root, `f72d16d`): `5 failed, 2958 passed, 266 skipped (51 via
requires_data), 8 deselected, 6 xfailed` in 249 s; `Total coverage: 83.34%`
(floor 80 holds). The five were the snapshot fingerprint guards in
`tests/test_backtest_regression.py` (the three-way proof had ignored that file
by path); marked in the last commit — the file reports `4 passed, 5 skipped`
on an empty root and `9 passed, 4 deselected` with the data, as in PR #524.
**Per-file floors without data** (`scripts/check_coverage_floors.py`):
`engine/data_connector.py` 82.46% < floor 88, `engine/wheel_runner.py` 76.84% <
floor 77 (the other five hold). This is the one CI consequence step 5 must
settle before the data is untracked.

## Unresolved / handoff

- **Operator, on the desktop (PowerShell, from the checkout):**
  `git fetch origin deep-history/bloomberg-raw claude/daybot-bloomberg-pull`,
  `python scripts/data_manifest.py materialize --root D:\swe-data`,
  `python scripts/data_manifest.py check --root D:\swe-data` — report the
  `checked 99 …` line. Then move the local-only stores under the root and set
  `SWE_DATA_ROOT` (DATA_POLICY §6). Only then may the step-5 PR merge.
- The day-bot ticks (491 MB) have no copy outside git; upload to Drive before
  `claude/daybot-bloomberg-pull` is deleted.
- Whether sandboxes should hold a small committed fixture subset so the §9.4
  smoke can run there is an explicit Operator choice (D31, rejected
  alternatives) — today they hold no data.
- **Step-5 blocker to settle first:** without data the per-file coverage floors
  fail for `engine/data_connector.py` (82.5% vs 88) and `engine/wheel_runner.py`
  (76.8% vs 77) while the aggregate holds (83.3% vs 80). Options: recalibrate
  those two floors to the no-data measurement minus 2pp (the floors' own rule;
  the desktop full lane keeps the real picture), or first add synthetic-fixture
  unit tests for the connector paths the data-backed suites covered. Decide in
  the step-5 Execution Prompt.
- History purge (3.01 GB of data blobs in the pack): separate decision.
