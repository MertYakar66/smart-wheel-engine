---
id: data-manifest-bundle-hint
title: The data_manifest hint: the bundle, not deleted branches
kind: fix
status: in-flight
terminal: remote-sandbox
pr:
decisions: [D31]
date: 2026-10-09
headline: materialize names the bundle restore for a deleted branch and the fetch for main; a test restores a bundle end to end
surface: [scripts/data_manifest.py, tests/test_data_manifest.py, scripts/pull_iv_surface.py, tests/test_deep_read_connector.py, FILE_MANIFEST.md]
---

## Goal
After D31 step 6 (#559) deleted the four data branches from GitHub,
`scripts/data_manifest.py materialize` still told whoever filled a data root
to run `git fetch origin <branch>` for a missing object, a command that now
fails, and its test pinned that text. Make the message, the docstrings and
the registry row say where the commits are now: the full-history bundle,
restored into a repository of its own and passed as `--repo`
(`docs/DATA_POLICY.md`, first fill, since #560). Prove that procedure with a
test. Campaign #544.

## What we tried
The card's six changes, as written: the message in `cmd_materialize`, the
module docstring and usage line, the test changes (one renamed, two new),
two stale code notes and one `FILE_MANIFEST.md` row.

## What worked
- The message now branches on the source: `main` gets `git fetch origin main`
  (with `--unshallow` for a shallow clone) or a restored bundle as `--repo`;
  any other branch gets the bundle restore. The two hints name
  `docs/DATA_POLICY.md`'s first fill.
- `test_materialize_from_a_restored_bundle` runs the documented procedure on
  the fixture: `git bundle create --all`, `git init --bare`,
  `git fetch <bundle> "+refs/*:refs/*"`, `git fsck --full --no-dangling`, then
  `materialize --repo <bare>` writes 3 files and `check` reads
  `3 ok, 0 missing, 0 mismatched`. A bare repository works as `--repo`
  unchanged: `materialize` only calls `git -C <repo> cat-file`.

## What didn't
- The sandbox's first fast-lane run stopped at collection (7 errors):
  `requests_mock` and `hypothesis` are dev extras the start hook does not
  install. `pip install -e ".[dev]"` (what CI uses) fixed it; nothing in the
  repository changed for it.

## How we fixed it
Message and docs only; `materialize`'s logic (what it writes, how it
verifies, its exit codes) is unchanged. The renamed test fails against
`origin/main`'s script and passes against this one.

## Evidence
- `python -m pytest tests/test_data_manifest.py tests/test_deep_read_connector.py -q`:
  `18 passed, 7 skipped`; the three named tests PASSED under `-v`.
- With `origin/main`'s `scripts/data_manifest.py` swapped in:
  `FAILED tests/test_data_manifest.py::test_materialize_never_overwrites_and_names_the_bundle_restore`,
  `1 failed, 14 passed`; restored: `15 passed`.
- `python -m pytest tests/ -m "not backtest_regression" -q`:
  `3245 passed, 273 skipped, 8 deselected, 6 xfailed, 171 warnings in 313.72s`.
- `ruff check` / `ruff format --check` on the four changed `.py` files:
  `All checks passed!`, `4 files already formatted`.

## Unresolved / handoff
Nothing new found. The pen sets `status` and `pr:` at merge.
