---
id: d33-tool-equals
title: "Drive consolidation: an area name may hold one key=value part (2a-ii)"
kind: feature
status: in-flight
terminal: sandbox
pr: 545
decisions: [D33, D34]
date: 2026-09-28
headline: scripts/drive_consolidate.py now takes an area name whose folder names may each hold one '=' between two plain runs (ticker=AAPL), so card 2a-ii can bring the Theta trees home one ticker folder at a time. A ticker area puts a file where a plan of the whole SmartWheelData area puts it, unless the file has a conflict or a duplicate outside the folder; malformed names stay refused; no other rule changed. 31 new tests; 13 fail against the old tool.
surface: [scripts/drive_consolidate.py, tests/test_drive_consolidate.py, TESTING.md, CHANGELOG.md]
---

## Goal

Card 2a-ii brings `option_history` (71,027 files, 154 names) and its banded backup
(51,729 files, 503 names) home from Drive's `SmartWheelData/data_processed/theta/`,
one chunk at a time (D34; `PROJECT_STATE.md` §0 B). Each name's folder is called
`ticker=<SYMBOL>`, and the tool refused any area name with `=`, so a chunk could not be
one ticker folder. The Operator confirmed the change on 2026-09-28 ("yes, go ahead
with the tool change"). The Execution Prompt is on campaign issue #544 and on PR #545.

## What we tried

1. **Where an `=` would travel.** Before changing the rule, every use of an area name
   was traced:
   - it names the destination, `data_archive/drive-legacy/<area>/<path>`, and the
     conflict folder `<area>/_conflicts/<id>/`;
   - it is written to JSON (census, plan, copy log) and to the CSV ledger through
     Python's `csv` module;
   - it never reaches rclone as a path, because downloads are staged under the Drive
     id. The Drive query quotes the folder name, and no code splits on `=`.
2. **The narrowest rule that serves.** Each folder name may hold one `=`, strictly
   inside: `[A-Za-z0-9._-]+(?:=[A-Za-z0-9._-]+)?`. A plain `=` anywhere would also
   have worked, but it admits `=AAPL`, `a=` and `a=b=c`, none of which a hive
   partition folder uses. It would also let an area name start with `=`, which a
   spreadsheet opening the ledger reads as a formula.
3. **An independent check** by four fresh contexts, each in its own clone, with a
   skeptic per lens (the rule, the tests, the records, Windows and Drive). It is
   posted on PR #545.

## What worked

- The rule above. `_check_areas` and `sweep`'s destination check share it, so both take
  `ticker=AAPL` and both still refuse the malformed forms. The checkers compared the
  old and new tool on over a million generated names: no name the old tool accepted
  is refused now, and every newly accepted name has at most one `=` per folder name,
  strictly inside.
- The end-to-end test plans a fake `SmartWheelData` tree twice: once as the whole area,
  as card 1 did, and once as the ticker folder alone, as 2a-ii will. Every row of the
  ticker plan has the same class and destination as the same Drive id's row in the
  whole plan. Nothing of the sibling `ticker=MSFT` enters the ticker plan. The ledger
  CSV reads back the same `area` and `dest` strings as `plan.json`. `copy` and `verify`
  on the ticker plan both exit 0, and each file's bytes sit under
  `.../option_history/ticker=AAPL/`, including a nested `expiration=2016-01-22` folder.

## What didn't

- **The old rule ended in `$`.** Python's `$` also matches before a trailing newline,
  so the old rule accepted `a/b\n`. The Windows-name check refused it next, so no such
  name ever got through. The new rule ends in `\Z` and refuses it itself.
- **My first records overstated two things.** The check found both:
  - they said a ticker area always puts a file where the whole-area plan does. Two
    cases differ, by design: a file whose natural path is taken goes to the chunk's
    own `<area>/_conflicts/<id>/`, and a file the whole plan calls a `duplicate` (the
    same bytes earlier in the tree) is a `copy` in the ticker plan unless its twin is
    already home;
  - the worklog said the whole-tree `ruff check .` and `ruff format --check .` were
    clean. Only CI's lint scope is.
- **My first tests let six mutants through.** Dropping digits from the rule, a `.`
  from the key, or digits from the value; letting a space or any character after the
  `=`; and restoring the old refusal message. Each passed every test. No accepted
  name had a digit, and no refused name had a bad character after the `=`. The
  banded backup's chunk name (`option_history_banded_backup_2026-06-01/ticker=AAPL`)
  was unguarded. The first commit also left a placeholder in the Evidence section.

## How we fixed it

- `scripts/drive_consolidate.py`: `_AREA_PART` and `AREA_NAME` (one comment line), and
  `_check_areas`' refusal message, which names the new rule. No other line changed.
- `tests/test_drive_consolidate.py`: 31 tests, each named `…key_value…`:
  - 7 accepted names, among them the banded backup's chunk name,
    `x/expiration=2016-01-22` and `a1.b/c`;
  - 2 checks that the rule itself refuses a trailing newline (`a/b\n`, `ticker=AAPL\n`);
  - a census of an area named `x/ticker=AAPL`;
  - 11 refused names, each refused by a census that writes no output, among them
    `ticker=AA PL` and `ticker=A$B`;
  - 3 checks that the refusal names the rule;
  - the end-to-end test above;
  - `sweep` into `data_archive/old/ticker=AAPL`, and 5 refused sweep destinations.
  The "Pins:" docstring gains one line, qualified as above. The six mutants are each
  caught now.
- The records say what the end-to-end test covers, and what it does not.

## Evidence

- Fail before, pass after: the base tool (`git show origin/main:scripts/drive_consolidate.py`,
  blob `051b83ee`), beside the new test file in a folder outside the checkout,
  `-k key_value`: `13 failed, 18 passed`. The 13 are the 6 accepted names that hold an
  `=`, the end-anchor test on `a/b\n` (the old rule's `$` accepted it), the census of
  `x/ticker=AAPL`, the 3 message checks, the end-to-end test and the sweep into
  `ticker=AAPL`. The refusal tests pass on both tools. The same in the checkout:
  `31 passed`.
- The six mutants, each run with `-k key_value`: no digits `3 failed`; no `.` in the key
  `1 failed`; no digits in the value `1 failed`; a space in the value `3 failed`; any
  value `2 failed`; the old message `3 failed`.
- The whole test file and the fast lane: see the Run Summary on PR #545 for this head.
  At `3b8a4d5` (21 tests), the whole file gave `222 passed, 1 skipped` and the fast
  lane, `python -m pytest tests/ -m "not backtest_regression" -q`, gave
  `3201 passed, 272 skipped, 8 deselected, 6 xfailed` (0 failed).
- Lint: `ruff check` and `ruff format --check` on CI's lint scope (`engine/ data/ tests/
  scripts/ utils/ dashboard/ backtests/regression/`) are clean. The whole tree reports
  410 findings and 72 files to reformat, the same counts as `main` `5e90b1f`, all in
  folders outside CI's scope that this change does not touch.
- The four structure checks are green; their output is in the Run Summary on the PR.

## Unresolved / handoff

- The PR merges only after desktop card 2a-i reports: 2a-i runs against the tool's
  blob `051b83ee`, and `main` must not move the tool under it.
- For card 2a-ii's driver:
  - **Conflicts and duplicates differ from card 1's plan, by design.** A conflict goes
    to the chunk's own `_conflicts/<id>/`. A file whose twin (the same bytes) lies in
    another ticker folder, or in the other Theta tree, is a `copy` in the chunk's plan,
    while card 1's plan calls it a `duplicate`. The driver must compare each chunk's
    copies with card 1's plan and stop, or defer the chunk, on any difference, as
    2a-i's does.
  - **The area name is a label.** The census checks the area's `folder` and `id`, not
    that the name's last part is that folder. The driver must build each name from
    card 1's path and take the folder from it, as 2a-i's does.
  - **cmd.exe splits an unquoted argument at `=`.** A `.bat` or `.cmd` wrapper must
    double-quote any argument that holds one. 2a-i passes areas in a JSON file and runs
    the tool from Git Bash, which avoids this.
- `CLAUDE.md` §4 and §9 name the whole-tree `ruff check .` and `ruff format --check .`,
  while CI lints a scope; on `main` the whole tree is not clean. Which wording the
  rule-books keep is the pen's call.
