---
id: d33-tool-equals
title: "Drive consolidation: an area name may hold one key=value part (2a-ii)"
kind: feature
status: in-flight
terminal: sandbox
pr:
decisions: [D33, D34]
date: 2026-09-28
headline: scripts/drive_consolidate.py now takes an area name whose folder names may each hold one '=' between two plain runs (ticker=AAPL), so card 2a-ii can bring the Theta trees home one ticker folder at a time. A ticker area puts each file where a plan of the whole SmartWheelData area puts it; malformed names stay refused; no other rule changed. 21 new tests; 8 fail against the old tool (the 7 that accept a new name, and the one that pins the rule's end anchor).
surface: [scripts/drive_consolidate.py, tests/test_drive_consolidate.py, TESTING.md, CHANGELOG.md]
---

## Goal

Card 2a-ii brings `option_history` (71,027 files, 154 names) and its banded backup
(51,729 files, 503 names) home from Drive's `SmartWheelData/data_processed/theta/`,
one chunk at a time (D34; `PROJECT_STATE.md` §0 B). Each name's folder is called
`ticker=<SYMBOL>`, and the tool refused any area name with `=`, so a chunk could not be
one ticker folder. The Operator confirmed the change on 2026-09-28 ("yes, go ahead
with the tool change"). The Execution Prompt is on campaign issue #544 and on the PR.

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

## What worked

- The rule above. `_check_areas` and `sweep`'s destination check share it, so both take
  `ticker=AAPL` and both still refuse the malformed forms.
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
  name ever got through. The new rule ends in `\Z` and refuses it itself. No name the
  old tool accepted is refused now.

## How we fixed it

- `scripts/drive_consolidate.py`: `_AREA_PART` and `AREA_NAME` (one comment line), and
  `_check_areas`' refusal message, which names the new rule. No other line changed.
- `tests/test_drive_consolidate.py`: 21 tests, each named `…key_value…`:
  - 4 accepted names;
  - 2 checks that the rule itself refuses a trailing newline (`a/b\n`, `ticker=AAPL\n`);
  - a census of an area named `x/ticker=AAPL`;
  - 9 refused names, each refused by a census that writes no output;
  - the end-to-end test above;
  - `sweep` into `data_archive/old/ticker=AAPL`, and 3 refused sweep destinations.
  The "Pins:" docstring gains one line.

## Evidence

- Fail before, pass after: the base tool (`git show origin/main:scripts/drive_consolidate.py`,
  blob `051b83ee`), beside the new test file in a folder outside the checkout,
  `-k key_value`: `8 failed, 13 passed`. The 7 acceptance tests fail, and so does the
  end-anchor test on `a/b\n`, which the old rule's `$` accepted. The refusal tests pass on
  both tools. The same in the checkout: `21 passed`.
- The whole test file and the fast lane: FAST_LANE_EVIDENCE
- `ruff check .` and `ruff format --check .` are clean. The four structure checks are
  green (their output is in the Run Summary on the PR).

## Unresolved / handoff

- The PR merges only after desktop card 2a-i reports: 2a-i runs against the tool's
  blob `051b83ee`, and `main` must not move the tool under it.
- A conflict destination still differs between a ticker plan and the whole plan
  (`<area>/_conflicts/<id>/`), by design. Card 2a-ii's driver must compare each chunk's
  copies with card 1's and stop on any difference, as 2a-i's does.
