---
id: d33-tool-equals
title: "Drive consolidation: an area name may hold one key=value part (2a-ii)"
kind: feature
status: in-flight
terminal: sandbox
pr: 545
decisions: [D33, D34]
date: 2026-09-28
headline: scripts/drive_consolidate.py now takes an area name whose folder names may each hold one '=' between two plain runs (ticker=AAPL), and nothing else, so card 2a-ii can bring the Theta trees home one ticker folder at a time. A ticker area's plan can differ from card 1's in any field, and no list of the cases is complete, so the 2a-ii driver must check each chunk against card 1's plan; malformed names stay refused; no other rule changed. 64 new tests; 14 fail against the old tool; the 18 mutants three checks found each fail.
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
3. **Three independent checks**, each by fresh contexts in their own clones, with a
   skeptic per finding. The first (four lenses: the rule, the tests, the records, Windows
   and Drive) checked `3b8a4d5`; the second checked the fixes at `04484d8`; the third
   checked `c0b3d95`. All three are posted on PR #545.

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
  In this tree nothing depends on objects outside the folder.

## What didn't

- **The old rule ended in `$`.** Python's `$` also matches before a trailing newline,
  so the old rule accepted `a/b\n`. The Windows-name check refused it next, so no such
  name ever got through. The new rule ends in `\Z` and refuses it itself.
- **My records kept claiming a complete list, and each check found it incomplete.**
  - First they said a ticker area always places a file where the whole-area plan does.
    The first check found two exceptions, conflicts and duplicates whose twin lies
    elsewhere, and my correction listed only those two.
  - The second check found two more: a second parent outside the chunk, and an MD5
    shared outside it. My next correction listed four.
  - The third check found at least five more, among them:
    - a conflict wholly inside the folder;
    - a name outside the chunk that takes the path first;
    - a path near the length limit;
    - an empty file already home at the whole plan's conflict path;
    - a credential-shaped folder, or one with a second parent, above the chunk.
  - The lesson: the chunk plan decides on its own anything that depends on what it does
    not see, so no list is complete. The records now state that rule, give examples,
    and require the driver to tolerate no difference.
  - Two smaller overstatements: the worklog said the whole-tree `ruff check .` and `ruff
    format --check .` were clean (only CI's lint scope is), and the records described the
    2a-ii driver's comparison in the present tense, though that driver is not written.
- **My tests let mutants through, three times.** At `3b8a4d5`, six passed every test:
  - dropping digits from the rule, `.` from the key, or digits from the value;
  - letting a space, or any character but whitespace, `/` and `=`, into the value;
  - restoring the old refusal message.
  The refused names pinned the value only against a second `=` and a newline. At
  `04484d8`, seven more passed:
  - dropping `_` from the value, or letting `/` into the value;
  - letting `\w` or other punctuation into either half;
  - making sweep check only the key half for Windows. Two of these, both in the key half,
    predate this change.
  At `c0b3d95`, five more passed, because the per-character test never varied the key of
  a `key=value` folder name, nor any folder name after the first.
  The first commit also left a placeholder in this fragment's Evidence section.

## How we fixed it

- `scripts/drive_consolidate.py`: `_AREA_PART` and `AREA_NAME` (one comment line), and
  `_check_areas`' refusal message, which names the new rule. No other line changed.
- `tests/test_drive_consolidate.py`: 64 tests, each named `…key_value…`:
  - 8 accepted names, among them the banded backup's chunk name,
    `x/expiration=2016-01-22`, `x/ticker=BRK_B` and `a1.b/c`;
  - 2 checks that the rule itself refuses a trailing newline (`a/b\n`, `ticker=AAPL\n`);
  - a census of an area named `x/ticker=AAPL`;
  - 12 refused names, each refused by a census that writes no output, among them
    `ticker=AA PL`, `ticker=A$B` and `ticker=/AAPL`;
  - 30 characters, each refused in four places: the value, a plain first folder name, the
    key of a later `key=value` name, and a later value. They are every printable ASCII
    character outside the rule's own, plus `Ä` and `٣`;
  - 3 checks that the refusal names the rule;
  - the end-to-end test above;
  - `sweep` into `data_archive/old/ticker=AAPL`, and 6 refused sweep destinations, among
    them `data_archive/ticker=A.`.
  The "Pins:" docstring gains one line, and the end-to-end test's comment states the rule
  above. All eighteen mutants the three checks found now fail at least one test.

## Evidence

- Fail before, pass after, at the final head: the base tool (`git show
  origin/main:scripts/drive_consolidate.py`, blob `051b83ee`), beside the new test file in
  a folder outside the checkout, `-k key_value`: `14 failed, 50 passed`. The 14 are the 7
  accepted names that hold an `=`, the end-anchor test on `a/b\n` (the old rule's `$`
  accepted it), the census of `x/ticker=AAPL`, the 3 message checks, the end-to-end test
  and the sweep into `ticker=AAPL`. The same in the checkout: `64 passed`.
- The eighteen mutants, each run with `-k key_value`, fail between 1 and 23 tests each.
  The Run Summary on PR #545 and its later comments give each count.
- The whole test file: `232 passed, 1 skipped` at `04484d8`, and `265 passed, 1 skipped`
  at `c0b3d95` and at the final head. The fast lane: locally at
  `3b8a4d5`, `3201 passed, 272 skipped, 8 deselected, 6 xfailed` (0 failed); at each later
  head, CI's Test Suite (3.11 and 3.12) ran it green, and no record gives its count.
- Lint: `ruff check` and `ruff format --check` on CI's lint scope (`engine/ data/ tests/
  scripts/ utils/ dashboard/ backtests/regression/`) are clean. The whole tree reports
  410 findings and 72 files to reformat, the same counts as `main` `5e90b1f`, all in
  folders outside CI's scope that this change does not touch.
- The four structure checks are green; their output is in the Run Summary on the PR.

## Unresolved / handoff

- The PR merges only after the pen has evaluated desktop card 2a-i's report. 2a-i ran
  against the tool's blob `051b83ee`, and `main` must not move the tool under it.
- **For card 2a-ii's driver, which is still to be written.**
  - **Every row, both ways.** A chunk's plan can differ from card 1's in any field (class,
    natural path, destination), wherever card 1's outcome depended on something the chunk's
    plan does not see, or decides on its own. For example: twins, second parents or
    ancestors outside the chunk; a name that takes the path first; a conflict inside or
    outside the folder; a path near the length limit. No list of these is complete, and
    none is to be accepted. So the driver must check each chunk as 2a-i's did. Drive under the chunk must be as card 1's
    census saw it. The chunk's fresh plan must copy exactly card 1's copies under the folder
    not yet home, at card 1's destinations. Every other row must keep card 1's class, except
    that a copy or duplicate whose bytes are now home may read `redundant`. Any other
    difference must stop the chunk, or defer it.
  - **Unresolved rows.** It must stop on any unresolved row in card 1's plan under the
    chunk, as 2a-i's does. `_check_areas` never refused a credential-shaped area name,
    before this change or after it. The key=value rule admits more of them, such as
    `x/token=1`. No Theta folder above a ticker is credential-shaped, and a later change
    could refuse such names.
  - **Deletion flags.** A chunk plan's `path_unique` and `deletable` are never card 1's,
    and never a forecast for card 3.
- **The area name is a label.** The census checks the area's `folder` and `id`, not
  that the name's last part is that folder. The driver must build each name from card
  1's plan, as `SmartWheelData/` plus the folder's `path` there, and take the folder from
  it, as 2a-i's does.
- **cmd.exe splits an unquoted argument at `=`.** A `.bat` or `.cmd` wrapper must
  double-quote any argument that holds one. 2a-i passes areas in a JSON file and runs
  the tool from Git Bash, which avoids this.
- `CLAUDE.md` §4 and §9 name the whole-tree `ruff check .` and `ruff format --check .`,
  while CI lints a scope; on `main` the whole tree is not clean. Which wording the
  rule-books keep is the pen's call.
