---
id: d33-drive-parent-id-2026-09-25
title: Record the SmartWheelData area's Drive parent id (N-7 of #534)
kind: docs
status: merged
terminal: sandbox
pr: 536
decisions: [D33]
date: 2026-09-25
headline: The tool's built-in Drive areas named the folder that holds SmartWheelData (OptionsEngine_Project, 1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4), and the inventory record did not. The id is now in DATA_INVENTORY §C.3, and the areas test ties each area's id, folder name and parent to one row of §C.3.
surface: [docs/DATA_INVENTORY.md, tests/test_drive_consolidate.py, CHANGELOG.md]
---

## Goal

Close finding N-7 of the #534 review. `DEFAULT_AREAS` in
`scripts/drive_consolidate.py` gives SmartWheelData's parent folder by id. The
record future agents read, `docs/DATA_INVENTORY.md` §C.3, gave only its path.
The Operator approved recording it on 2026-09-25: "go, and yes on the Drive-id
record".

## What we tried

The id was checked on Drive before it was written down. The read-only Drive
connector gives SmartWheelData (`1wCFPBf0o9PJMy2f2vy34S316XFc1Sq3e`) the parent
`1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4`, whose title is `OptionsEngine_Project`. That
matches the tool.

## What worked

- §C.3 row 2 now names the parent folder with its id.
- `test_the_default_areas_match_the_inventory_record_and_their_modes` now finds
  each area's own row of §C.3 by its id: exactly one row. That row must also carry
  the area's folder name. Its "Folder (id)" cell must carry the parent: the
  parent's id, or "top of My Drive" for an area at the top. The four day-bot areas' parent
  (`1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0`) was already recorded in row 4.

## What didn't

The first version of the test only checked that each parent id appeared somewhere
in the file. Codex's review of #536 showed the gap: both non-root parent ids were
already in the file, so giving SmartWheelData the day-bot parent would still have
passed. The test now reads the §C.3 table row by row.

The row-based version still let a wrong parent through when the wrong id was
another one in the same row, such as the area's own child or a sibling area. The
independent check of #536 found it. The parent is now looked for only in the
row's "Folder (id)" cell, and it must differ from the area's own id.

## How we fixed it

A one-line change to the record, and the areas test reads §C.3 cell by cell. The
tool is unchanged.

## Evidence

- Fail-before: with the id taken out of the record, the test fails with
  `AssertionError: SmartWheelData: parent 1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4`.
  With the id put back, it passes.
- Twelve deliberate breaks, each made and then undone, all fail the test.
  In the tool, SmartWheelData's parent becomes:
  - the day-bot parent (the gap Codex found, which the first version passed);
  - its own child's id (`bloomberg`);
  - its own id;
  - `root`.

  Also in the tool:
  - `vendor_swe_data`'s parent becomes a sibling's id;
  - a folder name is wrong.

  In the record:
  - the parent id is removed from row 2;
  - the parent id moves from row 2 to row 3;
  - the parent id moves into row 2's "What it holds" cell;
  - row 1 loses "top of My Drive";
  - the day-bot parent is removed from row 4;
  - an area id appears in two rows.

  The child and sibling cases passed the row-based version.
- The fast lane, ruff and the structure checks: see the Run Summary on the pull
  request.

## Unresolved / handoff

- An opportunistic fix in the same pull request: the D33 fragment
  (`d33-data-estate-2026-09-24-…`) still read `in-flight` after #533 merged. It
  now reads `merged`, with `pr: 533`.
- The test expects exactly four rows in §C.3. Card 2 adds the new `swe-data/`
  folder to the record; the same pull request updates this test.
- 64 older fragments still read `in-flight`. That is the history the September
  restart recorded, and it is left alone here.
