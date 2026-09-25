---
id: d33-drive-parent-id-2026-09-25
title: Record the SmartWheelData area's Drive parent id (N-7 of #534)
kind: docs
status: in-flight
terminal: sandbox
pr:
decisions: [D33]
date: 2026-09-25
headline: The tool's built-in Drive areas named the folder that holds SmartWheelData (OptionsEngine_Project, 1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4), and the inventory record did not. The id is now in DATA_INVENTORY §C.3, and the areas test checks every non-root parent id against the record as well as every area id.
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
- `test_the_default_areas_match_the_inventory_record_and_their_modes` also checks
  that every area whose parent is not `root` has its parent id in the record. The
  four day-bot areas' parent (`1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0`) was already
  recorded in row 4.

## What didn't

Nothing.

## How we fixed it

A one-line change to the record and a two-line addition to the test. The tool is
unchanged.

## Evidence

- Fail-before: with the id taken out of the record, the test fails with
  `AssertionError: SmartWheelData: parent 1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4`.
  With the id put back, it passes.
- The fast lane, ruff and the structure checks: see the Run Summary on the pull
  request.

## Unresolved / handoff

- An opportunistic fix in the same pull request: the D33 fragment
  (`d33-data-estate-2026-09-24-…`) still read `in-flight` after #533 merged. It
  now reads `merged`, with `pr: 533`.
- 64 older fragments still read `in-flight`. That is the history the September
  restart recorded, and it is left alone here.
