---
id: card3-code-part1
title: card 3 code: census owners, swe-data by id, trashlist
kind: feature
status: in-flight
terminal: desktop Executor
pr:
decisions: [D33, D34, D36]
date: 2026-10-15
headline: "scripts/drive_consolidate.py records each object's owner, reads swe-data/ by its id and the chain to OptionsEngine_Project, and gains trashlist (every object of the three consolidate areas listed, a folder or a blocker, with a proof whose sha256 the Operator approves) and recount (nothing changed since the proof, on the day of the trash). Read-only; no Drive was read."
surface:
  - scripts/drive_consolidate.py
  - tests/test_drive_consolidate.py
  - tests/test_drive_trashlist.py
  - docs/DATA_INVENTORY.md
---

## Goal

Card 3's one code run under D36's middle route (the Operator's choice of 2026-10-15): give
the Drive consolidation tool what card 3a needs to prove, file by file, that the three old
Drive folders (`swe-local-only`, `SmartWheelData`, `smart-wheel-engine`) hold nothing whose
bytes are not proven twice, on the desktop and in `swe-data/`, so the Operator can trash
those three folders by hand in Drive's web page after a recount. The Execution Prompt (on
the PR) and card 3's design (version 6 with addenda A, B and C, #544) specify it.

## What we tried

- Read rclone v1.74.4's source for the two claims the run depends on (cited in the PR's Run
  Summary): `backend query` with `--drive-auth-owner-only` only adds the `owners` field; and
  `backend query -M` returns Drive's `permissions`, while `lsjson -M` alone does not.
- The census gains `owned` and `trashed` per object, and the area folder's owner and
  modified time; the `census` command (never `census()`) reads the chain, `swe-data/` by id,
  and the sharing of each link, `swe-data/` and each area folder.
- `trashlist` classes every object of the three consolidate areas, and each area folder,
  from the census, never from the plan's rows; `recount` compares a fresher census with the
  proof and reads each twin again.

## What worked

- The five listing conditions, the folder rule and the accounting rule, each as one line
  of code that a mutation can remove; the proof as canonical JSON with no time in it.
- The fake rclone of `tests/test_drive_consolidate.py`, extended (owners, permissions, a
  trash, My Drive's root under its own id, the chain and `swe-data/` always present), serves
  both test files.

## What didn't

- A recount that matched proof rows to areas by the built-in names would miss a proof made
  from a census whose area names differ: it now finds each area by its folder's row (its id).
- Some rules are covered twice by design: `--out` that exists is refused before any work and
  again by the exclusive `mkdir`; `kind` is derived from the MIME type, so recount's `kind`
  comparison cannot change without the `mime` one. The load-bearing table in the Run
  Summary says which.

## How we fixed it

See the PR: `scripts/drive_consolidate.py` (census fields, `locate`, `trashlist`,
`recount`), the fake's extension, and `tests/test_drive_trashlist.py`.

## Evidence

The PR's Run Summary holds every command and its output: the test counts before and after,
the fast lane, ruff, the registry checks, the load-bearing table and CI.

## Unresolved / handoff

- The pen closes `docs/deadlines.md`'s row "no tool run reads Drive until the tool finds
  `swe-data/` by id" after the merge, and the matching sentence under `docs/DATA_INVENTORY.md`
  §C.3's table (this run changed only rows 1 and 3, as its prompt said).
- Card 3a (desktop, read-only on Drive): the six-area census, `swe-data/`'s census, the
  inventory, the plan, `bytecheck` if any check is pending, then `trashlist`.
