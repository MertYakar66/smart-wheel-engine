---
id: d39-data-raw
title: Desktop card for D39: the day-bot's data_raw home and into swe-data
kind: verification
status: merged
terminal: desktop
pr: 588
decisions: [D33, D34, D39]
date: 2026-10-17
headline: The day-bot's data_raw is home and in swe-data/ — 1,725 files and 34,816,381 B, exactly card 1's plan; the copy read 1,725 copied and verified, 0 not, the verify 1,725 ok, the upload 1,725 new with 0 errors, and the check both ways by MD5 1,725 the same with 0 on one side only; end check "no finding" and the manifest 144/0/0 by full hash.
surface: []
---

data_raw is home and in swe-data: 1,725 files, 34,816,381 B

## Goal

D39 point 1, and point 2 for its new subtree (campaign #544). Bring the day-bot's Drive folder
`_local_archive/data_raw` (`1uHSbrEaZoyW_Tgn1BimSz02KOJ18e616`, in
`1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0`) home into the data root under
`data_archive/drive-legacy/day-bot-local-archive/data_raw/`, never over an existing file and checking
every file by its bytes. Then put the new subtree into `swe-data/` on Drive
(`13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b`) and check it there both ways by MD5, so that the desktop and
`swe-data/` match again. The Drive original stays where it is. The run used
`scripts/drive_consolidate.py` as `main` holds it (git blob `6903c9c8…`) for steps 3 to 7 and for
step 8's filters, and rclone v1.74.4 for the upload and the check. The root's `_logs/` is not in
`swe-data/`; it goes up with D33 point 8's routine, so D39 point 2 is done only after that.

## What we tried

The card's steps, in order, each in one Bash call writing its whole output to its own file in
`%TEMP%` (`swe-d39-step<N>.txt`):

- **Step 0.** The worktree `.claude/worktrees/d39-data-raw` on the new branch
  `claude/d39-data-raw` from `origin/main` (`78a2116`). Its lines: `branch claude/d39-data-raw`,
  `blob 6903c9c82dbdea48382f9e78ce45632289777951`, `D39 on main`, `changes 0`, `SWE_DATA_ROOT` the
  root, `rclone v1.74.4`, `checked 144 manifest files: 144 ok, 0 missing, 0 mismatched (size only)`,
  three `absent:` lines (the staging folder, `_logs/d39-data-raw`, the subtree), `free
  1039245369344`, and no `STOPPED:` line.
- **Step 1, the Drive sign-in.** `about exit 0`, `== the Drive sign-in works`.
- **The briefing**, said in chat in the card's words (in no step file), **and the REGISTER note** on
  #544, before step 2.
- **Step 2, the root before.** `== the root outside _logs: 147812 files, 18,644,236,734 B, 0 link(s)
  not followed`, and `== against card 3a's listing of 2026-10-16: then 147812 files; now 147812;
  gone, added or changed 0`.
- **Step 3, the census of `data_raw`.** Start `2026-10-17T10:32:51Z`, end `2026-10-17T10:36:24Z`.
- **Step 4, the inventory of the root.** Start `2026-10-17T10:36:56Z`, end `2026-10-17T10:48:12Z`.
- **Step 5, the plan** (`--limit 0`). Start `2026-10-17T10:48:43Z`, end `2026-10-17T10:48:44Z`.
- **Step 6, the copy home.** Start `2026-10-17T10:49:27Z`, end `2026-10-17T11:04:51Z`.
- **Step 7, the verify.** Start `2026-10-17T11:05:23Z`, end `2026-10-17T11:05:34Z`.
- **Step 8, the upload into `swe-data/`.** Start `2026-10-17T11:06:15Z`, end `2026-10-17T11:44:34Z`.
- **Step 9, the check of the subtree both ways by MD5.** Start `2026-10-17T11:45:13Z`, end
  `2026-10-17T11:47:48Z`.
- **Step 10, the end check.** Its block prints no times.
- **Step 11, the start of this record.** `Already up to date.` (the branch was level with `main`),
  the new fragment, `exit 0`, and the six sha256 lines below.

Each of steps 3 to 9 ran once, under the empty letter: there is no lettered step file. Each step file
has its `== step done: …` and `== END step N` lines, and the rule for steps 3 to 9, run in the session
after each step (it writes no file), read `== the step is done` each time. No step ran again.

## What worked

Every step. The decisive lines, from the step files:

```
census: day-bot-local-archive/data_raw: 4946 objects; 3306 files, 35,125,927 B, and every id, equal to rclone's own listing
census: 1 areas, 4946 objects; swe-data/ is 13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b, in 1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4
census exit 0

inventory: 147812 files (147811 hashed, 1 credential-shaped, not read), 18,644,236,641 B; 0 links skipped
inventory exit 0

day-bot-local-archive/data_raw [read-only]: 4946 objects: copy 1725 (34,816,381 B), duplicate 1581 (309,546 B), folder 1640 (0 B)
to copy home: 1725 files, 34,816,381 B
swe-data will hold about 149536 files, 18,679,053,022 B
root files excluded as credential-shaped: 1
left to settle or listed: 0
plan exit 0
== classes: copy 1725, duplicate 1581, folder 1640
== the plan passes its checks

copy: 1725 to copy home; 0 already there with the right bytes; 1725 to fetch
copy: 1725 copied and verified; 0 not
copy exit 0

verify: 1725 copies: 1725 ok, 0 missing, 0 mismatched; 0 root files Drive objects match: 0 ok, 0 missing or changed; 0 object(s) still need a byte check
verify exit 0

rclone copy exit 0
copied (new): 1725
ERROR lines: 0
33.203 MiB / 33.203 MiB, 100%, 5.176 KiB/s, ETA 0s

rclone check exit 0
== combined: the same 1725; only on the desktop 0; only in swe-data/ 0; differ 0; could not be read 0
== files under the subtree on the desktop: 1725; compared without a checksum: 0; names rclone saw twice: 0; compared by MD5: yes
== the subtree is the same on the desktop and in swe-data/
```

Steps 3 to 5 each ended `== step done: <its output>, sha256 <its hash>` (the hashes are the ones in
Evidence), and steps 6 to 9 `== step done: the copy`, `… the verify`, `… the upload` and `… the
check`.

**The census matched card 1's of 2026-09-25 exactly** (4,946 objects, 3,306 files, 35,125,927 B),
and **the plan matched card 1's plan exactly**: copy 1,725 (34,816,381 B), duplicate 1,581
(309,546 B), folder 1,640. The plan had no `redundant`, `unresolved` or `needs-byte-check` row. A
duplicate has the same bytes as a file that is copied, so it comes home once, at its twin's path;
by the card, the plan's ledger names each one's twin. The copy's censuses before and after the
downloads each read the same counts as step 3 and matched rclone's own listing again.

**The end state**, step 10:

```
checked 144 manifest files: 144 ok, 0 missing, 0 mismatched
== the root outside _logs: then 147812 files; now 149537; gone 0, changed 0, added 1725 (under the subtree 1725, elsewhere 0); unreadable 0; the plan's copies 1725
== the end state: no finding
```

No `== beside the root, not removed:` line was printed: the tool's staging folder beside the root
did not exist at the end.

**The numbers agree with one another.** The plan's forecast for `swe-data/`, 149,536 files and
18,679,053,022 B, is the inventory's 147,811 hashed files and 18,644,236,641 B plus the 1,725
copies and their 34,816,381 B. The root outside `_logs` now holds 149,537 files, that forecast plus
the one credential-shaped file, which stays off Drive.

## What didn't

Nothing failed. No step printed an `ERROR:` or `STOPPED:` line or a traceback, no census differed
from rclone's listing, no step ran again, and no BLOCKED, WAITING or STOPPED note was posted.

What differed from the card was the times, from each step file's own start and end lines: step 3
took 3 min 33 s (the card said about 5 min), step 4 11 min 16 s (about 10 min), step 5 1 s (a
minute), step 6 15 min 24 s (about 20 min), step 7 11 s (about 10 min), step 8 38 min 19 s
(about an hour) and step 9 2 min 35 s (a few minutes). From step 3's start to step 9's end took
1 h 14 min 57 s; the card's estimates for those steps add up to about 1 h 46 min and a few minutes
for the check (its two and a half hours also covers step 10, the records and CI). The upload ran at
about 1.33 s a file (2,299 s for 1,725 files, folders not counted).

## How we fixed it

Nothing needed fixing. What ran is the card's sequence, from the worktree, with the tool's blob
checked before steps 3 to 6's commands (the verify's block and step 8's `filters` do not check it):

```
python scripts/drive_consolidate.py census --remote gdrive: --areas "$W/areas.json" --root "$ROOT" --out "$W/census.json"
python scripts/drive_consolidate.py inventory --root "$ROOT" --out "$W/root.json"
python scripts/drive_consolidate.py plan --census "$W/census.json" --inventory "$W/root.json" --out "$W/plan.json" --limit 0
python scripts/drive_consolidate.py copy --plan "$W/plan.json" --root "$ROOT" --remote gdrive:
python scripts/drive_consolidate.py verify --plan "$W/plan.json" --root "$ROOT"
python scripts/drive_consolidate.py filters | grep -vxF '/_logs/**' > "$W/filters.txt"
rclone copy "$ROOT/$SUB" "gdrive:$SUB" --drive-root-folder-id "$SWE" --immutable --checksum --ignore-checksum --ignore-size --no-update-dir-modtime --drive-use-trash=true --exclude-from "$W/filters.txt" --ignore-case -v --stats 10m --stats-one-line --stats-log-level NOTICE
rclone check "$ROOT/$SUB" "gdrive:$SUB" --drive-root-folder-id "$SWE" --checksum --exclude-from "$W/filters.txt" --ignore-case --disable ListR --combined "$W/check-combined.txt" --low-level-retries 100 -v
python scripts/data_manifest.py check --root "$ROOT"
```

`$ROOT` is `C:/Users/merty/Desktop/swe-data`, `$W` is `$ROOT/_logs/d39-data-raw`, `$SUB` is
`data_archive/drive-legacy/day-bot-local-archive/data_raw` and `$SWE` is
`13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b`. `areas.json` names the one area, read-only. The upload was an
rclone `copy`, never a `sync`, `move` or delete; `--drive-root-folder-id` made `swe-data/` rclone's
root, so it wrote only inside it, and `--immutable` would have stopped it rather than change a file
already there.

## Evidence

**Where the record lives:** outside git, in the data root's `_logs/d39-data-raw/`. Each output, by
sha256, from step 11's file:

| Output | sha256 |
| ------ | ------ |
| `census.json` (step 3) | `0c7862bcff43056405dc2955d1cc58f7a5da3f2fd1915d8fc0b0277a4d65ec03` |
| `root.json` (step 4) | `81d64f9c3e430a902dbe37f07e7697a345ebd908189d43147165d5c39183cd7b` |
| `plan.json` (step 5) | `fd534fbbd661e1b9b6eadfaf0f59e49ef130fc0d1bbbf72805a827f0de372790` |
| `before.tsv` (step 2) | `477e7ae362f14e10fdfb2020cd8f06276af7acb3f5d0a4a303af45822cc932ef` |
| `filters.txt` (step 8) | `4d40bf42f6608331c1529b548223830b2f57bb68cb36d9d69ff737a711f725bf` |
| `check-combined.txt` (step 9) | `076df093d386dff5f1576fb017213d7376db966a9cb4d6ce3c9c0f59b6278e75` |

By a listing of the folder in this session (not in a step file), it also holds `areas.json`, the
plan's ledger `plan_ledger.csv`, the copy's log `plan_copy-20261017T104927Z-e0ba29.jsonl` (step
6's `copy: log` line), rclone's own logs `upload.log` and `check.log`, and the pointers `done-3.txt`
to `done-9.txt`. rclone's logs and the combined list name every file, so none of these files is
posted to GitHub; they stay on the desktop for the pen's check.

**The step files**, every `swe-d39-step*.txt` in `%TEMP%`, will be posted in the Run Summary's
comments on this card's pull request (step 13).

**The notes posted on #544:**

| Note | Link |
| ---- | ---- |
| REGISTER | [#544 comment 6232849411](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-6232849411) |

The report on #544 follows in step 13, after this record is pushed.

## Unresolved / handoff

1. **The pen's check.** A context that did not write this card checks the run against the step
   files and the record on the desktop.
2. **D39 point 2 is not done yet.** The root's `_logs/`, this card's `_logs/d39-data-raw/` among
   it, is not in `swe-data/`: card 2c's filters left it out. The pen writes D33 point 8's routine
   into `docs/DATA_INVENTORY.md` §C and runs it once for `_logs/`; point 2 is done when that is
   checked.
3. **D39 point 5.** Pulling new data starts only after points 1 and 2 are done and checked; then
   the pen brings the Operator the plan for new data.
4. **Drive holds `data_raw` twice now**: the day-bot's original, which stays where it is (D39
   point 1), and the new copy in `swe-data/`. Removing the original is a step of its own, with the
   Operator's yes (D39 point 3).
5. **The sign-in.** The Operator's sign-in of 2026-10-16 ends 7 days after it was made, about
   14:48 UTC on 2026-10-23 (`docs/deadlines.md`).
