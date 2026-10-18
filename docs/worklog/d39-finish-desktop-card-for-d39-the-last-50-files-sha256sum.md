---
id: d39-finish
title: Desktop card for D39: the last 50 files, SHA256SUMS, _logs and the whole-root check
kind: verification
status: in-flight
terminal: desktop
pr:
decisions: [D33, D39]
date: 2026-10-18
headline: D39 points 1 and 2 are done and checked — the day-bot's last 50 files (177,171 B) are home with Drive's bytes; SHA256SUMS now lists 149,585 files, every old line unchanged; the 50, the new SHA256SUMS and the root's _logs/ (535 files) are in swe-data/; the check of the whole root both ways by MD5 reads 150,121 the same, 0 on one side only and 0 differ, with one known 3-byte stray file in swe-data/ counted apart; end check "no finding", the manifest 144/0/0 by full hash; the routine is in docs/DATA_INVENTORY.md §C.
surface: []
---

D39 points 1 and 2 are done and checked: 150,121 files the same both ways

## Goal

D39 points 1 and 2, as amended on 2026-10-18 (campaign #544), done and checked, so that D39 point 5
(new data) can start. Three things were left after D39's first card (#588):

- the 50 files of the day-bot's `_local_archive/` (`1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0`) that no
  census had covered, the 49 of `data_store/` (`1IGJ8_N8ALUt_HBjhcgylbTHt7iO4OJ4p`) and a manifest
  text file (`1gdtz8RlCha9Ut9XHB3esj01SZBSW_wiP`): home under
  `data_archive/drive-legacy/day-bot-local-archive/`, each at its own path, checked by its bytes;
- the root's `SHA256SUMS` (D33 point 3's second proof), which still listed card 2c's 147,810 files
  only, rebuilt and its hash recorded in git;
- `swe-data/` (`13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b`) brought level with the root: the 50, the new
  `SHA256SUMS` and the root's `_logs/`, by copies only, then the whole root checked against it both
  ways by MD5; and D33 point 8's routine written into `docs/DATA_INVENTORY.md` §C.

The Drive originals stay where they are. The run used `scripts/drive_consolidate.py` as `main` holds
it (git blob `6903c9c8…`) for its `sums` and `filters` and its `credential_shaped` check (steps 2, 3
and 8 import it), nothing else, and rclone v1.74.4 for the listings, the download, the uploads and
the checks.

## What we tried

The card's steps, in order, each in one Bash call writing its whole output to its own file in
`%TEMP%` (`swe-d39f-step<N>.txt`). Every step ran once, under the empty letter: there is no lettered
step file.

- **Step 0.** The worktree `.claude/worktrees/d39-finish` on the new branch `claude/d39-finish` from
  `origin/main` (`4d9c17d`). Its lines: `branch claude/d39-finish`, the mark (`behind/ahead of main
  0/0 · run mode change`), `blob 6903c9c82dbdea48382f9e78ce45632289777951`, `D39's amendment on
  main`, `changes 0`, `SWE_DATA_ROOT` the root, `rclone v1.74.4`, `RCLONE_ variables 0`, `checked 144
  manifest files: 144 ok, 0 missing, 0 mismatched (size only)`, `D39's card is done`, `SHA256SUMS
  e22d8579…`, five `absent:` lines, `free 1037186736128`, and no `STOPPED:` line.
- **Step 1, the Drive sign-in.** `about exit 0`, `the sign-in ends about 2026-10-23T14:48Z ; hours
  left 118.5`, `== the Drive sign-in works`.
- **The briefing**, said in chat in the card's words (in no step file), **and the REGISTER note** on
  #544, before step 2.
- **Step 2, the root before.** `== the root outside _logs: 149537 files, 18,679,053,115 B; _logs
  without this run's folder: 535 files, 4,131,321,023 B; 0 link(s) not followed`; `==
  credential-shaped names: outside _logs 1; under _logs 0`; `== against D39's card's listing of
  2026-10-17: then 147812 files; now 149537; gone 0, changed 0, added 1725; D39's copies 1725, the
  added files exactly those with their sizes: yes`; `== the root is as D39's card left it`.
- **Step 3, the listing of the 50.** Start `2026-10-18T16:20:42Z`, end `2026-10-18T16:20:47Z`;
  `lsjson exit 0 and 0; stderr lines 0`.
- **Step 4, the copy home.** Start `2026-10-18T16:21:38Z`, downloaded `16:21:51Z`, end `16:21:51Z`.
- **Step 5, the new `SHA256SUMS`.** `SHA256SUMS now e22d8579…` and `the kept copy e22d8579…` (card
  2c's list, kept before `sums` ran); start `2026-10-18T16:22:29Z`, end `2026-10-18T16:32:59Z` (the
  times around the tool's `sums`).
- **Step 6, the 50 and `SHA256SUMS` into `swe-data/`.** Start `2026-10-18T16:33:43Z`, end
  `2026-10-18T16:38:22Z`.
- **Step 7, `_logs/` into `swe-data/`.** Start `2026-10-18T16:38:55Z`, end `2026-10-18T17:22:57Z`.
- **Step 8, the check of the whole root both ways by MD5.** Start `2026-10-18T17:23:39Z`, end
  `2026-10-18T21:06:41Z`.
- **Step 9, the end check.** Its block prints no times.
- **Step 10, this run's folder into `swe-data/`.** Start `2026-10-18T21:11:05Z`, end
  `2026-10-18T21:11:54Z`.
- **Step 11, the start of this record.** `Already up to date.` (the branch was level with `main`),
  the new fragment, `exit 0`, nine sha256 lines (Evidence), and `== the routine is in
  docs/DATA_INVENTORY.md: the check 150,121 the same on 2026-10-18; SHA256SUMS 149585 files,
  ff8362bb5bab9ce6fae5073fea96cd6a606280c5f56b6eb93cb73e3a8feae46e`.

The rule for steps 3 to 8 and 10, run in the session after each of them (it writes no file), read
`== the step is done` each time.

## What worked

Every step. The decisive lines, from the step files:

```
== the archive's top level: 5 folders, 1 file(s)
== data_store: 49 files, 174,577 B, 27 folders; with the top-level file: 50 files, 177,171 B
== without both hashes 0; Google-format 0; a path twice (any case) 0; an id twice no; credential-shaped 0; a name Windows cannot hold 0; the longest desktop path 140 characters; empty folders 0
== the 50 files pass their checks

== home already with Drive's bytes 0; to fetch 50; something else in the way 0
rclone exit 0
ERROR lines: 0
== staged and checked: 50 of 50 with Drive's size, MD5 and SHA-256
== moved into the root: 50
== home: 50 of 50 with Drive's size, MD5 and SHA-256; other files under data_store 0; the staging folder removed
== the 50 files are home

sums: SHA256SUMS lists 149585 files, 18,652,097,382 B; its own sha256 ff8362bb5bab9ce6fae5073fea96cd6a606280c5f56b6eb93cb73e3a8feae46e
sums exit 0
== the old list 147810 lines; the new 149585; unchanged 147810, changed 0, gone 0; added 1775, with the SHA-256 D39's plan or Drive gave them 1775; D39's copies and the 50: 1775
== the new list passes its checks

the 50: rclone exit 0 and 0; copied (new) 50
SHA256SUMS in swe-data/ before: 1 1Ij65QivPI5aieUjRp8PLl-_81F7C91Zi 003f28c71d5cb1e1040baf9c690ee63c (card 2c's MD5 003f28c71d5cb1e1040baf9c690ee63c; the new list's b041ff970d074dac4376dba6919df04f)
SHA256SUMS: rclone exit 0; replaced 1; in swe-data/ after: 1 1Ij65QivPI5aieUjRp8PLl-_81F7C91Zi b041ff970d074dac4376dba6919df04f
ERROR lines: 0
== SHA256SUMS replaced in place, the same Drive id

files to send: 535
rclone copy exit 0
copied (new): 535
ERROR lines: 0
3.848 GiB / 3.848 GiB, 100%, 1.240 MiB/s, ETA 0s

rclone check exit 1
== combined: the same 150121; only on the desktop 0; only in swe-data/ 0; differ 0; could not be read 0
== the known stray file in swe-data/ (a checker's upload of 2026-10-18, 3 B; it waits for the Operator's yes to remove): 1
== files on the desktop the check covers: 150121; compared without a checksum: 0; names rclone saw twice: 0; compared by MD5: yes
== rclone's exit 1: as expected
== the root is the same on the desktop and in swe-data/

rclone copy exit 0; rclone check exit 0; copied (new) 22; ERROR lines 0
== this run's folder: 22 files; the same in swe-data/ 22; other lines 0
```

Steps 3 to 8 and 10 each ended `== step done: …` (step 3's names `fifty.json` and its sha256, step
5's the new list's sha256).

**The 50.** Drive's counts matched the pen's of 2026-10-18 exactly: 49 files in `data_store/`
(174,577 B, 27 folders, none empty) and the top-level text file, 50 files and 177,171 B in all, each
with both an MD5 and a SHA-256 on Drive. All 50 were downloaded to the staging folder beside the root,
checked there by size, MD5 and SHA-256, moved into the root by a rename that refuses an existing file,
and checked again at home: 50 of 50. The staging folder was removed (its empty folders only).

**`SHA256SUMS`.** The old list (card 2c's, 147,810 lines, sha256 `e22d8579…`) is kept as
`_logs/d39-finish/SHA256SUMS.card2c`. The new list holds 149,585 lines: every one of the 147,810
old lines unchanged, and 1,775 added (D39's 1,725 copies and the 50), each with the SHA-256 that
D39's plan or Drive gave it. 147,810 + 1,725 + 50 = 149,585. Its sha256 is
`ff8362bb5bab9ce6fae5073fea96cd6a606280c5f56b6eb93cb73e3a8feae46e`. In `swe-data/` it replaced card
2c's list in place, after the MD5 showed card 2c's list there (`003f28c7…`): the same Drive id
(`1Ij65QivPI5aieUjRp8PLl-_81F7C91Zi`) before and after, and the new MD5 (`b041ff97…`) after.

**The uploads.** The 50: 50 copied new. `_logs/`: 535 copied new, 3.848 GiB. This run's own folder,
last: 22 copied new, and 22 the same in `swe-data/`. No upload logged an `ERROR`, an `immutable file
modified` line or a duplicate name.

**The check of the whole root.** 150,121 files the same both ways by MD5, 0 only on the desktop, 0
only in `swe-data/`, 0 differ, 0 unreadable; every file the check covers on the desktop, 0 compared
without a checksum, 0 names seen twice. 150,121 is step 2's 149,537 files outside `_logs`, plus the
50, less the credential file, plus `_logs/`'s 535. The one known stray file in `swe-data/` was
counted apart (below); it is why rclone exited 1, and the step's own check found that exit
explained by it.

**The end state**, step 9:

```
checked 144 manifest files: 144 ok, 0 missing, 0 mismatched
Drive reads exit 0, 0 and 0; stderr lines 0
== the root outside _logs: then 149537 files; now 149587; gone 0, changed 1 (SHA256SUMS among them), added 50 (the 50 50, other 0); unreadable 0
== _logs without this run's folder: then 535 files; now 535; gone 0, changed 0, added 0
== on Drive: the 50 as step 3 read them yes; the archive's top folders 5; data_raw 3306 files, 35,125,927 B
== the end state: no finding
```

No `== beside the root, not removed:` line: the staging folder did not exist at the end. The
day-bot's archive on Drive is as step 3 read it, and its `data_raw` still holds 3,306 files and
35,125,927 B, card 1's counts.

## What didn't

No step failed. No step printed a `STOPPED:` line or a traceback, no step ran again, and no BLOCKED,
WAITING or STOPPED note was posted.

**One delivery failure, before step 11 ran.** The first Bash call meant to run step 11's block never
reached bash intact: bash answered `/usr/bin/bash: -c: line 75: unexpected EOF while looking for
matching `''` and ran nothing. No step 11 file was written and the worktree stayed clean, so it is
not counted as a run of step 11. The block is the card's longest, and the Bash tool on this desktop
is known to deliver a long heredoc to bash incomplete, so that bash parses part of the Python as
shell (it happened to an earlier D33 card). Sending the same text again would fail the same way. So
the definitions line and step 11's block were copied out of the card itself (sha256 `1c17e9b3…`),
lines 202 and 822 to 915, by `sed` into a script in the session's scratchpad (95 lines, 8,325 B,
sha256 `44d8d9d0bd952e4fe27ac8af3d76c4c16548c9ced3b19301d383ca89cecf09f3`, which `sed -n
'202p;822,915p'` of the card reproduces; these figures and bash's "line 75" are from the session's
own commands, shown in the Run Summary, not from a step file), checked with `bash -n`,
and run by one Bash call, `bash <that script>`. The text that ran is the card's, word for word; only
its way into bash changed. Steps 0 to 10 went through the Bash tool as the card wrote them.

**The times**, from the step files' own start and end lines, against the card's estimates: step 3
took 5 s (a minute), step 4 13 s (a few minutes), step 5 10 min 30 s (about 15 minutes), step 6
4 min 39 s (a few minutes), step 7 44 min 2 s (about an hour), step 8 3 h 43 min 2 s (about four
hours; card 2d's took 220 minutes) and step 10 49 s (a few minutes). From step 3's start to step
10's end took 4 h 51 min 12 s.

## How we fixed it

Nothing needed fixing. What ran is the card's sequence, from the worktree, with the tool's blob
checked at step 0 and again before `sums`. Its main commands (step 6 also read `swe-data/`'s top
level twice with `rclone lsjson --files-only --hash`, before and after the `copyto`):

```
rclone about gdrive:
rclone lsjson --hash --drive-root-folder-id "$LAID" gdrive:
rclone lsjson -R --hash --disable ListR --drive-root-folder-id "$DSID" gdrive:
rclone copy gdrive: "$STAGE/data_store" --drive-root-folder-id "$DSID" --files-from-raw "$W/todo-ds.txt" --checksum --immutable --low-level-retries 100 -v
rclone copy gdrive: "$STAGE" --drive-root-folder-id "$LAID" --files-from-raw "$W/one-text-file.txt" --checksum --immutable --low-level-retries 100 -v
python scripts/drive_consolidate.py sums --root "$ROOT"
python scripts/drive_consolidate.py filters | grep -vxF '/_logs/**' > "$W/filters.txt"
rclone copy "$ROOT/$DEST/data_store" "gdrive:$DEST/data_store" --drive-root-folder-id "$SWE" --immutable --checksum --ignore-checksum --ignore-size --no-update-dir-modtime --drive-use-trash=true --exclude-from "$W/filters.txt" --ignore-case -v
rclone copy "$ROOT/$DEST" "gdrive:$DEST" --drive-root-folder-id "$SWE" --files-from-raw "$W/one-text-file.txt" --immutable --checksum --ignore-checksum --ignore-size --no-update-dir-modtime --drive-use-trash=true -v
rclone copyto "$ROOT/SHA256SUMS" "gdrive:SHA256SUMS" --drive-root-folder-id "$SWE" --checksum --no-update-dir-modtime --drive-use-trash=true -v
rclone copy "$ROOT/_logs" "gdrive:_logs" --drive-root-folder-id "$SWE" --immutable --checksum --ignore-checksum --ignore-size --no-update-dir-modtime --drive-use-trash=true --exclude-from "$W/filters.txt" --exclude "/d39-finish/**" --ignore-case -v --stats 10m --stats-one-line --stats-log-level NOTICE
rclone check "$ROOT" gdrive: --drive-root-folder-id "$SWE" --checksum --exclude-from "$W/filters.txt" --exclude "/_logs/d39-finish/**" --ignore-case --disable ListR --combined "$W/check-combined.txt" --low-level-retries 100 -v --stats 10m --stats-log-level NOTICE
python scripts/data_manifest.py check --root "$ROOT"
rclone copy "$W" "gdrive:_logs/d39-finish" --drive-root-folder-id "$SWE" --immutable --checksum --ignore-checksum --ignore-size --no-update-dir-modtime --drive-use-trash=true --exclude-from "$W/filters.txt" --ignore-case -v
rclone check "$W" "gdrive:_logs/d39-finish" --drive-root-folder-id "$SWE" --checksum --exclude-from "$W/filters.txt" --ignore-case --disable ListR --combined <%TEMP%>/swe-d39f-w-combined.txt --low-level-retries 100 -v
```

`$ROOT` is `C:/Users/merty/Desktop/swe-data`, `$W` is `$ROOT/_logs/d39-finish`, `$DEST` is
`data_archive/drive-legacy/day-bot-local-archive`, `$STAGE` is
`C:/Users/merty/Desktop/swe-data.d39f-staging`, `$SWE` is `13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b`,
`$LAID` is `1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0` and `$DSID` is `1IGJ8_N8ALUt_HBjhcgylbTHt7iO4OJ4p`.
Every upload was an rclone `copy` or, for `SHA256SUMS` alone, `copyto`; never a `sync`, `move` or
delete. On the desktop, the one move was step 4's rename of the 50 from the staging folder into the
root, and the one removal its empty staging folders. `--drive-root-folder-id` made
`swe-data/` rclone's root, so it wrote only inside it, and `--immutable` would have stopped any copy
rather than change a file already there. The one replacement, `SHA256SUMS`, went by `copyto` only
after its MD5 in `swe-data/` showed card 2c's list. Step 9 also read the day-bot's archive on Drive
(two `lsjson` and one `rclone size` of `data_raw`, `1uHSbrEaZoyW_Tgn1BimSz02KOJ18e616`). The
routine in `docs/DATA_INVENTORY.md` §C was written by step 11's block, not by hand.

## Evidence

**Where the record lives:** outside git, in the data root's `_logs/d39-finish/` on the desktop, and
the same 22 files in `swe-data/_logs/d39-finish/` (step 10). Each output, by sha256, from step 11's
file:

| Output | sha256 |
| ------ | ------ |
| `la-top.json` (step 3, the archive's top level) | `b597bca74e5d8a2a910bae797116d5712ca7e928a9a036fe34818eaf2a7337a1` |
| `ds.json` (step 3, `data_store/`) | `1a9abc896eac22129c853dfb7140782ed7edcfe32cf53d6cd859171ca2245a13` |
| `fifty.json` (step 3, the 50 with their ids and hashes) | `e832eeb7fbc12fbc22fd09ae490e3d1eae9ec5902535675ea5124987c78a0001` |
| `before.tsv` (step 2, the root outside `_logs`) | `8bd39a79373d938f2f121133567129bdd15eb3452a19c55e5ca7476b2b11cae8` |
| `logs-before.tsv` (step 2, `_logs/`) | `1a8f331329611e77bdf642ed162f49e845687ee275270067e39353fd14b9636c` |
| `SHA256SUMS.card2c` (step 5, card 2c's list kept) | `e22d8579bed1ce7a81a90ccb1096d0397a29de8471f8b56f466f05f13f882989` |
| the root's `SHA256SUMS` (step 5, the new list) | `ff8362bb5bab9ce6fae5073fea96cd6a606280c5f56b6eb93cb73e3a8feae46e` |
| `filters.txt` (step 6) | `4d40bf42f6608331c1529b548223830b2f57bb68cb36d9d69ff737a711f725bf` |
| `check-combined.txt` (step 8) | `c1fa9773a1c836f5c088f01d0a748c46ab9cd679e7db7bfb8c60c2955be8e515` |

rclone's own logs (`copy.log`, `upload-fifty.log`, `upload-logs.log`, `check.log`), the to-do lists,
the combined list and the pointers `done-3.txt` to `done-8.txt` are there too. rclone's logs and the
listings name every file, so none of these files is posted to GitHub; they stay on the desktop and in
`swe-data/` for the pen's check. Step 10's own logs and its pointer are in `%TEMP%`
(`swe-d39f-w-*`, `swe-d39f-done-10.txt`), since nothing may write `_logs/d39-finish/` once it is
uploaded.

**The step files**, every `swe-d39f-step*.txt` in `%TEMP%`, are posted in the Run Summary on this
card's pull request (step 13).

**The notes posted on #544:**

| Note | Link |
| ---- | ---- |
| REGISTER | [#544 comment 6252201840](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-6252201840) |

The report on #544 follows in step 13, after this record is pushed.

## Unresolved / handoff

1. **The pen's check.** A context that did not write this card checks the run against the step
   files and the record on the desktop and in `swe-data/`.
2. **The stray file in `swe-data/`.** `data_archive/drive-legacy/day-bot-local-archive/fake-text-file.txt`
   (3 B, Drive id `1M3t2AZlHe2ksvzFIjNM8X-BTK7MeKRwH`), which a checker of this card uploaded by
   mistake on 2026-10-18, is still there; the card left it, as D39 point 3 asks. Step 8 counted it
   apart (1), so "no difference" holds for every root file, with this one file only in `swe-data/`.
   Removing it needs the Operator's yes.
3. **D39 point 5.** Points 1 and 2 are done and checked once the pen's check agrees; then the pen
   brings the Operator the plan for new data.
4. **Drive holds the day-bot's files twice**: the originals in `_local_archive/`, which stay where
   they are (D39 point 1), and the copies in `swe-data/`. Removing the originals is a step of its
   own, with the Operator's yes (D39 point 3). `_logs/` (D39's ledger of `data_raw`'s duplicates
   among it) is now in `swe-data/`, the condition #588's check set before any such removal.
5. **Long blocks.** Step 11's block reached bash only through a script file (What didn't). Cards
   with a block this long could ship it as a file to run, checked by sha256, as `PROJECT_STATE.md`
   §0 B already proposes for long cards.
6. **The sign-in.** The Operator's sign-in of 2026-10-16 ends 7 days after it was made, about
   14:48 UTC on 2026-10-23 (`docs/deadlines.md`).
