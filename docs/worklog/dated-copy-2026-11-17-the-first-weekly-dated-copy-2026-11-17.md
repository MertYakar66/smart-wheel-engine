---
id: dated-copy-2026-11-17
title: The first weekly dated copy, 2026-11-17
kind: verification
status: in-flight
terminal: desktop
pr:
decisions: [D29]
date: 2026-11-17
headline: the first weekly dated copy — six of the eight release-schedule files are saved under the data root and in swe-data/, the same both ways by checksum; BLS refused both of its addresses (HTTP 403), so there is no BLS copy for this date
surface: []
---

The first weekly dated copy is taken, in part. Copy date: **2026-11-17** (UTC). List: `schedules`.
Six of the eight addresses were saved; the two BLS addresses answered HTTP 403 and have no file. The
members list was not part of this run: the Operator has not released it.

## Goal

Keep this week's dated copy of the lists the Operator has released, as `DECISIONS.md` D29, "Amended
on 2026-11-17 (ruling 8)", allows. This is the first copy, run on the desktop under the first-copy
prompt of `docs/DATED_COPIES.md`. Success is: the copy is under the data root and in `swe-data/`, the
same both ways by checksum; nothing else under the data root changed; and this record is in one small
pull request.

## What we tried

The prompt's steps, in order, each once.

- **Before the gate.** The script's own test file, with `SWE_DATA_ROOT` unset for that one command.
  It had run on Linux only until now.
- **Gate G1.** "Your Drive sign-in is made. Go?" The pen relayed the Operator's answer, "Go"
  (2026-11-17, about 15:37 UTC). Nothing was fetched or written before it.
- **Step a.** A read-only listing of `swe-data/`, reached by its id, to test the Drive sign-in.
- **Steps b and c.** The manifest's 144 files checked by size, and the sha256 of
  `data_raw/sp500_constituents_current.csv` taken.
- **Step d.** `scripts/copy_dated_sources.py --lists schedules --date 2026-11-17`, once, with no
  retry.
- **Step e.** The two new folders copied to the same paths in `swe-data/` with `rclone copy
  --immutable`, the copy folder first and the run's log folder last.
- **Step f.** Both folders checked against `swe-data/` with `rclone check --checksum`.
- **Step g.** Steps b and c again.
- **Step h.** The repository's four document checks.

## What worked

- The test file passes on Windows: 155 passed.
- The sign-in works: step a ended with exit 0 and six names.
- Six files were saved, each with HTTP 200 and no redirect, and each passes the script's light check.
  The sha256 of each file on disk, taken again with `sha256sum`, equals the one the script printed.
- Both folders are in `swe-data/` and the same both ways: 0 differences; 7 matching files of 7 in the
  copy folder, and 1 of 1 in the log folder.
- Nothing else under the data root changed, as far as the two checks show: the manifest line and the
  constituents file's sha256 are the same before and after.

## What didn't

- **BLS refused both of its addresses.** `bls_news_release.htm` and `bls.ics` each answered HTTP 403
  on the first request, with no redirect. The script saves only a 200 answer, so neither has a file.
  The prompt allows one run of step d and no retry, so none was made. This run does not establish
  why BLS refused. The job names itself by the project and its public repository address, with no
  email address and never a browser's User-Agent (D29's amendment); whether BLS's answer depends on
  that is unverified.
- **So there is no BLS copy dated 2026-11-17.** The script takes a copy only on its own UTC date, so
  a BLS copy for this date can be taken only before 24:00 UTC on 2026-11-17, and only on a new
  instruction: a second run that day would fetch just the two missing addresses.

## How we fixed it

Nothing was repaired. The run went on with the six saved files, as the prompt says for exit 10.

## Evidence

**The saved files**, in `data_raw/macro_schedule/date=2026-11-17/` under the data root and at the same
path in `swe-data/`:

| File | Bytes | sha256 | Result |
| ---- | ----: | ------ | ------ |
| `fed_fomccalendars.htm` | 166065 | `19bc6b2f09042326b5ec7d3b61a7454885d7f7406009b3ebd3e4a1bd3bbd1cb9` | saved |
| `bea_schedule.htm` | 68831 | `1fa51654a9c361408a165848afd5acfa217c6bd15a1fdc760c078a0095b4efa6` | saved |
| `bea_schedule.ics` | 43459 | `cbebc02fc884f265b1ae3820134b5b0ac2202ace72974129b27665c2660f15cd` | saved |
| `bea_release_dates.json` | 12096 | `5774525c10ec63520b36b7647a11fad4d192f5d1d609074109f692acd29ce721` | saved |
| `census_calendar_listview.html` | 85284 | `4bfc41b64a8673d8dabe9e0d2861071c8b2024d91327fdfb95a602d1a14ea125` | saved |
| `census_calendar_listview_2026.html` | 89667 | `da7a83bce7e362d6bf465e0c3d2de2bc02ea31f2844734994c896dea69ec25ab` | saved |

**The addresses with no file:**

| File | Result |
| ---- | ------ |
| `bls_news_release.htm` | refused (HTTP 403) |
| `bls.ics` | refused (HTTP 403) |

**The run's own two files** (not copies of a source; they stay under the data root and in
`swe-data/`):

| File | Folder | Bytes | sha256 |
| ---- | ------ | ----: | ------ |
| `_copy-153858Z.json` | `data_raw/macro_schedule/date=2026-11-17/` | 14976 | `533f95a4b5803a3c7ba15d991ecd57d7a041eae256ea82b27f464044287ecd52` |
| `run-153858Z.log` | `_logs/dated-copies/2026-11-17/` | 706 | `906a78bd33276bf9bad76756e54cfb62b5c45499e61c4d34126056f26468f630` |

**Commands and their output.** Times are UTC, from `date -u` beside each command.

- The test file: `env -u SWE_DATA_ROOT python -m pytest tests/test_copy_dated_sources.py -q` gave
  `155 passed in 5.28s`, exit 0 (Python 3.12.10, win32).
- Step a, 15:38:45: `rclone lsf gdrive: --drive-root-folder-id 13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b
  --max-depth 1 --dirs-only` gave exit code 0, 6 names, 0 lines on stderr.
- Step b: `python scripts/data_manifest.py check --root "$SWE_DATA_ROOT" --size-only` gave `checked
  144 manifest files: 144 ok, 0 missing, 0 mismatched (size only)`.
- Step c: `sha256sum "$SWE_DATA_ROOT/data_raw/sp500_constituents_current.csv"` gave
  `c3007ff19acb57b49fdb6639be23081f567efb7b664a8fea483077cf2cda86e7`.
- Step d, 15:38:58 to 15:39:03: exit code 10. Its lines:

  ```
  fed_fomccalendars.htm  166065  19bc6b2f09042326b5ec7d3b61a7454885d7f7406009b3ebd3e4a1bd3bbd1cb9  saved
  bls_news_release.htm  -  -  refused (HTTP 403)
  bls.ics  -  -  refused (HTTP 403)
  bea_schedule.htm  68831  1fa51654a9c361408a165848afd5acfa217c6bd15a1fdc760c078a0095b4efa6  saved
  bea_schedule.ics  43459  cbebc02fc884f265b1ae3820134b5b0ac2202ace72974129b27665c2660f15cd  saved
  bea_release_dates.json  12096  5774525c10ec63520b36b7647a11fad4d192f5d1d609074109f692acd29ce721  saved
  census_calendar_listview.html  85284  4bfc41b64a8673d8dabe9e0d2861071c8b2024d91327fdfb95a602d1a14ea125  saved
  census_calendar_listview_2026.html  89667  da7a83bce7e362d6bf465e0c3d2de2bc02ea31f2844734994c896dea69ec25ab  saved
  ```

- Step e. The filter list: 61 lines, the tool's 62 less its `/_logs/**` line; it lets through all 7
  files of the copy folder and the 1 of the log folder. `rclone copy` of
  `data_raw/macro_schedule/date=2026-11-17`, 15:39:44 to 15:39:53: exit 0. `rclone copy` of
  `_logs/dated-copies/2026-11-17`, 15:40:03 to 15:40:09: exit 0. Both with the prompt's flags.
- Step f, 15:40:17 to 15:40:21. The copy folder: `0 differences found`, `7 matching files`, exit 0;
  7 files on disk. The log folder: `0 differences found`, `1 matching files`, exit 0; 1 file on disk.
- Step g: the same line as step b, and the same sha256 as step c.
- Step h: `working structure: 6 of 6 checks pass`; `worklog-index: OK (INDEX.md is current).`; `OK:
  FILE_MANIFEST.md accounts for every tracked file.`; `doc-currency: OK`.
- No `data_raw/sector_dated/` folder exists: the run named no `members` list.

## Unresolved / handoff

- **BLS.** There is no BLS copy for 2026-11-17, and the next weekly run will meet the same answer
  unless something changes. What BLS asks of an automated reader is not established here; finding
  that out, and any change to how the job names itself, is the pen's to plan and the Operator's to
  decide (D29's amendment records their choice).
- **`SHA256SUMS` and the manifest** do not list the new files, as D29's amendment says: the
  whole-root re-hash and `build` did not run.
- **The members list** waits for the Operator's word.
