---
id: drive-consolidate-2026-09-24
title: "D33 Drive consolidation tool: census, plan, copy home by id, never overwrite, never delete"
kind: feature
status: in-flight
terminal: sandbox
pr:
decisions: [D33, D31]
date: 2026-09-24
headline: scripts/drive_consolidate.py makes D33's consolidation mechanical. It lists every object in the old Drive areas, cross-checked against rclone size, and classifies each by bytes only (size, MD5 and SHA-256; a missing hash never matches). It copies Drive-only files home by id into data_archive/drive-legacy/, publishing each verified download with a hard link that never overwrites. It sweeps local stray data by bytes, through the same staging folder, and writes SHA256SUMS. It has no command that deletes or uploads. An independent review found one blocker (Drive can silently drop part of a query over several parents) and four should-fixes, all fixed. 27 of 27 deliberate breaks of the safety rules fail the tests.
surface: [scripts/drive_consolidate.py, tests/test_drive_consolidate.py, FILE_MANIFEST.md, TESTING.md, CHANGELOG.md]
---

## Goal

D33 (2026-09-24):
- Drive becomes one complete second copy, `swe-data/`.
- Everything that only the old Drive areas hold comes home first, never over an existing file.
- An old copy may go only once proven byte-identical.

This run builds the tool the three desktop cards use, and tests it against a fake Drive. The Execution Prompt is on the pull request.

## What we tried

1. **A first version** that also counted a git object as redundant when the history bundle held it. Codex's review of #533 showed that a git object is not a byte copy of the bundle, so the rule went: only bytes count.
2. **An independent review** by a context that did not write the tool. It had only the Execution Prompt and the files. It found:
   - **B1:** Drive sometimes answers `(A in parents) or (B in parents)` with nothing (Google issue 149522397). rclone works around this in its own listing, but `backend query` does not, so the census could silently skip whole subtrees. The review reproduced it.
   - **S1:** rclone's JSON omits a zero `size` (`size,omitempty`). Every empty file therefore became `needs-byte-check`, and a second round copied it again, until `plan` aborted.
   - **S2:** a re-plan after a copy could abort with "no free destination".
   - **S3:** `copyid` stops at its first failing pair, so one undownloadable object blocked its batch on every rerun.
   - **S4:** `copy` and `verify` passed while byte checks were pending.
   - **S5:** the SHA-256 half of the match, and area 4's read-only mode, had no test.
   - **Nits:** a per-row `deletable`; names rclone rewrites; a check-then-copy race; stdout on a Windows pipe; `.partial` leftovers; unguarded output paths; `bytecheck` not comparing a listed SHA-256; backslashes in area names; command-line length; `rclone.conf` and `.git/config` missing from the credential names; prompt drift.

## What worked

The fixes, one per finding:
- **B1.** An empty answer to a query over several parents is asked again one parent at a time. Every area's file count and bytes must also equal `rclone size`, which walks folder by folder; any difference stops the run.
- **S1.** A missing size with the empty file's MD5 is a size of 0. An empty file is redundant only when its listed MD5 and SHA-256 are the empty file's and an empty file already sits at its own destination.
- **S2.** A destination is assigned only once the bytes are settled. "No free destination" is recorded as unresolved, not fatal.
- **S3.** A failed batch is retried one object at a time, and each object's rclone error is logged.
- **S4.** `copy` refuses to run and `verify` fails while any byte check is pending.
- **N3 and N5.** `copy` downloads into `<root>.d33-staging` (beside the root, same volume). It re-hashes there, then publishes with a hard link, which fails if the destination exists. A mismatched download never reaches the root, and no `.partial` file lands in it.
- **The remaining nits:** all fixed, plus `sweep` for card 1.

The deliberate-break check pins every rule. Each rule was broken in turn, and every break made the suite fail, including all the rules the review listed as untested.

## What didn't

- **My first deliberate-break pass missed three rules.** One of the three breaks was built wrongly: "last row wins" hid a per-row `deletable`. Three tests were added, and the breaks were rebuilt correctly.
- **A gate masked by a pipe.** I ran `check_manifest_coverage.py | tail -1` in a chain that committed on success, so `tail`'s exit code hid the check's failure. The failure was benign: the registry rows named files not yet committed, and the pushed commit holds both. From now on each gate's own exit code is read, never a pipeline's.
- **A partial file could stay in the root.** I found this myself while preparing the pull request, after the review.
  - If a local copy failed part-way (a full disk, say), the file it had begun stayed where it was.
  - `sweep` also wrote straight to its destination, so a copy that failed its hash check stayed in the root.
  - Nothing was overwritten. But `sums` would have listed the bad file, and card 2 would have uploaded it.
  - Now a failed copy removes the file it created, which exclusive creation proves is its own. `sweep` copies through the staging folder and publishes with a hard link, as `copy` does.

## How we fixed it

See above. The tool has no command that deletes, moves or uploads anything. The only files it removes or replaces are its own:
- temporary downloads outside the root;
- the staging name of a file once it is published;
- the unfinished file a failed copy has just created;
- its output files, which never go inside the root except under `_logs/`.

## Evidence

```
$ python -m pytest tests/test_drive_consolidate.py -q
49 passed

$ deliberate breaks (each applied alone, then the suite run; the tool restored after)
CAUGHT  27 of 27:
  - M1-M3 SHA-256 in the plan, bytecheck and copy;
  - B1 no per-parent fallback; B1 no rclone size cross-check;
  - S1 zero size; S2 abort on no free destination; S3 no per-object retry; S4 copy with pending checks;
  - N1 deletable per row; N2 rewritten names; N3 publish overwrites;
  - .git/config; the same-object rule; copy ignores conflicts; census follows shortcuts;
  - inventory hashes credentials; read-only deletable; bytecheck trusts the listing;
  - empty matched by content; sweep overwrites; the output guard;
  - a failed copy keeps its partial file; sweep publishes without re-hashing; sweep copies
    straight to its destination; sweep's publish overwrites; sweep leaves its staging folder.

$ python -m pytest tests/ -m "not backtest_regression" -q   (with the first version of the tool)
3227 passed, 28 skipped, 8 deselected, 20 xfailed in 864.33s    pytest exit 0

$ rclone v1.68.2: strings in the binary, and backend/drive/drive.go
"Disabling ListR to work around bug in drive as multi listing (%d) returned no entries"   (drive.go:2126)
json:"size,omitempty,string"
failed copying %q to %q: %w
```

## Unresolved / handoff

- Card 1 runs the tool's tests on Windows; this run tested on Linux.
- A census mismatch can also come from duplicate folder names, which rclone's walk merges. If card 1 stops on one, the ledger's `path_unique` column shows where.
- Card 3 needs its own reviewed tool for deletion. This one has none, by design.
