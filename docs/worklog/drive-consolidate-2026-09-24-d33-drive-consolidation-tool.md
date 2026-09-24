---
id: drive-consolidate-2026-09-24
title: "D33 Drive consolidation tool: census, plan, copy home by id, never overwrite, never delete"
kind: feature
status: in-flight
terminal: sandbox
pr:
decisions: [D33, D31]
date: 2026-09-24
headline: scripts/drive_consolidate.py makes D33's consolidation mechanical. It lists every object in the old Drive areas, cross-checked against rclone size listing one folder at a time, and classifies each by bytes only (size, MD5 and SHA-256; a missing hash never matches). It copies Drive-only files home by id into data_archive/drive-legacy/, publishing each verified download with a hard link that never overwrites. It sweeps local stray data the same way and writes SHA256SUMS. Only bytes already in the root make a Drive object deletable, a link is never a home, and verify re-hashes every root file the plan relies on. It has no command that deletes or uploads. Two independent reviews and Codex's review of #534 found two blockers between them, all fixed; 68 of 68 deliberate breaks of the safety rules fail the tests.
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
- **Codex's automatic review of #534** found three more. Each was reproduced against the code before it was fixed:
  - **P1, links.** The plan could place a Drive object on an existing symbolic link whose target held the same bytes. `copy` and `verify` then passed with no file of the root's own. Also found while fixing it: on Windows, `os.walk` follows a junction (`os.path.islink` says no), so the inventory could count bytes outside the root, even on a Drive mount, as the root's own. Now:
    - the inventory lists links and junctions without following them;
    - the plan places nothing at or under one;
    - `copy`, `sweep` and `verify` refuse a destination with one on its path.
  - **P1, a stale inventory.** `sweep` counted a file as home because the inventory listed its bytes, even after that root file was removed or changed. It now re-hashes the root file first. The same trust sat in `verify`, which checked only the copies. It now also re-hashes every root file a Drive object was matched to, so a file changed since the inventory fails `verify`, instead of reaching card 3 with its Drive twin marked deletable.
  - **P2.** `verify` now refuses a plan made for another root, as `copy` does.

- **A second independent review, of #534 at `143f2ff`.** It had only the pull request and the Execution Prompt. Each finding below was checked against the code or the rclone source before it was fixed:
  - **B-1 (blocker).** Every `copy` and `duplicate` row was marked `deletable` before its bytes were home. A copy that never landed stayed deletable in every later plan, and so did the Drive twin that relied on it. Now `deletable` needs bytes already in the root: a `redundant` row, or a later row of the same object mirroring one. A copy becomes deletable only in the plan made after it lands.
  - **S-a.** `rclone size` used Drive's batched listing, the same OR query as the census, so it was not an independent check (rclone's `walk.ListR`; its own comment names `--disable ListR` as the switch). Now it runs with `--disable ListR` and lists one folder at a time. A file Drive lists without an MD5 is listed, not counted, because rclone neither counts nor downloads it (`newObjectWithExportInfo`). On a mismatch, the error names any folders that share a name under one parent.
  - **S-b.** `sweep --dest ./data/old`, `.` and `DATA/old` reached the live trees. Now a dot component, a live tree in any case, a credential-shaped name or an unsafe name is refused.
  - **S-c.** `sweep` dropped a data file with an unsafe name silently, and an unreadable folder vanished from `sweep`, `sums` and `inventory`. Now an unreadable folder stops the run, an unsafe data file name stops `sweep` before anything is copied, and each credential-shaped file or link it passes over is named.
  - **S-d and S-i:** the links and stale-inventory findings above, found independently.
  - **S-e.** One object `bytecheck` could not download stopped it, and with it `copy`. Now that object is listed unresolved and the rest are settled.
  - **S-f.** A staging name that could not be removed crashed `copy` with exit 1, which means "verify found a difference". Now it is a note. A read-only source no longer makes a read-only copy (times only, not the mode). Any unexpected error exits 4 or 2, never 1.
  - **S-g.** A folder with a second parent outside every area gave deletable children; trashing one by id would take it from the other project's folder too. Now any item with a second parent makes the paths under it ambiguous, so never deletable.
  - **S-h.** A later row of the same object was labelled `duplicate` even when the first row never came home. Now it mirrors the first row, and `bytecheck` downloads such an object once.
  - **S-j.** A git folder uploaded without its `.git` name would have had its `config` copied home. Now a `config` beside `HEAD` and `objects/` stays on Drive, unresolved.
  - **Nits, all taken but N-7:**
    - a hard link falls back to a copy only where the volume cannot hold one;
    - containment checks resolve links;
    - `census --root`;
    - no destination that looks credential-shaped;
    - `_logs` in any case;
    - no hash of a credential kept;
    - `*.env`;
    - Windows' case rules for name clashes (NTFS treats `ı` and `i` as one);
    - absolute paths for rclone;
    - the missing reserved names;
    - the copy log written per object.
  - **N-7, not taken:** the SmartWheelData area's parent id is written only in the tool. The census checks it on every run, and a wrong id stops the run before anything is planned.

## How we fixed it

See above. The tool has no command that deletes, moves or uploads anything. The only files it removes or replaces are its own:
- temporary downloads outside the root;
- the staging name of a file once it is published;
- the unfinished file a failed copy has just created (exclusive creation proves it is its own);
- its output files, which never go inside the root except under `_logs/`.

## Evidence

```
$ python -m pytest tests/test_drive_consolidate.py -q
60 passed

$ deliberate breaks (each applied alone, then the suite run; the tool restored after)
CAUGHT  36 of 36 (the runner: each break applied alone, then the file's tests with -x):
  - M1-M3 SHA-256 in the plan, bytecheck and copy;
  - B1 no per-parent fallback; B1 no rclone size cross-check;
  - S1 zero size; S2 abort on no free destination; S3 no per-object retry; S4 copy with pending checks;
  - N1 deletable per row; N2 rewritten names; N3 publish overwrites;
  - .git/config; the same-object rule; copy ignores conflicts; census follows shortcuts;
  - inventory hashes credentials; read-only deletable; bytecheck trusts the listing;
  - empty matched by content; sweep overwrites; the output guard;
  - a failed copy keeps its partial file; sweep publishes without re-hashing; sweep copies
    straight to its destination; sweep's publish overwrites; sweep leaves its staging folder;
  - the plan ignores links; copy's pre-scan ignores links; verify ignores links; the walk
    follows junctions; the inventory hashes file links; sweep trusts a stale inventory; sweep
    ignores a link at its destination; verify accepts another root; verify skips matched root files.

$ python -m pytest tests/ -m "not backtest_regression" -q   (on 143f2ff, before Codex's review)
3257 passed, 28 skipped, 8 deselected, 20 xfailed in 781.98s    pytest exit 0

$ rclone v1.68.2: strings in the binary, and backend/drive/drive.go
"Disabling ListR to work around bug in drive as multi listing (%d) returned no entries"   (drive.go:2126)
json:"size,omitempty,string"
failed copying %q to %q: %w
```

## Unresolved / handoff

- Card 1 runs the tool's tests on Windows; this run tested on Linux.
- A census mismatch can also come from two folders that share a name under one parent: rclone's walk by path cannot tell them apart. If card 1 stops on one, the error names them, and the pen decides what to do before anything changes on Drive.
- The per-object copy log (review 2, N-14) has no test of its own: only a crash in the middle of a batch would show the difference.
- Card 3 needs its own reviewed tool for deletion. This one has none, by design.
