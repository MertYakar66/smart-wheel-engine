---
id: drive-consolidate-2026-09-24
title: "D33 Drive consolidation tool: census, plan, copy home by id, never overwrite, never delete"
kind: feature
status: merged
terminal: sandbox
pr: 534
decisions: [D33, D31]
date: 2026-09-24
headline: scripts/drive_consolidate.py makes D33's consolidation mechanical. It lists every object in the old Drive areas, cross-checked against rclone size listing one folder at a time, and classifies each by bytes only (size, MD5 and SHA-256; a missing hash never matches). It copies Drive-only files home by id into data_archive/drive-legacy/, publishing each verified download with a hard link that never overwrites. It sweeps local stray data the same way and writes SHA256SUMS. Only bytes already in the root make a Drive object deletable, a link is never a home, and verify re-hashes every root file the plan relies on. It has no command that deletes or uploads. Seven independent reviews (the last in five rounds) and twenty-one Codex reviews of #534 found two blockers between them, all fixed, and the last review of each kind found nothing; all 144 deliberate breaks of the safety rules fail the tests.
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

- **Codex's second review, of `abf9bed`,** found one more P1, which I reproduced. A link named `SHA256SUMS.tmp` in the root made `sums` write through it: a file outside the root lost its bytes to the checksum list, and `SHA256SUMS` became a link to it. The plan, ledger and census outputs used the same fixed temp name. Now:
  - every output goes to a fresh name of the tool's own (an exclusive create), then replaces the destination;
  - a destination that is a link is refused;
  - the copy log refuses a link too.

- **A third independent review, of `abf9bed`,** found no blocker. It did find:
  - **The rule card 3 relies on most had lost its test.** A Drive id is deletable only if every row naming it is. My break for it ("N1 deletable truly per row") also made copies deletable, so a different test caught it. The record "N1 caught" was true of the break, not of the rule. The reviewer's four breaks of the rule (per row, first row wins, last row wins, any row) all passed the suite. Now a test of a read-only area nested inside a consolidate area, run in both area orders, catches all four.
  - **S-g, one level up.** An area root that also sits in another project's folder gave deletable children. The census now keeps the root's parents, and a second one makes every path in the area ambiguous.
  - **Two rules with no test:**
    - `verify` failing on a pending byte check alone;
    - an empty file Drive lists without a SHA-256 needing a byte check.
    Both are now tested.
  - **Nits, all taken:**
    - an empty file is home only under its own name by Windows' one-to-one case rule, not the wider clash key ("straße" is not "strasse");
    - the census withholds a credential's hashes by path, not by name, so `.git/config` too. The plan's own withholding then became redundant, and the break run showed it (no test could tell it apart), so it was dropped;
    - `sweep` names each folder it passes over (`.git`, `_locks`, `__pycache__`), with its file count;
    - the census error names folders with two parents;
    - the link tests make real junctions on Windows when symbolic links are refused;
    - the three backup checks are tested (links in `bytecheck`'s placement, the link re-check at publish, `sweep`'s failed publish: each failed file reported, and the rest still tried);
    - `SCHEMA` 3, with documents of another schema refused;
    - area names Windows cannot keep are refused;
    - a file with two parents comes home, never deletable, instead of staying unresolved;
    - the records now match the runs.

- **A fourth independent review, of `a585b67`,** found no blocker. Its two should-fixes and the nits worth taking:
  - **A regression of mine.** When files with two parents started coming home, such a file was judged by one path only, the first one Drive listed. A file in both `data/` and `secrets/` could therefore be copied home past the credential rule. The same gap applied to anything under a folder with two parents. Now every path to an object is checked, and an object with more than 64 paths stays on Drive, listed.
  - **A missing `root_parents`** in a census was read as "one parent". It now means ambiguous, so nothing in that area is deletable, and the schema checks on the census, `sweep`'s inventory and `bytecheck`'s inventory are tested.
  - **Two byte-risk paths in older code:**
    - a `_logs` folder that is a link into a live tree would let an output replace a data file, so outputs now refuse a link on their path under the root;
    - a copy log hard-linked to a data file would have had JSON appended to it, so the log now refuses a second hard link.
  - **Nits:**
    - a name clash is counted under every folder an object is in;
    - "an empty file is already here" now needs the exact name (a case variant is copied beside it: at worst one spare empty file), replacing the one-to-one case key, which still joined `ı`/`i`;
    - the output writer forgets a temp name once it is replaced;
    - `sweep`'s count of a passed-over folder notes unreadable parts instead of stopping.

- **Codex's third review, of `91f1c3e`,** found a P1 that the every-path fix left open. A parent outside the area has a path the census never saw, and that path could run through someone's `credentials/` folder. So a file whose in-area path was harmless was still copied home. Now anything with a parent the census never listed stays on Drive, unresolved, and the census keeps none of its hashes. That covers a parent outside the area, a second parent of the area root, and unknown root parents.
  - **One more bug, found while fixing it:** the census withheld hashes before it knew the area root's parents. Every object looked unknown to it, and a copy row then carried "withheld" as its MD5, which the copy refused as a mismatch. Fixed. A withheld row is never classed as anything but unresolved.
  - **An object reached through two areas** is now tested with nested areas, the one layout the census sees whole.

- **Codex's fourth review, of `3cf01d4`,** found that the rule was still judged one area at a time. Areas can nest.
  - With the inner area listed first, one Drive object was classed `copy` at `B/prices.csv`. The outer area saw the same object at `A/tokens/vendor/prices.csv` and left it on Drive.
  - In both orders, the inner area's census kept the object's hashes.

  I reproduced both. Now one verdict covers each Drive id across every area: a path that keeps an object on Drive in one area keeps it there in all of them, and the census keeps no hash of it in any.
  - **One more gap, found while building the break list:** a credential-shaped file that Drive lists without an MD5 has no hash to withhold. Only the plan's verdict kept it out of a byte check, and no test pinned that. One does now.

- **Codex's fifth review, of `1d7ac59`,** broke the S-j heuristic, which recognised a git folder by a `HEAD` file and an `objects/` folder. Drive lets two children share a name. A file named `objects`, listed after the folder of that name, hid the git folder, and its `config` was classed `copy`. A git folder uploaded without `HEAD` did the same. I reproduced both.
  - **The fix is a name rule, not a better heuristic.** Any file named exactly `config` or `config.worktree`, in any folder and in any case, is credential-shaped. A folder named `config` is not affected.
  - **The heuristic had only ever run on Drive.** In the root, a bare mirror's config (`mirror.git/config`) was hashed by the inventory and listed in `SHA256SUMS`, and the rclone exclusions for the upload let it through. The name rule covers both sides.
  - **What it costs:** a file named `config` holds settings, not market data. The plan lists every one it keeps on Drive or leaves out of the root's inventory.

- **Codex's sixth review, of `119fb58`,** found that `copy` trusted the plan. It fetched each object by id on the plan's word. A file renamed to `config`, or moved under a credential-shaped folder, between card 1 and card 2 would still land in `data_archive`, because its unchanged bytes pass the recorded hashes. I reproduced it.
  - **The fix:** the plan is treated as a snapshot. Just before its first download, `copy` runs a fresh census of the plan's areas, and so does `bytecheck`, which also downloads. Each object about to be fetched must sit at the same path and still be safe by the one-verdict rule. Otherwise nothing is fetched, and the run exits 3.
  - **Two checks, because each catches what the other misses:** a file filed under a second folder named `secrets` keeps its path, so only the safety verdict catches it. A harmless move keeps it safe, so only the path check catches that.
  - **Not covered, at first:** a change made on Drive while a copy is running.

- **Codex's seventh review, of `a93c1d8`,** took up exactly that gap. It renamed a file to `config` right after the one census, and `copy` fetched and published it. I reproduced it.
  - **The fix:** `copy` now stages every download first. After the last download is in, it runs a second census and publishes only what is unchanged. A changed object's download is removed, never published, and the run exits 3.
  - **`bytecheck` does the same.** On a change it keeps no verdict and no hash, and it leaves the plan file as it was.
  - **Why not a census per batch, as Codex suggested:** it costs a full Drive listing each time, and still leaves a window before each download. With the census after the downloads, only the local hard link happens between the last census and publication.
  - **What remains:** an object renamed during its own download is read before a census can see the change. Nothing it changed into is kept.

- **A fifth independent review, of `1d7ac59`,** found no blocker. It reproduced one should-fix and four nits. Its sixth finding, a partial `.git` upload without `HEAD`, was already fixed by the name rule above.
  - **Should-fix: an object updated between two areas' listings.** The census lists one area after another. An update in between gave one id two rows with different bytes: one redundant, the other mirrored as a duplicate. Both were deletable, while Drive's new bytes were nowhere in the root. Now an id listed differently in two areas stops the census. That also covers an object moved from one area to the next.
  - **Nits, all taken:**
    - a folder loop hid `secrets/`, and a file inside it was copied home. A loop is now a path that cannot be known.
    - a census from an earlier rule set was accepted. `SCHEMA` is 4 now, and the plan keeps no hash of an id it keeps on Drive, whatever the census kept.
    - a linked `_logs`, reached through another name for the root, let an output replace a data file. No output goes inside the root while `_logs` is a link, and a Windows `\\?\` path is compared the ordinary way.
    - `bytecheck` removed an empty `--tmp` folder it had not made.
  - **Two guards it suggested are not kept:** judging an empty path list as unsafe, and judging the row's own path as well. Once a loop is an unknown path, neither can ever fire, and no test could tell them apart.
  - **A limit of the fake, recorded:** by default the fake counts a file once per parent, while rclone's walk counts it once per path. So the tests of a plan with a folder that has two parents in one area test a census that real rclone stops first. One test now pins that stop, with the fake counting by path.

- **Codex's eighth review, of `78e8cd1`,** found two more, both reproduced:
  - **P1.** A download whose bytes no longer matched was kept as a mismatch and skipped the census after the downloads. An object changed to a credential during the copy, bytes and all, therefore left those bytes in the staging folder. Now every fetched object goes through that census, and a changed object's download is removed whatever its bytes.
  - **P2.** `same_path` treated `C:\swe-data` and `\\?\C:\swe-data` as two roots, so a later command refused the right root spelled the other way. It now compares them the ordinary way, as `_inside` does.

- **Codex's ninth review, of `f82df45`,** found two more, both reproduced:
  - **P1.** The census after the downloads compared paths only. An object edited in place, with its path unchanged, got through. If the edit came before its download, the new bytes stayed in staging as a mismatch. If it came after, the plan's old bytes were published while Drive held others. Now the size and hashes are compared too, wherever both listings have them, and an edited object counts as changed at either census.
  - **P2.** On Windows, `sha256sums` is the same file as `SHA256SUMS`, and `sums` hashed its old list into the new one. It now recognises its own list and temporary files in any letter case.

- **Codex's tenth review, of `2df6649`,** found a P1 in the control flow. If the census after the downloads failed (Drive not answering, or totals that moved), `copy` stopped before the step that removes changed downloads, so bytes that may have become a credential stayed in staging. I reproduced it by making the second census fail. Now, if anything stops the run before that census completes, every file this run downloaded into its own new staging folder is removed. `bytecheck` needed no change: it removes each download as soon as it is hashed.

- **Codex's eleventh review, of `98dc7e9`,** found no major issue.

- **A sixth independent review, of `98dc7e9`,** found no blocker, and nothing marked deletable wrongly. It reproduced three findings:
  - **Should-fix.** A changed download was removed only when the publish loop reached it, after the guarded section. A stop while publishing, such as Ctrl+C or a copy log that could not be appended, left it in staging. Now every changed download is removed right after the census, before anything is published.
  - **Nit, a regression of mine.** Every `bytecheck` without `--tmp` left an empty folder in the system temp folder, because `main` had made it. Now `bytecheck` is told it made the folder.
  - **Nit.** A folder loop through first parents stopped the plan with "deeper than 200". This failed closed. Now the path goes through a parent that does not lead back into the chain.

- **The break run on `98dc7e9` caught 118 of 120.** The two it missed:
  - **The safety verdict in the drift check.** The census withholds an unsafe object's hashes, so the same object also showed as "edited in place", and only the reason told them apart. The reason each change reports is now tested.
  - **`guard_out`'s check for a link anywhere on the output's path.** Since the check for a linked `_logs` came in, it only refused outputs that land in `_logs` or outside the root anyway, so it is removed. The containment check resolves links, and `write_outputs` refuses a linked destination.
  - **A lesson about my own check.** Earlier I had tried that second break by hand in a scratch folder and read it as caught. In fact another test failed there, because that folder has no `docs/`. Breaks are judged only in full worktrees now.

- **Codex's twelfth review, of `06cdc25`,** showed I was wrong to remove that check.
  - **P2.** Say `_logs` is a link to a folder outside the root. Then an output "into `_logs`" resolves outside the root, so neither containment check fires, and `write_outputs` could replace a file in that folder. The removed check had covered this, but no test did, so the break run could not show it.
  - **The fix.** A link or junction anywhere below the root on an output's path is now refused, however the root is spelled. Three tests pin it. The older check for a linked `_logs` keeps a case of its own: a side door straight into `data/`, which one more test now pins.
  - **P3.** `main` made `bytecheck`'s temporary folder before any check that can stop the run, so a wrong inventory or a change on Drive left the folder behind. `bytecheck` now makes it only after its checks.

- **Codex's thirteenth review, of `1b5da58`,** found two P2s. Neither risked a byte, and both are fixed:
  - `sweep` matched its skip names and code suffixes by exact case. On the desktop, `.GIT`, `__PYCACHE__`, `DATA_MANIFEST.JSON` and `MODEL.PY` would have been copied into the archive. It now uses the tool's case fold.
  - `plan --out` and `--ledger` naming the same file wrote the ledger and then replaced it with the plan. Now any command refuses, before writing, two outputs that name one file.

- **Codex's fourteenth review, of `c93ec3c`,** found a P2 of the same kind. `copy --log` could name the plan, and its appended lines corrupted the plan, so every later command failed to load it. The same mistake could replace a census, an inventory or an areas file. Now no command writes an output that resolves to one of its own inputs.

- **Codex's fifteenth review, of `05e3676`,** found the one input that check missed: `plan`'s `--about` file. I then listed every command's file arguments again, and none is missing now.

- **Codex's sixteenth review, of `90aa364`,** found a P1 for Python 3.11, which the repository supports.
  - **The finding.** `os.path.isjunction` arrived in Python 3.12. On 3.11 the tool took a Windows junction for an ordinary folder. The inventory would walk into it, and `copy`, `verify` and `sweep` would accept a destination under it.
  - **The fix.** Without `isjunction`, the tool reads the reparse tag that `os.lstat` reports on Windows, as Python 3.12's own `isjunction` does. Every link check goes through that one helper.
  - **The test.** The junction test runs through both routes. The 3.11 route fails on `90aa364`.

- **Codex's seventeenth review, of `d912ce3`,** found no major issue.

- **A seventh independent review, of `d912ce3`,** found no blocker, and nothing published, kept or marked deletable wrongly. It reproduced three should-fix items and five nits. All are fixed in `58fa44b`, each with a test that fails on `d912ce3`:
  - **Should-fix: another name for `_logs`.** A path through another name for `_logs`, with a link inside it leading out, never met a folder that resolves to the root itself, so the link check was skipped. An output could replace a file outside the root. Now any link on the output's path whose own folder is inside the root is refused, however the path gets there.
  - **Should-fix: a sweep source named like a credential folder.** The name rule applied only below the source. A file in `gdrive_credentials/` came home under a harmless path, and card 2 would have uploaded it. Now a source whose path, spelled or resolved, has a credential-shaped part is refused.
  - **Should-fix, in the tests: the junction test could not pass on Windows,** where the tool spells every path with `\\?\`. It now compares folders by identity, and runs with that spelling too.
  - **Nit: a `..` after a link (Linux only).** Linux goes up from where the link leads, while the tool wrote from the link's own folder, so an output could replace a live root file. Now such a path is refused, and every check is made on the path the tool writes.
  - **Nit: one unreadable download stopped every copy.** Say an antivirus holds one fresh download: the run stopped and discarded the rest, on every rerun. Now that object alone fails, and its download is removed.
  - **Nit: the copy log appended to any file,** such as the plan's ledger, which card 3 reads. Now it appends only to its own log, and never reads a file with a credential-shaped name.
  - **Nit: `\\?\Volume{…}` paths.** Without the prefix, they read as relative paths. Now only the drive-letter and share forms are accepted.
  - **Nit, in the tests: the mid-run link test on a desktop without the symbolic-link privilege.** The fake rclone now makes a junction there, as the tests' own probe does. That runs only on Windows, so card 1 runs it.

- **Codex's eighteenth review, of `58fa44b`,** found two P2s in the new log check, both reproduced. It judged only the log's own name, so it read an existing file in a `secrets/` folder. And it took any empty file for its own log, and appended to it.
  - **The fix, at the root.** Both came from appending to a file that existed before the run. Now the copy log is always a new file that the run creates exclusively, before any download. So an existing path of any kind is refused unopened: a file, a link or a folder. The tool never reads a log. Each run writes its own, named `<plan>_copy-<UTC time>-<random>.jsonl`.
  - **Four checks fewer.** This replaces the link, hard-link and plan-alias checks on the log, and the check that read it. The break run on `58fa44b` had already shown two of them to be redundant.

- **The seventh review's second round, of `58fa44b`,** confirmed its eight findings fixed. It found two more, both reproduced and fixed in `68ecd3f`:
  - **Should-fix: a link reached only through another link's target.** An alias outside the root pointed at a link in `_logs` that leads out. The check looked only at the parts of the path as spelled. Now it follows each link to where it leads, as the OS does, and checks the rest of the way from there. It follows at most 40 links, and refuses a loop.
  - **Should-fix, in the tests:** the simulated Windows spelling in the junction test cannot run on Windows itself, so it is skipped there. The plain cases already use the real prefix.
  - **Its three findings on the old log check** were already settled by the new log.

- **Codex's nineteenth review, of `68ecd3f`,** found no major issue.

- **The seventh review's third round, of `68ecd3f`,** found nothing that risks a byte. It found three things, each reproduced first and fixed in `2069c36`:
  - **Should-fix: a rerun that names its log was refused.** The first run's log was already there, so "a rerun resumes" failed for a card that spells `--log`. Now every run's log is `<name>-<UTC time>-<random>.jsonl`, and `--log` gives only the name. A file named by `--log` is never touched.
  - **Nit: an output under a volume mounted in a folder was refused.** Windows marks that folder with the junction tag, but its target is a volume, not a path to follow. The link walk now goes past it as a plain folder: it is that volume's top.
  - **Nit, Windows only: a dangling link as the log.** Windows's CREATE_NEW would follow it and create the file it points to. A path that exists in any form, a dangling link included, is now refused before the log is created.

- **Codex's twentieth review, of `2069c36`,** found a P2 in that new skip, reproduced. It matched any target starting with `\\?\Volume{`, so a link to a folder inside a volume was walked past, and a link in the root at its end was never checked. Now only a target that is exactly a volume's top is walked past. Any other volume target, and any GLOBALROOT name, is refused (`e5084d1`).

- **The seventh review's fourth round, of `2069c36`,** found the same gap as Codex, and one nit a card would trip on. `--log <root>\_logs` put the log beside `_logs`, at the root's top, so the run was refused after the census with a misleading message. Now a `--log` that names a folder puts the log in it, and the log's place is checked before the census (`6a00dcb`).

- **The seventh review's fifth round, of `6a00dcb`,** found no new hole and no regression. All 25 of its reproductions pass, except the recorded limit below (another name for the root that `realpath` keeps).

- **Codex's twenty-first review, of `6a00dcb`,** found no major issue.

- **The break runs of the last rounds.** The run on `58fa44b` missed four checks on the copy log: the check that read the log made them redundant, and all went together in `5254f8a`. The final run, on `6a00dcb`, caught 143 of 144. The miss is the copy log's place, which is checked twice on purpose: before the census, so a wrong place stops the run at once, and again just before the log is created. Removing both checks is caught.

## How we fixed it

See above. The tool has no command that deletes, moves or uploads anything. The only files it removes or replaces are its own:
- temporary downloads outside the root;
- the staging name of a file once it is published;
- the unfinished file a failed copy has just created (exclusive creation proves it is its own);
- its output files, which never go inside the root except under `_logs/`.

## Evidence

```
$ python -m pytest tests/test_drive_consolidate.py -q
201 passed, 1 skipped (the Windows-only test)

$ python mut_all.py .   (each deliberate break applied alone, then the file's tests with -x;
                          the tool byte-compared with its backup afterwards)
on 6a00dcb, in four slices: caught 36 of 36, 36 of 36, 36 of 36 and 35 of 36; patterns gone 0
  MISSED  the copy log's place not guarded: checked twice, before the census and just before
          the log is created; the break of both checks: CAUGHT (1 failed, 192 passed, 1 skipped)
  by rule:
  - SHA-256 decides (plan, bytecheck, copy);
  - the census's completeness (per-parent fallback, the rclone size check, ListR disabled,
    files without an MD5);
  - zero size; no free destination; per-object retry; copy and verify with pending checks;
  - deletable per id (per row, first row, last row, any row); copies not deletable;
    read-only never; a second parent (item or area root); same-object mirrors;
  - rclone-rewritten, reserved and case-clashing names; the empty file by its own name;
  - never overwrite: publish, sweep, conflicts, a file or link appearing mid-run, the
    fallback only where links are unsupported, no partial file;
  - links and junctions, on Python 3.11 and 3.12 (inventory, plan, bytecheck, copy, sweep, verify);
    a link in the root on the way to an output, followed hop by hop; a '..' after a link;
  - the copy log is always a new file; one unreadable download fails alone;
  - stale inventories (sweep, verify's matched root files); verify's root;
  - credentials: never read, no hash kept (census and plan), any file named config or
    config.worktree, *.env; outputs never land in the root or through a link;
  - paths: every path to an object judged, a parent the census never listed, the 64-path
    cap, and one verdict per Drive object across every area (the census and the plan);
  - the plan is a snapshot: copy and bytecheck census again before the first download
    and after the last (path and safety); nothing changed is published or kept;
  - the census stops on one id listed two ways; a loop is an unknown path; the plan
    keeps no hash of what it keeps on Drive; no output while _logs is a link;
    bytecheck removes only the temporary folder it made;
  - sweep: destinations, unsafe names, unreadable folders, named pass-overs, staging;
  - bytecheck carries on; absolute paths for rclone; exit codes; older documents refused.

$ python -m pytest tests/ -m "not backtest_regression" -q   (on the final code)
3409 passed, 29 skipped, 8 deselected, 20 xfailed in 862.13s    pytest exit 0   (on 6a00dcb)

$ rclone v1.68.2: strings in the binary, and backend/drive/drive.go, walk.go, operations.go
"Disabling ListR to work around bug in drive as multi listing (%d) returned no entries"   (drive.go:2126)
json:"size,omitempty,string"
failed copying %q to %q: %w
walk.ListR: "FIXME disable this with --no-fast-list ??? `--disable ListR` will do it..."
newObjectWithExportInfo: a non-Google file without an MD5 is skipped under --drive-skip-gdocs
```

## Unresolved / handoff

- Card 1 runs the tool's tests on Windows; this run tested on Linux.
- A census mismatch can also come from two folders that share a name under one parent: rclone's walk by path cannot tell them apart. If card 1 stops on one, the error names them, and the pen decides what to do before anything changes on Drive.
- The per-object copy log (review 2, N-14) has no test of its own: only a crash in the middle of a batch would show the difference.
- Card 3 needs its own reviewed tool for deletion. This one has none, by design. The third review's advice for that tool:
  - run `verify` on the same plan, and require exit 0, just before trashing anything;
  - re-check each Drive object's MD5, SHA-256 and modified time at trash time;
  - re-check a twin that lives in a live tree (`data/`, `data_raw/`, `data_processed/`) at trash time, since the next data pull may rewrite it.
- An object renamed during its own download is read before a census can see the change. The census after the downloads keeps it out of the root, and nothing it changed into is kept. Card 2 still runs the copy with nobody editing the old areas.
- The tests of a plan with a folder that has two parents in one area test a census that real rclone stops first (the fake counts once per parent unless asked to count by path).
- **Another name for the root that `realpath` keeps passes the containment checks.** Examples: a bind mount on Linux; on Windows, a loopback share such as `\\localhost\C$`. The seventh review found this in its second round, and it predates this run. Fixing it means comparing folders by identity, and FAT and exFAT report no file id. The cards spell every output path in the ordinary way.
- Card 1 must check whether the Desktop is redirected into OneDrive. If it is, the root and the staging folder beside it sync, and hard links behave differently: stop and report.
