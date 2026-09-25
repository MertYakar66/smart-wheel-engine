---
id: d33-data-estate-2026-09-24
title: "D33: the data estate after the MacBook; Drive a complete second copy"
kind: docs
status: merged
terminal: sandbox
pr: 533
decisions: [D33, D31]
date: 2026-09-24
headline: The Operator dropped the MacBook and asked for Drive to hold all the data with no duplicates. Codex reviewed the plan twice; every finding was checked and accepted. D33 records the approved plan, which copies and never moves, deletes only named proven files, and deletes the four data branches only after the history bundle restores from Drive.
surface: [DECISIONS.md, PROJECT_STATE.md, docs/DATA_INVENTORY.md, docs/DATA_POLICY.md, docs/deadlines.md, CHANGELOG.md]
---

## Goal

Record what the Operator decided on 2026-09-24 about where the data lives, once
the MacBook was dropped:

- "we will forget the macbook exists. we will pull from Theta Data or anywhere
  else from the beginning. However, we will keep Bloomberg data at all costs"
- "1. if Desktop has them certainly, no need to leave it on the github. delete
  them 2. Drive is our secondary source of storage, if something happens to
  desktop, Drive should have all the data 3. Theta subscription is no longer
  active. Once we are sure that our repo and working system/mechanism is
  efficient, we will discuss about collecting which data from where..etc so
  leave it as is until we figure out the repo structure 4. yes, start filling
  google drive gaps, make sure there are no duplicates. and the folder structure
  in the drive must also be clear/noted down for future agents/work"

The Operator approved a plan, asked for a second opinion from Codex first, and
then approved the revised plan and D33's wording ("yes").

## What we tried

1. **A move-based plan.** Old Drive files matching a desktop file by MD5 and size
   would be moved server-side into the new folder, and everything left over
   trashed after a mirror check.
2. **Codex review 1** ("agree with changes"). It raised eleven findings, two of
   them blockers.
3. **A copy-based plan**, with every finding accepted and checked.
4. **Codex review 2** ("agree with changes"). It raised four corrections, all
   accepted.

## What worked

Checking every review finding against evidence before accepting it:
- **rclone v1.68.2's Drive backend source.** A shortcut is listed as its target,
  with a composite id. A server-side `Move` updates the shortcut's own id, so it
  re-parents the pointer, not the data.
- **Local tests with rclone v1.68.2:**
  - `--files-from` combined with `--exclude-from` refuses to start ("the usage of
    --files-from overrides all other filters");
  - `move --ignore-existing` skips a destination with different bytes, and leaves
    the source in place;
  - `rclone checksum sha256 SHA256SUMS <dir>` reports missing, altered and extra
    files, and exits non-zero.
- **Local tests with git 2.43:**
  - a plain `git merge` silently replaced an ignored local file;
  - `--no-overwrite-ignore` aborted and kept it;
  - `git push --atomic --force-with-lease=<ref>:<sha> … :<ref>` deleted nothing
    when one branch had moved, and deleted both when both were at their expected
    commits.
- **GitHub:**
  - PR #507 has been closed since 2026-09-23 07:22:49 UTC, and its head ref is
    retained;
  - 20 CSV files under `staging/` (7,152,880 B) are tracked and outside the
    manifest;
  - `67b7134..origin/main` adds 8 files, deletes the 87 manifest files and
    modifies 58. None of the added paths is ignored by the desktop branch's
    `.gitignore`.

## What didn't

- **The move-based design.** Moving an old Drive file can move a shortcut instead
  of the data. `--files-from` cannot express a rename. `--ignore-existing` skips
  a file on its name alone. Trashing "everything left" deletes files no check ever
  compared.
- **Three of the pen's claims were wrong, and Codex corrected them:**
  - "Old versions stay on GitHub in `main`'s history and #507". In fact 32 data
    versions are reachable only from the four branches' histories.
  - "Drive keeps a SHA-256 for every ordinary file". rclone's Drive docs say a
    small fraction of files lack SHA-1 or SHA-256.
  - The first draft deleted `ibkr$p` in card 1, before any restore test.

## How we fixed it

D33, as approved:
- **Copy, never move.** Nothing is deleted until everything is proven, and then
  only from a named list, with the Operator's yes.
- **One Drive folder, `swe-data/`,** laid out like the root, checked both ways,
  with a `SHA256SUMS` list in both places.
- **Drive-only content comes home** to `data_archive/drive-legacy/<area>/<path>`,
  never over an existing file, with a ledger.
- **A duplicate needs the same size, MD5 and SHA-256.** A missing hash never
  matches.
- **The four branches go only after** the bundle restores from Drive's copy, and
  only if each is still at its checked commit.

The records now match: D31 status, `PROJECT_STATE.md` §0 A and B,
`docs/DATA_INVENTORY.md` §A (2b abandoned, 6 and 7), the §C tier rows, §C.1,
§C.2, and a new §C.3 listing the four Drive areas with their ids.
`docs/DATA_POLICY.md` §3 and §6 were updated, and the Theta subscription row in
`docs/deadlines.md` is closed.

**Opportunistic fix:** the Tier R row of `docs/DATA_INVENTORY.md` §C showed an
unrendered placeholder, `{tot/1e6:.1f} MB`. It now reads 312.8 MB (the §C.2 total
of 312,846,180 B).

## Evidence

```
# a scratch script in the pen's sandbox (not committed): the blob ids every commit introduced
# (git log --root -m --raw --no-renames) on the four branches, minus main's history, #507's head and the four tips
data blobs reachable from the four branches, held by neither main's history, PR #507's head, nor any tip: 32
  (e.g. c3fe8dee1 and 1c5dd1ad1, data/bloomberg/deep/sp500_liquidity__1994_2015.csv.gz, both changed by f7a8458 on deep-history/bloomberg-raw)

$ git ls-remote origin 'refs/pull/507/*'
24835719ffa6d83a2c5e1dce4a7605b356695ede	refs/pull/507/head

$ rclone copy src dst --files-from list.txt --exclude-from filter.txt --ignore-case
CRITICAL: Failed to initialise global options: failed to reload "filter" options: the usage of --files-from overrides all other filters, it should be used alone or with --files-from-raw

$ git push --atomic --force-with-lease=refs/heads/data1:$A --force-with-lease=refs/heads/data2:$A origin :refs/heads/data1 :refs/heads/data2
 ! [rejected]        (delete) -> data1 (atomic push failed)
 ! [rejected]        (delete) -> data2 (stale info)

$ python scripts/check_working_structure.py
working structure: 6 of 6 checks pass
```

## Unresolved / handoff

1. **The consolidation tool is next.** It covers:
   - the Drive census, taken by raw object queries so shortcuts are never
     followed;
   - the plan and the ledger;
   - copying by Drive id without overwriting;
   - the checksum list.

   Its tests must cover these cases:
   - same path, different content;
   - interrupted reruns;
   - a missing SHA-256;
   - shortcuts, Google-format files, duplicate names, credential names and
     Windows-unsafe names.
2. **Then desktop cards 1–3,** and step 6 between cards 2 and 3. Codex reviews
   card 3 before it runs.
3. **The day-bot project's `_local_archive/vendor_swe_*` stays read-only.** The
   Operator may later ask for those copies to be removed.
