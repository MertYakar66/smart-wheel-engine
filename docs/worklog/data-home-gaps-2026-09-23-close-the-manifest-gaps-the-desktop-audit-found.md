---
id: data-home-gaps-2026-09-23
title: Close the manifest gaps the desktop audit found (D31)
kind: fix
status: completed
terminal: sandbox
pr: 528
decisions: [D31]
date: 2026-09-23
headline: manifest 99 → 144 rows — the 16 tracked feature sidecars plus a 29-file data_archive of what only the non-main branches held (pre-2004 MOVE/SKEW/JPMVXYG7 among it); guard test; two Windows-only test fixes
surface: [data/DATA_MANIFEST.json, scripts/data_manifest.py, tests/test_data_manifest.py, tests/test_data_paths.py, docs/DATA_INVENTORY.md, docs/DATA_POLICY.md, DECISIONS.md]
---

## Goal

The desktop run of D31 steps 2–3 (#527, merged `fc03b94`) proved the root
byte-exact against the manifest and then showed that the proof has a blind
spot: `check` iterates manifest rows only, so "0 missing" says nothing about a
file the manifest never listed. It named two such gaps. Close both in the
manifest, so that step 5 (untrack) cannot drop a tracked file's only copy and
step 6 (delete the non-`main` branches) cannot lose a dataset — and add the
guard that would have caught the first gap.

## What we tried

1. Re-derived both gaps here from git, independently of the desktop's audit:
   `git ls-files` of the data trees on `main` against the manifest's paths
   (gap A), and, for each non-`main` branch, every blob reachable from it but
   not from `main`'s history (gap B).
2. Considered listing only the six branch-only datasets the desktop named.
   Rejected: it needs a judgement per file (is `main`'s version really a
   superset?). The rule adopted needs none — every distinct data file (by
   sha256) at any non-`main` branch tip that no manifest row carries.
3. Considered placing the branch files at their original paths under the root.
   Rejected: several paths are live on `main` with different bytes, and a
   branch-only path such as `data/bloomberg/vol_indices.csv` could be picked up
   by a loader. They go under `data_archive/<branch>/<git path>`, which no code
   reads, with `git_path` telling `materialize` where the bytes sat in git.
   Not `archive/`: the repository already has an `archive/` folder of retired
   code.

## What worked

- **Gap A, confirmed:** 87 data-shaped files are tracked under the data trees
  on `main`; the manifest had 71. The 16 missing are every
  `data/features/<group>/ticker=AAPL/{metadata.json,stats.json}` pair — the
  files `FeatureStore.get_metadata` / `get_stats` read (the desktop measured
  `None` from the root without them). Their blobs are identical at the
  manifest's `git:main` commit `69bf3b9` and at `main`.
- **Gap B, confirmed and larger than one file list.** Blobs reachable only from
  each branch (not from `main`'s history): `deep-history/bloomberg-raw` 73
  (1.22 GB: 13 already in the manifest, 24 at the tip, 36 older versions);
  `claude/daybot-bloomberg-pull` 20 (15 in the manifest, 5 small scripts/docs);
  `backup/drive-tier-c-2026-07-22` 162 (7.5 MB, docs/code);
  `data/drive-migration` 7 (310 KB, scripts).
- **The archive rule** over the four branch tips gives 29 distinct data files,
  312.8 MB (24 from `deep-history/bloomberg-raw`, 3 from the day-bot branch, 1
  each from the other two). The branches' copies of the sidecars are
  byte-identical to `main`'s, so the sidecar rows cover them.
- **The replay of the desktop's next round**, on a scratch root: the old
  99-row manifest first, then the new one —
  `materialize 144 manifest files: 99 already present, wrote 45, 0 mismatched (kept, not overwritten), 0 unavailable, 0 failed`,
  then `checked 144 manifest files: 144 ok, 0 missing, 0 mismatched`.
- **The unique history is really there**, read back from the archived files:
  `rates_fx_vol.csv` MOVE 1988-04-04 → 2026-06-04 (9,871 rows), JPMVXYG7
  1992-06-01 → (8,868), CVIX 2001-08-29 →; `vol_indices.csv` SKEW 1990-01-02 →
  (9,159), VXN 2001-02-02 →.

## What didn't

- **The first manifest pass (2026-09-18) took only `data/bloomberg/deep/` from
  `deep-history/bloomberg-raw`** and missed the sidecars on `main`. The
  materialize-then-check proof on the desktop could not see either gap, because
  both sides of it read the same manifest. The desktop's independent audit
  caught what the tool could not.
- **Two Windows-only failures in this campaign's own tests (#526):**
  `_make_root` wrote the fixture CSV with `write_text` (CRLF on Windows) and
  the throwaway repo's `core.autocrlf=true` stored LF, so the git round trip
  changed the bytes; `Path("/abs/ibkr")` is not absolute on Windows, so the
  override was re-rooted. CI (Ubuntu) could not see either.

## How we fixed it

- `data/DATA_MANIFEST.json`: 144 rows / 1,743,765,458 bytes — the 16 sidecars
  (`group: features`, `git_source: git:main`) and 29 `archive` rows
  (`data_archive/<branch>/<git path>`, `git_path`, `git_source` per branch);
  `git_sources` gains `backup/drive-tier-c-2026-07-22` @ `597cc6a` and
  `data/drive-migration` @ `2483571`. Every new row was computed from the git
  blob itself (size, sha256).
- `scripts/data_manifest.py`: `git_path` (default `path`) is what
  `materialize` reads; the `archive` group; `data_archive` joins `WALK_DIRS`;
  `build` carries `git_path` with `git_source`.
- `.gitignore`: `data_archive/`.
- Tests: `test_every_tracked_data_file_has_a_manifest_row` (red on the old
  manifest, naming the 16 sidecars; green now);
  `test_materialize_writes_archive_rows_from_their_git_path`; the committed
  manifest's archive rows are well-formed and no two rows carry the same bytes;
  the fixture writes bytes and pins `core.autocrlf=false`; the resolver test
  uses an absolute path from `tmp_path`.
- Docs: `DECISIONS.md` D31 status note; `docs/DATA_INVENTORY.md` §A (step
  table, new step 2b), §B, §C (tiers R and C), new §C.2 (the archive, file by
  file, and what only a git bundle can keep); `docs/DATA_POLICY.md` §3/§6;
  `FILE_MANIFEST.md` rows; `CHANGELOG.md`.

## Evidence

```
tracked data-shaped on main: 87 | in manifest: 71 | not in manifest: 16
=== deep-history/bloomberg-raw @ 68a48b2
commits not on main: 35 | blobs only reachable from this branch: 73 files, 1218605362 bytes
  in-manifest 13 blobs 373.4 MB | tip 24 blobs 241.8 MB | history-only 36 blobs 603.4 MB
=== claude/daybot-bloomberg-pull @ 2abf850
commits not on main: 16 | blobs only reachable from this branch: 20 files, 490733878 bytes
=== backup/drive-tier-c-2026-07-22 @ 597cc6a
commits not on main: 49 | blobs only reachable from this branch: 162 files, 7497610 bytes
=== data/drive-migration @ 2483571
commits not on main: 2 | blobs only reachable from this branch: 7 files, 309785 bytes

archive: deep-history 24 · daybot 3 · backup 1 · drive-migration 1 → 29 files, 312.8 MB
rows: 144 | bytes: 1743765458
counts: archive 29, bloomberg 22, broad_pull 27, deep 13, features 26, processed 1, raw 11, ticks 15

materialize 99 manifest files: 0 already present, wrote 99, 0 mismatched (kept, not overwritten), 0 unavailable, 0 failed
materialize 144 manifest files: 99 already present, would write 45, 0 mismatched (kept, not overwritten), 0 unavailable, 0 failed
materialize 144 manifest files: 99 already present, wrote 45, 0 mismatched (kept, not overwritten), 0 unavailable, 0 failed
checked 144 manifest files: 144 ok, 0 missing, 0 mismatched

tests/test_data_manifest.py + tests/test_data_paths.py: 18 passed
(before the manifest update: the new guard failed with
 "Left contains 16 more items, first extra item: 'data/features/dynamics/ticker=AAPL/metadata.json'")
```

## Unresolved / handoff

1. **Desktop, round 2:** `git pull`; `materialize` (writes the 45 new rows);
   `check` must read `checked 144 manifest files: 144 ok, 0 missing, 0 mismatched`.
   Step 5 is held on that line.
2. **Before step 6:** a full-history bundle of every branch on the desktop
   (`data_archive/git/`), `git bundle verify` clean and its heads equal to
   `git ls-remote --heads origin` — the only copy of the 36 older versions on
   `deep-history/bloomberg-raw` and of the branches' scripts and docs — then
   Drive copies of the ticks, `data_archive/` and the bundle, `rclone check`
   clean. Blocked on the Operator's `rclone config reconnect gdrive:`.
3. **The IBKR morning-pull task** (#527 handoff 5) writes to an explicit path
   inside the checkout; re-point its `--out` at the root.
4. **Step 2b:** the data laptop's local-only stores are not on the desktop.
