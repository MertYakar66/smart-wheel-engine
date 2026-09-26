---
id: data-home-desktop-round3-2026-09-23
title: "Desktop: history bundle, archive on Drive, Drive-only data pulled home (D31)"
kind: verification
status: merged
terminal: desktop
pr: 538
decisions: [D31]
date: 2026-09-23
headline: bundle built and proved to hold all 5 manifest git_sources and 1,101,549,144 B of history-only blobs, but the coverage gate FIRED (main moved to #530 mid-run); 29,262 Drive files pulled home, 29,260 checksum-identical and 2 ibkr conflicts kept on both sides; theta on Drive is only 13% of what the lost laptop held
surface: [docs/worklog/INDEX.md]
---

## Goal

Finish the desktop side of D31: make a full-history git bundle so the four data
branches become disposable, put `data_archive` and the bundle on Drive, re-point
the IBKR morning-pull task at the data root, and pull home every dataset that now
exists only on Google Drive — the data laptop is gone.

A gate fired at step 1, so **this fragment is pushed without a PR** per the card.

## What we tried

1. `git bundle create ... --all` from the checkout, exactly as the card specified.
2. When that died, diagnosing rather than retrying.
3. Rebuilding the bundle from a fresh full clone, leaving the checkout untouched.
4. Proving the bundle by reading data *out* of it, not by trusting `verify`.

## What worked

- Step 0's gate, first try.
- The IBKR task edit, applied and read back from the scheduler.
- All eight Drive children pulled home under `--ignore-existing`; 12,072 files
  verified, 2 conflicts preserved rather than overwritten.
- The bundle itself — once built from a source that actually had the bytes.

## What didn't

**`git bundle create --all` from this checkout, and the reason is the point.**

```
error: RPC failed; curl 18 transfer closed with outstanding read data remaining
error: 2275 bytes of body are still expected
fetch-pack: unexpected disconnect while reading sideband packet
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
fatal: could not fetch 4170997d0ff880fdb5f0d4df9f83f39e2c139b4b from promisor remote
error: pack-objects died
```

The checkout is a **blobless partial clone**:

```
core.repositoryformatversion 1
remote.origin.promisor true
remote.origin.partialclonefilter blob:none
```

It holds every commit and tree but fetches file contents on demand, so
`bundle create --all` had to re-download every historical blob — including the
very old versions the bundle exists to preserve — and the transfer died after
6m59s. No partial bundle file was written (`data_archive/git/` was empty
afterwards).

The danger is not the failure. It is that on a faster link this command can
**succeed** from a partial clone and produce a bundle that looks complete and
silently lacks the old blobs. A partial clone is not a safe source for a backup.

## How we fixed it

Built the bundle from a fresh **full** clone (`--no-checkout`, no filter), never
touching the checkout's config:

```
clone-exit=0
no promisor/filter config -> FULL clone confirmed
in-pack: 9071   size-pack: 1.58 GiB
```

Note the shape against the checkout's `in-pack: 9526, size-pack: 1.24 GiB` — the
partial clone has *more* object records but *fewer* bytes, exactly what a blobless
clone that has lazily back-filled some blobs looks like. With no promisor remote
configured on the new clone, git cannot lazily fetch, so a successful
`bundle create --all` there proves every object was already local.

## Evidence

### Step 0 — bring-up gate (PASS)

`main` fast-forwarded `b7bcd38..67b7134`, tree clean.

```
root: C:\Users\merty\Desktop\swe-data
checked 144 manifest files: 144 ok, 0 missing, 0 mismatched
exit=0
```

Re-run after ~3 GB of Drive data landed in the root — still `144 ok, 0 missing,
0 mismatched`, exit 0.

### Step 1 — the bundle

`C:\Users\merty\Desktop\swe-data\data_archive\git\smart-wheel-engine-all-refs-2026-09-23.bundle`

- size `1,698,024,795 B`
- sha256 `EE15DD9CBD0132FA4195947893F9A3531DA50A4DF69749BCBE8AB026CAF8FA84`
- written 2026-09-23 12:40:41

```
...smart-wheel-engine-all-refs-2026-09-23.bundle is okay
The bundle contains these 9 refs:
67b7134ad24fedda58a61518cba96a7c227696d8 refs/heads/main
67b7134ad24fedda58a61518cba96a7c227696d8 refs/remotes/origin/HEAD
597cc6af6e2b579d667ca24dc2b324cd766a2a58 refs/remotes/origin/backup/drive-tier-c-2026-07-22
2abf850d76b3de9a1e04301d333485e64061f3cd refs/remotes/origin/claude/daybot-bloomberg-pull
629237d01b76c5c26ac7ee919c3b4110b2a5e24d refs/remotes/origin/claude/project-restart-ai-agents-kot5jr
24835719ffa6d83a2c5e1dce4a7605b356695ede refs/remotes/origin/data/drive-migration
68a48b245cea285d98f86ab7e6ba1b5cdca7002d refs/remotes/origin/deep-history/bloomberg-raw
67b7134ad24fedda58a61518cba96a7c227696d8 refs/remotes/origin/main
67b7134ad24fedda58a61518cba96a7c227696d8 HEAD
The bundle records a complete history.
The bundle uses this hash algorithm: sha1
verify-exit=0
```

### Step 1 — the coverage gate (FIRED)

```
bundle holds 4 of 6 GitHub refs at the same commit; 2 differ
  DIFFERS  refs/remotes/origin/claude/project-restart-ai-agents-kot5jr: GitHub 5db6847c0, bundle 629237d01
  DIFFERS  refs/remotes/origin/main: GitHub f1c0066e0, bundle 67b7134ad
coverage exit=1
```

**Both refs moved after the bundle was cut, by other actors, mid-run.**

`main` advanced because **D31 step 5 merged while this run was in flight**:

```
f1c0066 Merge pull request #530: refactor(data-home): git tracks no market data (D31 step 5)
b3a1d4e fix(data-home): every script resolves its data paths through engine.paths (D31 step 5)
629237d Merge origin/main (#529) into the D31 step-5 branch
14a1ae6 docs(worklog): D31 step 5 — the final no-data CI-form result
40dfe69 docs(data-home): changelog and worklog for D31 step 5
76c3ecd test(data-home): CI without data — floors and three tests that read the checkout (D31 step 5)
81993a8 refactor(data-home): untrack the market data from git (D31 step 5)

 134 files changed, 742 insertions(+), 7675787 deletions(-)
```

`claude/project-restart-ai-agents-kot5jr` is a third party's live branch; it moved
twice during this run (`ab89a16` → `629237d` → `5db6847`).

The four **data-carrying** branches are held at exactly their GitHub commits.
Only `main` (fully preserved on GitHub and in this checkout) and a foreign feature
branch differ.

### Step 1 — the bundle proved by readback, not by trust

`git bundle verify` is git's own claim. These are independent:

Fetched `deep-history/bloomberg-raw` straight out of the bundle into an empty
repo (`tip: 68a48b245cea285d98f86ab7e6ba1b5cdca7002d`) and compared extracted
blobs to the root's archived copies:

```
FILE                                         SIZE  MATCH    sha256
rates_fx_vol.csv                           583872  IDENTICAL f9b86c2b3821e900f13dc177…
vol_indices.csv                            943706  IDENTICAL eadea41788193e6f565fc405…
sp500_iv_history.csv                           20  IDENTICAL e130dcc6f9210bc176b3e2e4…
spx_correlation.csv                        352416  IDENTICAL 541e6a73944b5d8d2af9db4d…
sp500_institutional.csv                     23534  IDENTICAL ca286693fc888e6248e6cfe6…

bundle-blob == root-archive : 5 identical, 0 differ
```

**Every one of the manifest's five `git_sources` is inside the bundle** — which is
what makes the bundle a self-sufficient recovery source now that #530 has
untracked the data from `main`:

```
  PRESENT  69bf3b933  fix(ibkr-adapter): null-NAV risk_view + provenance staleness
  PRESENT  68a48b245  archive: preserve 2026-06-04 deep-history session transcript
  PRESENT  2abf850d7  data(day-bot): tick auto-persist (+1 files)
  PRESENT  597cc6af6  docs(backup): features verified 0 diff — 8/9 Tier-C children
  PRESENT  24835719f  fix(manifest): register data-migration files, drop phantom dat

git cat-file -s 69bf3b933:data/bloomberg/sp500_ohlcv.csv -> 63446582
```

History-only content the bundle preserves (across all its refs):

```
paths under data/ seen anywhere in history: 114
blob versions NOT at any ref tip (only history holds them): 136
their total size: 1,101,549,144 B (1,051 MiB)
largest contributors:
     246,673,591 B  3 old version(s)  data/bloomberg/sp500_vol_iv_full.csv
     171,963,397 B  9 old version(s)  data/bloomberg/deep/sp500_ohlcv__1994_2018.csv.gz
     139,325,527 B  2 old version(s)  data/bloomberg/sp500_liquidity.csv
     123,209,004 B  2 old version(s)  data/bloomberg/sp500_ohlcv.csv
     101,755,030 B  8 old version(s)  data/bloomberg/deep/sp500_liquidity__1994_2015.csv.gz
      96,772,128 B  1 old version(s)  data/bloomberg/broad_pull/iv_surface/sp500_iv_surface.csv.gz
      58,684,159 B  1 old version(s)  data/bloomberg/broad_pull/per_name/vol_term_rv.csv.gz
      30,129,416 B  2 old version(s)  data/bloomberg/deep/sp500_vol_iv_full__1994_2012.csv.gz
```

The card's estimate was 36 versions / 603 MB for `deep-history/bloomberg-raw`
alone; the figure above spans every ref in the bundle, so it is a superset, not a
contradiction. Either reading, **over a gigabyte of data exists only inside git
history** and the bundle is now its only copy outside GitHub.

### Step 2 — IBKR morning-pull task (APPLIED)

Backup: `_logs\ibkr_task_before_2026-09-23.xml` (1,648 B).

```
BEFORE: /c py -3.12 C:\Users\merty\swe-ops\scripts\ibkr_gateway_pull.py --out C:\Users\merty\Desktop\smart-wheel-engine\data_processed\ibkr\portfolio_snapshot.json >> C:\Users\merty\swe-ops\logs\ibkr_pull.log 2>&1
AFTER : /c py -3.12 C:\Users\merty\swe-ops\scripts\ibkr_gateway_pull.py --out C:\Users\merty\Desktop\swe-data\data_processed\ibkr\portfolio_snapshot.json >> C:\Users\merty\swe-ops\logs\ibkr_pull.log 2>&1
```

Read back from the scheduler after `Set-ScheduledTask`: `Execute = cmd`,
`WorkingDirectory = ''`, `State = Ready` — both unchanged. Task not run.
The checkout's `data_processed\ibkr` had **not** reappeared.

Separately: the task's last run (08:17:40 today) returned `2147946720` =
`0x800710E0`, "the operator or administrator has refused the request". It was
already failing before this edit; not this card's surface.

### Step 3 — data_archive to Drive (GATE PASSED)

30 local files (29 archived data files + the bundle), 1.9 GB, uploaded
12:42:56 → 13:02:55.

```
2026/09/23 13:02:58 NOTICE: Google drive root 'data_archive': 0 differences found
2026/09/23 13:02:58 NOTICE: Google drive root 'data_archive': 30 matching files
check exit=0
```

New folder id: **`data_archive = 1hCmngYyGwSHvkCT_BmF-EC-t8fkUA7xi`** (a child of
`swe-local-only` = `1JwPWszfyggUDT1vYaRjZ8nlHEDR3vEOn`).

This is the first off-machine copy of the bundle, and the first copy of the 29
archived branch-only files anywhere but this desktop.

The step-4 `ibkr` conflict files were deliberately archived **after** this gate,
so that `30 matching files` means exactly what the card specified rather than
silently becoming a different number. A second, additive upload then took
`data_archive` to `0 differences found` · `33 matching files` — see the stray-file
note below for why 33 and not 32.

### Step 4 — Drive → desktop, all eight children

`--ignore-existing` throughout, so no desktop file was ever rewritten.

| child | Drive objects | Drive bytes | `rclone check --checksum --one-way` |
|---|---:|---:|---|
| `data_raw` | 11 | 2,373,074 | `0 differences found` · 11 matching |
| `trade_universe` | 1 | 166,350 | `0 differences found` · 1 matching |
| `vol_indices` | 2 | 762,936 | `0 differences found` · 2 matching |
| `data_processed_root` | 5 | 95,078 | `0 differences found` · 5 matching |
| `validation` | 23 | 23,287,908 | `0 differences found` · 23 matching |
| `ibkr` | 19 | 1,210,565 | **2 differences** · 17 matching |
| `option_premium` | 155 | 1,905,629,509 | `0 differences found` · 155 matching |
| `features` | 11,858 | 1,218,784,927 | `0 differences found` · 11858 matching |

12,072 of 12,074 files verified identical. The two `ibkr` differences are both
cases where **this desktop's file is newer than Drive's**:

| path | desktop | Drive |
|---|---|---|
| `portfolio_snapshot.json` | 5,279 B · 2026-07-18 01:42 | 4,493 B · 2026-06-10 17:04 |
| `portfolio_history.json` | 2,968 B · 2026-07-18 17:07 | 2,896 B · 2026-06-11 20:12 |

Both sides kept. Drive's versions were copied to
`<root>\data_archive\drive-swe-local-only\ibkr\`; the desktop's newer files were
never touched. See `_logs\differ_ibkr.txt`.

**A stray file was created here and is deliberately left in place.** The first
attempt at that copy ran through bash, where `"$DEST\\$p"` collapsed to a literal
`ibkr$p` instead of `ibkr\<name>`; `rclone copyto` then created a file actually
named `ibkr$p`, and the additive upload carried it to Drive. It is byte-identical
to `portfolio_history.json` (`680a5e8ecef61a1375d272798040a5afc397f04dda452584886a16de619ad1d7`),
so it holds no unique data. This card forbids deleting a file anywhere, so it was
**not** removed — it is the reason the archive counts 33 rather than 32, and it
should be deleted by the Operator at
`<root>\data_archive\drive-swe-local-only\ibkr$p` and on Drive.
Also note the exit code in that first attempt read `0` only because `$?` followed
a `| tail` pipeline — it was `tail`'s status, not rclone's. Check rclone's own
exit code, not a pipeline's.

`features` was the long pole: 12:00:49 → 12:34:17 for 11,858 files. The 26
manifest-tracked feature files already present were skipped by
`--ignore-existing` and still came back as matching, so Drive's copies are
byte-identical to them.

### Step 5 — theta: the backup is not a backup

```
Total objects: 17.188k (17188)
Total size: 1.245 GiB (1337169896 Byte)
```

Against the ~132,862 files / ~11 GB the laptop held, Drive has **~13% of the
files and ~11% of the bytes**. `data_processed\theta` was ABSENT on both this
desktop and the checkout before today. The upload was never verified, and it was
substantially incomplete.

Copy started detached 2026-09-23 11:55:08 (PID 31476), first log lines:

```
2026/09/23 11:55:10 INFO  : _chains.out: Copied (new)
2026/09/23 11:55:10 INFO  : _chains.err: Copied (new)
2026/09/23 11:55:10 INFO  : _ivsmoke.err: Copied (new)
2026/09/23 11:55:10 INFO  : _chains_rerun.log: Copied (new)
2026/09/23 11:55:10 INFO  : _ivsmoke.start: Copied (new)
2026/09/23 11:55:10 INFO  : _manifest.json: Copied (new)
```

Log: `_logs\theta_pull.log`.

The copy finished inside this session rather than needing a later run:

```
Transferred:   	    1.245 GiB / 1.245 GiB, 100%, 1.399 MiB/s, ETA 0s
Transferred:        17188 / 17188, 100%
Elapsed time:     50m58.2s
```

`17188 files, 1,337,169,896 B` on disk — exactly the object count and byte count
Drive reported — and zero `ERROR`/`Failed to` lines in the log.

The per-file checksum pass also completed inside the session (12:47:46 →
13:08:50):

```
2026/09/23 13:08:50 NOTICE: Local file system at //?/C:/Users/merty/Desktop/swe-data/data_processed/theta: 0 differences found
2026/09/23 13:08:50 NOTICE: Local file system at //?/C:/Users/merty/Desktop/swe-data/data_processed/theta: 17188 matching files
theta check exit=0
```

`_logs\differ_theta.txt` is empty. **What Drive holds is now fully and provably
on this desktop; what the laptop held is not.**

### Step 6a — untracked data under the checkout

13 untracked non-`.py` files, 373,354,604 B, **every one byte-identical at the
root** (all under `data/bloomberg/deep/`). Nothing lives only in the checkout.

### Step 6b — the swe-ops clone

```
fatal: not a git repository: C:/Users/merty/Desktop/smart-wheel-engine/.git/worktrees/swe-ops
```

`C:\Users\merty\swe-ops` is an **orphaned worktree**: its `.git` file says
`gitdir: C:/Users/merty/Desktop/smart-wheel-engine/.git/worktrees/swe-ops`, that
directory does not exist, and `git worktree list` on the primary reports only the
primary. So `log`/`status` are impossible there — the card's step 6b commands
cannot run by construction.

Audited by filesystem walk instead. Contents frozen at 2026-06-12 (it still
carries `AGENTS.md` and `COMMIT_GUIDE.md`, which #522 deleted from main):

- 53 files identical at the root
- 7 files differing — all **older and smaller** June copies (`sp500_ohlcv.csv`
  62,443,071 vs the root's 63,446,582, etc.)
- 3 files absent at the root, none of them data: `EXTRACTION_GUIDE.md` (tracked
  on `main` and present in the checkout), `sp500_iv_history.csv` (20 bytes,
  header only: `date,ticker,iv_30d`), `.gitkeep` (2 bytes)

**swe-ops holds no unique data.** It remains load-bearing: the scheduled task
executes `swe-ops\scripts\ibkr_gateway_pull.py`.

### Invariants (OPERATING_MODEL.md §7)

- **No file deleted or overwritten** — every Drive pull used `--ignore-existing`;
  the two `ibkr` conflicts were preserved on both sides; nothing was moved.
- **`flex_credentials.json` never left this machine** — it is 93 B at
  `<root>\data_processed\ibkr\`, and a recursive search of Drive's `ibkr` folder
  for `credential|secret|token` returned nothing. No step of this card uploads
  the folder it lives in.
- **No data committed** — this branch commits two files, both under
  `docs/worklog/`.
- **The four data branches untouched beyond fetching** — verified by tip:
  `68a48b2`, `2abf850`, `597cc6a`, `2483571`, unchanged from GitHub.
- **No history rewrite, no `git rm`/`clean`/`stash`/`reset --hard`/`--prune`.**

## Unresolved / handoff

- **The step-1 coverage gate is unresolved and needs the Operator's call.** The
  bundle is complete for every data-carrying ref and for all five manifest
  `git_sources`; it lacks only `main@f1c0066` (#530, itself safely on GitHub) and
  a foreign branch that moves continuously. Re-cutting the bundle will keep
  racing that branch. Decide whether "every GitHub ref" or "every ref that
  deletion would destroy" is the standard.
- **Do not merge `origin/main` into this branch.** #530 deletes 7,675,787 lines
  of tracked data; merging applies those deletions to this checkout's working
  tree, removing ~1.4 GB of data files from the desktop. That is forbidden by
  this card. This also means `INDEX.md` here is generated without #530's two new
  fragments, so the merge-ref worklog-index guard may flag it — that is expected
  and is why no PR was opened.
- **Roughly 9.8 GB of theta data appears to be gone.** Drive holds 13% of the
  laptop's file count. If the laptop is recoverable at all, this is the last
  chance. Otherwise `docs/DATA_INVENTORY.md` §C.1 should record theta as
  partial, not backed up.
- **`DATA_INVENTORY.md` §C row B′/§C.1 remain stale** on the tick Drive copy from
  the previous round, and now also need the `data_archive` folder id and the
  theta shortfall.
- **Delete the stray `data_archive\drive-swe-local-only\ibkr$p`**, locally and on
  Drive. It is an exact duplicate of `portfolio_history.json` created by a shell
  quoting bug in this run; this card's "never delete a file" constraint meant it
  could not be cleaned up here.
- **The IBKR task's `0x800710E0` failures** predate this run and are unexplained.
- **`SWE_DEEP_TEST_DATA` is still unset**; pointing it at `<root>\data\bloomberg`
  unlocks 8 tests that skip today.
- Helper scripts left at `<root>\_logs\`: `bundle_covers_github.py`,
  `untracked_audit.py`, `tree_vs_root.py`.

## Superseded at the close (the pen, 2026-09-26)

Card 1 (#538) settled or overtook several items above. The text above is kept as
written.

- **The coverage gate.** D33 point 7 set the standard: the four exact branch
  commits. Card 1 restored all four from the desktop's bundle. The restore from
  Drive's copy is still to come, in card 2.
- **"Do not merge `origin/main`".** Card 1 merged it under guard (`e13220f`), after
  the Operator's yes. It removed the 87 checkout copies, 566,845,655 B by the
  manifest (not ~1.4 GB), each first proved identical to the root's copy.
- **The morning pull.** Card 1 unregistered it, so swe-ops is no longer
  load-bearing, and its `0x800710E0` failures no longer matter.
- **"Theta … appears to be gone / Drive holds 13%"** measured only
  `swe-local-only/theta`. Card 1's census found a second, far fuller Theta upload in
  `SmartWheelData/data_processed/theta`. Card 1b confirms its size.
- **"`flex_credentials.json` never left this machine".** The Drive search here
  covered only `swe-local-only`'s `ibkr`. Card 1 found an older credential-shaped
  `flex_credentials.json` (674 B) in SmartWheelData's `ibkr`. It has never been
  opened.
- **Still open:** the stray `ibkr$p` (card 3), and `SWE_DEEP_TEST_DATA`.
