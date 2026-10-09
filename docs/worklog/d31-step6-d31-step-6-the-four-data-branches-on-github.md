---
id: d31-step6
title: D31 step 6: the four data branches on GitHub
kind: verification
status: in-flight
terminal: desktop
pr:
decisions: [D31, D33]
date: 2026-10-09
headline: The four data branches deleted from GitHub in one atomic push with a lease on each; main and refs/pull/507/head untouched, and exactly four of the remote's 532 refs went.
surface: []
---

**The four data branches deleted; main untouched.**

## Goal

Remove the four data-carrier branches from GitHub, now that their whole history is
proven restorable from the full-history bundle, which is on the desktop and on
Drive. D31 step 6 under D33 point 7: the branches go only after the bundle
restores from Drive's copy into an empty repository with `git fsck --full` clean
and the four exact commits present (card 1, #538, and card 2d, #556), and only if
each branch is still at its checked commit — one atomic push, with a lease on
each.

| branch | checked commit |
| --- | --- |
| `deep-history/bloomberg-raw` | `68a48b245cea285d98f86ab7e6ba1b5cdca7002d` |
| `claude/daybot-bloomberg-pull` | `2abf850d76b3de9a1e04301d333485e64061f3cd` |
| `backup/drive-tier-c-2026-07-22` | `597cc6af6e2b579d667ca24dc2b324cd766a2a58` |
| `data/drive-migration` | `24835719ffa6d83a2c5e1dce4a7605b356695ede` |

## What we tried

The card's five steps in order, with its one gate between steps 2 and 3:

1. **Step 0** — fetched without `--prune`, branched `claude/d31-step6` from
   `origin/main`, and took the Executor mark.
2. **Step 1** — checked every premise, read only: the desktop root, the four tips
   on GitHub, `refs/pull/507/head`, the manifest's `git_sources`, the bundle's
   size and sha256, the bundle's heads, the four commits in this checkout, and the
   open pull requests.
3. **Step 2** — wrote four local keep-refs under `refs/kept/d31-step6/`, never
   pushed.
4. **An independent read-only audit, before the gate.** Five auditors in separate
   contexts checked the card's premises, what breaks when the four refs vanish,
   the safety net, the card's bash blocks as shell code, and the governance; each
   concern raised was then handed to a skeptic to refute. Nineteen agents, no
   writes anywhere. It found two true defects in the card's `<context>` (below)
   and settled the push's one load-bearing safety property: `--force-with-lease`
   is enforced on a *deletion* at the installed git 2.53.0.windows.1 — upstream's
   release-gating test `t/t5533-push-cas.sh` at tag `v2.53.0` is titled "compare &
   swap push force/delete safety" and asserts that a delete refspec with a stale
   lease is refused and the branch survives.
5. **G1** — the gate, with both defects put to the Operator before the question.
   Answer: `yes`.
6. **Step 3** — one `git push --atomic --porcelain` with four
   `--force-with-lease=refs/heads/<branch>:<tip>` options and four
   `:refs/heads/<branch>` delete refspecs.
7. **Step 4** — read GitHub back, read only.

## What worked

- **The push went through on its first and only attempt**: four `[deleted]` lines
  in name order, `Done`, `push exit 0`. Step 3's own gate read
  `keep-refs: 4 of 4` and named the step 1 file and G1 record it had checked.
- **Exactly four refs went, and nothing else.** The pre-push audit counted the
  remote's full advertisement at 532 refs (1 `HEAD`, 5 `refs/heads/*`, 526
  `refs/pull/*`, 0 tags). After the push it is 528: 1 `HEAD`, 1 `refs/heads/*`
  (`main`), 526 `refs/pull/*`, 0 tags.
- **`main` is untouched** at `ed85e1d9193861f25931145202d20207b962640e`, step 0's
  commit, so step 4 printed no `NOTE:` about it.
- **`refs/pull/507/head` is untouched** at `24835719ffa6d83a2c5e1dce4a7605b356695ede`,
  as D33 point 7 requires. It is now the only ref on GitHub pointing at any of the
  four tips.
- **The four keep-refs hold their checked commits** after the push, and the four
  `refs/remotes/origin/<branch>` tracking refs are gone, which is what the push
  itself does.

## What didn't

The run hit no `STOPPED:` line and needed no rerun. Three inaccuracies in the
card were found before the push, none of which changed what the push did:

1. **"no local branch holds those tips" is false for one of the four.**
   `refs/heads/data/drive-migration` exists in this checkout at exactly
   `24835719…`. Harmless, and in the safe direction — it is an extra holder of
   that tip. It had no effect on the push: both the delete refspec and its lease
   name the *remote* ref, so the local branch was untouched and is still there.
2. **This checkout is a blobless partial clone**
   (`remote.origin.partialclonefilter=blob:none`, `remote.origin.promisor=true`),
   so step 1's `held in this checkout` test — `git cat-file -e <tip>^{commit}` —
   proves the commit object, not the whole tree. With `GIT_NO_LAZY_FETCH=1`, 75
   blobs across the four tip trees were absent locally (32 of 984, 41 of 1171, 1
   of 1446, 1 of 1363), and after the push 74 of them can no longer be fetched
   from GitHub — the exception is the one under `data/drive-migration`, whose tip
   survives as `refs/pull/507/head`. **Every one of the 75 is a code, doc, test,
   dashboard or regression-snapshot file; not one is a data payload**
   (`.gitignore`, `AGENTS.md`, `engine/wheel_runner.py`,
   `backtests/regression/snapshots/s27_ivpit_24t_100k.json`,
   `docs/DATA_INVENTORY.md`, and so on — the full list is in the Run Summary).
   So the card's "`scripts/data_manifest.py materialize` still finds them there"
   holds for the data: all 144 manifest rows resolve to a blob present in this
   checkout right now, 0 missing, checked with lazy fetch disabled and using each
   row's real `git_path` (29 of the 144 rows carry one — `materialize` builds its
   spec as `<commit>:<git_path or path>`, `scripts/data_manifest.py` line 321).
   What the keep-refs are *not* is a complete copy of the four branches' trees;
   the complete copy is the bundle.
3. **Step 0's `files already here:` line can never be empty**, because the brace
   group's `> "$F"` redirect creates the file before the body's `ls` runs. The
   first run printed `files already here: swe-d31s6-step0.txt`, listing the file
   step 0 was itself writing. Cosmetic: the already-ran test is a separate,
   exact one (`[ -e "$TMPD/swe-d31s6-step3.txt" ]`), and it printed nothing
   correctly.

One number in this run's own working was wrong before it was corrected: an
intermediate check of mine reported 29 manifest rows whose blob was missing
locally. That was a path-mapping error — it resolved `<commit>:<path>` for every
row and ignored `git_path`, so it was testing paths that do not exist in those
commits. Redone with the spec `materialize` actually uses, the count is 0 missing
of 144.

## How we fixed it

Nothing needed fixing to make the push work. The two real defects were reported to
the Operator at G1, in front of the card's question and clearly separated from it,
so that the yes was given against the true state: that the keep-refs preserve the
data the manifest names but not the branches' full trees, and that one local
branch still holds one of the tips. The Operator answered `yes`, recorded word for
word in `swe-d31s6-g1.txt`.

The data's safety does not rest on the keep-refs in any case. Independently of
git, all 57 manifest rows whose `git_source` is one of the four branches are
present on the desktop root and verified by full sha256 against the manifest: 57
ok, 0 mismatched, 0 absent, 1,176,919,803 B. Drive holds a byte-identical second
copy of the root (card 2c and 2d, #554 and #556), and the bundle holds the
complete history on both the desktop and Drive.

## Evidence

Each step wrote one file in `%TEMP%` under `noclobber`. No step was rerun, so
each took its first name. Pasted whole.

`swe-d31s6-step0.txt`:

```
files already here: swe-d31s6-step0.txt 
From https://github.com/MertYakar66/smart-wheel-engine
   1363946..ed85e1d  main       -> origin/main
Switched to a new branch 'claude/d31-step6'
branch 'claude/d31-step6' set up to track 'origin/main'.
Report from the Executor, Sir — branch `claude/d31-step6` · HEAD `ed85e1d` · pushed nothing yet · behind/ahead of main 0/0 · run mode change
branch claude/d31-step6
changes 0
main ed85e1d9193861f25931145202d20207b962640e
== END step 0
```

`swe-d31s6-step1.txt`:

```
SWE_DATA_ROOT=C:\Users\merty\Desktop\swe-data
checked 144 manifest files: 144 ok, 0 missing, 0 mismatched (size only)
597cc6af6e2b579d667ca24dc2b324cd766a2a58	refs/heads/backup/drive-tier-c-2026-07-22
2abf850d76b3de9a1e04301d333485e64061f3cd	refs/heads/claude/daybot-bloomberg-pull
24835719ffa6d83a2c5e1dce4a7605b356695ede	refs/heads/data/drive-migration
68a48b245cea285d98f86ab7e6ba1b5cdca7002d	refs/heads/deep-history/bloomberg-raw
ed85e1d9193861f25931145202d20207b962640e	refs/heads/main
at its checked commit: deep-history/bloomberg-raw 68a48b245cea285d98f86ab7e6ba1b5cdca7002d
at its checked commit: claude/daybot-bloomberg-pull 2abf850d76b3de9a1e04301d333485e64061f3cd
at its checked commit: backup/drive-tier-c-2026-07-22 597cc6af6e2b579d667ca24dc2b324cd766a2a58
at its checked commit: data/drive-migration 24835719ffa6d83a2c5e1dce4a7605b356695ede
pull/507 head 24835719ffa6d83a2c5e1dce4a7605b356695ede
manifest git_sources: {'git:main': '69bf3b933a8adee8e8e8727ca280a9d2af99acb8', 'git:deep-history/bloomberg-raw': '68a48b245cea285d98f86ab7e6ba1b5cdca7002d', 'git:claude/daybot-bloomberg-pull': '2abf850d76b3de9a1e04301d333485e64061f3cd', 'git:backup/drive-tier-c-2026-07-22': '597cc6af6e2b579d667ca24dc2b324cd766a2a58', 'git:data/drive-migration': '24835719ffa6d83a2c5e1dce4a7605b356695ede'}
bundle: 1698024795 B, sha256 ee15dd9cbd0132fa4195947893f9a3531da50a4df69749bcbe8ab026caf8fa84
bundle heads: 9
68a48b245cea285d98f86ab7e6ba1b5cdca7002d refs/remotes/origin/deep-history/bloomberg-raw
2abf850d76b3de9a1e04301d333485e64061f3cd refs/remotes/origin/claude/daybot-bloomberg-pull
597cc6af6e2b579d667ca24dc2b324cd766a2a58 refs/remotes/origin/backup/drive-tier-c-2026-07-22
24835719ffa6d83a2c5e1dce4a7605b356695ede refs/remotes/origin/data/drive-migration
held in this checkout: 68a48b245cea285d98f86ab7e6ba1b5cdca7002d
held in this checkout: 2abf850d76b3de9a1e04301d333485e64061f3cd
held in this checkout: 597cc6af6e2b579d667ca24dc2b324cd766a2a58
held in this checkout: 24835719ffa6d83a2c5e1dce4a7605b356695ede
open pull requests' heads and bases: none
== END step 1
```

`swe-d31s6-step1.heads` (the bundle's 9 heads; its `main` is the bundle's own
vintage of 2026-09-23, not today's):

```
67b7134ad24fedda58a61518cba96a7c227696d8 refs/heads/main
67b7134ad24fedda58a61518cba96a7c227696d8 refs/remotes/origin/HEAD
597cc6af6e2b579d667ca24dc2b324cd766a2a58 refs/remotes/origin/backup/drive-tier-c-2026-07-22
2abf850d76b3de9a1e04301d333485e64061f3cd refs/remotes/origin/claude/daybot-bloomberg-pull
629237d01b76c5c26ac7ee919c3b4110b2a5e24d refs/remotes/origin/claude/project-restart-ai-agents-kot5jr
24835719ffa6d83a2c5e1dce4a7605b356695ede refs/remotes/origin/data/drive-migration
68a48b245cea285d98f86ab7e6ba1b5cdca7002d refs/remotes/origin/deep-history/bloomberg-raw
67b7134ad24fedda58a61518cba96a7c227696d8 refs/remotes/origin/main
67b7134ad24fedda58a61518cba96a7c227696d8 HEAD
```

`swe-d31s6-step2.txt`:

```
kept: refs/kept/d31-step6/deep-history/bloomberg-raw 68a48b245cea285d98f86ab7e6ba1b5cdca7002d
kept: refs/kept/d31-step6/claude/daybot-bloomberg-pull 2abf850d76b3de9a1e04301d333485e64061f3cd
kept: refs/kept/d31-step6/backup/drive-tier-c-2026-07-22 597cc6af6e2b579d667ca24dc2b324cd766a2a58
kept: refs/kept/d31-step6/data/drive-migration 24835719ffa6d83a2c5e1dce4a7605b356695ede
== END step 2
```

`swe-d31s6-g1.txt` — the Operator's answer, word for word:

```
yes
```

`swe-d31s6-step3.txt`:

```
step 1 file: C:/Users/merty/AppData/Local/Temp/swe-d31s6-step1.txt; G1 record: C:/Users/merty/AppData/Local/Temp/swe-d31s6-g1.txt; keep-refs: 4 of 4
To https://github.com/MertYakar66/smart-wheel-engine
-	:refs/heads/backup/drive-tier-c-2026-07-22	[deleted]
-	:refs/heads/claude/daybot-bloomberg-pull	[deleted]
-	:refs/heads/data/drive-migration	[deleted]
-	:refs/heads/deep-history/bloomberg-raw	[deleted]
Done
push exit 0
== END step 3
```

`swe-d31s6-step4.txt`:

```
ed85e1d9193861f25931145202d20207b962640e	refs/heads/main
gone from GitHub: deep-history/bloomberg-raw
gone from GitHub: claude/daybot-bloomberg-pull
gone from GitHub: backup/drive-tier-c-2026-07-22
gone from GitHub: data/drive-migration
main ed85e1d9193861f25931145202d20207b962640e
pull/507 head 24835719ffa6d83a2c5e1dce4a7605b356695ede
kept locally: deep-history/bloomberg-raw 68a48b245cea285d98f86ab7e6ba1b5cdca7002d
kept locally: claude/daybot-bloomberg-pull 2abf850d76b3de9a1e04301d333485e64061f3cd
kept locally: backup/drive-tier-c-2026-07-22 597cc6af6e2b579d667ca24dc2b324cd766a2a58
kept locally: data/drive-migration 24835719ffa6d83a2c5e1dce4a7605b356695ede
== END step 4
```

Beyond the card's steps, read only, to prove "every other ref is untouched" and to
settle the two defects above:

```
$ git ls-remote origin | wc -l
528
$ git ls-remote origin | awk '{print $2}' | sed 's#^\(refs/[a-z]*\)/.*#\1/*#' | sort | uniq -c
      1 HEAD
      1 refs/heads/*
    526 refs/pull/*
$ git ls-remote --tags origin | wc -l
0
$ git ls-remote origin | grep -E '^(68a48b24…|2abf850d…|597cc6af…|24835719…)'
24835719ffa6d83a2c5e1dce4a7605b356695ede	refs/pull/507/head

$ for b in <the four>; do echo "origin/$b -> $(git rev-parse -q --verify "refs/remotes/origin/$b" || echo GONE)"; done
origin/deep-history/bloomberg-raw -> GONE
origin/claude/daybot-bloomberg-pull -> GONE
origin/backup/drive-tier-c-2026-07-22 -> GONE
origin/data/drive-migration -> GONE

$ git config --get remote.origin.partialclonefilter ; git config --get remote.origin.promisor ; git --version
blob:none
true
git version 2.53.0.windows.1

$ for t in <the four tips>; do GIT_NO_LAZY_FETCH=1 git rev-list --objects --missing=print --no-walk "$t" | ... ; done
68a48b245cea285d98f86ab7e6ba1b5cdca7002d  objects_at_tip=984   missing=32
2abf850d76b3de9a1e04301d333485e64061f3cd  objects_at_tip=1171  missing=41
597cc6af6e2b579d667ca24dc2b324cd766a2a58  objects_at_tip=1446  missing=1
24835719ffa6d83a2c5e1dce4a7605b356695ede  objects_at_tip=1363  missing=1

$ git for-each-ref --format='%(objectname) %(refname)' refs/heads/ | grep <the four tips>
24835719ffa6d83a2c5e1dce4a7605b356695ede refs/heads/data/drive-migration

$ # every manifest row resolved as materialize does, <commit>:<git_path or path>
rows with an explicit git_path: 29 of 144
manifest rows resolvable from git in THIS checkout, no lazy fetch: present=144 missing=0

$ # the 57 branch-sourced manifest rows, full sha256 against the manifest, on the root
rows whose git_source is one of the four branches: 57
branch-sourced rows on the desktop root: 57 ok by full sha256, 0 mismatched, 0 absent; 1,176,919,803 B
```

## Unresolved / handoff

1. **The records that still name the four branches, for the pen.** None of these
   is CI or a test; each is a human procedure or a note that now points at a ref
   that no longer exists:
   - `docs/DATA_POLICY.md`'s first-fill step 1 and `docs/DATA_INVENTORY.md` §B,
     both of which open with `git fetch origin <the four>` — that command now
     exits non-zero. §B already carries the replacement two lines further down
     ("until a branch is deleted (step 6), after which the desktop root, Drive
     and the full-history bundle are where those bytes live");
   - `docs/DATA_INVENTORY.md`'s lines that call the branches the holders of
     Tier B, B′ and R;
   - `scripts/data_manifest.py`'s docstring and its materialize hint
     (`git fetch origin <branch>`), which `tests/test_data_manifest.py` asserts.
     That is a code change and needs its own Execution Prompt;
   - `docs/FRESH_LAB_BOX_SETUP.md`, and `scripts/pull_iv_surface.py`'s note on
     pushing slices to `deep-history/bloomberg-raw`.
2. **Where the four commits live now.** In the full-history bundle,
   `data_archive/git/smart-wheel-engine-all-refs-2026-09-23.bundle`
   (1,698,024,795 B, sha256 `ee15dd9cbd0132fa4195947893f9a3531da50a4df69749bcbe8ab026caf8fa84`),
   on the desktop and on Drive; and in this desktop's four
   `refs/kept/d31-step6/<branch>` refs, which are never pushed. On GitHub,
   `24835719…` also survives as `refs/pull/507/head`.
3. **A correction the next card should carry.** The keep-refs are a complete copy
   of the *data* the manifest names (all 144 rows resolve locally) but not of the
   four branches' full trees: 74 code, doc, test and snapshot blobs at those tips
   are now neither local nor fetchable. A fresh checkout, or a full checkout of
   those tips on this desktop, must come from the bundle. Any future card that
   repeats the card's line 44 — "the desktop then keeps the commits … and
   `materialize` still finds them there" — should say "keeps the data" rather
   than "keeps the commits", and should not repeat "no local branch holds those
   tips", which was never true for `data/drive-migration`.
4. **Still held, and not this run's:** the history purge (the data blobs in
   `main`'s pack), which stays the Operator's separate later decision; and
   `refs/pull/507/head`, which D33 point 7 keeps on GitHub until that purge.
   `refs/heads/data/drive-migration` is still on this desktop; nothing in this
   run deleted a local branch, and nothing should without the Operator's yes.
