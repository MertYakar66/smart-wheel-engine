---
id: d33-card2c
title: Desktop card 2c: swe-data on Drive, checked both ways, and the restore tests (D33)
kind: verification
status: in-flight
terminal: desktop
pr:
decisions: [D33]
date: 2026-10-07
headline: swe-data/ on Drive holds all 147,811 baseline files and 147,810 of them match the root by MD5, but a duplicate Drive folder left one file uncheckable, so the check stopped for the pen and the restore tests did not run.
surface: []
---

stopped at step 3: STOPPED: swe-data/ (or the root) holds two objects of one name (2026/10/06 22:56:11 NOTICE: data_archive/drive-legacy/SmartWheelData/data_processed/theta/option_history/ticker=FDX/expiration=20201113: Duplicate directory found in destination - ignoring); 1 such line(s) in check/a1/06-check.txt: the pen decides

## Goal

D33 point 3 makes the desktop root the main copy of the data and Drive the second: one
folder, `swe-data/`, at the top of My Drive, laid out like the root, holding every file in
the root except the credential-shaped names and the root's own run logs (`_logs/`). Two
checks were to prove it complete: rclone's two-way checksum check, and a checksum list kept
in both places (`SHA256SUMS`), whose own hash is recorded in git. The card then had to show
that the copy restores: the manifest's 144 engine files pulled back from `swe-data/` into an
empty folder and checked, and card 1's full-history bundle pulled back and fetched into an
empty repository (D33 point 7 and D31 step 6, the precondition for ever deleting the four
data branches).

Cards 1 to 2b only ever read Drive. This card was the first to write to it.

## What we tried

One desktop run of the driver's `run` mode, four steps, each skipped if already DONE:

1. **plan** — check the Drive sign-in and My Drive's top; write `SHA256SUMS`; take the
   root's inventory as the baseline; prove that rclone, given the tool's exclusions, sees
   exactly the baseline's files and that `SHA256SUMS` matches the inventory; pin it all.
2. **upload** — make `swe-data/` once at the top of My Drive, pin its id, then `rclone copy`
   the root into it by that id, with `--immutable` so no file there is ever written over and
   rclone's own post-copy check (and with it rclone's one deletion) turned off.
3. **check** — `rclone check` of the root against `swe-data/`, both ways, by checksum; then
   `rclone size` of `swe-data/` with no filter at all, to catch anything extra or twice.
4. **restore** — the manifest's 144 files and the bundle, pulled back from `swe-data/` into
   new folders outside the root, and verified.

Steps 1 and 2 passed. Step 3 stopped the run. Step 4 never started.

## What worked

**The plan, in full.** The baseline came to **147,811 files, 18,644,236,641 B**, inside the
147,000–149,000 and 18.4–18.9 GB the Operator approved. The tool's inventory read 147,812
files (147,811 hashed, 1 credential-shaped and not read), 18,644,236,641 B, 0 links skipped.
rclone, given the tool's exclusions, saw **exactly those 147,811 files** — no more, no fewer.
`SHA256SUMS` was written into the root before the baseline was taken and lists 147,810 files
(the baseline but itself), 18,617,103,830 B, with its own sha256
`e22d8579bed1ce7a81a90ccb1096d0397a29de8471f8b56f466f05f13f882989`. That sha256 is the
fingerprint D33 asks to be recorded in git, and this fragment is where it is recorded. The
one credential-shaped file in the root, `data_processed/ibkr/flex_credentials.json`, was
excluded by name and never opened, never uploaded.

**The upload, in full.** `swe-data/` was made once at the top of My Drive (`rclone mkdir`),
its id pinned in `_logs\d33-card2c\swe-data.id`, and every later Drive call reached the
folder by that id alone. The copy ran to `exit 0` and the driver's own line reads:
**147,811 files uploaded in this attempt; 147,811 in all attempts** — exactly the baseline,
so no file was sent more than once, and no folder had to be re-sent. The upload log holds
**0 errors** and **0 "Removing failed copy" lines**: `--immutable` never tripped on a file
with other bytes, and nothing was deleted on Drive. The driver logged **0 retries** across
the whole run.

**The comparison itself, as far as it reached.** `rclone check` compared by **MD5** (it named
its checksum: `Using md5 for hash comparisons`) and found **147,810 matching files** of the
baseline's 147,811.

**The root is untouched.** The end check compared the root with the baseline by name, size
and bytes: then 147,812 files, now 147,812; **gone 0, size changed 0, bytes changed 0 of
147,811 hashed, added 0**. The manifest check by full hash reads **144 ok, 0 missing, 0
mismatched**. The end check's verdict is `== endstate: no finding`.

## What didn't

**One file could not be checked, and that stopped the run.** Drive holds **two folders named
`expiration=20201113`** under
`data_archive/drive-legacy/SmartWheelData/data_processed/theta/option_history/ticker=FDX/`.
rclone reported the duplicate and ignored one of the two, and the single `data.parquet`
inside the ignored folder then could not be resolved:

```
2026/10/06 22:56:11 NOTICE: data_archive/.../ticker=FDX/expiration=20201113: Duplicate directory found in destination - ignoring
2026/10/07 00:26:23 ERROR : data_archive/.../ticker=FDX/expiration=20201113/data.parquet: file not in Google drive root ''
2026/10/07 02:21:55 NOTICE: Google drive root '': 1 files missing
2026/10/07 02:21:55 NOTICE: Google drive root '': 1 differences found
2026/10/07 02:21:55 NOTICE: Google drive root '': 1 errors while checking
2026/10/07 02:21:55 NOTICE: Google drive root '': 147810 matching files
```

**Why it happened.** This is the hazard the card names under "rclone on Drive": when a reply
from Drive is lost, rclone retries a create that had in fact already worked, and Drive ends
up holding two objects of one name. It is a Drive-side artefact of the upload, not a defect
in the data: the desktop root holds that `data.parquet` intact, `SHA256SUMS` carries its
hash, and the end check confirms the root unchanged. What is wrong is that Drive has one
redundant *container* too many, and rclone cannot see past it.

**What the stop cost.** The check's second half never ran: `rclone size` of `swe-data/` with
no filter (`07-size.txt`) was not reached, so **nothing extra or twice on Drive has been
ruled out by count and bytes**. The restore tests never ran, so **neither restore has been
demonstrated from `swe-data/`** — which means D31 step 6 is *not* satisfied and the four data
branches must not be deleted on the strength of this run.

**Why nothing was done about it.** Clearing a duplicate needs a *deletion on Drive*. This
card forbids every deletion anywhere, and says so explicitly: clearing this "needs a removal,
which only a later card can make, with the Operator's yes". So the run stopped, reported, and
left Drive exactly as it found it.

## How we fixed it

Nothing was fixed, and nothing should have been: the card's own rule is that this stop is the
pen's to resolve. What the run did instead was stop cleanly and record completely —
`swe-data/` is left in place with all 147,811 files in it, the duplicate folder untouched,
the root unchanged, and every number above readable from a file under `_logs\d33-card2c\`.

The shape of a fix, for whoever picks it up, is a one-file problem and not a re-upload: the
redundant Drive folder is removed (with the Operator's yes, in a card that is allowed to
delete), and then the check and the restore tests are re-run from where they stopped. The
upload itself need not be repeated — a rerun skips every file `swe-data/` already holds with
the same bytes.

## Evidence

**What ran, in order.** The Drive sign-in was checked before each of the driver's steps and
answered `exit 0` every time (`plan/a1/00-signin.txt`, `upload/a1/00-signin.txt`,
`check/a1/00-signin.txt`). My Drive's top was read three times per step by the tool's two
areas and by name: `0 object(s) named swe-data` before the plan and before the upload,
`1 object(s) named swe-data` before the check. Then `SHA256SUMS` and the inventory; rclone's
own view of the root; `swe-data/` made once by name and pinned by id; the upload; the check,
both ways; and no further step.

**The rclone commands that made and checked the copy** (`$ROOT` and `$WORK` as the step file
prints them; `<the folder's Drive id>` stands for the pinned id of `swe-data/`, which the
step files carry in full):

```
rclone mkdir --drive-root-folder-id <My Drive's id> gdrive:swe-data

rclone copy $ROOT gdrive: --drive-root-folder-id <the folder's Drive id> --immutable \
  --checksum --ignore-checksum --ignore-size --no-update-dir-modtime --drive-use-trash=true \
  --exclude-from $WORK\plan\a1\filters.txt --ignore-case -v --stats 10m --stats-one-line \
  --stats-log-level NOTICE

rclone check $ROOT gdrive: --drive-root-folder-id <the folder's Drive id> --checksum \
  --exclude-from $WORK\plan\a1\filters.txt --ignore-case --disable ListR \
  --combined $WORK\check\a1\combined.txt --low-level-retries 100 -v --stats 10m \
  --stats-one-line --stats-log-level NOTICE
```

`rclone size` of `swe-data/` with no filter was the next command and did not run. The pen
builds the refresh routine for `docs/DATA_INVENTORY.md` §C.3 from the three above.

**The timings**, from the logs under `$WORK`:

| What | Log | Time |
| ---- | --- | ---- |
| `sums` (wrote `SHA256SUMS`) | `plan/a1/02-sums.txt` | exit 0 (10.6 min) |
| `inventory` (the baseline) | `plan/a1/03-inventory.txt` | exit 0 (2.2 min) |
| `rclone mkdir` | — | exit 0 (0.0 min) |
| `rclone copy` (the upload) | `upload/a1/05-upload.txt` | exit 0 (3166.9 min) |
| `rclone check` (both ways) | `check/a1/06-check.txt` | exit 1 (220.2 min) |
| `rclone size` | `check/a1/07-size.txt` | did not run |
| the restore tests | `restore/a1/08-files.txt`, `10-bundle.txt` | did not run |

The upload took **52.8 hours** (3,166.9 min), against the card's estimate of about a day and
a half. Step 0 counted **132,053 folders** below the root, and rclone makes each folder on
Drive as it makes a file, so the copy created about **279,864 Drive objects in 190,014 s —
about 1.5 a second**, against the card's model of 2. That shortfall is the whole of the
overrun. The check took **3.7 hours** against an estimated 7, because it stopped before
`rclone size`.
The upload log holds 148,128 per-file and progress lines. The driver logged **0 `RETRY:`
lines** in the whole run.

**The end state** (step 4, `endstate`):

```
== the root now, against the baseline (2026-10-04T14:53:49+00:00): names, sizes and bytes
  then 147812 files; now 147812; gone 0, size changed 0, bytes changed 0 of 147811 hashed (credential-shaped names by size only), added 0
== uploaded by this card, in all attempts: 147811 files
== the steps: plan DONE; upload DONE; check not done; restore not done
== no $SCR folder: no restore test ran
== endstate: no finding
```

and the manifest, by full hash: `checked 144 manifest files: 144 ok, 0 missing, 0 mismatched`.
There were **no findings**: the root was not changed by this run beyond `SHA256SUMS`, which
the plan wrote before the baseline was taken.

**Where the record lives.** Under `_logs\d33-card2c\` in the data root, which nothing
uploads: the driver's own run log, the step folders `plan\a1\`, `upload\a1\` and `check\a1\`
with their pins and logs, and `swe-data.id`. The attempts' fingerprint files, by sha256:

| Pin | sha256 |
| --- | ------ |
| `plan/a1/FINGERPRINTS.txt` | `dc95fefde96677c696077dfd4333f56480d3542b9b5548e1bfaa1f378d2871c8` |
| `upload/a1/FINGERPRINTS.txt` | `e0508324a248bec920e36a9a30592f32679a27a4e13f24f7934195d39ff1f136` |
| `check/a1/FINGERPRINTS.txt` | `24189f023a0b4280ce26f293e2971e2addf19ab954d870c1470242f98f29a277` |
| `restore/a1/FINGERPRINTS.txt` | absent: the step never ran |

`check/a1/FINGERPRINTS.txt` was written by the end check, not by the check step, because the
check step stopped before it could pin its own. There is no `$SCR` folder, so the restore
tests left no copies anywhere.

**The notes posted on #544:**

- REGISTER — https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-5981169086
- STOPPED — https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-6027270841

## Unresolved / handoff

1. **The duplicate Drive folder is the pen's decision.** Drive holds two folders named
   `expiration=20201113` under
   `data_archive/drive-legacy/SmartWheelData/data_processed/theta/option_history/ticker=FDX/`.
   Removing one needs the Operator's yes in a card that may delete on Drive. Until then
   `rclone check` cannot read the one `data.parquet` behind it.
2. **`swe-data/` is not yet proven complete.** 147,810 of 147,811 files match the root by
   MD5, but `rclone size` with no filter never ran, so nothing extra or duplicated on Drive
   has been ruled out by count and bytes. The duplicate folder is itself one known extra
   object.
3. **Neither restore test has run.** D31 step 6 is therefore **not** satisfied: the four data
   branches (`deep-history/bloomberg-raw`, `claude/daybot-bloomberg-pull`,
   `backup/drive-tier-c-2026-07-22`, `data/drive-migration`) must not be deleted on the
   strength of this run.
4. **`docs/DATA_INVENTORY.md` §C.3 is still to be written**, by the pen at the close, from
   this fragment: the folder, its id (in the step files), `SHA256SUMS`' own sha256, and the
   three rclone commands above.
5. **Keeping Drive current after future data changes** was out of scope here; the pen records
   the routine at the close.
6. The checkout still carries `stash@{0}`, the stray paste set aside by card 2a-i. Left as it
   is, as every card since has left it.
