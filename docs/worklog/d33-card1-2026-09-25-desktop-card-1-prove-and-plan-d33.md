---
id: d33-card1-2026-09-25
title: "Desktop card 1: prove and plan (D33)"
kind: verification
status: merged
terminal: desktop
pr: 538
decisions: [D33]
date: 2026-09-25
headline: morning-pull task exported and retired; tool 190 passed/12 skipped on Windows; main merged into round 3 removing only the 87 checkout copies (87/87 byte-identical to the root's manifest, 0 gone and 0 changed among 235 untracked files); 20 staging CSVs (7,152,880 B) swept home; the 1.7 GB bundle restores with fsck clean and all four tips; Drive census 7 areas / 332,772 objects all equal to rclone size, plan bytechecked with 120,139 files (10,716,801,880 B) to copy home; root 144/0/0 at the start and at the end; nothing on Drive changed and nothing was deleted from the root or Drive
surface: [docs/worklog/INDEX.md, CHANGELOG.md]
---

## Goal

Prepare D33's consolidation on the desktop, proving the two things the card names:
**nothing is deleted** from the root or from Drive, and **nothing on Drive changes**.
The merge of `main` into `claude/data-home-desktop-round3` removes only the 87
checkout copies of data files the root already holds — so the run had to prove
those 87 are home *before* merging, not after.

Round 3 (2026-09-23) had said explicitly not to merge `main`, because merging
deletes those 87 files from the checkout. This card lifts that hold with a guard.

## What we tried

In the order the card set, each block's real output saved under
`C:\Users\merty\Desktop\swe-data\_logs\d33-card1\`:

0. Pin `main`, read `main`'s v4 rule-books out of git, run preflight.
1. Export and retire the "SWE IBKR Morning Pull" scheduled task **first**, because
   a successful run would overwrite the root's `portfolio_snapshot.json`.
2. Test `scripts/drive_consolidate.py` from `main` in `%TEMP%`, outside both the
   checkout and the root.
3. Guarded merge: prove the 87 by bytes, record all 235 untracked/ignored files,
   merge with `--no-overwrite-ignore`, resolve the one `INDEX.md` conflict by
   regenerating it, re-check, push plainly.
4. Four `sweep` runs to bring local stray data home by bytes.
5. Restore the 1.7 GB history bundle into an empty bare repo and read data out of it.
6. Drive `census` (read-only) → root `inventory` → `plan` → space check →
   `bytecheck` → full saved summary → manifest re-check.

## What worked

Everything, on its first run, with one intentional non-zero exit (step 3's merge
conflict, which the card predicted).

- **The task retired cleanly.** Exported 3,278 B / 1,592 characters of UTF-16 with
  one `<Task` line, logon type `InteractiveToken`, and **0 password/token/secret
  matches outside it**. It was `Ready`, not `Running`; after `Unregister-ScheduledTask`
  the task count is **0**.
- **The guard held.** All **87 of 87** checkout copies `main` deletes were already
  byte-identical to the root's manifest copies — **0 absent, 0 differ** — *before*
  the merge. Of the 13 added paths, none was already present.
- **`--no-overwrite-ignore` clobbered nothing.** 235 untracked or ignored files
  before the merge; **0 gone, 0 changed** after.
- **The tool in the merged checkout is the one that was tested** (`tool files as
  tested: 0` differences).
- **The bundle is real history, not a pointer.** sha256 matched, `fsck exit 0`,
  `config` count **0** (it owes nothing to this clone or its remote), no
  `alternates` file, and all four data-branch tips resolve `ok`.
- **Drive was only read.** Seven areas, 332,772 objects, every area's totals equal
  to `rclone size`; `swe-data exists: False`.
- **The root is unchanged where it matters**: 144 ok / 0 missing / 0 mismatched at
  step 0, at step 3 and again at step 6.

## What didn't

Nothing failed, but three results are worth recording because they contradict what
round 3 expected:

- **swe-ops held no new data.** The card expected "~7 older June copies" to come
  home from `C:\Users\merty\swe-ops`. All **61** data files there (49 + 11 + 1) were
  **already in the root by bytes**, so sweeps 1–3 copied **0 files**. Reported, not
  treated as a stop, per the card.
- **Nothing needed a byte check.** The plan classed **0** rows `needs-byte-check`:
  every Drive object listed a size and a usable MD5. So the space check passed
  trivially (largest 0 B against 1,195,674,525,696 B free on C:), `bytecheck`
  downloaded nothing, and `%TEMP%\swe-bytecheck-card1\` was never left behind. The
  plan is still marked `bytechecked` (`2026-09-25T20:11:30+00:00`).
- **The census is slow, not stuck.** It enumerates 332,772 Drive objects and prints
  one line only as each area finishes. The first area landed within minutes and the
  remaining six took far longer; that shape is normal, not a hang.

## How we fixed it

Nothing needed fixing. The single conflict — `docs/worklog/INDEX.md`, the magnet
file — was resolved the way OPERATING_MODEL.md §6 requires: **regenerated** from
the union of both sides' fragments (248 lines), never `--theirs`. The merge
commit is `e13220f`, pushed as `936ce5a..e13220f`.

## Evidence

Every figure above comes from a file in `C:\Users\merty\Desktop\swe-data\_logs\d33-card1\`.

| file | bytes | what it holds |
| --- | ---: | --- |
| `main.sha` | 41 | `9851f4b2fc94e16448baa3ba00de6cf8c1bf7ee6` |
| `about.json` | 119 | `rclone about gdrive:` |
| `step0.txt` | 726 | preflight |
| `step1_list.txt` | 238 | the task before it was retired |
| `ibkr_task_export.xml` | 3,278 | the task's definition, UTF-16 |
| `step1_export.txt` | 141 | export and secret check |
| `step1_unregister.txt` | 198 | the unregister |
| `step2_tests.txt` | 3,862 | the tool's test run |
| `added.txt` | 661 | the 13 paths the merge adds |
| `deleted.txt` | 4,041 | the 87 paths the merge deletes |
| `manifest_main.json` | 38,636 | `main`'s manifest |
| `others_before.json` | 16,675 | the 235 untracked/ignored files |
| `step3_checks.txt` | 263 | pre-merge checks |
| `step3_merge.txt` | 190 | the merge |
| `step3_resolve.txt` | 321 | INDEX.md regenerated, merge committed |
| `step3_after.txt` | 528 | after-merge checks |
| `step3_push.txt` | 168 | the push |
| `root_before_sweep_1..4.json` | 8,771,909 each | the root before each sweep |
| `step4_sweep_1..4.txt` | 775 / 778 / 782 / 1,926 | the four sweeps |
| `step5_bundle.txt` | 266 | the bundle restore |
| `step6_census.txt` | 785 | the Drive census |
| `census.json` | 149,785,395 | the census itself |
| `step6_inventory.txt` | 571 | the root inventory |
| `root.json` | 8,776,473 | the inventory itself |
| `step6_plan.txt` | 1,406 | the plan |
| `plan.json` | 311,257,331 | the plan itself, marked `bytechecked` |
| `plan_ledger.csv` | 131,740,434 | the plan's ledger |
| `step6_space.txt` | 104 | the space check |
| `step6_bytecheck.txt` | 1,333 | the byte check |
| `step6_summary.txt` | 1,512 | the full plan summary + manifest re-check |

**Step 0 — preflight.** Python 3.12.10; branch `claude/data-home-desktop-round3` at
`936ce5a`, working tree clean; `main in HEAD: 1` (not yet merged); the tool matched
`main`'s copy; `SWE_DATA_ROOT` = `C:\Users\merty\Desktop\swe-data`; 144 ok / 0 missing /
0 mismatched; no staging folder; the Desktop is `C:\Users\merty\Desktop`, not redirected
into OneDrive; rclone v1.74.4 with a `gdrive:` remote; Drive free 5,448,959,487,271 B.

**Step 1 — the morning pull is retired.**

```
TaskName       : SWE IBKR Morning Pull
TaskPath       : \
State          : Ready
LastRunTime    : 2026-09-25 11:50:32 AM
LastTaskResult : 2147946720
portfolio_snapshot.json 5279 B, modified 2026-07-18 01:42
```

`2147946720` is `0x800710E0` — the failure round 3 recorded, so no run of this task
ever wrote the root's snapshot. After the unregister: `state now: Ready`,
`unregister exit 0`, task count **0**. The snapshot file is untouched at 5,279 B.

To restore the task if it is ever needed, from Git Bash:

```bash
powershell.exe -NoProfile -Command "Register-ScheduledTask -TaskName 'SWE IBKR Morning Pull' -TaskPath '\' -Xml (Get-Content -Raw -Encoding Unicode 'C:\Users\merty\Desktop\swe-data\_logs\d33-card1\ibkr_task_export.xml')"
```

**Step 2 — the tool passes on Windows.** `190 passed, 12 skipped in 118.33s`,
`pytest exit 0`, in `%TEMP%\swe-tooltest-card1`. Every skip is the allowed kind:
nine are `links are not available here` (Developer Mode off — `WinError 1314`, plus one
`WinError 2`), one is Windows resolving `..` by name as the tool does, and two are
`native()` already spelling paths with `\\?\`.

**Step 3 — the guarded merge.** Base `67b7134`; prefetch exit 0; the dry merge showed
exactly one conflict, `docs/worklog/INDEX.md`; 13 added paths, none already present;
87 deleted paths, of which **87 match the manifest, 0 absent, 0 differ**; 235 untracked
or ignored files recorded. After the merge: 0 gone, 0 changed; `tool files as tested: 0`;
`worklog-index: OK`; 144 ok / 0 missing / 0 mismatched; working structure 6 of 6.
Merge commit `e13220f`, pushed `936ce5a..e13220f` with a plain push — this branch is
never rebased or force-pushed, because it carries a merge commit.

**Step 4 — the last local strays.** Each sweep re-inventories the root first, so nothing
is brought home twice. All four exit 0 and end `staging folder gone`.

- swe-ops `data`: 49 data files, **49 already in the root by bytes**, 0 to copy, 13 code or log files skipped.
- swe-ops `data_raw`: 11 data files, **11 already in the root**, 0 to copy.
- swe-ops `data_processed`: 1 data file, **1 already in the root**, 0 to copy, 1 code or log file skipped.
- the checkout's `staging\`: 20 data files, 0 already in the root, **20 to copy (7,152,880 B)**, 25 code or log files
  skipped — 15 `blue_chips` CSVs, 4 `casy` CSVs and `fundamentals_pit/sp500_fundamentals_pit.csv` (4,926,159 B).
  `sweep: copied and verified 20 file(s)`.

No `PASSED` line appeared in any sweep: no credential-shaped file sits in those folders.

**Step 5 — the bundle restores.**
`swe-data\data_archive\git\smart-wheel-engine-all-refs-2026-09-23.bundle` (1,698,024,795 B)
hashes to `EE15DD9CBD0132FA4195947893F9A3531DA50A4DF69749BCBE8AB026CAF8FA84`, fetched into
an empty bare repo at `%TEMP%\swe-bundle-card1` (1.6 GB on disk). `fsck exit 0`; the
`remote.|promisor|partialclone` config count is **0**; no `objects/info/alternates`; and
all four data-branch tips resolve:

```
ok 68a48b245cea285d98f86ab7e6ba1b5cdca7002d
ok 2abf850d76b3de9a1e04301d333485e64061f3cd
ok 597cc6af6e2b579d667ca24dc2b324cd766a2a58
ok 24835719ffa6d83a2c5e1dce4a7605b356695ede
```

**Step 6 — census, inventory, plan, byte check.** Drive was only read.

```
census: swe-local-only: 46019 objects; 29310 files, 6,991,080,522 B, equal to rclone size
census: SmartWheelData: 281631 objects; 148788 files, 16,257,882,742 B, equal to rclone size
census: smart-wheel-engine-git: 64 objects; 32 files, 53,723 B, equal to rclone size
census: day-bot-local-archive/vendor_swe_data: 94 objects; 73 files, 269,788,882 B, equal to rclone size
census: day-bot-local-archive/vendor_swe_data_raw: 15 objects; 12 files, 2,373,076 B, equal to rclone size
census: day-bot-local-archive/vendor_swe_data_processed: 3 objects; 2 files, 166,352 B, equal to rclone size
census: day-bot-local-archive/data_raw: 4946 objects; 3306 files, 35,125,927 B, equal to rclone size
census: 7 areas, 332772 objects; swe-data exists: False
```

The root inventory: 29,397 files (29,396 hashed, **1 credential-shaped, not read**),
7,935,118,331 B, 0 links skipped. The space check: `0 to byte-check; 0 B listed; largest
0 B; 0 with no listed size`, against `C: free 1195674525696`. `bytecheck exit 0`, no
`CHANGED` line, and the plan carries `bytechecked: 2026-09-25T20:11:30+00:00`.

The plan the pen will review:

```
swe-local-only [consolidate]: 46019 objects: copy 2 (0 B), folder 16709 (0 B), redundant 29308 (6,991,080,522 B)
SmartWheelData [consolidate]: 281631 objects: copy 118367 (10,681,538,612 B), duplicate 700 (92,979,972 B), folder 132843 (0 B), redundant 29720 (5,483,363,484 B), unresolved 1 (674 B)
smart-wheel-engine-git [consolidate]: 64 objects: copy 32 (53,723 B), folder 32 (0 B)
day-bot-local-archive/vendor_swe_data [read-only]: 94 objects: copy 13 (393,164 B), duplicate 16 (211,371,992 B), folder 21 (0 B), redundant 44 (58,023,726 B)
day-bot-local-archive/vendor_swe_data_raw [read-only]: 15 objects: duplicate 1 (2 B), folder 3 (0 B), redundant 11 (2,373,074 B)
day-bot-local-archive/vendor_swe_data_processed [read-only]: 3 objects: duplicate 1 (2 B), folder 1 (0 B), redundant 1 (166,350 B)
day-bot-local-archive/data_raw [read-only]: 4946 objects: copy 1725 (34,816,381 B), duplicate 1581 (309,546 B), folder 1640 (0 B)
to copy home: 120139 files, 10,716,801,880 B
swe-data will hold about 149535 files, 18,651,920,211 B
Drive free: 5,448,959,487,271 B
Drive margin after swe-data: 5,430,307,567,060 B
root files excluded as credential-shaped: 1
  excluded  data_processed/ibkr/flex_credentials.json
left to settle or listed: 1
  unresolved        SmartWheelData/data_processed/ibkr/flex_credentials.json  credential-shaped name: never read or copied
```

The one unresolved row is the credential-shaped file, named but never opened, on both
sides. No key, token or password value appeared in any output of this run.

The manifest re-check at the end of step 6: **144 ok, 0 missing, 0 mismatched**.

## Unresolved / handoff

- **The plan is written and byte-checked; no object has been copied.** Card 2 reviews it;
  card 3 acts on it. `plan.json` is 311 MB and `plan_ledger.csv` 131 MB — read them with a
  script, not by hand.
- **Never re-run `plan` against this `census.json`/`root.json` pair**: it would replace the
  checked plan and lose the `bytechecked` stamp.
- **Left in `%TEMP%` on purpose**, and safe to delete once card 3 is done:
  `swe-mark-card1.py`, `swe-tooltest-card1\`, `swe-bundle-card1\` (1.6 GB).
  `swe-bytecheck-card1\` does not exist: nothing needed downloading, so the tool removed
  its own empty temporary folder.
- **The stray `ibkr$p` was not touched**, as the card required; it waits for card 3.
- **The morning pull no longer exists.** Anything that assumed a 07:30 ET snapshot refresh
  must be re-pointed, or the task restored with the command above.
- **`main` may have moved.** This branch carries a merge commit, so it is pushed plainly —
  never rebased, never force-pushed.

## Corrections at the close (the pen, 2026-09-26)

The pen's review of this record found these points wrong or too strong. The text
above is kept as the Executor wrote it; these lines correct it.

- **The cards.** Card 2 copies home, builds `swe-data/` on Drive and proves it, with
  nothing deleted. Card 3 cleans up (D33). The record above has card 2 as the plan's
  review and card 3 running `copy`.
- **The merge's guard** was the card's `PRESENT` check: none of the 13 paths `main`
  adds existed, so git had nothing to overwrite. Git ignores `--no-overwrite-ignore`
  on a true merge like `e13220f`, so that flag protected nothing here.
- **"No `PASSED` line"** shows that no credential-shaped *data* file was found.
  `sweep` skips code and log files (39 here) before its credential check, so their
  names were not examined.
- **The snapshot.** Its size and modified time were the same before and after
  (5,279 B, 2026-07-18 01:42), so no run rewrote it after 2026-09-23. "Byte for
  byte" and "ever" go further than that evidence.
- **The byte check** had nothing to settle (0 `needs-byte-check` rows) and made no
  Drive query. Its stamp means only that it ran. Card 2's `copy` takes a fresh census
  before any download.
- **"235 untracked files"** means untracked or ignored files outside `.claude/`,
  `dashboard/` and Python caches. "0 changed" means the same size and modified time.
- **swe-ops** held no new data. That confirms round 3's finding; it does not
  contradict it.
- **The census found more than the record says:**
  - `SmartWheelData/data_processed/theta` holds a Theta upload far fuller than
    `swe-local-only`'s. Card 1b settles its size from the plan.
  - `SmartWheelData/data_processed/ibkr` holds a credential-shaped
    `flex_credentials.json`, never opened.
  - Both are with the Operator (PROJECT_STATE.md §0 B).
