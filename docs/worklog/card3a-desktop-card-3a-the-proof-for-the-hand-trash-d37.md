---
id: card3a
title: Desktop card 3a: the proof for the hand trash (D37)
kind: verification
status: in-flight
terminal: desktop
pr:
decisions: [D33, D37]
date: 2026-10-16
headline: The proof is made: swe-local-only, SmartWheelData and smart-wheel-engine each read "clear to trash by hand: yes" with 0 blockers; proof sha256 0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9; the root unchanged and the manifest 144/0/0.
surface: []
---

the proof is made: sha256 0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9

## Goal

Card 3a of D37 (point 2; campaign #544). Prove that every file in the three old Drive
folders, `swe-local-only`, `SmartWheelData` and `smart-wheel-engine` (the tool's area
`smart-wheel-engine-git`), has both of its copies: its desktop twin in the data root and its
twin in `swe-data/` on Drive; every folder there is accounted for (one parent, owned, holding only
such files and folders). With that proof the Operator can later trash each clear folder by
hand. The run only read Drive, wrote only the new folder `_logs\card3a\` in the root, and ran
`scripts/drive_consolidate.py` as `main` holds it after #576 (git blob `6903c9c8…`) through
`trashlist`, one command per step. The pen's check of the proof, the Operator's yes to its
sha256, and the trash itself are D37 points 3 and 4, outside this card.

## What we tried

The card's steps, in order, each in one Bash call writing its whole output to its own file in
`%TEMP%` (`swe-card3a-step<N>.txt`):

- **Step 0.** The worktree `.claude/worktrees/card3a` on the new branch `claude/card3a` from
  `origin/main` (`9cd074f`). Its lines: `branch claude/card3a`, `blob
  6903c9c82dbdea48382f9e78ce45632289777951`, `changes 0`, `SWE_DATA_ROOT` the root, `rclone
  v1.74.4`, `checked 144 manifest files: 144 ok, 0 missing, 0 mismatched (size only)`, `absent:
  C:/Users/merty/Desktop/swe-data.d33-staging`, `absent: C:/Users/merty/Desktop/swe-data/_logs/card3a`,
  `free 1042385104896`, and no `STOPPED:` line.
- **Step 1, the Drive sign-in.** `about exit 0`, `== the Drive sign-in works`.
- **The briefing and the REGISTER note** on #544, before step 2.
- **Step 2, the root before.** `== the root outside _logs: 147812 files, 18,644,236,734 B, 0 link(s)
  not followed; written C:/Users/merty/Desktop/swe-data/_logs/card3a\before.tsv`.
- **Step 3, the census of `swe-data/`.** Start `2026-10-16T10:01:21Z`, end `2026-10-16T14:48:17Z`.
- **A pause the card did not have.** While step 3 ran, the Operator wrote in chat: "Before you
  start step 4, stop and wait for my word: I will make a fresh Drive sign-in first. Then run step 1
  again under the next letter, and go on to step 4 only when it reads == the Drive sign-in works."
  The Executor asked the Operator to sign in only after step 3 had ended, because the census's
  rclone was using `rclone.conf` and, by an earlier session's note on rclone v1.74.4 (not checked
  again in this run), a token refresh rewrites that file. After step 3 ended the Operator
  signed in again and said so; **step 1b** read `about exit 0` and `== the Drive sign-in works`.
- **Step 4, the census of the six areas.** Start `2026-10-16T15:33:10Z`, end `2026-10-16T21:02:31Z`.
- **Step 5, the inventory of the root.** Start `2026-10-16T21:02:54Z`, end `2026-10-16T21:13:31Z`.
- **Step 6, the plan** (`--limit 0`). Start `2026-10-16T21:13:56Z`, end `2026-10-16T21:14:22Z`.
- **Step 7, `trashlist`.** Start `2026-10-16T21:14:52Z`, end `2026-10-16T21:17:55Z`.
- **Step 8, the end check**, then this record.

Each of steps 3 to 7 ran once, under the empty letter, and the rule for steps 3 to 7 read `== the
step is done` after each. No step ran again but step 1, whose second run (`step1b`) the Operator
asked for.

## What worked

Every step. The decisive lines, from the step files:

```
census: swe-data: 279862 objects; 147811 files, 18,644,236,641 B, and every id, equal to rclone's own listing
census: 1 areas, 279862 objects; swe-data/ is 13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b, in 1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4

census: swe-local-only: 46019 objects; 29310 files, 6,991,080,522 B, and every id, equal to rclone's own listing
census: SmartWheelData: 281630 objects; 148787 files, 16,257,882,068 B, and every id, equal to rclone's own listing
census: smart-wheel-engine-git: 64 objects; 32 files, 53,723 B, and every id, equal to rclone's own listing
census: day-bot-local-archive/vendor_swe_data: 94 objects; 73 files, 269,788,882 B, and every id, equal to rclone's own listing
census: day-bot-local-archive/vendor_swe_data_raw: 15 objects; 12 files, 2,373,076 B, and every id, equal to rclone's own listing
census: day-bot-local-archive/vendor_swe_data_processed: 3 objects; 2 files, 166,352 B, and every id, equal to rclone's own listing
census: 6 areas, 327825 objects; swe-data/ is 13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b, in 1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4

inventory: 147812 files (147811 hashed, 1 credential-shaped, not read), 18,644,236,641 B; 0 links skipped

to copy home: 0 files, 0 B
root files excluded as credential-shaped: 1
left to settle or listed: 0
byte checks pending: 0
```

The plan put every Drive file of the six areas in `redundant` (a twin in the root) and every
other object in `folder`.

**The three verdicts**, from step 7:

| Folder | listed | folder | blocker | sharing | Verdict |
| ------ | ------ | ------ | ------- | ------- | ------- |
| `swe-local-only` (`1JwPWszfyggUDT1vYaRjZ8nlHEDR3vEOn`) | 29,310 (6,991,080,522 B) | 16,710 | 0 | the owner alone | clear to trash by hand: yes |
| `SmartWheelData` (`1wCFPBf0o9PJMy2f2vy34S316XFc1Sq3e`) | 148,787 (16,257,882,068 B) | 132,844 | 0 | the owner alone | clear to trash by hand: yes |
| `smart-wheel-engine` (area `smart-wheel-engine-git`, `1dA_fq1MorvsqUWeVxqR0aAaJ9XMwEjUY`) | 32 (53,723 B) | 33 | 0 | the owner alone | clear to trash by hand: yes |

All three are in `1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4`. Each area's `kinds:` line reads files
without an MD5 0, Google-format files 0 and shortcuts 0. The blockers' count: `== blockers: 0 in
all`. No `WARNING:` line was printed.

**The proof's sha256** is the same in all three places step 7 prints it, and step 8 hashed the
bytes again:

```
trashlist: sha256 0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9
== summary sha256: 0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9
== step done: C:/Users/merty/Desktop/swe-data/_logs/card3a/list, proof sha256 0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9
```

**The end state**, step 8:

```
checked 144 manifest files: 144 ok, 0 missing, 0 mismatched
== the root outside _logs: then 147812 files; now 147812; gone 0, added 0, size, time or file id changed 0; unreadable 0
== the proof: C:/Users/merty/Desktop/swe-data/_logs/card3a/list/trashlist.json, sha256 0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9; its summary names 0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9
== the end state: no finding
```

## What didn't

Nothing failed. No step printed an `ERROR:` or `STOPPED:` line or a traceback, no census differed
from rclone's listing, no step ran again for a failure, and no BLOCKED, WAITING or STOPPED note was
posted.

What differed from the card:

- **The pause before step 4 and the second sign-in**, at the Operator's word (above). It cost the
  time between step 3's end (`14:48:17Z`) and step 4's start (`15:33:10Z`).
- **The times.** From each step file's own start and end lines: step 3 took 4 h 47 min (the card
  said about 4½ h), step 4 5 h 29 min (about 5 h), step 5 10 min 37 s, step 6 26 s, and step 7
  3 min 3 s (the card said about 15 min).

## How we fixed it

Nothing needed fixing. What ran is the card's sequence, from the worktree, with the tool's blob
checked before each command:

```
python scripts/drive_consolidate.py census --remote gdrive: --preset swe-data --root "$ROOT" --out "$W/swe.json"
python scripts/drive_consolidate.py census --remote gdrive: --preset card2b --root "$ROOT" --out "$W/census6.json"
python scripts/drive_consolidate.py inventory --root "$ROOT" --out "$W/root.json"
python scripts/drive_consolidate.py plan --census "$W/census6.json" --inventory "$W/root.json" --out "$W/plan.json" --limit 0
python scripts/drive_consolidate.py trashlist --plan "$W/plan.json" --census "$W/census6.json" --swe-census "$W/swe.json" --root "$ROOT" --out "$W/list"
```

`$ROOT` is `C:/Users/merty/Desktop/swe-data` and `$W` is `$ROOT/_logs/card3a`. Step 7 piped
`trashlist`'s output through `sed`, so any `sharing:` line other than `the owner alone` would
reach its file only as "not the owner alone, or not known"; all three read `the owner alone`.

## Evidence

**Where the record lives:** outside git, in the data root's `_logs\card3a\`. Each output, by
sha256, from step 9's file:

| Output | sha256 |
| ------ | ------ |
| `swe.json` (step 3) | `cac53a6428643b613025909e26f2c5313ceb33b128037220b0b0fe8b121744ca` |
| `census6.json` (step 4) | `67d421b0eb2cfbe4e52bf79785b3b27946ba981448e6bc32ec563305d2553213` |
| `root.json` (step 5) | `f4fd593891c5baa70f26730b522dfa4e0ed2aa4cf9b832fd06cd20286e28fcce` |
| `plan.json` (step 6) | `b848848b6d45a8f8d4c129a76c4fa73a417b753d5758ca07ad0b7e8ba8432d36` |
| `list/trashlist.json` (step 7, **the proof**) | `0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9` |
| `list/trashlist.csv` | `6e5afcc4d2588e9eef75218bde3eb128c73dc0070dd7d323301242fa432836dd` |
| `list/kept.csv` | `70e050912f0d5d5428c3b2e20a6cac341c5f143a0b53326eafbfd20f8230baa9` |
| `list/sharing.json` | `33341bd4e21e9790c0427a8135d1b7a76f760015356b1c2f40882b0baa419696` |
| `list/trashlist_summary.json` | `a6bdc5c7de1d47d6a8302c2d76e6b93f4d08d1ad49e011ebeb620751e3b6ca3c` |

The folder also holds step 2's listing `before.tsv`, the plan's ledger `plan_ledger.csv`, and the
pointers `done-3.txt` to `done-7.txt`. None of these files is posted to GitHub; they stay on the
desktop for the pen's check.

**The step files**, every `swe-card3a-step*.txt` in `%TEMP%` (`step1b` among them), are posted
whole in the Run Summary's comments on this card's pull request.

**The notes posted on #544:**

| Note | Link |
| ---- | ---- |
| REGISTER | [#544 comment 6213016954](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-6213016954) |

The report on #544 follows in step 11, after this record is pushed.

## Unresolved / handoff

1. **D37 point 3.** A context that did not write this card checks the proof and its four companion
   files on the desktop, then brings the sha256
   `0ae2d62dcff86179135441f05b997f6a4700f99c88996c8b1d0051e8b29a07d9` to the Operator for a yes.
   The trash, the recount and the rehearsal come after that (D37 point 4).
2. **The writer pause stays** (D37 point 5) until the check after the trash.
3. **For the pen, reading the numbers** (each computed from the step files):
   - `swe-data/`'s census reads 279,862 objects, 147,811 of them files, so 132,051 others. The card
     quotes card 2c's count of `swe-data/`'s folders as 132,053, two more. The census matched
     rclone's own listing id for id, so this is what Drive holds now; why the two counts differ
     was not looked into.
   - Step 2 totals the root at 18,644,236,734 B and step 5 at 18,644,236,641 B, 93 B less. Step 5
     lists the one credential-shaped file by size only and never opens it, and its total
     presumably leaves that file out; this was not checked.
   - `trashlist`'s `folder` count is one above the plan's for each area (16,710 against 16,709,
     132,844 against 132,843, 33 against 32): `trashlist` also classes the area folder itself.
   - The three folders list 178,129 files, while the root holds 147,812. A `redundant` file is one
     with the same size, MD5 and SHA-256 as a root file, so Drive's duplicates share a twin, and
     `trashlist` hashes each desktop twin once (the tool's docstring and its `_Twins` class).
4. **The sign-in.** The Operator's second sign-in of 2026-10-16 ends 7 days after it was made
   (Testing mode). The recount on the day of the trash needs a working sign-in of its own.
