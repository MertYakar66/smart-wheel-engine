---
id: d33-card2a-iii
title: Desktop card 2a-iii: the last three chunks of card 2a-ii (D33, D34)
kind: verification
status: in-flight
terminal: desktop
pr:
decisions: [D33, D34]
date: 2026-10-01
headline: card 2a-ii's last three chunks came home, each on its first attempt of this run — 11,506 files, 581,529,511 B — completing all three Theta folders at card 1b's counts to the byte; all 15 chunks DONE at 112,382 files and 9,555,430,166 B; end check "no finding" against a baseline taken before any copy; manifest 144/0/0 by full hash.
surface: []
---

every chunk home

## Goal

D33 makes the desktop root the main copy; D34 keeps Theta the way Bloomberg is kept. Card 2a-ii
(2026-09-29 to 30, PR #548) cut card 1's Theta plan into 15 chunks and brought 12 home — 100,876
files — then the desktop's Drive sign-in died mid-card: `c13-banded-RCL-to-WBD` failed six attempts
across two runs, every one with `Error 401: Invalid Credentials`, its stop became final, and c14 and
c15 were never tried. The Operator reconnected Drive on 2026-09-30 at 13:03 UTC.

This card finishes those three chunks with the same tool, the same driver and the same checks:

| chunk | Drive objects | copy | copy bytes |
| --- | ---: | ---: | ---: |
| `c13-banded-RCL-to-WBD` | 19,994 | 9,997 | 417,118,592 |
| `c14-banded-WDAY-to-ZTS` | 2,666 | 1,333 | 54,725,633 |
| `c15-deep365` | 3,372 | 176 | 109,685,286 |
| **in all** | **26,032** | **11,506** | **581,529,511** |

Card 2a-ii's own handoff asked for one addition, and this card is where it landed: **a credential
probe before anything reads Drive**, reported as a stop the pen clears rather than as a retry. That
is step 2, and it is the only procedural change from 2a-ii.

## What we tried

1. **Step 0** — preflight on the checkout and the root: the tool's blob, a clean tree,
   `SWE_DATA_ROOT`, rclone, the manifest by size, card 1's `plan.json` by bytes and sha256, card
   2a-ii's four pins, the work folder's shape (12 chunks DONE; c13 with six attempts and two marks;
   c14 and c15 untried), `drive-legacy` as 2a-ii left it, no staging folder, free space.
2. **Step 1** — fetch the driver from its comment on #544 and gate it on its sha256
   (`2fd7af96…`). Every later step re-checks it before it runs.
3. **Step 2 — the credential probe.** One read-only `rclone about gdrive:`, whose quota lines the
   block filters out of its file. This is the check card 2a-ii asked for.
4. **Step 3 — the baseline.** The driver's `endstate` run *before* a byte is copied. The Operator did
   not know whether the Dashboard's `update` had run since 2026-09-30T10:06Z ("3. I dont know"), so
   the run takes its own "before" picture instead of assuming one, and accepts findings there only
   under `data_processed/ibkr/`.
5. **Step 4 (`run`)** — census, inventory, plan, copy, verify, chunk by chunk, unattended. Each chunk
   copies only after its fresh plan matches card 1's.
6. **Step 5** — `endstate` again, compared with step 3's baseline, then the manifest by full hash.

## What worked

All three chunks came home and verified on their first attempt of this run — c13 as `a7` (its
seventh overall, after 2a-ii's six failures), c14 and c15 as `a1`. No `RETRY:` line, no stop, no
`NOT PINNED` line:

| chunk | attempt | copied | bytes | census | inventory | copy | verify |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| c13-banded-RCL-to-WBD | a7 | 9,997 | 417,118,592 | 17.2 min / 19,994 obj / 94 areas | 1.9 min | 80.3 min | 9997 ok, 0 missing, 0 mismatched |
| c14-banded-WDAY-to-ZTS | a1 | 1,333 | 54,725,633 | 2.7 min / 2,666 obj / 22 areas | 8.4 min | 12.0 min | 1333 ok, 0 missing, 0 mismatched |
| c15-deep365 | a1 | 176 | 109,685,286 | 3.5 min / 3,372 obj / 1 area | 1.6 min | 8.5 min | 176 ok, 0 missing, 0 mismatched; 1506 root files ok |
| **in all** | | **11,506** | **581,529,511** | | | | `0 object(s) still need a byte check` throughout |

All three fresh censuses found Drive exactly as card 1 saw it — `the same ids, paths, sizes and
hashes` — and all three plans copied exactly card 1's copies not home yet, each with `card 1's
copies already home with the same bytes: 0`. c13's plan matching for the second time (2a-ii checked
it before its copy failed) also confirms that its six failed attempts published nothing.

**All three Theta folders are now complete, at card 1b's counts to the byte.** Summing the fifteen
chunks' copy-log totals by folder:

| folder under `SmartWheelData/data_processed/theta/` | chunks | files | bytes |
| --- | --- | ---: | ---: |
| `option_history` | c01–c08 | 65,802 | 7,474,676,020 |
| `option_history_banded_backup_2026-06-01` | c09–c14 | 46,404 | 1,971,068,860 |
| `option_history_deep365` | c15 | 176 | 109,685,286 |
| **in all** | | **112,382** | **9,555,430,166** |

The banded backup stood at 35,074 of 46,404 when 2a-ii stopped; 35,074 + 9,997 + 1,333 = 46,404, so
c13 and c14 closed it exactly. The three folder totals are card 1b's own, unchanged since card 1.

**c15's duplicate count fell to 0 and its redundant count rose by exactly the same 700 files and
bytes** — the shift card 2a-ii predicted, now measured:

```
  duplicate  now        0 (0 B)   card 1      700 (92,979,972 B)
  redundant  now     1506 (271,492,800 B)   card 1      806 (178,512,828 B)
```

271,492,800 − 178,512,828 = 92,979,972 B, card 1's duplicate bytes to the byte. And c15's verify
went further than the other two chunks': `1506 root files Drive objects match: 1506 ok, 0 missing or
changed`. So all 1,506 redundant objects, the 700 among them, were byte-checked against copies
already home — the 700 repeats really are the `option_history` files c01–c08 brought home, and
nothing was re-fetched to prove it.

**The sign-in held for the whole run.** Step 2 answered `about exit 0` before anything else read
Drive, and neither `Error 401` nor `invalid_grant` appears anywhere in step 4's output. The probe
cost one API call and about a second.

**The baseline answered the Operator's third question with evidence.** Step 3 read
`== endstate: no finding`: of the root's 31,620 files at chunk 0's first inventory
(2026-09-29T10:40:24Z), 0 gone, 0 resized, 0 bytes changed of 31,619 hashed, and the 100,876 added
were all inside this card's folders. So in the 24 hours after card 2a-ii's own end check, no file in
the root changed name, size or bytes and nothing was added outside `drive-legacy` — either the
Dashboard's `update` did not run, or it rewrote nothing. Step 5 then found the same "no finding"
with 11,506 more files, all in this card's folders.

## What didn't

**Nothing failed in this run.** Three observations worth leaving behind anyway:

1. **`inventory` time does not track the root's size.** It hashes nothing new between chunks, yet it
   took 1.9 min at 132,496 files (c13), 8.4 min at 142,493 (c14) and 1.6 min at 143,826 (c15) — a
   4.4× spread with the root growing monotonically. The variance is the OS file cache and the disk,
   not the work. A card that budgets per-chunk time from the file count will misjudge it; the census
   is the predictable cost.
2. **The whole run hung on c13's one remaining attempt, and nothing in the card could have saved it.**
   c13 carried two `RUN-STOP` marks, so three failures in this run would have made its stop final and
   ended the copying at once. Step 2's probe reduces that exposure but does not remove it: the sign-in
   could still have died at minute 40 of c13's 80-minute copy. The run was lucky as well as checked.
3. **The 3-hour estimate was generous.** Step 4 ran 2h16m47s (10:24:01Z to 12:40:48Z), between a
   baseline that began at 10:13:31Z and an end check that began at 12:42:22Z. c13's census — 17.2 min
   for 19,994 objects — remains the single largest fixed cost, and it is paid three times per chunk
   (once standalone, twice inside `copy`).

Neither the card's 2-hour wait nor its look-in hang rule ever fired.

## How we fixed it

Nothing needed fixing, and nothing was fixed: this card runs a fixed tool (git blob `be2986e8…`) and
a fixed driver (sha256 `2fd7af96…`) and edits neither. The one change from card 2a-ii is procedural —
step 2's credential probe and step 3's baseline — and both are in the card, not in code.

The record for all fifteen chunks stays in one place, `_logs\d33-card2a-ii\`: c13's new attempt `a7`
sits beside 2a-ii's failed `a1`–`a6` and its two `RUN-STOP` marks, which this card leaves untouched.
A reader of that folder sees the failure and the success side by side.

## Evidence

Step 5, the end check, over the whole root:

```
== card 1's plan.json: unchanged since prepare
== the root now, against chunk 0's first inventory (2026-09-29T10:40:24+00:00): names, sizes and bytes
  then 31620 files; now 144002; gone 0, size changed 0, bytes changed 0 of 31619 hashed
  (credential-shaped names by size only), added 112382, of which outside data_archive/drive-legacy/: 0;
  under data_archive/drive-legacy/ but outside this card's folders: 0
== in this card's folders under data_archive/drive-legacy/: 112382 files, 9,555,430,166 B
  the copy logs record 112382 files, 9,555,430,166 B; in this card's folders: other bytes 0,
  card 1's bytes without a log record 0, logged but absent 0
== no staging folder
== endstate: no finding
== against the baseline (swe-card2a-iii-step3.txt): the same findings, and 11506 files added,
  all in this card's folders; 15 of 15 chunks DONE
== every chunk of card 2a-ii is home: 112,382 files, 9,555,430,166 B
```

`python scripts/data_manifest.py check --root <root>`: `checked 144 manifest files: 144 ok, 0
missing, 0 mismatched` — by full hash, not size only.

**The credential was never read.** Each of the three plans reported `root files excluded as
credential-shaped: 1` / `excluded  data_processed/ibkr/flex_credentials.json`: the inventory counted
it by size and never opened it (`143826 files (143825 hashed, 1 credential-shaped, not read)`).

**Download rate.** Each `copy` takes the census twice around its downloads, so subtracting twice the
chunk's standalone census time from its copy time estimates the downloading: c13 80.3 − 2×17.2 =
45.9 min, c14 12.0 − 2×2.7 = 6.6 min, c15 8.5 − 2×3.5 = 1.5 min — 54.0 min (3,240 s) for 11,506
files and 581,529,511 B, about **0.28 s a file** and **179 KB/s**, on files averaging 50 KB. This is
an estimate: the censuses inside `copy` are not separately timed.

**Where the record lives:** `_logs\d33-card2a-ii\` in the desktop root, outside git (D31). This
card's new pins:

| file | sha256 |
| --- | --- |
| `c13-banded-RCL-to-WBD/a7/FINGERPRINTS.txt` | `863a03ad2c67fd1be0ee76c592ba72086e1717d7465c6f60986ab3a98e65d3df` |
| `c14-banded-WDAY-to-ZTS/a1/FINGERPRINTS.txt` | `9d12c1e99eaf39b13c89a8ba148e110d7f9fc0ef08562bb04af66bbd0e9b574d` |
| `c15-deep365/a1/FINGERPRINTS.txt` | `7139752c2e8544fe807f360294a2a5037f489aed59f3d032f13c70660ab1b76d` |

Card 2a-ii's four pins (`chunks.json`, `left.json`, `summary.json`, `card1-plan.sha256`) and c13's
`a1`–`a6` pins were re-read unchanged at step 0; their values are in 2a-ii's fragment.

**Notes posted on #544:**
[REGISTER](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-5929456204).
No BLOCKED or STOPPED note was needed.

## Unresolved / handoff

1. **For the pen at the close:** this fragment's `status:` is `in-flight` and its `pr:` is empty —
   set them to `merged` and the PR number when it lands, and add the PR number and merge commit to
   the CHANGELOG entry.
2. **The Drive sign-in has a clock on it.** It signs in through a Google Cloud app of the Operator's
   own (confirmed 2026-09-30). If that app is still in "Testing", Google ends the sign-in seven days
   after it was made — about **2026-10-07 13:00 UTC**. Any Drive card after that date should expect a
   fresh login, and should run a probe like step 2 before it reads Drive. Publishing the app would
   remove the clock; that is the Operator's call.
3. **Card 2b is what remains on Drive**: the rest of `SmartWheelData` (including the two empty files
   directly in `data_processed/theta`), the day-bot's areas (`data_raw` among them), `swe-local-only`
   and the git upload. None of it was touched here.
4. **Card 3 (deletion) now has a complete Theta set to work from.** All 112,382 of card 1's Drive-only
   Theta files are home and verified, so the Drive copies of c13–c15 are no longer the only copy — the
   caveat 2a-ii raised is closed. Nothing has been deleted: D33 defers every deletion to card 3.
5. **c13's failure record stays in place.** `RUN-STOP-1.txt`, `RUN-STOP-2.txt` and attempts `a1`–`a6`
   are still in `_logs\d33-card2a-ii\c13-banded-RCL-to-WBD\`, beside the successful `a7`. The card's
   constraints forbid deleting them, and they are the evidence that the six attempts published
   nothing.
6. **Worth considering for the next long card:** step 2's probe proved cheap and the right shape, but
   it only guards the start. A probe between chunks — or a stop that distinguishes a 401 from an
   outage *during* a copy, which this card's step 4 rule already does by reading the chunk's log —
   would shorten the next dead-credential run further.
