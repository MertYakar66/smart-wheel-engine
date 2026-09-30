---
id: d33-card2a-ii
title: Desktop card 2a-ii: Drive's option_history trees, by ticker folder (D33, D34)
kind: verification
status: merged
terminal: desktop
pr: 548
decisions: [D33, D34]
date: 2026-09-30
headline: 100,876 of card 1's 112,382 Drive-only Theta files (8,973,900,655 B) came home into data_archive/drive-legacy/SmartWheelData/ across 12 chunks, each on its first attempt; option_history is complete; c13-c15 (11,506 files, 581,529,511 B) stopped on an expired Drive OAuth credential (Error 401) and wait for a later run; end check "no finding", manifest 144/0/0.
surface: []
---

stopped at c13-banded-RCL-to-WBD: STOPPED: c13-banded-RCL-to-WBD: 3 attempts in each of 2 runs; the last: census exited 2; its output is in c13-banded-RCL-to-WBD/a6/01-census.txt

## Goal

D33 makes the desktop root the main copy, and D34 keeps Theta the way Bloomberg is kept. Card 2a-i
(2026-09-28, PR #546) brought home 2,223 files and deferred `option_history_deep365`, because all
700 of its duplicates repeat files in `option_history` — a folder this card brings home. So this
card, 2a-ii, takes the three folders that were left:

| folder under `SmartWheelData/data_processed/theta/` | copy | copy bytes |
| --- | ---: | ---: |
| `option_history` | 65,802 | 7,474,676,020 |
| `option_history_banded_backup_2026-06-01` | 46,404 | 1,971,068,860 |
| `option_history_deep365` | 176 | 109,685,286 |
| in all | 112,382 | 9,555,430,166 |

Each ticker folder is one area, so every destination is the one card 1's plan gives. Consecutive
folders form a chunk of at most 20,000 Drive objects and 1 GiB, and a chunk copies only after its
own fresh census, inventory and plan have been checked object by object against card 1's plan.

## What we tried

1. **Step 0** — preflight on the checkout and the root: the tool's blob, a clean tree,
   `SWE_DATA_ROOT`, rclone v1.74.4, manifest 144/0/0 by size, card 1's `plan.json` at
   311,257,331 B and sha256 `d5ffbf44…`, `drive-legacy` as 2a-i left it (2,223 files,
   1,078,864,189 B), and none of the three destination folders present. Clean on the first run.
2. **Step 1** — fetch the driver from its comment on #544 and gate it on its sha256.
3. **Step 2 (`prepare`)** — read card 1's 311 MB `plan.json` once, cut 15 chunks, and check the
   account against card 1b's published counts.
4. **Step 3 (`plan0`)** — census, inventory and plan for chunk 0, checked against card 1's, then
   stop at G1 and ask the Operator before a byte is copied.
5. **Step 4 (`run`)** — copy and verify, chunk by chunk, unattended.
6. **Step 4 again (`step4b`)** — after c13's three attempts failed, the card's 2-hour wait and one
   rerun.
7. **Step 5 (`endstate`)** — hash the whole root and compare it with chunk 0's first inventory.

## What worked

Step 2 cut the three folders into 15 chunks and accounted for every file row in card 1's plan:
`== card 1's plan, as prepare read it, matches card 1b's counts, and every file row is in one part`.
Nothing at all was left for card 2b — no ticker folder failed the settle rules, and the two loose
objects (one directly in each of `option_history` and the banded backup) are already home, so they
cost nothing to copy.

Twelve chunks came home and verified on the first attempt (`a1`), with no retry:

| chunk | copied | bytes | census | copy | verify |
| --- | ---: | ---: | ---: | ---: | --- |
| c01-oh-AAPL-to-BIIB | 7,858 | 1,053,919,701 | 18.0 min / 18,964 obj | 79.6 min | 7858 ok, 0 missing, 0 mismatched; 1624 root files ok |
| c02-oh-BKNG-to-CSX | 8,412 | 931,206,898 | 18.3 min / 19,406 obj | 87.0 min | 8412 ok, 0 missing, 0 mismatched; 1291 root files ok |
| c03-oh-CVS-to-GILD | 8,524 | 811,513,871 | 17.6 min / 19,014 obj | 84.5 min | 8524 ok, 0 missing, 0 mismatched; 983 root files ok |
| c04-oh-GM-to-LLY | 8,016 | 1,034,629,685 | 16.5 min / 17,464 obj | 77.5 min | 8016 ok, 0 missing, 0 mismatched; 716 root files ok |
| c05-oh-LMT-to-NEM | 9,506 | 1,021,520,195 | 18.7 min / 19,874 obj | 88.2 min | 9506 ok, 0 missing, 0 mismatched; 431 root files ok |
| c06-oh-NFLX-to-ROP | 8,861 | 1,055,731,202 | 16.7 min / 17,938 obj | 78.9 min | 8861 ok, 0 missing, 0 mismatched; 108 root files ok |
| c07-oh-ROST-to-UNP | 9,571 | 1,067,953,880 | 17.9 min / 19,142 obj | 82.1 min | 9571 ok, 0 missing, 0 mismatched; 0 root files |
| c08-oh-UPS-to-XYZ | 5,054 | 498,200,588 | 9.4 min / 10,108 obj | 44.1 min | 5054 ok, 0 missing, 0 mismatched; 0 root files |
| c09-banded-AAPL-to-COF | 5,185 | 222,044,257 | 14.8 min / 19,954 obj | 54.5 min | 5185 ok, 0 missing, 0 mismatched; 4792 root files ok |
| c10-banded-COIN-to-GPN | 9,956 | 431,111,161 | 15.4 min / 19,954 obj | 77.2 min | 9956 ok, 0 missing, 0 mismatched; 21 root files ok |
| c11-banded-GRMN-to-MCD | 9,971 | 416,100,642 | 15.2 min / 19,942 obj | 76.8 min | 9971 ok, 0 missing, 0 mismatched; 0 root files |
| c12-banded-MCHP-to-QQQ | 9,962 | 429,968,575 | 15.8 min / 19,924 obj | 80.3 min | 9962 ok, 0 missing, 0 mismatched; 0 root files |
| **in all** | **100,876** | **8,973,900,655** | | | `0 object(s) still need a byte check` throughout |

`option_history` is complete: c01–c08 are exactly its 65,802 files and 7,474,676,020 B, matching
card 1b to the byte. The banded backup stands at 35,074 of 46,404.

Every one of those twelve fresh plans matched card 1's exactly. The `Drive` line read `the same
ids, paths, sizes and hashes` in all twelve — no object had moved, changed size or changed hash
since card 1's census — and `copies now: exactly card 1's copies not home yet`
equalled card 1's full count each time, with `card 1's copies already home with the same bytes: 0`.

## What didn't

**c13-banded-RCL-to-WBD stopped the run: the Drive OAuth credential expired mid-card.** Its plan
was checked and matched card 1's (19,994 objects, the same ids, paths, sizes and hashes; 9,997
copies, 417,118,592 B). Its copy then ran all 500 download batches and failed:

```
ERROR: Drive query failed after 3 tries: '<the parent folder's Drive id>' in parents and name = 'ticker=RCL' and trashed = false…
NOTICE: Failed to backend: couldn't find root directory ID: googleapi: Error 401: Request had invalid authentication credentials.
Reason: authError, Message: Invalid Credentials
```

The same 401 came back on every later attempt, `a2` to `a6`, each one failing its census in about
12 seconds. Six attempts across two runs, over the 3h10m from a1's start to a6's end:

| attempt | run | what failed | wall |
| --- | --- | --- | ---: |
| a1 | 1 | `copy` exited 2 after 500/500 batches | 47.7 min |
| a2, a3 | 1 | `census` exited 2 | 0.2 min each |
| a4, a5, a6 | 2 | `census` exited 2 | 0.2 min each |

**Nothing at all was published by those six attempts.** The copy log's every record reads
`download failed: … Error 401`, so no file was downloaded, let alone linked into place; step 5
confirms it (`c13 … its copy logs record 0 files copied, 0 B`), and no staging folder was left
behind. c14 and c15 were never attempted: the driver stops at the first chunk it cannot settle.

The 2-hour wait the card prescribes did not help, and could not: this is not an outage but a dead
refresh token. rclone re-mints an access token on its own while the refresh token lives; a 401
`Invalid Credentials` on the *root-directory lookup* means the refresh itself was refused. Clearing
it needs `rclone config reconnect gdrive:` — an interactive Google login, which no Execution Prompt
can perform and which this card's constraints put outside the Executor's reach (`rclone version`
is the only rclone command it may run). The Operator was told at 09:59Z, during the five-minute gap
before attempt a6, that the window to save the run was open; no reconnection happened, and a6
failed like the rest.

Worth recording for the next card: **the card's retry ladder cannot distinguish a transient outage
from a credential that will never come back.** Both read as "a stop a rerun may clear", and the
ladder spends its six attempts and 2-hour wait on both. A credential check before a chunk's first
census — one cheap Drive call whose 401 is reported as a stop the pen must clear, not as a retry —
would have ended the run at a2 (07:47Z) instead of a6 (10:05Z), sparing the 2-hour wait and
the second run, with the same outcome.

## How we fixed it

Nothing was fixed in code: this card runs a fixed tool and a fixed driver, and edits neither. The
run was recorded as it stood. The copying ended by the card's own rule (a chunk that fails three
attempts in each of two runs is final), and the card went on to step 5, the record, the pull
request and the Run Summary. Whatever was copied stays: it is verified, and the next card plans
c13–c15 again from card 1's plan.

## Evidence

Step 5, the end check, over the whole root (`== endstate: no finding`):

```
== card 1's plan.json: unchanged since prepare
== the root now, against chunk 0's first inventory (2026-09-29T10:40:24+00:00): names, sizes and bytes
  then 31620 files; now 132496; gone 0, size changed 0, bytes changed 0 of 31619 hashed
  (credential-shaped names by size only), added 100876, of which outside data_archive/drive-legacy/: 0;
  under data_archive/drive-legacy/ but outside this card's folders: 0
== in this card's folders under data_archive/drive-legacy/: 100876 files, 8,973,900,655 B
  the copy logs record 100876 files, 8,973,900,655 B; in this card's folders: other bytes 0,
  card 1's bytes without a log record 0, logged but absent 0
== no staging folder
== endstate: no finding
```

`python scripts/data_manifest.py check --root <root>`: `checked 144 manifest files: 144 ok,
0 missing, 0 mismatched` — by full hash, not size only.

**Timings.** The three censuses a chunk needs dominated the run: 18 minutes for a 19,000-object
`option_history` chunk, 15 minutes for a banded chunk (more folders, fewer objects each). Removing
twice each chunk's census time from its copy time leaves 522.1 minutes of downloading for 100,876
files — **0.31 s a file**, or about 286 KB/s on files averaging 89 KB. The twelve DONE chunks took
18h27m of wall clock (2026-09-29T12:05:06Z to 2026-09-30T06:32:23Z), and the whole card 19h59m
across two runs.

**Where the record lives:** `_logs\d33-card2a-ii\` in the desktop root, outside git. Its pins:

| file | sha256 |
| --- | --- |
| `card1-plan.sha256` | `8069760be2a5e57ea377856a73ce63c8286a762115c18e8e79419e19515fc5e2` |
| `chunks.json` | `63028951d1035ea06d932dbdc6592dad4b1cedc58136b8ef94470c6f0b0a6b14` |
| `left.json` | `bfeded36676351b9d33302f4e82f899efb0bb7ff68b023b6adf127e102f26342` |
| `summary.json` | `2cb3785c186597d43dcc7c6b3133afab4dd85975c18f3b0c01cd6a61302996e9` |
| `c01…/a1/FINGERPRINTS.txt` | `e65c974644a8508479450013ad4ee5a8193b7e3c78a26998db73034d3fceb8f5` |
| `c01…/a1/FINGERPRINTS-2.txt` | `181b3c5617995983ea27cc4c9bd4d881430d5c876775cce3d08c91b4c99386bf` |
| `c02…/a1/FINGERPRINTS.txt` | `14eb99615af54c84956aa16d3821ea5bb1c4b45f3a0a56873e88e9979325567c` |
| `c03…/a1/FINGERPRINTS.txt` | `c07fe91e90fb9a6cfd47f5999fae8c3d110070f1dc0730a5d493b44cab815b48` |
| `c04…/a1/FINGERPRINTS.txt` | `a6b49c0f1c86a7f89b4cb856e059397ba8bda1f4c14db14bbfb8fe93ed5eabf9` |
| `c05…/a1/FINGERPRINTS.txt` | `595b5bf5417a49b9e97c4c41f5c72fe388f269f90d5bb01a2349e34de89f08bd` |
| `c06…/a1/FINGERPRINTS.txt` | `242e22c110a30254df9baa3e856148f1addac509621ce1df46de9c3fb395c898` |
| `c07…/a1/FINGERPRINTS.txt` | `9f1fb6702946a1fa3a2f6841158fb860ef19cbf076be31bc7b94d1ced38f86dc` |
| `c08…/a1/FINGERPRINTS.txt` | `78a1d8f46dd6ac76338a431a21d8410bbf85c3a5d308d39b09e2b6a8e5d4d91d` |
| `c09…/a1/FINGERPRINTS.txt` | `b60013ffc00d2d044e46af51a0d678cf123aaaa0ce293495180512276e062a33` |
| `c10…/a1/FINGERPRINTS.txt` | `be204846fe5fa6be3f22d736b6fe41a1bbd3021011d561e2620d0b3ebdd7be0c` |
| `c11…/a1/FINGERPRINTS.txt` | `9bd2b182461f1bd2270cee0f1583a53f232c2d9000f28a5fa8499bb7b685e698` |
| `c12…/a1/FINGERPRINTS.txt` | `b696e59762f2ddde419228389ad14099367c68168d60e129f176fd52807d705a` |
| `c13…/a1/FINGERPRINTS.txt` | `c4184a2e5a99a5366af3bfb04ea0fcf10aa02cb86ed28bb3cb9ffc7a5e9077f7` |
| `c13…/a2/FINGERPRINTS.txt` | `f50bceb19b0befde7c106da64a3a3305513b2b0e6f677c37eb9d9f292e0bfcd6` |
| `c13…/a3/FINGERPRINTS.txt` | `8a23c23cbd2f6e21852c00e660b9ae2ba0b3c89a4a8c251fdf154d401cad14a4` |
| `c13…/a4/FINGERPRINTS.txt` | `5c472b3c0586106c4a6415f1380cb047c4ba5012bf2c2d3f0f0d7fafffffecb5` |
| `c13…/a5/FINGERPRINTS.txt` | `a473a681602fd6c10d5b57bb0a85297c38bd4e741353b9279fe72c899478adae` |
| `c13…/a6/FINGERPRINTS.txt` | `010c3fe3e3548c830dfd8746bf37a835d68a96efdd5f7d06fbd07b99749b2eff` |

The five `c13` pins from `a2` to `a6` were written by step 5, for attempts the run left unpinned.
`c13/a1`'s was written by the run itself, after the failed copy.

**Notes posted on #544:**
[REGISTER](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-5889902748),
[STOPPED](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-5908996446).

## Unresolved / handoff

1. **The Drive credential has to be reconnected before any further Drive card runs.**
   `rclone config reconnect gdrive:` on this desktop, by the Operator. Until then every census
   fails in 12 seconds, so a rerun of this card would burn its attempts and stop again.
2. **c13, c14 and c15 are still on Drive only** — 11,506 files, 581,529,511 B:

   | chunk | files | bytes | note |
   | --- | ---: | ---: | --- |
   | c13-banded-RCL-to-WBD | 9,997 | 417,118,592 | plan already checked and matched card 1's |
   | c14-banded-WDAY-to-ZTS | 1,333 | 54,725,633 | never attempted |
   | c15-deep365 | 176 | 109,685,286 | never attempted; its 700 duplicates repeat `option_history` files that are now home, so a fresh plan will read them as `redundant` |

   Card 1's plan covers them unchanged, and the chunks are already cut in
   `_logs\d33-card2a-ii\chunks.json`. The pen decides whether the next run re-enters this card or
   takes a card of its own; either way nothing needs re-planning from scratch.
3. **The banded backup is partial** (35,074 of 46,404 files). Nothing reads
   `data_archive/drive-legacy`, so a partial folder harms nothing — but card 3 must not treat the
   Drive copies of c13–c15 as deletable, because they are not home yet.
4. **A suggestion for the next card, for the pen to weigh:** a credential probe before each chunk's
   first census, reported as a stop the pen clears rather than a retry. See "What didn't".
