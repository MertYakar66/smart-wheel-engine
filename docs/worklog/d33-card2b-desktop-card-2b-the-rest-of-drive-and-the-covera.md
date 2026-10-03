---
id: d33-card2b
title: Desktop card 2b: the rest of Drive, and the coverage evidence (D33, D34)
kind: verification
status: merged
terminal: desktop
pr: 552
decisions: [D33, D34]
date: 2026-10-03
headline: every Drive file in the six areas now has a byte-identical copy at home — 3,809 files and 47,691,144 B came home, exactly card 1's remainder, and the verify re-hashed all 178,216 non-folder Drive objects' home counterparts with 0 missing, 0 mismatched and 0 unresolved; one copy attempt was lost to a transient DNS failure and the driver's own retry carried it; end check "no finding" and manifest 144/0/0 by full hash.
surface: []
---

every Drive file home, but for 0 unresolved

## Goal

D33 makes the desktop root the main copy and brings home anything that exists only in the old
Drive folders, into `data_archive/drive-legacy/<area>/<path>`, never over an existing file.
D34 keeps the Theta collection the way Bloomberg is kept and leaves the day-bot's `data_raw` out
(the Operator, 2026-09-26).

Cards 2a-i, 2a-ii and 2a-iii (#546, #548, #550) worked from card 1's plan and brought home every
Drive-only file in the Bloomberg folders and five Theta subfolders: 114,605 files,
10,634,294,355 B. Card 1's plan left 3,809 files and 47,691,144 B untouched — files sitting
directly in big folders that a per-chunk card could not reach.

This card is the closing move and the evidence card 3 needs. It censuses **whole areas** — the
tool's own six (`swe-local-only`, `SmartWheelData`, `smart-wheel-engine-git`, and the day-bot's
`vendor_swe_data`, `…_raw` and `…_processed`), card 1's seven less the day-bot's `data_raw` —
copies home whatever still exists only on Drive, and then verifies the *whole* plan, so that every
Drive file in those areas is shown to have a copy at home with the same bytes. Nothing is deleted;
Drive is only read.

## What we tried

1. **Step 0** — preflight on the checkout and the root: the tool's blob
   (`be2986e8396718c927630b7bceaca1efc0320d43`), a clean tree, `SWE_DATA_ROOT`, rclone v1.74.4,
   the manifest by size, card 1's `plan.json` by bytes (311,257,331 B) and sha256 (`d5ffbf44…`),
   `drive-legacy` exactly as card 2a-iii left it, no staging folder, no work folder, 1.17 TB free.
   The branch `claude/d33-card2b` was cut from `origin/main` at `4e42139`.
2. **Step 1** — fetch the driver from its comment on #544 and gate it on its sha256
   (`591e21660f4852a473167c95c4815bf208f0b453e16ac87e59f780bed93b59a0`). Every later step
   re-checks it before it runs.
3. **Step 2 — the Drive sign-in.** One read-only `rclone about gdrive:`, whose quota lines the
   block filters out of its file so they never reach the public Run Summary. `about exit 0`.
4. **The briefing and the REGISTER note** on #544 — this card has no gate, so the briefing is how
   the Operator learns what the run needs (desktop awake, no Dashboard update, no checkout
   changes).
5. **Step 3 (`run`)** — the `plan` step (sign-in, census of the six areas, hashed inventory of the
   root, plan, then the check against card 1's plan), and the `copy` step (sign-in, the tool's
   copy on that plan, then its verify of the *whole* plan, then the coverage lines). Unattended.
6. **Step 4 (`endstate`)** — the root compared with the plan's own inventory, taken right after
   the census and about 10 hours before any copy was published; then the manifest by full hash.

## What worked

**The fresh plan reproduced card 1's remainder to the byte.** The census read 327,825 objects
across the six areas, every area's object, file and byte counts *equal to `rclone size`*, and
`swe-data exists: False`. The plan then asked for exactly `to copy home: 3809 files, 47,691,144 B`
— card 1's forecast for this card, exactly — and `left to settle or listed: 0`. The 3,809 split by
area the way card 1b's table said it would: `SmartWheelData` 3,762 (47,244,257 B),
`smart-wheel-engine-git` 32 (53,723 B), the day-bot's `vendor_swe_data` 13 (393,164 B) and
`swe-local-only` 2 (0 B, two empty files in `theta`).

**The check against card 1's plan passed with one expected difference and nothing else.** Drive
objects now 327,825 against card 1's 327,826 in these areas: **gone 1, new 0, changed 0**. The one
gone is `SmartWheelData/data_processed/ibkr/flex_credentials.json` (card 1: file, unresolved) — the
credential-shaped file the pen moved to Drive's trash on 2026-09-26, and a *name*, never a value.
There were **no** `NEW`, `CHANGED`, `TWIN`, `NOT NOW`, `ELSEWHERE`, `UNRESOLVED`, `BYTE CHECK` or
`LOST` lines at all. The class transitions tell the campaign's story in one table: `copy ->
redundant 114,605` (card 2a's copies, now home and recognised as such), `copy -> copy 3,809` (this
card's work), `redundant -> redundant 59,084`, `duplicate -> redundant 705`, `duplicate ->
duplicate 13`. And `not copied now, though on Drive as card 1 saw them: 0`.

**The copy and the verify.** `191/191 batches`, then `copy: 3809 copied and verified; 0 not`. The
verify re-hashed everything: `3809 copies: 3809 ok, 0 missing, 0 mismatched; 143690 root files
Drive objects match: 143690 ok, 0 missing or changed; 0 object(s) still need a byte check`.

**The coverage claim, with the condition it depends on.** Of the 178,216 non-folder Drive objects
in the plan: 3,809 copied home by this card, 174,394 already home (the verify re-hashed the root
file each matches), 13 the same bytes as one of those, and **0 unresolved** — so `== every Drive
file in the six areas has a copy at home with the same bytes, but for the 0 unresolved`. The claim
holds for Drive as the plan's census saw it at 2026-10-02T13:32:41Z, and the copy step's **two**
later full censuses read the same object, file and byte counts in every area, which is what rules
out Drive having changed under the plan.

**The end state.** Against the plan's inventory: `gone 0, size changed 0, bytes changed 0 of
144001 hashed`, `added 3809, of which outside data_archive/drive-legacy/: 0`, and `under
data_archive/drive-legacy/ but not this card's copies: 0`. The copies reconcile three ways — the
plan has 3,809 / 47,691,144 B, the copy logs record 3,809 / 47,691,144 B, `at home with other
bytes 0`, `the planned bytes without a log record 0`, `logged but absent 0`. `drive-legacy` now
holds 118,414 files and 10,681,985,499 B, which is card 2a's 114,605 plus this card's 3,809 — and
is also, to the byte, card 1's entire `copy` class (118,414 files, 10,681,985,499 B). Card 1's plan
is now wholly home. `== endstate: no finding`, and the manifest reads 144/0/0 by full hash.

## What didn't

**Copy attempt a1 was lost to a transient DNS failure, 312.7 minutes in.** Its pre-download census
was partway through `SmartWheelData` when rclone could not resolve the hostname:

```
ERROR: rclone size failed for <the folder's Drive id>: … dial tcp: lookup www.googleapis.com:
getaddrinfow: This is usually a temporary error during hostname resolution …
```

`exit 2` at 2026-10-02T18:56:26Z. Not the credential — a dead sign-in stops the run outright with
no retry, and all three sign-in checks in this run passed. Nothing had been downloaded yet, so
nothing was lost but time.

**The lesson worth carrying.** A copy attempt restarts from the *start of its step*, which means it
re-censuses all six areas, and a census of this Drive costs about five hours (the plan's took
307.4 min). So one momentary resolver failure cost a whole pass — a1 spent 312.7 min and published
nothing — and the run made **four** full-area census passes in all: the plan's, a1's failed one,
and the two inside a2's 624.6 min. That is most of its 21 hours. The whole-area census is also the
only way to reach files sitting directly in big folders, so it cannot simply be narrowed. A
census-cache, reused by a retry when an area's object, file and byte counts are unchanged, would
have spared three of those four passes. That is a proposal for the pen, not a change made here.

## How we fixed it

The driver's own retry rule carried it, with no human in the loop: five minutes after a1's exit it
checked the sign-in again and ran attempt a2 from the start of the copy step. a2 re-censused the
six areas (identical counts), fetched only what was not already home with the right bytes — all
3,809, since a1 had published nothing — censused again, and verified. `== copy: DONE (a2: 3809
copied and verified in this attempt; 3809 in all attempts)`. The tool never overwrites: it
publishes a verified download with a hard link or an exclusive create, and a destination holding
other bytes stops the copy. Step 4 confirms after the fact that nothing but this card's 3,809
copies was added and nothing at all was changed.

## Evidence

Timings, from the tool's own logs under `_logs\d33-card2b\`:

| what | log | result |
| --- | --- | --- |
| census, 6 areas, 327,825 objects | `plan/a1/01-census.txt` | `exit 0 (307.4 min)` |
| inventory, 144,002 files / 18,569,412,686 B | `plan/a1/02-inventory.txt` | `exit 0 (10.4 min)` |
| plan, 3,809 / 47,691,144 B to copy | `plan/a1/03-plan.txt` | `exit 0 (0.5 min)` |
| copy attempt a1 | `copy/a1/05-copy.txt` | `exit 2 (312.7 min)` — DNS |
| copy attempt a2, 191/191 batches | `copy/a2/05-copy.txt` | `exit 0 (624.6 min)` |
| verify, whole plan | `copy/a2/06-verify.txt` | `exit 0 (10.0 min)` |

The run was `driver run 20261002T082519Z` (2026-10-02T08:25:19Z) to 2026-10-03T05:36:03Z, about
21 hours; the end check was `driver endstate 20261003T053656Z`. The inventory left the one
credential-shaped root file unread by name (`data_processed/ibkr/flex_credentials.json`): 144,001
of 144,002 files hashed.

The record lives outside git, under `_logs\d33-card2b\` (`areas.json`, `card1-plan.sha256`, the two
driver logs, `plan/a1/`, `copy/a1/`, `copy/a2/`, and the `DONE` marks). Its pins:

| pin | sha256 |
| --- | --- |
| `plan/a1/FINGERPRINTS.txt` (8 files) | `52a62d581a969e9e85485cafcfababfd17fe1c965788a6191507e503710a1c2b` |
| `copy/a1/FINGERPRINTS.txt` (2 files) | `28e51bbd92800c4795da7863694a184694bf1c6b1e0b92a54612493a79d0fc75` |
| `copy/a2/FINGERPRINTS.txt` (4 files) | `0ceed1bf4af7caaf1ffcfb3ca88a930258afdf0bb6c4a71f978f115be81fbbe0` |

The card's plan is `plan/a1/plan.json`, sha256
`acd03b5898697b18f818d7f2939392b6186efd9cc9fd977238ea036b12fc793f`; step 4 re-read it and card 1's
and reported both unchanged.

Posted on #544: the REGISTER note
([5948148724](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-5948148724)).
No BLOCKED, WAITING or STOPPED note was needed. Step 0's and step 4's manifest checks both read
144/0/0 — the first by size, the second by full hash.

## Unresolved / handoff

- **Card 3 may now treat Drive's six areas as a second copy.** The coverage is proven for Drive as
  the plan's census saw it (2026-10-02T13:32:41Z), with two later full censuses agreeing. Card 3
  should re-probe before it deletes anything, as card 2a-iii's handoff also said.
- **The day-bot's `data_raw` is still outside all of this.** It has never been censused, planned or
  copied (D34, the Operator, 2026-09-26). Card 3 must not read this card's coverage as covering it.
- **Still card 2c:** `swe-data/` on Drive, `SHA256SUMS`, and the restore tests.
- **The credential-shaped file stays unresolved by design.** `data_processed/ibkr/flex_credentials.json`
  is excluded from the inventory by name and never opened; its Drive twin has been gone since card
  1 and is the single `GONE` line in this card's comparison.
- **The Drive sign-in held for the whole 21-hour run**, so the 7-day Testing-mode expiry that card
  2a-iii flagged did not bite here. Step 2's probe and the driver's per-step checks stay worth
  keeping for any long card.
- **A proposal for the pen** (an Executor may only propose): give the tool a census-cache that a
  retried attempt reuses when an area's object, file and byte counts are unchanged. On this run it
  would have spared three of the four full-area census passes over 327,825 unchanged objects, and
  it would make a transient network error cost minutes instead of a five-hour pass.
- **The pen, at the close (2026-10-03).** No number from a step file changed. On the census-cache
  proposal: the slow part is the listing itself, and knowing that an area's counts are unchanged
  takes a fresh listing. The copy step's two re-listings are also the evidence that Drive did not
  change under the plan, which the coverage rests on. A cache could save that time safely only by
  reading Drive's own change feed, a larger change, so it is not adopted for now (`PROJECT_STATE.md`
  §0 B). After the report, the Executor also saved a note to Claude's local memory on the desktop
  (the Operator's paste shows it). That is not one of the project's records: this fragment, PR #552
  and #544 are.
