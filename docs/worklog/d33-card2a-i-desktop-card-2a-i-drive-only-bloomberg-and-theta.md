---
id: d33-card2a-i
title: Desktop card 2a-i: Drive-only Bloomberg and Theta files, chunk by chunk (D33, D34)
kind: verification
status: in-flight
terminal: desktop
pr:
decisions: [D33, D34]
date: 2026-09-28
headline: 2,223 Drive-only files (1,078,864,189 B) came home into data_archive/drive-legacy/SmartWheelData/ across c0-bloomberg, c1-index_reference and c2-option_history_delisted; c3-option_history_deep365 deferred to 2a-ii; end check "no finding", manifest 144/0/0.
surface: []
---

c3 waits for 2a-ii

## Goal

D33 makes the desktop root the main copy, and D34 keeps Theta the way Bloomberg is kept. Both
leave the same job: the files that exist only in the old Drive folders have to come home, into
`data_archive/drive-legacy/SmartWheelData/<path>`, never over an existing file, and nothing gets
deleted before card 3.

Card 1 (2026-09-25) took the Drive census and wrote the plan; card 1b (2026-09-26) counted it
(PR #538). This card is the first that actually copies. It works in four chunks — the twelve
Bloomberg files first, then three Theta folders — and each chunk only copies after its own fresh
census, inventory and plan have been checked, object by object, against card 1's plan.

## What we tried

1. **Step 0** — preflight on the checkout and the root. It stopped on the first run: the working
   tree was not clean.
2. **Steps 1–2** — write the driver, in two halves, and gate all three halves on the sha256s the
   card gives.
3. **Step 3 (`prepare`)** — read card 1's 311 MB `plan.json` once, derive each chunk's `areas.json`
   and `expect.json`, and check the four chunks against card 1b's published counts.
4. **Step 4 (`plan0`)** — census, inventory and plan for chunk 0, checked against card 1's, then
   stop at the gate G1 and ask the Operator before a single byte is copied.
5. **Step 5 (`run`)** — copy and verify, chunk by chunk, unattended.
6. **Step 6 (`endstate`)** — hash the whole root and compare it with chunk 0's first inventory.

## What worked

All three live chunks came home and verified on the first attempt (`a1`), with no retry and no
stop:

| chunk | copied | bytes | verify | wall |
| --- | ---: | ---: | --- | ---: |
| c0-bloomberg | 12 | 484,917,228 | 12 ok, 0 missing, 0 mismatched; 74 root files 74 ok | 1.9 min |
| c1-index_reference | 1,261 | 493,572,169 | 1261 ok, 0 missing, 0 mismatched; 593 root files 593 ok | 14.6 min |
| c2-option_history_delisted | 950 | 100,374,792 | 950 ok, 0 missing, 0 mismatched; 884 root files 884 ok | 12.3 min |
| **in all** | **2,223** | **1,078,864,189** | `0 object(s) still need a byte check` throughout | ~37 min |

Every chunk's fresh plan matched card 1's exactly — the `Drive` line read `the same ids, paths,
sizes and hashes` in all three, and `copies now: exactly card 1's copies not home yet` equalled
card 1's full count each time (0 already home, since this was the first copying card).

The deferral rule did its job unprompted. `c3-option_history_deep365` was marked `DEFERRED` at
step 3, before any Drive read for it, because all 700 of its duplicates repeat files in
`data_processed/theta/option_history` — a folder 2a-ii brings home, not this card. Copying c3 now
would have put those bytes under c3's path instead of the path card 1 chose.

## What didn't

**The card's heredoc could not be carried by this session's Bash tool.** Steps 1 and 2 write the
driver with `cat > "$DRV.1" <<'PY' … PY`, a command of about 23 KB. Bash rejected it outright:

```
/usr/bin/bash: -c: line 183: unexpected EOF while looking for matching `''
```

Line 183 is `os.makedirs(a)  # a fresh folder: never an earlier attempt's` — a line ending in an
apostrophe. A quoted heredoc should not parse its body as shell at all, so the body was clearly
not reaching bash intact. Two scratchpad probes separated the cause from the noise:

- a small quoted heredoc containing the same apostrophes, the same `$`, and Python triple quotes
  round-tripped perfectly — so the content is not the problem;
- the same probe showed a doubled `\\` collapsing to `\`, confirming an escaping layer sits between
  the tool call and bash — but single `\0`, `\r\n` and `\n` survive, and the driver has no `\\`.

So this is a **payload-size transport limit in the harness, not a content fault**. The fix was not
to retry the same mechanism: a verbatim rerun would have failed identically and burned one of the
card's three attempts for nothing.

**The working tree was not clean at step 0.** `engine/features/dynamics.py` had a pasted
`https://claude.ai/api/organizations/…/files/…/contents` URL where a blank line belonged (mtime
2026-09-28 13:14 local, minutes before this run began at 10:47Z, so pasted in after card 1b —
every other timestamp in this fragment is UTC). It was not a card artefact and it was not ours, and
it made the file a `SyntaxError`, so the engine could not import it. Per the card, a stop in step 0
is reported and nothing else is written, so the run stopped and asked.

## How we fixed it

**The unclean tree.** Reported it to the pen rather than restoring it — a working-tree change is
not the Executor's to discard. The Operator chose to keep it rather than drop it:

```
git stash push -m "stray paste in dynamics.py, set aside before card 2a-i" -- engine/features/dynamics.py
```

It is `stash@{0}`; the file parses again; step 0b then read `changes 0`.

**The heredoc.** Wrote the two halves with the Write tool, which places exact bytes without passing
them through a shell, then ran the card's verification unchanged. The card's own sha256 gates are
what decide whether the driver is the card's driver, and all three passed on the first try:

| file | sha256 | card |
| --- | --- | --- |
| first half | `d8b9869667febfd4d5bb93789676048466d3fe3102e4861e4801ef73af9db3b7` | matches |
| second half | `9cedc6490e10ad2a0ae8901e8ad2320d575bcfb8681a2b094fed7f49d51a177b` | matches |
| whole driver | `606455bbd576c41093a0474b68a001a25cfa16e4e4a6933684574773f012e2f8` | matches |

The deviation is recorded as a `NOTE:` at the top of `swe-card2a-i-step1.txt` and
`swe-card2a-i-step2.txt`, so it travels with the evidence rather than only with this fragment.

## Evidence

**The end check (step 6) found nothing.**

```
== card 1's plan.json: unchanged since prepare
== the root now, against chunk 0's first inventory (2026-09-28T10:50:26+00:00): names, sizes and bytes
  then 29397 files; now 31620; gone 0, size changed 0, bytes changed 0 of 29396 hashed
  (credential-shaped names by size only), added 2223, of which outside data_archive/drive-legacy/: 0
== under data_archive/drive-legacy/: 2223 files, 1,078,864,189 B
  the copy logs record 2223 files, 1,078,864,189 B; under data_archive/drive-legacy/:
  other bytes 0, card 1's bytes without a log record 0, logged but absent 0
== no staging folder
== endstate: no finding
```

and `python scripts/data_manifest.py check --root <root>` read
`checked 144 manifest files: 144 ok, 0 missing, 0 mismatched` by full hash.

**Step 3 against card 1b's published counts** (PR #538, comments 5846529836 and 5846540733):

| chunk | objects | copy | copy bytes | dup | redundant |
| --- | ---: | ---: | ---: | ---: | ---: |
| c0-bloomberg | 96 | 12 | 484,917,228 | 0 | 74 |
| c1-index_reference | 3,714 | 1,261 | 493,572,169 | 0 | 593 |
| c2-option_history_delisted | 3,678 | 950 | 100,374,792 | 0 | 884 |
| c3-option_history_deep365 | 3,372 | — | — | 700 | — (DEFERRED) |

`== the chunks match card 1b's counts`.

**Timings**, from each attempt's `01-census.txt`, `02-inventory.txt` and `05-copy.txt`:

| chunk | census (objects) | inventory | copy | verify |
| --- | --- | ---: | ---: | ---: |
| c0-bloomberg | 0.2 min (96) | 2.1 min | 1.9 min | 0.0 min |
| c1-index_reference | 3.9 min (3,714) | 0.5 min | 14.6 min | 0.1 min |
| c2-option_history_delisted | 3.7 min (3,678) | 0.5 min | 12.3 min | 0.0 min |

Download rate, by the card's rule — c0's `05-copy.txt` is nearly all download, so bytes over its
minutes: **484,917,228 B / 1.9 min ≈ 4.06 MiB/s**. For c1 and c2, `05-copy.txt` minutes less twice
their `01-census.txt` minutes gives **≈ 1.15 MiB/s** (493,572,169 B / 6.8 min) and **≈ 0.33 MiB/s**
(100,374,792 B / 4.9 min). The card budgeted 0.417 MiB/s from round 3 and about 1.5 hours; the copy
run took **37 minutes** end to end (11:00:44Z to 11:38:11Z), c3 being deferred.

**Secrets.** The one credential-shaped file in the root,
`data_processed/ibkr/flex_credentials.json`, was excluded by every inventory
(`29397 files (29396 hashed, 1 credential-shaped, not read)`) and compared by size only in the end
check. Neither the tool nor the driver opened it. No key, token or password value appeared in any
output. Card 1's credential-shaped file on Drive sits in `data_processed/ibkr`, outside all four
chunks.

**Where the record lives:** `<SWE_DATA_ROOT>\_logs\d33-card2a-i\`, outside git. Card 1's plan was
pinned at `prepare` and unchanged at the end: sha256
`d5ffbf4439ff856af372c54b4ad58a06957a4b5a02e66c716597dabcfe8f5f70`. The attempt pins:

| pin | sha256 |
| --- | --- |
| `c0-bloomberg/a1/FINGERPRINTS.txt` (the plan G1 was asked about) | `bcb833e023049af7d17638593ebd55f2f6bbc02c3853298c7362fa5ce4515e1b` |
| `c0-bloomberg/a1/FINGERPRINTS-2.txt` (its copy and verify) | `01d3713e2f0ab6d588cc9c2279c21e16b458688c7e9bb794c9576c58690ca8dc` |
| `c1-index_reference/a1/FINGERPRINTS.txt` | `e82f2146dabd529bc691065e0dfb4a0100a5b5585d0e805e5ce86a401ce563bd` |
| `c2-option_history_delisted/a1/FINGERPRINTS.txt` | `32b0905b4f582da6f278103a9e2a6d169a39290ecb71e4c271d3dae7b2b93bef` |

The driver ran only the tool of git blob `051b83ee0e25e7e058724ced02e22935005a85f7`, the one cards 1
and 1b used, and was the only caller of it.

**c3's `DEFERRED` line, verbatim:**

```
700 of its duplicates repeat files this card does not bring home: data_processed/theta/option_history (700)
```

## Unresolved / handoff

- **c3-option_history_deep365 waits for 2a-ii.** 176 files, 109,685,286 B, plus the 700 duplicates
  whose twins live in `option_history`. Once 2a-ii has brought `option_history` home, c3's own plan
  will resolve those 700 as `redundant` and it can run. Re-running `prepare` in a later card will
  re-derive the `DEFERRED` mark from card 1's plan, so nothing here needs undoing first.
- **2a-ii is still blocked on the tool.** It covers `option_history` and its banded backup, by
  ticker, and the tool as it stands refuses `=` in an area name (`ticker=AAPL`). That change is the
  gate on the rest of the Theta collection.
- **The harness cannot carry a ~23 KB Bash heredoc.** Any later card that writes a driver this way
  will hit the same wall. Either split the driver into more, smaller halves, or write the file with
  the editing tool and keep the sha256 gate as the check — the gate is what makes either safe.
- **`stash@{0}` is still on this checkout** — the stray paste in `engine/features/dynamics.py`, set
  aside at the Operator's instruction, not dropped. It is nobody's open work as far as this run
  could tell, but it is not the Executor's to discard.
- **Nothing was deleted, anywhere** — not in the root, not in the staging folder (there is none),
  not on Drive. Card 3 still owns every deletion.
