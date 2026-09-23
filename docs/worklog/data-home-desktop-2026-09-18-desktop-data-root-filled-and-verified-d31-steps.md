---
id: data-home-desktop-2026-09-18
title: Desktop data root filled and verified (D31 steps 2-3)
kind: verification
status: completed
terminal: desktop
pr:
decisions: [D31]
date: 2026-09-23
headline: desktop root filled from git and proved 99 ok / 0 missing / 0 mismatched; engine reads it; fast lane 3192 passed with 0 requires_data skips; day-bot ticks backed up to Drive (0 differences, 15 files)
surface: [data/DATA_MANIFEST.json, scripts/data_manifest.py, engine/paths.py, docs/DATA_INVENTORY.md, docs/DATA_POLICY.md]
---

## Goal

Execute D31 steps 2 and 3 on the operator's main Windows desktop: create a data
root outside the checkout, materialize every one of the 99 manifest files into
it from the git objects, prove it byte-for-byte, move the local-only stores
under it, point the engine at it with `SWE_DATA_ROOT`, and prove the engine and
the test lane against it. The sandbox had already proved the same tools on a
scratch copy (PR #526, merged as `b7bcd38`); nothing had been run on this
desktop. Step 3's `check` line is the artifact D31 step 5 (untracking the data
from git) is held on.

**Date note.** The task card, its slug, and the log filename all carry
`2026-09-18` — the date of the Operator's D31 ruling and of the card. The run
itself happened on **2026-09-23**. The card's identifiers are kept verbatim so
the Strategist finds what it expects; the dates below are the real ones.

## What we tried

Straight execution of the card's ten steps, in order, each gate checked before
the next. The only judgement calls were in step 6 (which stores to move) and
step 9 (Drive backup); both are written up below.

## What worked

1. **Root creation.** `C:\Users\merty\Desktop\swe-data`, a sibling of the
   checkout on the same volume (so step 6's moves are renames, not copies).
   Confirmed `C:\Users\merty\Desktop` is **not** OneDrive-redirected (the shell
   folder is the literal path; OneDrive lives at `C:\Users\merty\OneDrive`), so
   1.5 GB of data does not enter a sync mount. 1122 GB free.
2. **Fetch, dry-run, materialize, check, census** — all gates hit their expected
   output first time (verbatim in Evidence).
3. **Move of the local-only stores** — 50 files / 218,544,446 bytes, count and
   byte total identical before and after. Nothing overwritten, nothing deleted,
   git tree still clean.
4. **`SWE_DATA_ROOT`** set at User scope; all ten `engine/paths.py` accessors
   resolve under the root.
5. **The §9.4 smoke and the fast lane ran against the root**, with **zero**
   `requires_data` skips — the data-backed tests executed rather than skipping,
   which is the point of the root.
6. **Step 9, on the second attempt.** After the Operator approved the OAuth
   consent, the 15 day-bot tick files were copied to `swe-local-only/ticks` and
   verified `0 differences found · 15 matching files`. This is the first backup
   those 491 MB have ever had, and it retires the "only copy anywhere" status
   that `DATA_INVENTORY.md` §C row B′ records for them.

## What didn't

**The fast lane has 2 failures (3192 passed). Both are pre-existing,
Windows-only test artifacts in the D31 test files themselves. Neither is caused
by this run, by the fill, or by `SWE_DATA_ROOT`.** Both tests are fully
self-contained (`tmp_path` plus monkeypatched env); neither reads the real root.
Three independent proofs, all in Evidence:

- both fail **identically with `SWE_DATA_ROOT` unset**;
- **CI is green on Ubuntu for the exact commit `b7bcd38`** that contains them;
- the Windows mechanism for each is demonstrated directly.

`tests/test_data_manifest.py::test_materialize_fills_an_empty_root_and_verifies`
— the fixture's `_make_root` writes the CSV with `Path.write_text`, which on
Windows emits **CRLF** (38 bytes on disk vs 36 LF-only), so `build` hashes CRLF
bytes. The fixture then commits it into a throwaway git repo that has **no
`.gitattributes`**, where the system-level `core.autocrlf=true` normalises the
blob to **LF**. `materialize` writes the LF blob and the sha256 disagrees, so
the command returns 1. The two `.gz` fixtures use `write_bytes` and are
unaffected — exactly matching the observed "wrote 2 ... 1 failed" on the `.csv`
alone.

`tests/test_data_paths.py::test_narrow_overrides_win_and_are_themselves_rerooted`
— the test asserts POSIX semantics: `Path("/abs/ibkr").is_absolute()` is
**`False`** on Windows (root but no drive), so `paths._override` treats it as
relative and re-roots it; `parts` is `('\\','abs','ibkr')` and `joinpath` on a
leading separator resets to the drive root, giving `C:\abs\ibkr` instead of
`/abs/ibkr`.

The real repository is **not** exposed to the CRLF mechanism: its
`.gitattributes` pins `* text=auto eol=lf`, and the checkout's working-tree copy
of `data/bloomberg/sp500_ohlcv.csv` hashes **identically** to the manifest
(`23c1dd8b...`, 63,446,582 bytes). The manifest was generated from the git
objects, and `materialize` streams raw blob bytes, so the two agree by
construction.

**Step 9 (Drive backup of the ticks) was BLOCKED, then unblocked by the Operator
and COMPLETED.** First attempt failed: `rclone` v1.74.4 is installed and a
`gdrive:` remote exists, but its OAuth token returned `invalid_grant` and the
remedy — `rclone config reconnect gdrive:` — is an interactive browser consent
an agent cannot give. The Operator authorised the re-auth and approved the
consent screen in the browser; the flow then ran from this session and the copy
and its verification completed. **The day-bot ticks now have a second,
checksum-verified copy** (evidence below), which removes the *ticks* half of the
D31 step-6 precondition. Gap B (the deep branch's unique files) still blocks
step 6.

**Not moved in step 6, with reasons** — the card lists these; they are absent,
or moving them would have broken an invariant:

| Named in the card | Disposition |
|---|---|
| `data_processed\theta`, `\sim`, `\corporate_actions`, `\edgar` | **Absent** on this desktop — it is not the data laptop. Nothing to move. |
| `data\features\<group>\ticker=*` | Present, but **all eight are the git-tracked AAPL sample**, and all ten `features` files are already in the manifest and were materialized into the root. Moving them would have deleted tracked files from the working tree and hit the "destination already exists" gate. Left in place. |
| `data_processed\trade_universe` | Tracked **and** in the manifest (already at the root); not in the card's move list. Left in place. |
| `data_processed\.gitkeep` | Tracked; not a `*.json` / `*.parquet`. Left in place. |

## How we fixed it

Nothing needed fixing in the repository — the two failures are pre-existing
upstream test artifacts, and this card owns only the worklog. They are reported
here and handed to the Strategist rather than patched: the fix touches
`tests/test_data_manifest.py` / `tests/test_data_paths.py`, outside this card's
`owns`. Suggested fixes, for whoever takes them:

- `test_data_manifest.py::_make_root` — use `write_bytes` (or
  `write_text(..., newline="\n")`) so the fixture is byte-stable on Windows;
- `test_data_paths.py` — assert against a value that is absolute on the host
  (e.g. `C:\abs\ibkr` on Windows), or mark the POSIX-absolute assertion
  `skipif(os.name == "nt")`.

Both are test-only and neither touches the decision-layer trio.

## Evidence

All commands run from `C:\Users\merty\Desktop\smart-wheel-engine` on `main` @
`b7bcd38` (branch `claude/data-home-desktop-check`, identical tree), Python
3.12.10, pandas 2.3.3. `$ROOT = C:\Users\merty\Desktop\swe-data`.

**Step 0 — bring-up**

```
b7bcd38 Merge pull request #526: feat(data-home): desktop data root, checksum manifest, materialize-from-git (D31 steps 1-4)
pandas 2.3.3
engine.paths OK; data_root= None
```

**Step 2 — the two data-only branches**

```
 * branch            deep-history/bloomberg-raw -> FETCH_HEAD
 * branch            claude/daybot-bloomberg-pull -> FETCH_HEAD
68a48b245cea285d98f86ab7e6ba1b5cdca7002d  branch 'deep-history/bloomberg-raw'
2abf850d76b3de9a1e04301d333485e64061f3cd  branch 'claude/daybot-bloomberg-pull'
```

Both match the manifest's `git_sources` exactly; the third source,
`git:main` = `69bf3b93...`, is an ancestor of `main`.

**Step 3 — dry run**

```
materialize 99 manifest files: 0 already present, would write 99, 0 mismatched (kept, not overwritten), 0 unavailable, 0 failed
```

**Step 4 — materialize**

```
materialize 99 manifest files: 0 already present, wrote 99, 0 mismatched (kept, not overwritten), 0 unavailable, 0 failed
```

**Step 5 — the proof (the line D31 step 5 is held on)**

```
root: C:\Users\merty\Desktop\swe-data
checked 99 manifest files: 99 ok, 0 missing, 0 mismatched
```

Exit code 0, full sha256 (not `--size-only`).

```
root: C:\Users\merty\Desktop\swe-data
group         manifest   present   size-ok        MB
bloomberg           22        22        22     264.6
broad_pull          27        27        27     297.0
deep                13        13        13     373.4
features            10        10        10       2.7
processed            1         1         1       0.2
raw                 11        11        11       2.4
ticks               15        15        15     490.7
missing overall: 0
```

**Step 6 — what moved** (`Move-Item`, same volume; pre-move and post-move census
identical)

| Moved to `$ROOT\data_processed\` | files | bytes |
|---|---:|---:|
| `ibkr\` (incl. `flex_credentials.json`, `uploads\`, `_backup\`) | 5 | 1,894,362 |
| `option_premium\` | 15 | 192,504,162 |
| `validation\` | 23 | 23,287,908 |
| `__btcompare_s32_friction_24t_1m.json` | 1 | 2,772 |
| `__btcompare_s35_oos_24t_100k.json` | 1 | 1,849 |
| `_inventory_scan.json` | 1 | 18,624 |
| `_schema_dump.json` | 1 | 62,485 |
| `theta_capabilities.json` | 1 | 9,348 |
| `vol_indices.parquet` | 1 | 558,775 |
| `vol_indices_wide.parquet` | 1 | 204,161 |
| **total** | **50** | **218,544,446** |

After the move the checkout's `data_processed/` holds only its two tracked
entries (`.gitkeep`, `trade_universe/`), and `git status --porcelain` is empty.
`flex_credentials.json` moved with its folder to the root and is not tracked.

```
root: C:\Users\merty\Desktop\swe-data
checked 99 manifest files: 99 ok, 0 missing, 0 mismatched
extra data files under the root not in the manifest: 49
```

49 EXTRA = the 50 moved files minus `_inventory_scan.json`, which is in the
tool's `SKIP_NAMES`. Root totals reconcile exactly: **150 files /
1,649,399,086 bytes** = 99 manifest (1,430,853,322) + 50 moved (218,544,446) +
the fast-lane log.

**Step 7 — the root is known**

```
User-scope value now: C:\Users\merty\Desktop\swe-data
C:\Users\merty\Desktop\swe-data
C:\Users\merty\Desktop\swe-data\data\bloomberg
C:\Users\merty\Desktop\swe-data\data_processed\theta
```

**Step 8 — the engine on the root** (OPERATING_MODEL.md §9.4)

```
connector: MarketDataConnector
ticker  ev_dollars     iv  premium  prob_profit  strike
  MSFT      110.13 0.3992    7.662       0.8571   363.0
   XOM       80.14 0.2723    1.768       0.8857   130.0
  AAPL      -12.61 0.2657    3.971       0.8286   294.0
   UNH      -20.69 0.3339    6.925       0.9143   399.5
nulls per column: {'ev_dollars': 0, 'iv': 0, 'premium': 0, 'prob_profit': 0, 'strike': 0}
missing from ranking: ['JPM']
drops_summary: {'total_dropped': 1, 'by_gate': {'event': 1}}
drops detail: [{'ticker': 'JPM', 'gate': 'event', 'reason': 'event_lockout:earnings@2026-10-13 (+/-5d buffer)'}]
```

All five tickers accounted for: four rank with non-null `ev_dollars` / `iv` /
`premium`, one drops on a **named** gate. Healthy per §9.4.

Two staleness warnings printed, both pre-existing properties of the data (the
frontier is 2026-07-02), not of this run: the OHLCV frontier is 83 days behind
the wall clock, and the earnings-calendar overlay snapshot (`asof=2026-07-03`)
is 82 days old.

The EV figures are **identical** to the sandbox's (110.13 / 80.14 / -12.61 /
-20.69) even though this machine has the option-premium rail. That is the
expected result, not a coincidence: the rail's accessor is dormant and off the
EV path, so its presence cannot move a ranking.

```
tests\test_deep_read_connector.py ..........                             [100%]
============================= 10 passed in 24.53s =============================
```

**Step 8 — the fast lane**

```
= 2 failed, 3192 passed, 22 skipped, 8 deselected, 20 xfailed, 171 warnings in 504.63s (0:08:24) =
```

```
requires_data: no data root  -> count = 0
any 'requires_data' mention  -> count = 0
```

Full log: `$ROOT\fastlane_desktop_2026-09-18.log` (local only — `.log` is in the
manifest tool's `SKIP_SUFFIXES`, so it never appears as an EXTRA).

The 22 skips are all environmental and named: **8** gated on
`SWE_DEEP_TEST_DATA` (5 survivorship-harness, 1 R6-Lehman, 2 deep-IV sentinel),
12 `ThetaTerminal not running on 127.0.0.1:25503`, 1 `SWE_LIVE_PREFLIGHT`, 1
Theta option-history larder. **None** is a `requires_data` / "no data root" skip.

**Bonus — the 8 `SWE_DEEP_TEST_DATA` skips are now unnecessary, and running them
proves the deep slices *functionally*.** That gate wants a directory containing
`deep/sp500_vol_iv_full__1994_2012.csv.gz` — i.e. a Bloomberg dir — and the root
now has one. Pointing it at `<root>\data\bloomberg` **for one process only**
(User and Machine scope deliberately left unset):

```
SWE_DEEP_TEST_DATA=C:\Users\merty\Desktop\swe-data\data\bloomberg
sentinel gate file present: True
tests\test_deep_iv_sentinel.py ..                                        [ 16%]
tests\test_survivorship_harness.py .........                             [ 91%]
tests\test_survivorship_r6_lehman.py .                                   [100%]
======================= 12 passed in 111.99s (0:01:51) ========================
```

This is the strongest functional evidence in the run for the **deep** group (13
files, 373 MB): the checksum proof says the bytes are right, and these 12 tests
say the engine can actually read and reason over them from the root — the
sentinel-nulling invariant and the survivorship/R6-Lehman proofs all hold. It
goes beyond the card, changes no persistent state, and is reported as an extra.

Proof the 2 failures are pre-existing and Windows-only:

```
SWE_DATA_ROOT in this process: '' (expect empty)
FAILED tests/test_data_manifest.py::test_materialize_fills_an_empty_root_and_verifies
FAILED tests/test_data_paths.py::test_narrow_overrides_win_and_are_themselves_rerooted
============================== 2 failed in 1.09s ==============================
```

```
{"conclusion":"success","displayTitle":"Merge pull request #526: feat(data-home): desktop data root, checksum...","headSha":"b7bcd38e886148a1ec8dc8af9f540755a11ab7f4","workflowName":"CI"}
```

```
write_text bytes on disk: b'date,ticker,close\r\n2026-01-02,AAPL,1\r\n'
contains CRLF: True | len on disk: 38 vs LF-only len: 36

Path('/abs/ibkr').is_absolute() on Windows: False
parts: ('\\', 'abs', 'ibkr')
root.joinpath(*parts) -> C:\abs\ibkr
```

**Step 9 — Drive backup of the ticks: first BLOCKED, then DONE**

First attempt, before the re-auth:

```
rclone v1.74.4
--- listremotes ---
gdrive:
```

```
ERROR : error listing: couldn't list directory: ... couldn't fetch token:
invalid_grant: maybe token expired? - try refreshing with "rclone config reconnect gdrive{...}:"
```

Not retried at the time — a deterministic credential expiry whose remedy is an
interactive browser OAuth. The Operator then authorised the reconnect and
approved the consent screen. `rclone.conf` was backed up first
(`rclone.conf.bak-2026-09-23-preauth`, 684 B) so the step was reversible.

```
2026/09/23 10:13:05 NOTICE: Waiting for code...
2026/09/23 10:35:31 NOTICE: Got code
```

Token replaced — `rclone.conf` sha256 `9518CEE6…` → `5AF7596D…` (the file is the
same 684 bytes, so size alone would have been a false negative; the hash is the
proof). The remote then worked, listing the nine `swe-local-only` children whose
IDs match the §C.1 record exactly. A tenth was created:

```
rclone mkdir gdrive:ticks --drive-root-folder-id 1JwPWszfyggUDT1vYaRjZ8nlHEDR3vEOn
ticks   1wnhr4kLZt6FBpuhUCSbc6hk7Igz7JFop
```

Copy (explicit-ID addressing, empty remote path — the §C.1 pattern):

```
2026/09/23 10:41:20 INFO  : SPY_ticks_2026-06-17.csv.gz: Copied (new)
2026/09/23 10:41:20 INFO  :   467.986 MiB / 467.986 MiB, 100%, 1.509 MiB/s, ETA 0s
copy EXIT=0

rclone size gdrive: --drive-root-folder-id 1wnhr4kLZt6FBpuhUCSbc6hk7Igz7JFop
Total objects: 15
Total size: 467.986 MiB (490719019 Byte)
```

490,719,019 B is byte-exact against the manifest's `ticks` group total. Proof:

```
rclone check <root>\data_raw\bloomberg\ticks gdrive: \
  --drive-root-folder-id 1wnhr4kLZt6FBpuhUCSbc6hk7Igz7JFop --checksum --one-way
2026/09/23 10:41:37 NOTICE: Google drive root '': 0 differences found
2026/09/23 10:41:37 NOTICE: Google drive root '': 15 matching files
EXIT=0
```

This is Drive-vs-local **MD5** identity — independent of the manifest's sha256,
exactly as §C.1 notes of the other Tier-C rows. Uplink measured ~1.5 MiB/s, far
better than the ~1 Mbps the 2026-07-22 record assumed. **Row B′ of
`DATA_INVENTORY.md` §C ("Backup: none yet") and the §C.1 table are now stale and
need a `ticks` row — flagged, not edited: both are outside this card's `owns`.**

**Independent verification (the circle broken).** `materialize` wrote the files
and `check` verified them against the *same* manifest, so the proof is circular
if the manifest itself is wrong. A separate read-only audit re-derived byte
identity **without** `scripts/data_manifest.py`: hashing each root file with its
own `hashlib` and comparing against `git cat-file blob <commit>:<path>` for the
matching source commit.

- **All 99 rows are byte-identical three ways** — disk == manifest sha256 ==
  git blob — totalling exactly **1,430,853,322 bytes**, zero mismatches,
  covering all 7 groups and all 3 source commits (largest tick file
  `SPY_ticks_2026-06-11.csv.gz` 49,630,842 B and largest deep slice
  `sp500_iv_surface__2019_2026.csv.gz` 72,500,413 B included). A second,
  independent hashing pass over all 99 reproduced the same total and the same
  zero mismatches.
- **No CRLF corruption**, despite `core.autocrlf=true`: `.gitattributes` pins
  `* text=auto eol=lf`, `git cat-file blob` applies no smudge filter, and disk
  bytes equal blob bytes for all 99.
- **The engine reads the root by path, not by result.** Proven by instrumenting
  `builtins.open` / `pandas.read_csv`: with the variable in-process, the AAPL
  OHLCV bytes came from `...\swe-data\data\bloomberg\sp500_ohlcv.csv` with
  **zero** opens under the checkout; `WheelRunner`'s connector carries
  `_data_dir = C:\Users\merty\Desktop\swe-data\data\bloomberg`.
  `engine/data_connector.py` imports `engine.paths` and calls
  `paths.resolve(data_dir)`. This proof shape was necessary: the checkout still
  holds a byte-identical tracked copy of every file the connector reads, so a
  run *without* the variable succeeds silently and is indistinguishable by any
  output. Results can never prove which tree was read; resolved paths can.
- **`SWE_DATA_ROOT` is set at User scope only** — `Process` and `Machine` are
  empty. Shells already open (including this session's) do **not** see it and
  must inject it inline; newly spawned shells will.

### Two gaps the audit found that `check` is structurally blind to

`check` iterates **manifest rows only**. There is no code path that reports a
file which is *needed but unlisted*, so "0 missing" is a statement about the
manifest, not about completeness. Two real gaps follow, both verified here with
direct commands. Neither affects steps 2–3; **both block later D31 steps.**

**Gap A — 16 git-tracked data files have no manifest row and are not at the
root.** Every `data/features/<group>/ticker=AAPL/{metadata.json,stats.json}`
pair, for all 8 groups. 87 data-shaped paths are tracked under
`data/`+`data_raw/`+`data_processed/`; the manifest holds 71 from `git:main`;
the 16 sidecars are the difference. Measured consequence: with the root active,
`FeatureStore.get_metadata('technical','AAPL')` and `get_stats(...)` return
**`None`**; against the checkout they return a full `FeatureMetadata`
(`row_count=2065`, 30 columns, `version=5`) and a 30-entry stats list. **D31
step 5's `git rm` would delete the only copies**, and `materialize` could never
restore them.

**Gap B — 7 data paths exist only on `origin/deep-history/bloomberg-raw`**: not
on `main`, not in the manifest, not at the root. One
(`sp500_iv_history.csv`, 20 B) is the empty stub D28 retired. The other six
carry history `main` does not have:

| Branch-only file | Bytes | Unique vs the `broad_pull` successor on `main` |
|---|---:|---|
| `rates_fx_vol.csv` | 583,872 | **MOVE from 1988-04-04** vs `sp500_vol_indices.csv` MOVE from 2004-01-01 (~16 yr); **JPMVXYG7 from 1992-06-01** — no `broad_pull` column exists at all; CVIX from 2001-08-29 vs 2004 |
| `vol_indices.csv` | 943,706 | **SKEW from 1990-01-02** (~14 yr), VXN from 2001-02-02 (~3 yr) vs `broad_pull` start 2004-01-01 |
| `vix_futures_curve.csv` | 790,646 | UX1–UX7 from **2004-03-26** vs `broad_pull` 2006-01-03 (~1.75 yr) |
| `sp500_short_interest.csv` | 2,427,311 | carries `short_interest_pct_float` / `float_pct` / `shares_out` — columns `DATA_INVENTORY` §6E records as **entitlement-blocked** in the broad pull (`broad_pull` is wider in time, narrower in columns) |
| `sp500_macro_calendar.csv` | 16,448 | 16 KB vs `broad_pull`'s 36 KB, different schema — **likely superseded** |
| `spx_correlation.csv` | 352,416 | both start 2006-01-03 — **superseded** |

**Deleting that branch (D31 step 6) would permanently destroy the only copies of
pre-2004 MOVE, SKEW and JPMVXYG7 history.**

The 17 data files `main` and the branch *share* but disagree on are not a loss:
`main` is the refreshed side on 16 of 17. The exception is
`sp500_vol_iv_full.csv` (branch 99,119,562 B vs main 63,102,578 B), where the
pre-2018 tail lives in the manifest's own deep slices
(`sp500_vol_iv_full__1994_2012`, `__2012_2018`) rather than in the monolith —
by design, per `DATA_INVENTORY` §0.

### Points the audit raised and this run settles

- *"No pre-move inventory exists, so silent loss of an untracked file cannot be
  ruled out."* A pre-move census **was** taken (in Evidence above): 50 files /
  218,544,446 bytes, byte-for-byte identical after the move. `Move-Item` within
  one volume is a rename, so contents are not rewritten. The empty
  `ibkr/_backup` directory at the root was already empty before the move — the
  pre-move count of 5 files under `ibkr/` equals the post-move count.
- *"`flex_credentials.json` is protected only by a blanket `data_processed/`
  rule."* Not so: `.gitignore:14` is `*_credentials.json`, a path-independent
  named secret rule (`git check-ignore -v scripts/flex_credentials.json` →
  `.gitignore:14:*_credentials.json`). The file has never been committed on any
  ref. Protection survives a folder reshuffle.
- *"`materialize` copied the deep slices rather than moving them."* It cannot
  write into the checkout at all — `scripts/data_manifest.py:262` writes only
  `root / f["path"]`. The checkout's gitignored `data/bloomberg/deep/` (13
  files, 357 MB, mtimes 2026-06-09/07-22) **predates this run**. The ~373 MB
  now existing in both places is pre-existing duplication, not something this
  run created, and `data/bloomberg/deep/` was not in the card's move list.

## Unresolved / handoff

1. **D31 step 3 is satisfied on this desktop.** The gating line is
   `checked 99 manifest files: 99 ok, 0 missing, 0 mismatched`. Step 5 (untrack
   the data) is unblocked on *this* criterion; its own CI per-file coverage
   floors (`engine/data_connector.py` 82.5% vs 88, `engine/wheel_runner.py`
   76.8% vs 77) still have to be settled inside that PR.
2. **D31 step 5 has a new precondition (Gap A).** Add the 16
   `data/features/*/ticker=AAPL/{metadata.json,stats.json}` sidecars to the
   manifest and materialize them into the root *before* anything is `git rm`-ed;
   otherwise the untracking PR deletes the only copies and `FeatureStore`
   metadata/stats silently become `None` on the root. A `build` against a root
   that contains them regenerates the rows; the underlying cause is that the
   manifest was generated from a tree that did not carry them.
3. **D31 step 6 had two blockers; one is now cleared.**
   - *Ticks — CLEARED.* The 15 day-bot tick files (490,719,019 B) are copied to
     `swe-local-only/ticks` (`1wnhr4kLZt6FBpuhUCSbc6hk7Igz7JFop`) and verified
     `0 differences found · 15 matching files`. `claude/daybot-bloomberg-pull`
     is no longer the only copy: the files now exist on the branch, on this
     desktop's root, and on Drive.
   - *Deep branch (Gap B) — STILL BLOCKING.* `origin/deep-history/bloomberg-raw`
     holds six data files that exist nowhere else, three of them carrying
     history no `broad_pull` panel has (MOVE 1988, SKEW 1990, JPMVXYG7 1992).
     They must be manifested and materialized — or consciously written off —
     before that branch is deleted. **Deleting it today is irreversible data
     loss.** Note this applies to the *deep* branch only; on the evidence above,
     `claude/daybot-bloomberg-pull` is now safe to delete.
   - *Doc follow-up:* `DATA_INVENTORY.md` §C row B′ still says the ticks have no
     backup, and §C.1's table has no `ticks` row. Both are now wrong. Outside
     this card's `owns`, so flagged rather than fixed.
4. **Two Windows-only test failures** are open upstream, in files this card does
   not own. They do not affect CI. Fixes proposed above.
5. **The "SWE IBKR Morning Pull" scheduled task now writes to a path this run
   emptied — Dashboard terminal, please re-point it.** The task (07:30 ET, backed
   by the separate `C:\Users\merty\swe-ops` clone) runs
   `scripts/ibkr_gateway_pull.py` with an **explicit absolute**
   `--out C:\Users\merty\Desktop\smart-wheel-engine\data_processed\ibkr\portfolio_snapshot.json`.
   Step 6 moved that directory to the root, and an explicit path is exactly what
   `SWE_DATA_ROOT` does **not** re-root. The script does
   `out.parent.mkdir(parents=True, exist_ok=True)`, so the next run will
   silently **recreate** `data_processed/ibkr/` inside the checkout and write
   there — a second, diverging copy rather than a visible failure. Either change
   `--out` to the root path or drop `--out` and let `SWE_IBKR_DATA_DIR` /
   `paths.ibkr_dir()` resolve it. Not fixed here: `swe-ops` is outside this
   card's `owns`, and `data_processed/ibkr/` plus the portfolio pipeline belong
   to the Dashboard terminal (OPERATING_MODEL.md §9.10). *(Unrelated
   pre-existing condition: its log currently ends in
   `ConnectionRefusedError: [WinError 1225]` — IB Gateway is not running.)*
6. **This desktop is not the data laptop.** No Theta corpus, no `sim`,
   `corporate_actions` or `edgar` store, and only the tracked AAPL feature
   sample — so `data_processed/theta` resolves under the root but is empty, and
   any Theta-backed work still belongs on the other machine.
7. **Consider setting `SWE_DEEP_TEST_DATA` at User scope**, now that the root
   holds the deep slices: `C:\Users\merty\Desktop\swe-data\data\bloomberg`
   unlocks 8 tests that skip today and that pass here (see Evidence). Left unset
   deliberately — the card authorised setting `SWE_DATA_ROOT` and no other
   persistent variable. Worth a one-line decision from the Strategist, since it
   turns 8 environmental skips into a standing regression gate on the deep data.
8. **The data frontier is 2026-07-02**, 83 days stale as of this run. Any
   "today" scan needs a refresh first (`docs/DATA_POLICY.md` §5).
