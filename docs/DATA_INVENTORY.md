# DATA INVENTORY — what data the engine uses, where it lives, how to verify it

_The single source of truth for the data layer. **§A–§C are the current
location map (updated 2026-09-18, DECISIONS D31).** §0–§7 that follow are the
detailed dataset census (schemas, row/date counts) as-of 2026-06-22; the daily
frontier has since advanced to **2026-07-02**. Trust the machine manifest for
exact bytes, this census for structure and coverage._

> **Companion machine file:** [`data/DATA_MANIFEST.json`](../data/DATA_MANIFEST.json)
> — every dataset the engine owns: path · size · sha256 · dataset group · the git
> object it was first taken from (144 files, 1.74 GB: generated 2026-09-18 from the
> git objects themselves, completed 2026-09-23 with the 16 feature sidecars and the
> `data_archive/` branch archive, §C.2). `python scripts/data_manifest.py check --root <root>`
> proves a root complete (exit 1 on any missing or altered byte); `materialize`
> fills a root from git; `census` shows presence by group; `build` regenerates the
> manifest after a refresh.

**Provider note.** Bloomberg prices are **split-adjusted**; Theta prices are **raw**. Never mix them.

## §A. Status — the data lives on the operator's desktop, not in git

Operator ruling 2026-09-18 (D31): every dataset, past and future, must be
accessible on the main Windows desktop; git is not a data store; Google Drive
stays the (delayed) backup, never a source of truth. **Ruling 2026-09-24 (D33):**
the MacBook is outside the data estate. Drive becomes the complete second copy:
one folder, `swe-data/`, laid out like the root and checked both ways. Old Drive
copies are removed only when proven identical (§A step 7, §C.3). `SWE_DATA_ROOT` names the
data root (`engine/paths.py` re-roots `data/`, `data_raw/`, `data_processed/`
under it; unset = the repository folder, the old behaviour — `docs/DATA_POLICY.md`
§6). The campaign runs in verified steps:

| Step | What | Status (2026-09-24) |
|---|---|---|
| 1 | Manifest of everything GitHub holds | **done** — 99 files at ruling time; **144 files / 1.74 GB** since 2026-09-23, after the desktop audit (#527) found two gaps: the 16 tracked feature sidecars (`data/features/*/ticker=AAPL/{metadata,stats}.json`) and the data that exists only on non-`main` branches (the 29-file archive, §C.2). `tests/test_data_manifest.py` now fails if any tracked data file lacks a row |
| 2 | Bring the git-held datasets onto the desktop | **done** — `C:\Users\merty\Desktop\swe-data`, 2026-09-23 (#527, then the 45 rows #528 added) |
| 3 | Verify the desktop root by checksum | **done** — `checked 144 manifest files: 144 ok, 0 missing, 0 mismatched` (2026-09-23, after #528; the first pass read 99/99 and was confirmed by an independent three-way byte audit, #527) |
| 4 | CI / sandbox posture without data (`requires_data` skips, no data root) | **done** |
| 5 | Untrack the data from git (no history rewrite) | **done 2026-09-23** — the 87 tracked data files left the index (`git rm --cached`; every earlier commit still holds them, so `materialize` can still read them from history); `.gitignore` keeps data out, and `tests/test_data_manifest.py` fails if a data file is tracked again; CI runs without data, with the two per-file floors that only held with data recalibrated to the no-data measurement (`scripts/check_coverage_floors.py`). A `git pull` of this change removes the tracked copies from a checkout's working tree — expected; the root holds them |
| 6 | Delete the four non-`main` branches | **held, under D33.** Three conditions: (a) the root checks 144/0/0; (b) Drive's `swe-data/` equals the root, checked both ways (step 7, card 2); (c) the full-history bundle restores from Drive's copy into an empty repository, with `git fsck --full` clean and the four exact commits present. Then one atomic push deletes the four, with a lease on each; it refuses if any branch has moved. Afterwards, 32 older data versions that only the branches' histories hold (not `main`'s history, not #507's retained head) live only in the bundle, on the desktop and on Drive. #507 has been closed since 2026-09-23. The ticks' Drive copy is done (2026-09-23: `swe-local-only/ticks`, 15 matching, 0 differences). Card 1 (2026-09-25) re-proved (a) and restored the desktop's copy of the bundle; (c), the restore from Drive's copy, is still open |
| 2b | The data laptop's local-only stores onto the desktop | **abandoned 2026-09-24 (D33):** "we will forget the macbook exists". The MacBook's Theta corpus (~132,862 files, ~11 GB), its feature shards and `sim` are not recovered; Theta is collected again later, from a source not yet chosen. The root holds everything Drive's `swe-local-only` held, checked on 2026-09-23 (the desktop's round 3, recorded on `claude/data-home-desktop-round3`): `theta` 17,188 files, `features` 11,858, `option_premium` 155 and `ibkr` 19. Two `ibkr` files differed, so both versions are kept, Drive's under `data_archive/drive-swe-local-only/ibkr/`. **Under review since card 1 (2026-09-25):** Drive's `SmartWheelData/data_processed/theta` (uploaded 2026-07-12/13) holds a far fuller Theta upload than `swe-local-only`, very likely the full corpus; card 1b confirms the count from the plan before this line is corrected (PROJECT_STATE §0 B). The Operator ruled on 2026-09-26 that Theta gets Bloomberg's rule: nothing that holds it is deleted until the desktop copy and the Drive copy are both proven. |
| 7 | Drive consolidation (D33): one folder, `swe-data/`, laid out like the root; the old Drive areas (§C.3) checked object by object | **approved 2026-09-24; card 1 done 2026-09-25 (#538):** the census of 7 areas, 332,772 objects, each equal to `rclone size`; the plan, 120,139 files (10,716,801,880 B) to copy home, 0 needing a byte check, 1 unresolved (a credential-shaped file, §C.3); nothing deleted, Drive unchanged. The checking tool comes first, then three desktop cards: (1) prove and plan, with no deletions and no Drive writes; (2) copy home what only Drive holds, build `swe-data/`, check it both ways and run the restore tests from Drive, with no deletions; (3) clean up the proven duplicates from a named list, with the Operator's yes |

## §B. Fill and verify the desktop root (once)

```powershell
git fetch origin deep-history/bloomberg-raw claude/daybot-bloomberg-pull backup/drive-tier-c-2026-07-22 data/drive-migration
python scripts/data_manifest.py materialize --root D:\swe-data   # creates only what is missing; verifies every byte; never overwrites
python scripts/data_manifest.py check --root D:\swe-data         # expect: checked 144 manifest files: 144 ok, 0 missing, 0 mismatched
python scripts/data_manifest.py census --root D:\swe-data        # presence by dataset group
```

`materialize` is safe to re-run after the manifest grows: files already present
and byte-identical are counted, only the new rows are written (on the desktop,
2026-09-23: `144 manifest files: 99 already present, wrote 45`).

Then move or copy the local-only stores (§C, Tier C) under the root and set
`SWE_DATA_ROOT` for the account (`docs/DATA_POLICY.md` §6).

**Since 2026-09-23 git tracks no data** (step 5): a fresh clone has none, and a
`git pull` of that change removed the tracked copies from existing checkouts.
`materialize` still fills a root from the commits the manifest names, because
untracking rewrote nothing — until a branch is deleted (step 6), after which the
desktop root, Drive and the full-history bundle are where those bytes live.

## §C. Where each dataset lives

| Tier | What | Under the data root | Git (2026-09-18) | Backup |
|---|---|---|---|---|
| **A — served** | 10 `_FILES` monoliths + `broad_pull/` panels — 49 files, 562 MB | `data/bloomberg/` | untracked 2026-09-23 (step 5); in `main`'s history | Drive `data/bloomberg` mirror (`1xpRvaQglsmcUuTKgVKHR39_3H-vbdIFh`): 48/48 sha256-verified 2026-07-21, sizes re-matched 2026-09-18 |
| **B — deep** | 13 gz slices, 1994→2026 + delisted — 373 MB (§2) | `data/bloomberg/deep/` | branch `deep-history/bloomberg-raw` @ `68a48b2` only (gitignored on `main`) | Drive `deep` (`1m_9LQNtbHzQo7MG5t3OxAINCXiwkhkna`): 13/13 present, byte-exact sizes 2026-09-18 |
| **B′ — ticks** | 15 SPY/QQQ day-bot tick files — 491 MB | `data_raw/bloomberg/ticks/` | branch `claude/daybot-bloomberg-pull` @ `2abf850` only | the desktop root (verified 2026-09-23, #527) and Drive `swe-local-only/ticks` (`1wnhr4kLZt6FBpuhUCSbc6hk7Igz7JFop`): 15/15, `rclone check --checksum --one-way` clean 2026-09-23 |
| **A′ — small samples** | `data/features` AAPL sample (26 files: 10 parquet + the 16 `metadata.json` / `stats.json` sidecars `FeatureStore` reads), `data_raw` yfinance/ohlcv/constituents (11), `data_processed/trade_universe` (1) | as named | untracked 2026-09-23 (step 5); in `main`'s history | Drive `swe-local-only/` (§C.1) |
| **R — branch archive** | every distinct data file at the tip of a non-`main` branch that no other row carries — 29 files, 312.8 MB (§C.2); read by no code | `data_archive/<branch>/<path>` | the four non-`main` branches only | Drive `swe-local-only/data_archive` (`1hCmngYyGwSHvkCT_BmF-EC-t8fkUA7xi`), together with the full-history bundle: `rclone check --checksum --one-way` 0 differences, 33 matching (2026-09-23, the desktop's round 3) |
| **C — local-only** | Theta corpus (partial), option-premium rail, feature shards, vol_indices, validation, ibkr (credentials excluded) | `data_processed/**`, `data/features/**` | never | **On the desktop root, checked 2026-09-23** (#527 and the desktop's round 3): everything Drive's `swe-local-only/` held (§C.1), including `theta` (17,188 files, 1,337,169,896 B), `features` (11,858), `option_premium` (155) and `ibkr` (19). Drive `swe-local-only/` still holds the older copies (§C.1, §C.3). **Not recovered, by D33:** the MacBook's full Theta corpus (~132,862 files, ~11 GB), its feature shards and `sim`; Theta is collected again later. `corporate_actions` and `edgar` never existed on the laptop. **Under review since card 1 (2026-09-25):** Drive's `SmartWheelData/data_processed/theta` (uploaded 2026-07-12/13) holds a far fuller Theta upload than `swe-local-only`, very likely the full corpus; card 1b confirms the count from the plan before this line is corrected (PROJECT_STATE §0 B). The Operator ruled on 2026-09-26 that Theta gets Bloomberg's rule: nothing that holds it is deleted until the desktop copy and the Drive copy are both proven. |

### §C.1 — `swe-local-only/` Tier-C backup record (root `1JwPWszfyggUDT1vYaRjZ8nlHEDR3vEOn`)

_Salvaged verbatim on 2026-09-18 from branch `backup/drive-tier-c-2026-07-22` (its only unique
content) so the branch can be deleted. The `theta` row was still uploading on 2026-07-22 and has
**not** been re-verified since; every other row was `rclone check --checksum` clean that day.
2026-09-23: that upload never finished. Drive holds 17,188 of the ~132,862 files (1.245 GiB of ~11 GB),
The desktop pulled all 17,188 home and checked them on 2026-09-23 (0 differences). The rest stays on the MacBook, which is outside the data estate (D33). **Under review since card 1 (2026-09-25):** Drive's `SmartWheelData/data_processed/theta` (uploaded 2026-07-12/13) holds a far fuller Theta upload than `swe-local-only`, very likely the full corpus; card 1b confirms the count from the plan before this line is corrected (PROJECT_STATE §0 B). The Operator ruled on 2026-09-26 that Theta gets Bloomberg's rule: nothing that holds it is deleted until the desktop copy and the Drive copy are both proven._

Copied with `rclone copy … gdrive: --drive-root-folder-id <child-id> --checksum` (explicit-ID
addressing, empty remote path) and verified with `rclone check … --checksum --one-way`
(Drive-vs-local **MD5** byte-identity — this is NOT the manifest's sha256; both proofs are
independent). Backed up / verified **2026-07-22**.

| Child (folder id) | Local source | `rclone check --checksum` |
|---|---|---|
| `theta` (`13sjqmRt389zaGi4iiA6xFSoDeRd1QzSp`) | `data_processed/theta/` | the July upload never finished: 17,188 files, 1,337,169,896 B. Pulled to the desktop root and checked 2026-09-23: ✅ 0 differences · 17,188 files |
| `option_premium` (`1s9ARxD8EDKUG_vRVdD4C-nGjGdkjNO9-`) | `data_processed/option_premium/` (1.8 GB) | ✅ 0 differences · 155 files |
| `features` (`1DFNY72PZBUcbQOxyvBX0BwPIrBCZe1A4`) | `data/features/` (~1.2 GB, 11,858 files; `_locks/`, `_backfill_log.csv`, `*.log` excluded) | ✅ 0 differences · 11,858 files |
| `vol_indices` (`1qHskhi0NOuwUuGHgQGAh6CKpbdzoE7us`) | `data_processed/vol_indices.parquet` + `_wide.parquet` | ✅ 0 differences · 2 files |
| `validation` (`1DImzxUuXxXODIG3uldKBsLx-f1-TZxCT`) | `data_processed/validation/` (22 MB) | ✅ 0 differences · 23 files |
| `data_processed_root` (`1spBVAgdZLyrLXZ7SgInrR2i7tMwhG62a`) | loose `data_processed/*.json` (incl. `_inventory_scan.json`) | ✅ 0 differences · 5 files |
| `data_raw` (`15ZGdTlLtMVr4ShIgpQq3bDw9tYrVme02`) | `data_raw/**` (git-tracked; incl. `sp500_constituents_current.csv`) | ✅ 0 differences · 11 files |
| `trade_universe` (`10JMptvhJsau459DLCH0tnJgzhwjpxwt4`) | `data_processed/trade_universe/` (git-tracked) | ✅ 0 differences · 1 file |
| `ibkr` (`1pr3fkf7zPWZxC8_sAtwdNOs8aGJujPDG`) | `data_processed/ibkr/` — **`flex_credentials.json` EXCLUDED (never uploaded)** | ✅ 0 differences · 19 files |
| `ticks` (`1wnhr4kLZt6FBpuhUCSbc6hk7Igz7JFop`) — added 2026-09-23 from the desktop | `data_raw/bloomberg/ticks/` (the 15 day-bot tick files, 490,719,019 B) | ✅ 0 differences · 15 files (2026-09-23) |

**Skipped (stated):** `data_processed/sim/` (regenerable paper-book outputs), `data_processed/.gitkeep` (empty marker).
**Absent on this laptop:** `financial_news/storage/sentiment.sqlite` (news-sentiment store — `financial_news/` holds only source code, no DB), `data_processed/{news_sentiment,corporate_actions,edgar}` (not present), `SWE_DEL_OUT`/`SWE_OUT_PATH` off-tree scratch (Windows defaults, absent on macOS).

### §C.2 — `data_archive/`: what only the other branches held (2026-09-23)

The desktop's audit (#527) showed that `check` can only vouch for rows the manifest
has, and that `deep-history/bloomberg-raw` carried data `main` never had. The rule
that closes it needs no judgement: **every distinct data file (by sha256) at the
tip of a non-`main` branch that no other manifest row carries gets a row**, at
`data_archive/<branch>/<the path it had in git>` with `git_path` naming that git
path. `materialize` writes them like any other row; nothing reads them. With them,
a root that passes `check` holds every data file any branch tip carried, so deleting
a branch cannot lose a dataset. 29 files, 312,846,180 bytes:

| Archive path (under `data_archive/`) | Bytes | Note |
|---|---:|---|
| `backup-drive-tier-c-2026-07-22/data/data_manifest.json` | 26,468 | the #507 draft's own manifest |
| `claude-daybot-bloomberg-pull/data/bloomberg/sp500_corporate_actions.csv` | 2,622,980 | an earlier or later version than `main`'s |
| `claude-daybot-bloomberg-pull/data/bloomberg/sp500_vol_iv_full.csv` | 62,135,640 | an older, longer monolith; its pre-2018 tail also lives in the `deep` slices |
| `claude-daybot-bloomberg-pull/data/bloomberg/treasury_yields.csv` | 491,813 | an earlier or later version than `main`'s |
| `data-drive-migration/data/data_manifest.json` | 24,890 | the #507 draft's own manifest |
| `deep-history-bloomberg-raw/data/bloomberg/rates_fx_vol.csv` | 583,872 | **MOVE from 1988-04-04, JPMVXYG7 from 1992-06-01** (no `broad_pull` column), CVIX from 2001 |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_analyst.csv` | 34,223 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_corporate_actions.csv` | 873,011 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_credit_risk.csv` | 21,401 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_dividends.csv` | 3,969,517 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_fundamentals.csv` | 108,585 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_historical_fundamentals.csv` | 1,361,430 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_index_membership.csv` | 981,797 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_institutional.csv` | 23,534 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_iv_history.csv` | 20 | the 20-byte stub D28 retired |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_iv_snapshot_today.csv` | 24,904 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_liquidity.csv` | 70,365,248 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_macro.csv` | 798,230 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_macro_calendar.csv` | 16,448 | older schema, likely superseded by `broad_pull/macro_calendar` |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_ohlcv.csv` | 62,443,071 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_sector_etfs.csv` | 1,569,197 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_short_interest.csv` | 2,427,311 | carries `short_interest_pct_float` / `float_pct` / `shares_out`, entitlement-blocked in the broad pull (§6E) |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_vix_full.csv` | 394,626 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/sp500_vol_iv_full.csv` | 99,119,562 | an older, longer monolith; its pre-2018 tail also lives in the `deep` slices |
| `deep-history-bloomberg-raw/data/bloomberg/spx_correlation.csv` | 352,416 | superseded (same start as `broad_pull`) |
| `deep-history-bloomberg-raw/data/bloomberg/treasury_yields.csv` | 105,769 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/vix_futures_curve.csv` | 790,646 | UX1–UX7 from **2004-03-26** (`broad_pull` 2006) |
| `deep-history-bloomberg-raw/data/bloomberg/vix_term_structure.csv` | 235,865 | an earlier or later version than `main`'s |
| `deep-history-bloomberg-raw/data/bloomberg/vol_indices.csv` | 943,706 | **SKEW from 1990-01-02**, VXN from 2001 (`broad_pull` starts 2004) |

**What the archive does not hold, and what does.** Older *versions* inside a
branch's history — 36 superseded file versions (603 MB) on
`deep-history/bloomberg-raw`, mostly the deep slices at earlier stages of the
June pull — plus the branches' scripts and docs (the day-bot's `pull_ticks.py`,
`pull_bars.py`, `pull_events.py`, `DAYBOT_PULL_MANIFEST.md`; the June session
transcripts). Those live only in git history, so before any branch is deleted the
desktop writes a full-history bundle of every branch
(`data_archive/git/<name>.bundle`, `git bundle verify` clean, its heads equal to
`git ls-remote`) and Drive gets a copy (§A step 6).

Done on 2026-09-23 (the desktop's round 3): `data_archive/git/smart-wheel-engine-all-refs-2026-09-23.bundle` (1,698,024,795 B), proved by reading data back out of it, with a copy on Drive. Counted exactly on 2026-09-24: **32** data versions are reachable only from the four branches' histories. Neither `main`'s history nor #507's retained head (`refs/pull/507/head`) holds them. After step 6 the bundle is their only copy, which is why D33 requires it to restore from Drive's copy first.

### §C.3 — Google Drive before the consolidation: four areas (2026-09-24)

_Found by the pen on 2026-09-24 through a read-only Drive connector, by searching
folder names (D33). Card 1's census (2026-09-25) lists every object in them; its
totals are below the table. Folders unrelated to this project are not recorded._

| Area | Folder (id) | What it holds | Treatment (D33) |
|---|---|---|---|
| 1 | `swe-local-only/`, top of My Drive (`1JwPWszfyggUDT1vYaRjZ8nlHEDR3vEOn`) | the Tier-C backup of 2026-07-22 (§C.1); `ticks` (2026-09-23); `data_archive/` (`1hCmngYyGwSHvkCT_BmF-EC-t8fkUA7xi`), holding the 29-file branch archive, the full-history bundle and `drive-swe-local-only/` | consolidate: copy home anything unique, then remove the proven duplicates |
| 2 | `SmartWheelData/` (`1wCFPBf0o9PJMy2f2vy34S316XFc1Sq3e`, created 2026-07-12), in The_Works › Projects › OptionsEngine_Project (`1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4`) | `data/`, with the Tier-A `bloomberg` mirror (`1xpRvaQglsmcUuTKgVKHR39_3H-vbdIFh`, its `deep` child `1m_9LQNtbHzQo7MG5t3OxAINCXiwkhkna`) and `features/`. `data_processed/` (`1vsBgmmRX0W9au30MQeEl0ha9kQld2vo7`), with a second, older `theta` upload (`1tUUfH_Ogq1RL2rubTKf5rzUWDBKIKo3i`), `option_premium`, `ibkr`, `validation` and `trade_universe`. `data_raw/`. `archive/` (`186O38w7OD0_QjPcsjGRaCOKG-0DJGPVD`), with `bloomberg_snapshot_2026-03-20`. `swe-deep-history/` (`13b5QUa-KV0fsaf0_f8N5n7ykqZ6m1Jd-`, §5), whose `sp500_vol_iv_full__1994_2026_FULL.csv.gz` (58,316,365 B) no manifest row carries. `staging/`, three checksum lists and some code files. Card 1 found two more things here. `data_processed/theta` appears to hold the full Theta corpus (under review). `data_processed/ibkr/` holds a credential-shaped `flex_credentials.json` (674 B, on Drive since 2026-07-12), never opened and shared with nobody (its sharing settings, checked 2026-09-26). The Operator rotates the Flex token and deletes this copy by hand (2026-09-26); no tool or agent opens, copies or deletes it (D33 point 5) | consolidate |
| 3 | `smart-wheel-engine/`, top of My Drive (`1dA_fq1MorvsqUWeVxqR0aAaJ9XMwEjUY`, 2026-04-23) | a partial upload of the repository's `.git` folder | consolidate, like areas 1 and 2: its files come home under `data_archive/drive-legacy/`, and the old copies go only once proven byte-identical to files in `swe-data/`. A git object is not byte-identical to the bundle that holds it, so the bundle does not count as a copy |
| 4 | `_local_archive/`, inside Projects › Day_Trading_Bot (`1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0`, 2026-07-21) | `vendor_swe_data` (`1EewDv70haKPVzjmhTDvo27lLsMjlqyzY`), `vendor_swe_data_raw` (`1_u85pi25w-H5HynRH3-WvGv1tMd8MW71`), `vendor_swe_data_processed` (`1WXeonbDMTT_VsGDizQD32Rw0V14xLxHE`): copies of this project's data, with folder dates of 2026-06-02. `data_raw` (`1uHSbrEaZoyW_Tgn1BimSz02KOJ18e616`) is the day-bot's own data: card 1 found that it matches nothing in the root, and the Operator ruled on 2026-09-26 to leave it out | read only: copy home anything unique, and delete nothing, because the folder belongs to the day-bot project. `data_raw` is left out: nothing in it is copied home (2026-09-26) |

Card 1's census and plan, 2026-09-25 (`swe-data/_logs/d33-card1/` on the desktop). Every
area's totals equal `rclone size`, and `swe-data exists: False`:

- area 1, `swe-local-only`: 29,310 files, 6,991,080,522 B. The plan has 29,308
  redundant (the root holds their bytes) and 2 copy, which are empty files.
- area 2, `SmartWheelData`: 148,788 files, 16,257,882,742 B. The plan has 118,367
  copy (10,681,538,612 B), 700 duplicate, 29,720 redundant and 1 unresolved.
- area 3, `smart-wheel-engine-git`: 32 files, 53,723 B, all copy.
- area 4, `_local_archive`, read only. `vendor_swe_data`: 73 files, 269,788,882 B.
  `vendor_swe_data_raw`: 12 files, 2,373,076 B. `vendor_swe_data_processed`: 2 files,
  166,352 B. `data_raw`: 3,306 files, 35,125,927 B, of which 1,725 copy and 1,581
  duplicate; left out by the Operator's ruling (2026-09-26).

In all, card 1's plan lists 120,139 files (10,716,801,880 B) to copy home, into
`data_archive/drive-legacy/<area>/`. Without `data_raw`'s 1,725 (34,816,381 B),
card 2 copies 118,414 files (10,681,985,499 B), subject to its fresh census.

After the consolidation (card 2), this project has one Drive folder: `swe-data/`
at the top of My Drive, laid out exactly like the desktop root. Card 2 records its
id and layout here, with the routine that keeps it current.

---

_The sections below (§0–§7) are the detailed census as-of 2026-06-22 — retained for
structure, coverage and schemas. Exact current size + sha256 per file: the manifest._

---

## 0. Reconciliation — what changed vs the prior inventory (stale → byte-true)

The previous doc claimed daily files end `2026-03-20`, treasury starts `2021-05`, and
corporate-actions is a 2-byte stub. `origin/main` has since been refreshed/backfilled.
Corrected deltas (all byte-verified 2026-06-22):

| File | Prior inventory (stale) | `origin/main` (byte-true) |
|---|---|---|
| `sp500_ohlcv.csv` | 988,809 · 2018→**2026-03-20** · 503 nm | **1,014,920 · 2018-01-02→2026-06-04 · 511 nm** |
| `sp500_vol_iv_full.csv` | 1,361,615 · **2015**→2026-03-20 · 503 | **1,037,278 · 2018-01-02→2026-06-04 · 510** (pre-2018 IV now in deep slices) |
| `sp500_liquidity.csv` | 1,362,737 · 2015→2026-03-20 · 503 | **1,388,848 · 2015-01-02→2026-06-04 · 511** |
| `treasury_yields.csv` | 1,254 · **2021-05-07**→2026-05-05 | **8,458 · 1994-01-03→2026-06-05** (full curve; `rate_1m` blank pre-2001) |
| `sp500_corporate_actions.csv` | **EMPTY 2-byte stub** (0 rows) | **52,442 rows · 1962-05-23→2026-06-05 · 481 nm** |
| `sp500_historical_fundamentals.csv` | 30,347 · 2015→2026-02-28 | **79,198 · 1990-01-01→2026-05-10** |
| `sp500_index_membership.csv` | 22,690 · 2015→2026-01-01 | **72,696 · 1990-04-01→2026-04-01** |
| `sp500_macro.csv` | 17,320 · 2015-01-01→2026-03-20 | **56,180 · 1990-01-02→2026-06-04** |
| `sp500_sector_etfs.csv` | 29,954 · 2015→2026-03-20 | **66,824 · 1998-12-22→2026-06-05** |
| `vix_term_structure.csv` | 2,094 · 2018→2026-03-20 | **9,200 · 1990-01-02→2026-06-04** |
| `sp500_vix_full.csv` | 16,955 · →2026-03-20 | **17,274 · 2015-01-02→2026-06-05** |
| `sp500_vol_dvd.csv` | 988,837 · →2026-03-20 | **988,837 · →2026-03-20 — UNCHANGED (laggard; not refreshed)** |
| Theta `option_history/` | ≈185.2M rows · 70 nm | **390,119,692 rows · 154 nm** (pull grew to completion) |

> The broad-pull **currency refresh** (§6) carries each refreshed daily series further to
> **2026-06-18** (latest bar = today, gate-confirmed) — staged, not yet integrated.

---

## 1. Bloomberg — monolith CSVs (`data/bloomberg/`; tracked on `origin/main` until 2026-09-23)

Universe ≈ 503–511 current S&P 500 names. "Date field" names the column the range is read
from (event tables key on ex/announce/as-of dates, not a daily `date`).

| File name | Type / title | Date range (verified) | Rows | Names | Date field |
|---|---|---|---|---|---|
| `sp500_ohlcv.csv` | Daily equity **OHLCV** (split-adj) | 2018-01-02 → 2026-06-04 | 1,014,920 | 511 | `date` |
| `sp500_vol_iv_full.csv` | Daily **implied + realized vol** (put/call IV, RV 30/60/90/260d) | 2018-01-02 → 2026-06-04 | 1,037,278 | 510 | `date` |
| `sp500_vol_dvd.csv` | Daily **vol + dividend-yield** panel | 2018-01-02 → **2026-03-20** ⚠️ | 988,837 | 503 | `date` |
| `sp500_liquidity.csv` | Daily **liquidity** (avg vol / turnover / shares out) | 2015-01-02 → 2026-06-04 | 1,388,848 | 511 | `date` |
| `sp500_historical_fundamentals.csv` | **Fundamentals** time series | 1990-01-01 → 2026-05-10 | 79,198 | 503 | `date` |
| `sp500_macro.csv` | **Macro** indicators (daily) | 1990-01-02 → 2026-06-04 | 56,180 | — | `date` |
| `sp500_sector_etfs.csv` | **Sector-ETF** OHLC (daily) | 1998-12-22 → 2026-06-05 | 66,824 | — | `date` |
| `sp500_vix_full.csv` | **VIX** full history (daily) | 2015-01-02 → 2026-06-05 | 17,274 | — | `date` |
| `vix_term_structure.csv` | **VIX term structure** (daily) | 1990-01-02 → 2026-06-04 | 9,200 | — | `date` |
| `treasury_yields.csv` | **Treasury yield** curve (daily, full tenor + sofr) | 1994-01-03 → 2026-06-05 | 8,458 | — | `date` |
| `sp500_dividends.csv` | **Dividend events** (declared/ex/record/pay + amount) | ex-date 1962-05-31 → 2027-03-12 (fwd-declared) | 50,230 | 427 | `ex_date` |
| `sp500_earnings.csv` | **Earnings** (EPS actual/est + announce date) | 1980-01-31 → 2028-01-19 (fwd-est) | 49,379 | 503 | `announcement_date` |
| `sp500_earnings_yf.csv` | Earnings (yfinance backfill) | 2008-01-17 → 2026-08-24 | 12,242 | 498 | `announcement_date` |
| `sp500_corporate_actions.csv` | **Corporate actions** (splits/M&A/rights) | announce 1962-05-23 → 2026-06-05 | 52,442 | 481 | `announcement_date` |
| `sp500_index_membership.csv` | **Index membership / weights** | as-of 1990-04-01 → 2026-04-01 | 72,696 | — | `as_of_date` |
| `sp500_analyst.csv` | **Analyst** ratings / targets | point-in-time snapshot | 503 | 503 | — |
| `sp500_credit_risk.csv` | **Credit risk** (Altman-Z, S&P rating) | snapshot | 503 | 503 | — |
| `sp500_fundamentals.csv` | **Fundamentals** snapshot (GICS, PE, beta, dvd yld…) | snapshot | 503 | 503 | — |
| `sp500_fundamentals_yf.csv` | Fundamentals snapshot (yfinance) | snapshot | 503 | 503 | — |
| `sp500_institutional.csv` | **Institutional / float** snapshot | snapshot | 503 | 503 | — |
| `sp500_iv_snapshot_today.csv` | Single-day **IV snapshot** (30/60d ATM) | snapshot | 503 | 503 | — |

> **Laggard flag:** `sp500_vol_dvd.csv` alone still ends **2026-03-20** — every other daily
> panel was refreshed to 06-04/06-05. The broad-pull currency refresh does **not** include a
> `vol_dvd` tail (manifest: `vol_iv` ATM refresh "N/A — current via skew-surface 100%MNY col"),
> so this remains the one un-refreshed daily monolith. Wiring/refresh consumers should treat it
> as the stale series.

> **Non-canonical larder assets (6).** Six of the tracked CSVs above are read by
> **neither** the connector's `_FILES` map (`engine/data_connector.py:122-132`)
> **nor** `data/consolidated_loader.py` — they are regeneratable side-panels,
> yfinance parallels, or point-in-time snapshots kept as a local larder, not live
> engine inputs: `sp500_historical_fundamentals.csv`, `sp500_sector_etfs.csv`,
> `sp500_earnings_yf.csv`, `sp500_fundamentals_yf.csv`, `sp500_institutional.csv`,
> and `sp500_iv_snapshot_today.csv`. (The empty legacy `sp500_iv_history.csv` was
> a 7th until it was retired in the 2026-07 D28 dead-code pass.) Each has a
> `scripts/pull_*` producer but **no live consumer**; treat them as reference /
> backfill stock,
> not part of the EV data path. (By contrast `sp500_vol_dvd`, `sp500_macro`,
> `sp500_vix_full`, `sp500_index_membership` and `sp500_analyst` *are* consumed
> by `consolidated_loader.py`, so they are canonical for the consolidated / deep
> path even though they are outside `_FILES`.)

---

## 2. Bloomberg — deep-history archive (`data/bloomberg/deep/` under the data root; D31)

Held on git branch `deep-history/bloomberg-raw` until step 6 (`materialize` copies it onto the
desktop, §B); Drive `deep` folder `1m_9LQNtbHzQo7MG5t3OxAINCXiwkhkna` holds all 13 (byte-exact sizes 2026-09-18). Gzipped CSV. **Dated slices** = current S&P names,
split-adj; **`__delisted`** slices = survivorship-complete (~1,000+ tickers incl. dead
names, back to 1990). Byte-confirmed 2026-06-22 (unchanged from the prior pass).

| File name | Type / title | Date range (verified) | Rows | Names |
|---|---|---|---|---|
| `sp500_ohlcv__1994_2018.csv.gz` | Deep **OHLCV** | 1994-01-03 → 2017-12-29 | 2,083,270 | 449 |
| `sp500_ohlcv__delisted.csv.gz` | OHLCV incl. **delisted** | 1990-01-02 → 2026-06-05 | 2,383,622 | 1,015 |
| `sp500_vol_iv_full__1994_2012.csv.gz` | Deep **vol/IV** (byte-identical to Drive copy, 28,297,508 B) | 1994-01-03 → 2012-06-29 | 1,661,191 | 436 |
| `sp500_vol_iv_full__2012_2018.csv.gz` | Deep **vol/IV** | 2012-07-02 → 2017-12-29 | 630,743 | 476 |
| `sp500_vol_iv__delisted.csv.gz` | Vol/IV incl. **delisted** | 1990-01-02 → 2026-06-05 | 2,408,183 | 1,016 |
| `sp500_liquidity__1994_2015.csv.gz` | Deep **liquidity** | 1994-01-03 → 2014-12-31 | 1,987,751 | 457 |
| `sp500_liquidity__delisted.csv.gz` | Liquidity incl. **delisted** | 1990-01-01 → 2026-06-05 | 2,393,425 | 1,011 |
| `sp500_iv_surface__2005_2011.csv.gz` | **IV moneyness/skew surface** (5 tenor × 5 mny) | 2005-01-03 → 2011-12-30 | 685,310 | 430 |
| `sp500_iv_surface__2012_2018.csv.gz` | IV surface | 2012-01-03 → 2018-12-31 | 795,760 | 470 |
| `sp500_iv_surface__2019_2026.csv.gz` | IV surface | 2019-01-02 → 2026-06-04 | 912,802 | 501 |
| `delisted_status.csv` | Delisting-status map | (no date col) | 1,016 | 1,016 |
| `ohlcv_dropped_ticks.csv` | Gate audit — dropped bad ticks | 1994-01-21 → 2009-01-22 | 97 | 32 |
| `ohlcv_dropped_ticks__delisted.csv` | Gate audit (delisted) | 1990-03-26 → 2006-12-18 | 51 | 37 |

> The on-disk `sp500_iv_surface__*` deep slices (2005→2026) are the historical companion to
> the **staged fresh** 5×5 surface in §6 (`iv_surface/sp500_iv_surface.csv.gz`, 2010→06-17).

---

## 3. Theta — option/market data (`data_processed/theta/`, gitignored, local-only)

The unit is a directory of parquet shards; counts are byte-true as-of **2026-06-22**. All
raw (unadjusted). The `option_history` larder pull is **complete and static** (DONE flag
2026-06-17; see `docs/THETA_ENTITLEMENT_RETEST_2026-06-17.md`) — counts are a static snapshot.

| File / path | Type / title | Date coverage (verified) | Rows | Names |
|---|---|---|---|---|
| `option_history/` | **Full-depth EOD option chains** — all strikes, C+P, OI; no greeks/IV. **Complete** (larder pull, DONE 2026-06-17). | expirations 2016-01-08 → 2026-08-21; obs 2016-01-04 → 2026-06-17 | **390,119,692** (71,027 files) | 154 |
| `option_history_banded_backup_2026-06-01/` | EOD option chains, **Δ-banded** strikes + OI (static backup) | expirations 2016-01-15 → 2026-05-22; obs 2017-06-23 → 2026-05-22 (sampled) | 66,574,386 (51,729 files) | 503 |
| `option_history_deep365/` | **Top-mega-cap term-structure depth** (0–365 DTE; Phase B). **Staging — out-of-ranker.** | expirations 2016-01-08 → 2026-06-18; obs 2016-01-04 → 2026-06-17 | 17,528,832 (1,682 files) | 8 |
| `option_history_delisted/` | **Delisted/acquired-name survivor-bias chains** (Phase D). **Staging — out-of-ranker.** | expirations 2016-01-08 → 2024-02-16; obs 2016-01-04 → 2023-12-13 | 9,707,709 (1,834 files) | 10 |
| `index_reference/option_history/` | **Index/ETF GEX-reference chains** (Phase C roots + SPY/QQQ Phase-2). **Out-of-ranker.** | expirations 2016-01-08 → 2026-07-31; obs 2016-01-04 → 2026-06-18 | 38,854,575 (1,853 files) | 6 |
| `chains/` | Full-chain **snapshots** w/ greeks+IV+quotes+OI | snapshots 2026-04-23 → 2026-06-05 | 116,214 (1,521 files) | 495 |
| `iv_surface/` | Per-name **IV-surface snapshots** (strike×right×δ×iv×mid×dte) | snapshots 2026-04-23 → 2026-06-01 | 364,192 (558 files) | 502 |
| `iv_surface_history/` | **IV-surface daily time series** (pilot) | 2026-04-13 → 2026-06-03 | 53,725 (108 files) | 4 (A, AAPL, ABBV, ABNB) |
| `iv_history/` | **ATM-IV daily time series** (`iv_atm`) | 2015-01-02 → 2026-03-20 | 1,291,775 | 497 |
| `index_options_chains/` | Index-option full-chain snapshots | snapshots 2026-04-23 → 2026-06-01 | 8,508 (21 files) | 8 indices* |
| `index_options_surfaces/` | Index-option IV surfaces | snapshots 2026-04-23 → 2026-06-01 | 66,230 (21 files) | 8 indices* |
| `option_ohlc/` | **Per-contract daily OHLC + bid/ask** (near expirations) | expirations 2026-06-18 → 2026-07-10 | 46,886 (1,507 files) | 502 |
| `stocks_eod/` | **Equity EOD OHLCV** (underlying) | 2024-04-23 → 2026-03-20 | 233,912 | 493 |
| `vix_family/vix_family.parquet` | **VIX term-structure** index OHLC | 2023-04-24 → 2026-04-22 | 3,030 | VIX, VIX3M, VIX6M, VIX9D |

\* Index universe (8): SPX, SPXW, NDX, RUT, DJX, VIX, XSP, AAPL. `iv_surface` ETF add-ons (8):
SPY, QQQ, DIA, IWM, XLE, XLF, XLK, XLV.

> **`option_history` grew sharply** since the 2026-06-08 pass (≈185M rows / 70 names →
> **390M rows / 154 names**) — the larder puller ran to completion (DONE flag 2026-06-17). `iv_history` is static at the
> prior numbers (still ends 2026-03-20). The **three staging trees** `option_history_deep365`,
> `option_history_delisted`, and `index_reference/option_history` were added by the 2026-06-17
> enrichment run (`docs/THETA_ENRICH_RUNBOOK_2026-06-17.md`) — survivor-bias / term-structure /
> index-GEX inputs that feed dormant subsystems and **never enter `rank_candidates_by_ev`**.
> On-disk rosters are partial vs plan: deep365 8/20 (AAPL, AMZN, AVGO, GOOG, GOOGL, META, MSFT,
> NVDA), delisted 10/45 (ABMD, ATVI, FRC, PXD, RE, SBNY, SGEN, SIVB, SPLK, TWTR), index_reference
> 6 roots (NDX, QQQ, RUT, SPX, SPY, XSP — VIX root not yet present). Greeks/IV history is
> **404 / not-entitled** (`docs/THETA_ENTITLEMENT_RETEST_2026-06-17.md`), so every `option_history*`
> tree carries EOD OHLC + bid/ask + OI only. Only `corporate_actions/` holds no data (Theta
> corp-actions 404). **No Theta API was hit to produce this — only local parquet footers were read.**

---

## 4. Derived / other stores (`data_processed/`, gitignored)

> **Not re-verified this pass** — `scripts/inventory_data.py` does not scan these; carried
> forward from the 2026-06-08 pass. Treat counts as last-known, not 2026-06-22-fresh.

| File / path | Type / title | Date range (last known) | Rows | Git |
|---|---|---|---|---|
| `data_processed/vol_indices.parquet` | **Vol-index** long series (VIX/VVIX/SKEW/MOVE/GVZ/OVX/VXN/VIX3M/6M/9D) | 2011-05-31 → 2026-05-22 | 35,062 | ignored |
| `data_processed/vol_indices_wide.parquet` | Same, wide (per-index close cols) | 2011-05-31 → 2026-05-22 | 3,783 | ignored |
| `data_processed/trade_universe/2025-11-22_trade_universe.csv` | Ranked **trade universe** snapshot | 2025-11-22 | 1,066 | ignored |
| `data_processed/ibkr/wheel_ledger.json` (+ portfolio/ev_calibration files) | Real **IBKR portfolio** state (acct U17853958) | live account | — | ignored |
| `data_raw/sp500_constituents_current.csv` | Current S&P constituents | snapshot | — | tracked |

> Note: `data_processed/_inventory_scan.json` is (re)written by `scripts/inventory_data.py`
> on each run (gitignored).

---

## 5. Google Drive `swe-deep-history/` — status (not re-checked this pass, remote)

Folder contains a **partial** mirror (2 of 12 planned deep-history files): `README.txt`,
`MANIFEST.txt`, `sp500_vol_iv_full__1994_2012.csv.gz` (28,297,508 B, byte-identical to the
on-disk deep slice), and `sp500_vol_iv_full__1994_2026_FULL.csv.gz` (58,316,365 B — a
convenience concat, **not** in git, **not** byte-reproducible from the current monolith).
The complete 13-file deep set in §2 is the superset (restored from
`deep-history/bloomberg-raw`). _Remote not accessed during this regeneration._

**2026-09-24:** this folder sits in Drive area 2 (§C.3). The consolidation checks each file,
and copies `…_FULL.csv.gz` home if the root lacks it (D33).

---

## 6. Staged broad-pull data (branch `claude/bloomberg-broad-pull-2026-06-17`, PENDING INTEGRATION)

The **31 staged files (~25 logical datasets)** pulled in the 2026-06-17/18 broad Bloomberg
session. **Committed on the broad-pull branch only (held, not on `main`)**, under `staging/`. Byte-scanned 2026-06-22
(rows/ranges/columns from the actual CSV/gz bytes; cross-checked against
`staging/BROAD_PULL_MANIFEST.md`). This is the input map for `docs/WIRING_CAMPAIGN.md`.

### 6A. Currency refresh — `staging/currency_refresh/` (frontier 06-05 → 06-18)

| File | Columns | Range | Rows | Names |
|---|---|---|---|---|
| `sp500_ohlcv__2026-06-05_2026-06-18.csv` | date,ticker,open,high,low,close,volume (rotated; KLAC 10:1 seam flagged) | 2026-06-05→06-18 | 5,075 | 508 |
| `sp500_liquidity__2026-06-05_2026-06-18.csv` | date,avg_vol_30d,turnover,shares_out,ticker | 2026-06-05→06-18 | 5,080 | 508 |
| `treasury_yields__2026-06-06_2026-06-18.csv` | date,rate_1m…rate_30y,sofr | 2026-06-08→06-18 | 9 | — |
| `vix_term_structure__2026-06-05_2026-06-18.csv` | date,vix,vix_3m,vix_6m | 2026-06-05→06-18 | 10 | — |

### 6B. Vol / rates / macro — `staging/macro_vol/` + `staging/macro_rates/`

| File | Columns | Range | Rows |
|---|---|---|---|
| `macro_vol/sp500_vol_indices.csv` | vix,vvix,skew,vxn,rvx,ovx,gvz,move,vxeem,cvix | 2004-01-01→2026-06-17 | 5,847 |
| `macro_vol/spx_correlation.csv` | cor1m,cor3m,cor6m | 2006-01-03→2026-06-17 | 5,146 |
| `macro_vol/credit_spreads.csv` | ig_oas,hy_oas | 2004-01-02→2026-06-16 | 5,647 |
| `macro_vol/vix_futures_curve.csv` | ux1…ux7 | 2006-01-03→2026-06-18 | 5,150 |
| `macro_rates/ois_sofr_curve.csv` | ois_1m…ois_30y,sofr_on,sofr_1y…10y | 2001-12-04→2026-06-18 | 6,393 |
| `macro_rates/real_yields.csv` | tips_2/5/10/30y,infl_swap_2/5/10y | 2000-01-03→2026-06-18 | 6,900 |
| `macro_rates/fed_funds.csv` | fed_target,ff_fut_front | 2000-01-03→2026-06-18 | 6,850 |
| `macro_rates/macro_surprise.csv` | citi_surprise_usd,citi_surprise_g10 | 2003-01-01→2026-06-18 | 6,044 |
| `macro_rates/fx.csv` | dxy,eurusd,usdjpy,gbpusd | 2000-01-03→2026-06-18 | 6,904 |
| `macro_rates/commodities.csv` | wti,gold,copper,natgas | 2000-01-04→2026-06-18 | 6,652 |
| `macro_rates/global_vol.csv` | vstoxx,vhsi,vnky,vkospi,cdx_ig_5y,cdx_hy_5y | 2000-01-03→2026-06-18 | 6,880 |
| `macro_rates/sector_factor_etfs_ohlcv.csv` | date,open,high,low,close,volume,etf (15 ETFs) | 1998-01-02→2026-06-18 | 94,646 |

### 6C. Skew surface + macro calendar

| File | Columns | Range | Rows | Names |
|---|---|---|---|---|
| `iv_surface/sp500_iv_surface.csv.gz` | **5 tenor × 5 mny** `iv_{30,60,90,180,365}d_{90,95,100,105,110}` | 2010-01-04→2026-07-02 | **1,949,796** | 514 |
| `macro_calendar/sp500_macro_calendar.csv` | event,ticker,name,country,release_datetime/date/time (11 events) | sched 2025-01-02→2027-12-08 | 352 | 11 |
| `macro_calendar/sp500_macro_releases.csv` | event,ticker,date,actual (11 events) | 2015-01-01→2026-06-17 | 4,724 | 11 |

### 6D. Per-name panels — `staging/per_name/` + `staging/dividend_pit/` + `staging/short_interest/`

| File | Columns | Range | Rows | Names |
|---|---|---|---|---|
| `per_name/returns_micro.csv` | tot_return,px_bid,px_ask | 2010-01-04→2026-06-18 | 1,874,882 | 511 |
| `per_name/vol_term_rv.csv.gz` | atm_iv_{30,60,90,180,365,730}d, rv_{10,20,30,60,90,120,180,260}d | 2010-01-04→2026-07-02 | 1,967,985 | 515 |
| `per_name/options_sentiment.csv.gz` | pc_oi_ratio,pc_vol_ratio,oi_call,oi_put,news_sent (**32.0 MB gzipped**; raw was 102.3 MB = 97.5% of GitHub's 100 MiB per-blob limit — gzipped 2026-07-02, refreshes must stage `.csv.gz`) | 2010-01-01→2026-06-18 | 1,998,083 | 511 |
| `per_name/beta_shares.csv` | beta_raw,shares_out | 2010-01-29→2026-06-30 (M) | 94,117 | 515 |
| `per_name/fundamentals_q.csv` | revenue,oper_inc,net_income,ebitda,eps,tot_asset,tot_liab,fcf,cfo,roe,nd_to_ebitda,gross_margin | 2010-01-01→2026-05-31 (Q) | 31,479 | 511 |
| `per_name/fundamentals_ext_q.csv` | roic,oper/net/ebitda_margin,debt_to_equity,int_coverage,dvd_payout,sales_growth,trail_fcf | 2010-01-01→2026-05-31 (Q) | 31,470 | 511 |
| `per_name/estimates_m.csv` | best_eps,best_sales,best_ebitda,best_target,best_pe,best_rating,analyst_count | 2010-01-29→2026-06-30 (M) | 93,195 | 516 |
| `per_name/estimates_fwd.csv` | best_{ebitda,eps,sales}_{1bf,2bf} | 2010-01-29→2026-06-30 (M) | 93,683 | 516 |
| `per_name/valuation_m.csv` | px_to_book,ev_to_ebitda,px_to_sales,pe,peg | 2010-01-29→2026-06-30 (M) | 89,588 | 513 |
| `per_name/sp500_snapshot_bdp.csv` | rtg_sp/moody/fitch,gics_sector/ind_grp/industry/sub_ind,inst_pct,free_float_pct,float_shares,next_earnings_dt (**`next_earnings_dt` CONSUMED**: PIT-gated forward-calendar overlay in `get_next_earnings`/`get_recent_earnings` — the D3-1 earnings-lockout fix; participates only for `as_of >= asof`; APPENDED vintages 2026-06-18 + 2026-07-03, newest-eligible-wins PIT) | as-of 2026-06-18→2026-07-03 | 1,027 | 516 |
| `dividend_pit/sp500_dividend_yield_pit.csv` | dvd_yld_12m,dvd_yld_ind,dvd_sh_12m (**DATED**) | 2010-01-29→2026-06-30 (M) | 72,875 | 423 |
| `short_interest/sp500_short_interest.csv` | short_interest,short_int_ratio (biweekly) | 2015-01-15→2026-06-15 | 134,546 | 515 |

### 6E. Coverage / entitlement caveats (from `BROAD_PULL_MANIFEST.md`)

- **Storage:** `iv_surface` (96.8 MB), `vol_term_rv` (58.7 MB), and — since 2026-07-02 — `options_sentiment` (32.0 MB) committed **gzipped** (raw CSVs exceed or approach GitHub's 100 MB limit; round IV to 2 dp). Loaders must read `.gz`. *(Byte sizes here and in §6 are decimal MB = 10⁶ B, not MiB.)*
- **Manifest vs bytes:** the manifest reports `credit_spreads` ending `2026-06-17`; the actual staged bytes end **2026-06-16** (one trading day earlier) — the §6B value is byte-true; do not "correct" it to the manifest.
- **Winsorization flags:** `options_sentiment` `pc_vol`/`news_sent` and several per-name level series carry outliers flagged for winsorization — clamp at load.
- **Entitlement-blocked (manifest bucket F, all-NaN — NOT pulled):** short-interest `pct_of_float` + borrow rate; `CDS_SPREAD_*`; rating `WATCH`/`OUTLOOK`; ESG scores; per-strike OI/greeks (use-Theta); `NFCI`; long IV tenors `7/14d` & DAY-named; `BEST_PERIOD_END_DT`. Substitutes used where noted (e.g. `SHORT_INTEREST`+`SHORT_INT_RATIO` for SI; `VOLATILITY_nD` for `nDAY_HV`).
- **PIT shape:** per-name fundamentals/estimates are **period-end dated** (filing-lag PIT not captured); `snapshot_bdp` is a **single as-of (2026-06-18)** (ratings/GICS/ownership are current values, not history).
- **Skew surface:** only moneyness `{90,95,100,105,110}` populate (wings `{80,120}` empty); the `100%MNY` column **is** current ATM IV.

---

## 7. Column schemas & dtypes (every dataset)

Verified 2026-06-25 from the on-disk parquet **arrow schemas** and **pandas CSV dtype inference**.
CSV dates are stored as strings; `volume` is `float64` in the Bloomberg/Theta-EOD CSVs (not int).
Theta `option_history*` trees share one 22-column schema and carry **no greeks/IV**.

### 7.1 Bloomberg monolith CSVs

| File | Columns (dtype) |
|---|---|
| `sp500_ohlcv.csv` | `date`(str), `ticker`(str), `open` `high` `low` `close`(f64), `volume`(f64) |
| `sp500_vol_iv_full.csv` | `date`(str), `hist_put_imp_vol` `hist_call_imp_vol`(f64), `volatility_30d` `_60d` `_90d` `_260d`(f64), `ticker`(str) |
| `sp500_vol_dvd.csv` | `date`(str), `ticker`(str), `vol_30d` `dvd_yld` `turnover`(f64) |
| `sp500_liquidity.csv` | `date`(str), `avg_vol_30d` `turnover` `shares_out`(f64), `ticker`(str) |
| `sp500_historical_fundamentals.csv` | `date`(str), `ticker`(str), `pe_ratio` `eps` `revenue` `ebitda` `book_value_per_share`(f64) |
| `sp500_macro.csv` | `date`(str), `open` `high` `low` `close`(f64), `instrument`(str) |
| `sp500_sector_etfs.csv` | `date`(str), `open` `high` `low` `close`(f64), `volume`(f64), `etf`(str) |
| `sp500_vix_full.csv` | `date`(str), `close`(f64), `instrument`(str) |
| `vix_term_structure.csv` | `date`(str), `vix` `vix_3m` `vix_6m`(f64) |
| `treasury_yields.csv` | `date`(str), `rate_1m` `rate_3m` `rate_6m` `rate_2y` `rate_5y` `rate_10y` `rate_30y` `sofr`(f64) |
| `sp500_dividends.csv` | `declared_date` `ex_date` `record_date` `payable_date`(str), `dividend_amount`(f64), `dividend_frequency` `dividend_type`(str), `ticker`(str) |
| `sp500_earnings.csv` / `_yf.csv` | `year/period`(str), `announcement_date` `announcement_time`(str), `earnings_eps` `comparable_eps` `estimate_eps`(f64), `ticker`(str) |
| `sp500_corporate_actions.csv` | `announcement_date` `effective_date` `action_type`(str), `ratio` `amount`(f64), `ticker`(str) — _populated on `origin/main` (was an empty stub on the stale branch)_ |
| `sp500_index_membership.csv` | `member_ticker_and_exchange_code`(str), `percentage_weight`(f64), `as_of_date`(str) |
| `sp500_analyst.csv` | `best_analyst_rating` `best_eps` `best_sales` `best_target_price` `tot_analyst_rec`(f64), `ticker`(str) |
| `sp500_credit_risk.csv` | `altman_z_score` `interest_coverage_ratio`(f64), `rtg_sp_lt_lc_issuer_credit`(str), `ticker`(str) |
| `sp500_fundamentals.csv` / `_yf.csv` | `ticker`(str), `30day_impvol_100.0%mny_df` `best_pe_ratio` `beta_raw_overridable` `cur_mkt_cap` `eqy_dvd_yld_12m` `free_cash_flow_yield` `pe_ratio` `return_com_eqy` `tot_debt_to_tot_eqy` `volatility_30d`(f64), `gics_industry_group_name` `gics_sector_name`(str) |
| `sp500_institutional.csv` | `eqy_free_float_pct` `eqy_inst_pct_sh_out` `eqy_sh_out`(f64), `ticker`(str) |
| `sp500_iv_snapshot_today.csv` | `ticker`(str), `30day_impvol_100.0%mny_df` `60day_impvol_100.0%mny_df` `volatility_30d`(f64) |

### 7.2 Bloomberg deep-history gz

| File group | Columns (dtype) |
|---|---|
| `sp500_ohlcv__*.csv.gz` | `date`(str), `ticker`(str), `open` `high` `low` `close` `volume`(f64) |
| `sp500_vol_iv*__*.csv.gz` | `date`(str), `hist_put_imp_vol` `hist_call_imp_vol` `volatility_30d` `_60d` `_90d` `_260d`(f64), `ticker`(str) |
| `sp500_liquidity__*.csv.gz` | `date`(str), `avg_vol_30d` `turnover` `shares_out`(f64), `ticker`(str) |
| `sp500_iv_surface__*.csv.gz` | `date`(str), `iv_{30,60,90,180,360}d_{90,95,100,105,110}`(f64) — 25 tenor×moneyness cols, `ticker`(str) |
| `delisted_status.csv` | `ticker` `name` `window`(str), `ohlcv_rows` `voliv_rows` `liq_rows` `dropped`(i64), `status`(str) |
| `ohlcv_dropped_ticks*.csv` | `date` `ticker`(str), `open` `high` `low` `close` `volume`(f64), `check_failed`/`which` `window`(str) |

### 7.3 Theta parquet

| Dataset | n | Columns (arrow type) |
|---|---|---|
| `option_history` / `_deep365` / `_delisted` / `index_reference` / banded backup | 22 | `symbol`(str), `expiration`(str), `strike`(f64), `right`(str), `created`(str), `last_trade`(str), `open` `high` `low` `close`(f64), `volume`(i64), `count`(i64), `bid_size`(i64), `bid_exchange`(i64), `bid`(f64), `bid_condition`(i64), `ask_size`(i64), `ask_exchange`(i64), `ask`(f64), `ask_condition`(i64), `open_interest`(f64), `ticker`(str) — **no greeks / IV** |
| `chains` | 23 | `symbol` `expiration`(ts) `strike` `right` `delta` `theta` `vega` `rho` `epsilon` `lambda` `iv` `iv_error` `underlying_timestamp` `underlying_price` `bid` `ask` `bid_size` `ask_size` `open_interest` `mid` `ticker` `snapshot_date` (+ `__index_level_0__`) |
| `index_options_chains` | 21 | as `chains` minus `iv_error`, `underlying_timestamp` |
| `iv_surface` / `index_options_surfaces` | 10 | `strike`(f64) `right`(str) `delta`(f64) `iv`(f64) `mid`(f64) `expiration`(ts) `dte`(i64) `ticker`(str) `snapshot_date`(str) (+ `__index_level_0__`) |
| `iv_surface_history` | 8 | `date`(ts) `ticker`(str) `expiration`(ts) `dte`(i64) `strike`(f64) `right`(str) `iv`(f64) `mid`(f64) |
| `iv_history` | 4 | `iv_atm`(f64) `ticker`(str) `source`(str) `date`(ts) |
| `option_ohlc` | 12 | `open` `high` `low` `close`(f64) `volume`(i64) `bid` `ask`(f64) `ticker`(str) `expiration`(str) `strike`(f64) `right`(str) `date`(ts) |
| `stocks_eod` | 7 | `open` `high` `low` `close`(f64) `volume`(f64) `ticker`(str) `date`(ts) |
| `vix_family` | 7 | `open` `high` `low` `close`(f64) `symbol`(str) `source`(str) `date`(ts) |

### 7.4 Derived parquet

| File | Columns (arrow type) |
|---|---|
| `vol_indices.parquet` | `date`(ts) `open` `high` `low` `close`(f64) `symbol`(str) `source`(str) |
| `vol_indices_wide.parquet` | `date`(ts) + per-index closes: `gvz_close` `move_close` `ovx_close` `skew_close` `vix_close` `vix3m_close` `vix6m_close` `vix9d_close` `vvix_close` `vxn_close`(f64) |

---

_Regenerate: against an `origin/main` checkout, `.venv/Scripts/python.exe scripts/inventory_data.py`
→ `data_processed/_inventory_scan.json` (monoliths + deep + Theta). For the staged §6, byte-scan
`staging/` on the broad-pull branch separately — `inventory_data.py` does not cover it._
