---
id: r7-deep-iv-sentinel
title: "R7: null the deep-IV 134217.7 sentinel on the assembled vol_iv read"
kind: fix
status: complete
terminal: desktop
pr:
decisions: []
date: 2026-06-06
headline: On the assembled (deep) vol_iv read, null the corrupt implied-vol sentinel (~134217.7) above a 10,000 floor while keeping the row — chosen over the early "IV>500%" note because on-bytes inspection of the delisted panel showed real distressed-name IVs of 500-1196% that a 500 cut would wrongly discard. Stacked on the deep-read branch. Other R7 items (drop .xlsx, shard bid_ask, deprecate vol_dvd) are R1-merge-time data ops — NOTED, not executed.
surface: [engine/data_connector.py, tests/test_deep_iv_sentinel.py]
---

## Goal
R7 hygiene from the data-layer QA. Of the four R7 items, only the deep-IV sentinel
is code-expressible and in-scope for an unattended main-targeted run; the rest are
data-file operations that belong with the deferred R1 merge.

## What we did
- `engine/data_connector._load_assembled`: for `key == "vol_iv"`, NULL
  `hist_put_imp_vol` / `hist_call_imp_vol` above `_DEEP_IV_SENTINEL_FLOOR`
  (10,000), keeping the row (realized-vol columns stay usable). Only on the
  assembled/deep path — the OFF/monolith path is untouched (default-OFF
  byte-identity preserved).
- `tests/test_deep_iv_sentinel.py` (gated on `SWE_DEEP_TEST_DATA`): sentinel
  nulled (0 IV > floor), real extremes (500-1000%) preserved, and a sentinel row
  is KEPT with NaN IV (not dropped).

## The threshold decision (refinement of the roadmap)
The roadmap said "null any IV > ~500%". Inspecting the panels on the bytes:
- `__1994_2012`: 945 values >500, **all == 134217.7** (the sentinel); none in
  [500,1000). So >500 == sentinel here.
- `__delisted`: 485 values >500 — but 19 in [500,1000) (max 934) and ~20 in
  [1000, ~1200) that are **NOT** the sentinel (1061, 1080, 1196, …); 427 are the
  134217.7 sentinel.
A 500 cut would discard real distressed-name implied vols. The sentinel sits alone
at ~134217.7 with a clean gap below, so a **10,000 floor** removes only the
corrupt sentinel and never real data. This is NOT a §2 question (the ranker is
untouched; this is data-intake hygiene on the deep path).

## NOT executed (R1-merge-time data ops, deferred)
- Drop `sp500_short_interest.csv.xlsx` (binary dup) — a refresh-branch data file,
  not on main; belongs to the R1 data intake.
- Shard `sp500_bid_ask.csv` before 100 MB — refresh-branch data op, R1 intake.
- `sp500_vol_dvd.csv` deprecation — NOTED only; do NOT remove the
  `consolidated_loader:352 load_vol_dvd()` hook unattended (loader behaviour
  change → needs review). All three remain in the roadmap R7 row.

## Evidence
- `pytest tests/test_deep_iv_sentinel.py` (SWE_DEEP_TEST_DATA set) -> 2 passed.
- `pytest tests/test_deep_read_connector.py` -> 10 passed (no over-null regression).
- ruff clean; manifest OK; lane-claim OK (data_connector is not a trio file).
- Stacked on `claude/deep-read-activation` (#335); base = that branch.

## Unresolved / handoff
Default-OFF (only the deep assembled path nulls). When deep-read is adopted (R1 +
re-baseline), this nulling is already in place. The .xlsx/bid_ask/vol_dvd data ops
ride with R1.
