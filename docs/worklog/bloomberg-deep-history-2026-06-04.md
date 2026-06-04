---
id: bloomberg-deep-history-2026-06-04
title: Bloomberg OHLCV rotation fix + make-current + deep-history backfill (2026-06-04)
kind: docs
status: in-flight
terminal: lab
pr:
decisions: []
date: 2026-06-04
headline: Fixed the OHLCV column-rotation defect, made all Tier-1 + context data current to 2026-06-04, deepened all single-series context to inception, generalized the pullers to contiguous backfill, and began the per-name deep-history backfill (vol_iv incl. the 2008 GFC) — deep data routed to a gz buffer branch to keep the connector monoliths <100 MB.
surface: [scripts/_bbg_panel.py, scripts/pull_ohlcv.py, scripts/pull_liquidity.py, scripts/pull_vol_iv.py, scripts/pull_vix_term_structure.py, scripts/pull_context_index.py, data/bloomberg/]
---

## Goal

Execute the deep-history + make-current pull from the transient lab terminal:
FIX the OHLCV rotation defect, make every Tier-1/context series current to
2026-06-04, deepen cheap context to inception, and backfill the expensive
per-name panels (OHLCV/IV/liquidity) toward the 1994 IV floor, newest-first and
contiguous, committing each chunk. Data only — **no engine/connector edits**.

## Environment

`C:\Anaconda3\python.exe` (3.13.9) venv at `C:\Users\mertmert\bbg-venv`;
`blpapi 3.26.4.2`, `xbbg 1.2.6`, `pandas 3.0.3`; `PYTHONUTF8=1`. API entitled.
git identity `Mert Yakar <mertyakar.my@gmail.com>`.

## STORAGE ARCHITECTURE (operator decision — important)

GitHub's HARD file limit is **100 MB** (the earlier 50 MB notes were soft
warnings). The deep per-name backfill pushes the monoliths past 100 MB, so:

- **Refresh branch `data/bloomberg-refresh-2026-06-02`** keeps the
  connector-read MONOLITHS (`sp500_ohlcv.csv`, `sp500_liquidity.csv`,
  `sp500_vol_iv_full.csv`) **FROZEN at their <100 MB pushed floors** (vol_iv at
  the 2012-07-02 floor, 94.5 MB). The engine baseline stays clean & pushable.
- **Deep history goes to SEPARATE gz files** under `data/bloomberg/deep/`, pushed
  to a transient **buffer branch `deep-history/bloomberg-raw`** (NEVER merged).
  gz keeps them well under 100 MB (the 2007–2012 vol_iv slice is 8.7 MB gz).
- NOT Git LFS (quota/cost). The permanent home is **Google Drive**, set up
  later from the laptop (see DEFER).

## What was done

### Phase 1 — OHLCV rotation defect FIXED (commit 8fdf612, pushed)
Connector (`engine/data_connector.py:202-208`) renames `open->high, high->close,
close->open`; the stored CSV must be ROTATED (stored `open` = true HIGH/row-max,
stored `low` = true LOW). The prior 2026-03-23+ delta was CANONICAL → inverted.
Fix: restored the clean rotated base from origin/main (->2026-03-20), re-pulled
2026-03-23->2026-06-04 with field map `{PX_HIGH->open, PX_LAST->high,
PX_LOW->low, PX_OPEN->close, PX_VOLUME->volume}`. Post-build GATE: **0 violations
across 1,014,916 non-NaN rows**. Verified offline (round-trip) + live.

### Phase 2 — make-current + cheap context to inception (all pushed)
- **2a Tier-1 current (8fdf612):** ohlcv re-pulled to 06-04; liquidity + vol_iv
  to 06-04.
- **2b vix_term_structure (e386a51):** 2018->03-20 → **1990-01-02 -> 2026-06-04**
  (9,200 rows; vix_3m 2002, vix_6m 2008).
- **2d context indices to inception (ccd821d):**
  - `vix_futures_curve` 20,111 -> **36,779** (UX1-7 from 2004-03-26)
  - `vol_indices` 22,968 -> **43,244** (SKEW 1990 … VXEEM/VIX9D 2011)
  - `spx_correlation` 8,605 -> **15,396** (COR1M/3M/6M from 2006)
  - `rates_fx_vol` 8,716 -> **24,842** (MOVE 1988 / JPMVXYG7 1992 / CVIX 2001)

### Phase 3a — per-name deep backfill (begun; newest-first; floor 1994)
- **window 1 (8dc6f59, pushed in the refresh monolith):** vol_iv 2012-07 ->
  2015-01 (+280,302). This is the frozen floor of the refresh-branch monolith
  (vol_iv now 2012-07-02 -> 2026-06-04, 1,668,021 rows, 94.5 MB).
- **windows 2-3 (relocated to the gz buffer branch):** vol_iv 2007-07 -> 2012-06
  (+526,932). **The 2008 GFC IV is captured.** Stored as
  `data/bloomberg/deep/sp500_vol_iv_full__1994_2012.csv.gz` (526,932 rows,
  2007-07-02 -> 2012-06-29, 436 tickers, 8.7 MB gz) on
  `deep-history/bloomberg-raw`. The full uncompressed scratch monolith
  (2007-07 -> 2026-06-04) is also at `C:\Users\mertmert\deep_scratch\` on the box.

## The machinery

`scripts/_bbg_panel.py` (NEW) — shared engine. Fills FORWARD gap [max+1 -> END]
then walks BACKWARD [min-1 -> FLOOR] **newest-first in ~30-month windows**,
rewriting after EVERY window (durable). Per-window: pull members in 30-ticker
`bdh` -> pivot long->wide -> merge+dedupe(date,ticker keep-last) -> sort -> write
-> optional validate() (the OHLCV gate). Pullers are thin configs.

Env knobs: `SWE_PULL_END`, `SWE_PULL_FLOOR`, `SWE_PULL_MODE`
(forward|backfill|both), `SWE_BACKFILL_CHUNK_MONTHS`, `SWE_BACKFILL_MAX_WINDOWS`,
`SWE_OUT_PATH` (write to a scratch path instead of the monolith — used for deep
backfill), `SWE_PULL_LIMIT`, `SWE_PULL_NO_WRITE`.

Cheap-context pullers: `scripts/pull_vix_term_structure.py`,
`scripts/pull_context_index.py <dataset>` (per-ticker pulls so one bad ticker
can't null the batch). Offline tests (`test_pullers_offline.py`, kept off-repo)
prove window tiling + the rotation round-trip/gate. All passed.

## Gotchas

- xbbg 1.2.6 → long narwhals; `.to_native()` + `pivot_table`; `PYTHONUTF8=1`.
- Historical windows use **today's** SPX members (post-IPO names absent in old
  windows; departed names keep old rows). Survivorship is inherent.
- Meter: ~3-6M points this session, **zero throttling** (one 30-month per-name
  window ≈ 1.9M points). Full 3-panel backfill to 1994 ≈ 20-40M → will hit the
  cap; pull-until-error, per-window write keeps what landed, resume.

## RESUME RUNBOOK

Terminal up + logged in; venv active; `cd` repo; `$env:PYTHONUTF8='1'`.

### Continue the deep backfill (vol_iv shown; ohlcv/liquidity analogous)
Grow the off-monolith scratch with `SWE_OUT_PATH`, then re-carve the gz and push
to the buffer branch. The refresh-branch monolith is NEVER advanced.
```
$env:SWE_OUT_PATH='C:\Users\mertmert\deep_scratch\sp500_vol_iv_full.csv'
$env:SWE_PULL_MODE='backfill'; $env:SWE_BACKFILL_MAX_WINDOWS='2'
python scripts/pull_vol_iv.py            # grows scratch 2004..2007, etc.
# re-carve deep slice (date < 2012-07-02) -> data/bloomberg/deep/*.csv.gz, then:
git checkout deep-history/bloomberg-raw
git add -f data/bloomberg/deep/sp500_vol_iv_full__1994_2012.csv.gz
git commit -m "deep: vol_iv backfill chunk" ; git push origin deep-history/bloomberg-raw
git checkout data/bloomberg-refresh-2026-06-02
```
For OHLCV/liquidity: first SEED the scratch from the frozen monolith
(`copy data\bloomberg\sp500_ohlcv.csv <scratch>`), then same loop with
`SWE_OUT_PATH=<scratch>` and a deep filename `sp500_ohlcv__1994_2018.csv.gz` /
`sp500_liquidity__1994_2015.csv.gz`. (OHLCV scratch keeps the rotated map; its
gate still applies.)

### Not done this session (lower urgency / bespoke)
- **Phase 2c snapshots** (`sp500_fundamentals`, `credit_risk`, `institutional`,
  `analyst`, `iv_snapshot_today`): point-in-time `bdp`; re-pull anytime with the
  prior session's verified field lists.
- **sp500_macro:** stale to 2026-03-20; OHLC-per-instrument; refresh + backfill
  to 1990 (+ IG/HY OAS + financial-conditions proxy if entitled). Verify the
  per-instrument bbg map on one instrument first.
- **Phase 4 IV moneyness/skew surface:** `sp500_iv_surface.csv`,
  `{30,60,90}DAY_IMPVOL_{90,95,100,105,110}.0%MNY_DF`, LONG
  `date,ticker,tenor_days,moneyness_pct,iv`. Verify the field floor on one
  ticker; prioritise put side + crisis windows.

## DEFER to laptop/overnight (NOT the metered terminal)
Move the deep gz files from `deep-history/bloomberg-raw` into **Google Drive**
(permanent home); convert to Parquet partitioned by year; add the connector
read-path that assembles recent(git monolith) + deep(Drive); then delete the gz
buffer branch. That connector wiring is the real fix and is out of scope here.
Also deferred (wrong channel): option chains (Theta), dealer GEX (subscription),
`borrow_rate_net` + macro `importance`/deep-history (BQL/Excel), `treasury_yields`
(yfinance).
