---
id: bloomberg-deep-history-2026-06-04
title: Bloomberg OHLCV rotation fix + make-current + deep-history backfill (2026-06-04)
kind: docs
status: in-flight
terminal: lab
pr:
decisions: []
date: 2026-06-04
headline: Fixed the OHLCV column-rotation defect, made all Tier-1 + context data current to 2026-06-04, deepened all single-series context to inception, generalized the pullers to contiguous backfill, COMPLETED the vol_iv deep-history backfill to the 1994 floor (captures the 2000-2002 and 2008 crises), and routed deep data to a gz buffer branch (rclone/Drive staged) to keep the connector monoliths <100 MB.
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
- **windows 2-8 (gz buffer branch `deep-history/bloomberg-raw` @ 61c2183):**
  vol_iv backfilled 2012-06 all the way to the **1994 IV floor — COMPLETE**.
  `data/bloomberg/deep/sp500_vol_iv_full__1994_2012.csv.gz` = 1,661,191 rows,
  **1994-01-03 -> 2012-06-29**, 436 tickers, 27.0 MB gz. **Both the 2000-2002
  dot-com bear (+9/11) and the 2008 GFC IV are captured.** With the frozen
  monolith the vol_iv panel is contiguous 1994 -> 2026. Full uncompressed scratch
  (1994-2026, 3,329,212 rows) also at `C:\Users\mertmert\deep_scratch\` on the box.
  Pulled with **zero throttling** (~14M data points across the session).

## 2026-06-05 — Phase 3b: OHLCV + liquidity deep backfill COMPLETE (to 1994)

Resumed on a fresh lab terminal (cloned the repo, recreated `bbg-venv`
[blpapi 3.26.4.2 / xbbg 1.2.6 / pandas 3.0.3], live API gate). Completed the two
remaining per-name panels to the 1994 floor, newest-first, with the SAME proven
machinery (`scripts/pull_ohlcv.py` / `pull_liquidity.py` + `_bbg_panel.py`,
`SWE_PULL_MODE=backfill` `SWE_BACKFILL_MAX_WINDOWS=1`). Each window grew an
**off-monolith scratch** (`C:\Users\mertmert\deep_scratch\`); the deep slice
(date < monolith floor) was carved → gz → pushed to `deep-history/bloomberg-raw`
after every chunk (durable on a transient box). Monoliths stayed FROZEN (verified
unmodified on the refresh branch). Commits went through a dedicated **git worktree
pinned to the buffer branch** instead of checkout-switching the main tree — same
result, no 850-file churn, scripts always present for the next pull.

- **OHLCV** `data/bloomberg/deep/sp500_ohlcv__1994_2018.csv.gz` —
  **1994-01-03 → 2017-12-29**, 2,083,270 rows, 449 tickers, 31.7 MB gz.
  Joins the frozen monolith (2018-01-02 → 2026-06-04) → unbroken 1994 → 2026.
  Final buffer SHA `5296515`.
- **Liquidity** `data/bloomberg/deep/sp500_liquidity__1994_2015.csv.gz` —
  **1994-01-03 → 2014-12-31**, 1,987,751 rows, 457 tickers, 20.9 MB gz.
  Joins the frozen monolith (2015-01-02 → 2026-06-04). Final buffer SHA `c814bc5`.

All three deep panels (vol_iv, OHLCV, liquidity) are now on the buffer branch and
join their frozen monoliths for contiguous **1994-01-03 → 2026-06-04** coverage.
All three gz are < 100 MB → GitHub-resident, no Drive needed for these.

### OHLCV bad-tick policy (operator-approved 2026-06-05)
The rotation gate (`open == row-max` AND `low == row-min`) tripped on **97 rows**
across the 1994–2010 windows — genuine bad Bloomberg prints on illiquid names
(true open/close a few cents OUTSIDE the reported high/low), **not** a rotation
defect: the field map is identical to the clean 2010–2018 windows, which passed
100%, and the per-window failure rate stayed 0.000–0.014% (never trended up as
the universe thinned). Decision: **drop only the gate-failing rows**, audit each
to `data/bloomberg/deep/ohlcv_dropped_ticks.csv` (ticker, date, OHLCV, which
check failed, window), under a **0.5%-per-window safety valve** (exceed → STOP as
a systematic signal). Breakdown: 51 `open!=max`, 46 `low!=min`. The carved gz
independently RE-GATES clean (0 violations). Liquidity has no OHLC invariant →
no gate, no drops.

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

### Deep backfill — ALL THREE PANELS DONE to 1994 (vol_iv + ohlcv + liquidity) [2026-06-05]
vol_iv, ohlcv, AND liquidity are now fully backfilled to the 1994 floor (deep gz on
the buffer branch `deep-history/bloomberg-raw`; see the 2026-06-05 section above for
spans/SHAs + the OHLCV bad-tick audit). The loop below is retained for reference /
any future per-name panel: grow an off-monolith scratch with `SWE_OUT_PATH`, then carve
the gz and push to the buffer branch. The refresh-branch monolith is NEVER advanced.
The proven loop (shown for vol_iv; swap the script + scratch + deep filename for
ohlcv `sp500_ohlcv__1994_2018.csv.gz` / liquidity `sp500_liquidity__1994_2015.csv.gz`):
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
- **Phase 2c snapshots** — **DONE 2026-06-05.** Refreshed all five point-in-time
  `bdp` snapshots (`sp500_fundamentals`, `credit_risk`, `institutional`,
  `analyst`, `iv_snapshot_today`) to today (503 members) and codified the verified
  field lists + per-file schema/ticker-format/column-order in the new reusable
  `scripts/pull_snapshots.py` (`SWE_SNAP_SMOKE` / `SWE_SNAP_ONLY` knobs;
  3-ticker smoke before the full run). Committed to the refresh branch.
- **sp500_macro:** stale to 2026-03-20; OHLC-per-instrument; refresh + backfill
  to 1990 (+ IG/HY OAS + financial-conditions proxy if entitled). Verify the
  per-instrument bbg map on one instrument first.
- **Phase 4 IV moneyness/skew surface:** `sp500_iv_surface.csv`,
  `{30,60,90}DAY_IMPVOL_{90,95,100,105,110}.0%MNY_DF`, LONG
  `date,ticker,tenor_days,moneyness_pct,iv`. Verify the field floor on one
  ticker; prioritise put side + crisis windows.

## Google Drive (rclone) — STAGED; upload pending operator permission
rclone v1.74.2 at `C:\Users\mertmert\rclone\rclone-v1.74.2-windows-amd64\rclone.exe`.
Remote `gdrive` authorized via browser OAuth (token in `%APPDATA%\rclone\rclone.conf`);
`rclone about gdrive:` shows **4.79 TiB free**; `gdrive:swe-deep-history/` created and
the t.txt round-trip verified. The deep-DATA upload is BLOCKED by Claude Code's
auto-mode classifier (data-exfiltration hard block) and the agent cannot self-grant
the permission. **Nothing is lost** — the deep gz is on the buffer branch. To finish
the Drive copy, EITHER the operator runs it, OR adds this allow rule to
`~/.claude/settings.local.json` (then the agent runs it):
```
permissions.allow += "PowerShell(& 'C:/Users/mertmert/rclone/rclone-v1.74.2-windows-amd64/rclone.exe' *)"
```
Upload + verify (forward-slash paths to match the rule):
```
& 'C:/Users/mertmert/rclone/rclone-v1.74.2-windows-amd64/rclone.exe' copy 'C:/Users/mertmert/deep_scratch/sp500_vol_iv_full__1994_2012.csv.gz' gdrive:swe-deep-history/ -P
& 'C:/Users/mertmert/rclone/rclone-v1.74.2-windows-amd64/rclone.exe' lsl gdrive:swe-deep-history/
```
Confirm bytes on Drive before deleting any local copy (the gz is also on the buffer branch).

## DEFER to laptop/overnight (NOT the metered terminal)
Move the deep gz files from `deep-history/bloomberg-raw` into **Google Drive**
(permanent home); convert to Parquet partitioned by year; add the connector
read-path that assembles recent(git monolith) + deep(Drive); then delete the gz
buffer branch. That connector wiring is the real fix and is out of scope here.
Also deferred (wrong channel): option chains (Theta), dealer GEX (subscription),
`borrow_rate_net` + macro `importance`/deep-history (BQL/Excel), `treasury_yields`
(yfinance).
