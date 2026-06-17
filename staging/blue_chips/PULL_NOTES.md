# Blue-chip backfill fragments (#355) — session notes (2026-06-17)

Supervised Bloomberg session, branch `claude/phase1a-casy-bloomberg-pull`. Phase 1A, #355.
Same validated method as `staging/casy/` (see `staging/casy/PULL_NOTES.md`). **Fragments only —
monoliths + trio byte-untouched; Phase 1B integration + re-baseline deferred to a no-Bloomberg clone.**

> **⚠️ CORRECTION (2026-06-17, found during Phase-1B integration; see `pull_batch1_un.py`).**
> The first batch-1 pull used the committed `<NAME> UW` (NASDAQ) tickers for **WMT/KMB/CPB/DPZ/PLTR**,
> which return only the truncated recent rows for these **NYSE** names — i.e. it re-pulled the same thin
> data (WMT 122→122 etc.) and did **not** backfill history. The wrong `UW` code is exactly *why* they
> were thin on `main`. The original "open/close exact" validation gave false confidence (it compared
> UW-to-UW); the tell was the row count not growing. **Confirmed:** `WMT UW Equity` has 0 rows in
> Jan-2018; `WMT UN Equity` has the full 2018→2026 history. **Fixed:** the five `*_ohlcv.csv` fragments
> were re-pulled as `<NAME> UN Equity` (WMT/KMB/CPB/DPZ 2117 rows from 2018; PLTR 1426 from its
> 2020-09-30 IPO; OHLC integrity verified). Stored ticker = `<NAME> UN Equity`; Phase-1B integration
> drops the old wrong `<NAME> UW Equity` rows. *(Batch-1 vol_iv was already full on `main` and is
> untouched — note the pre-existing oddity that `<NAME> UW` vol_iv carries full history while `UW`
> ohlcv did not; out of scope here, flagged for the record.)*

## Scope (runbook A2)
10 names with truncated OHLCV history on `main`; all get an OHLCV backfill (2018-01-02→2026-06-04).
5 are also IV-thin (committed vol_iv == 52 rows) and get a vol_iv backfill too.

| name | ohlcv ticker | committed ohlcv bars | committed vol_iv rows | vol_iv backfill? |
|---|---|---|---|---|
| WMT | WMT UW Equity | 122 | 2117 | no (IV full) |
| KMB | KMB UW Equity | 255 | 2117 | no |
| CPB | CPB UW Equity | 450 | 2117 | no |
| DPZ | DPZ UW Equity | 356 | 2117 | no |
| PLTR | PLTR UW Equity | 380 | 1422 (full post-IPO) | no |
| VEEV | VEEV UN Equity | 52 | 52 | **yes** |
| COHR | COHR UN Equity | 52 | 52 | **yes** |
| LITE | LITE UW Equity | 52 | 52 | **yes** |
| SATS | SATS UW Equity | 52 | 52 | **yes** |
| VRT | VRT UN Equity | 52 | 52 | **yes** |

## Method (identical to CASY)
- OHLCV column scramble reproduced: `open←PX_HIGH, high←PX_LAST, low←PX_LOW, close←PX_OPEN, volume←PX_VOLUME`. Confirmed per name via cross-match (PX_LAST/PX_LOW are the unambiguous best matches for high/low, diff ≪ alternatives).
- vol_iv: `30DAY_IMPVOL_100.0%MNY_DF`→`hist_put_imp_vol==hist_call_imp_vol`; `VOLATILITY_30/60/90/260D`→`volatility_*`.
- Ticker exactly as committed; date range 2018-01-02→2026-06-04.

## Validation vs committed (truncated) overlap rows — vintage note
**Adjustment basis is consistent with the committed universe** — proven by `open`(=PX_HIGH) and
`close`(=PX_OPEN) matching **exactly to the cent** on every overlap row for all names. So the
backfilled 2018–2025 history is on the same split/dividend-adjustment basis as the rest of the universe.

The recent overlap rows (which Phase 1B replaces anyway) carry minor **current-revision-vintage** drift,
larger than CASY's because these are higher-volume names with more post-close corrections:
- `high`(=PX_LAST) / `low`(=PX_LOW): per-component close/low revisions, ≤~2.7 abs (<1%).
- `volume`: 2 rows/name. The big one is **2026-03-20** (March triple-witching, also the main pull's
  old `end_date`): committed carried a stale preliminary value (e.g. WMT 6.77M) vs the correct final
  (37.95M). The fresh pull **corrects** a stale witching-day volume. The other is the 2026-06-04
  frontier-day finalization (tiny).

Per-name OHLCV validation (maxabsdiff over committed overlap): open/close = 0.000000 for all;
high/low small (WMT 1.15/0.46, KMB 0.29/0.63, CPB 0.07/0.00, DPZ 2.70/0.00, PLTR 0.65/0.51);
volume 2 revised rows/name (witching + frontier).

### Vintage record (for the re-baseline)
Each backfilled name carries the **current** revision vintage on its recent-overlap high/low/volume
and (for the 5 IV-thin names) IV, exactly as noted for CASY. Raw open/close exact; backfill history
adjustment-consistent. The Phase-3 re-baseline absorbs the recent-row deltas (those rows are replaced).
