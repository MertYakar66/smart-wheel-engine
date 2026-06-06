# Connector Deep-Read + Survivorship Harness — Design

> **Status: DESIGN (2026-06-05).** Companion to
> `DATA_LAYER_ACTIVATION_ROADMAP.md`. This specifies *how* to wire the
> campaign's deep + delisted data into the engine without breaking the
> `CLAUDE.md` §2 invariant. **It is a design — no connector / harness code is
> changed by the work on this branch** (only the R0a credit-rating diagnostic
> fix in `wheel_runner.py`). Execute after operator review.

## 0. The governing constraint

`CLAUDE.md` §2: **no tradeable candidate bypasses `EVEngine.evaluate`.** The
decision-layer trio (`ev_engine.py`, `wheel_runner.py`, `candidate_dossier.py`)
is CI-gated. Therefore the deep-read must land **entirely below** the connector's
public accessors, so the trio receives *longer/wider series through the same
method signatures* and is not itself restructured.

The seam is exact. In `engine/data_connector.py`:

- `_load(key)` reads **one CSV** (`_FILES[key]`), normalizes the `ticker`
  column via `normalize_ticker`, parses date columns, caches by key.
- Every accessor (`get_ohlcv`, `get_iv_history`, `get_liquidity`,
  `get_fundamentals`, …) calls `_load(key)` then filters. `get_ohlcv` applies
  the rotation un-rename (`open→high, high→close, close→open`) **above**
  `_load`.

**Assemble inside `_load`** and all of that is transparent: the rotation rename,
the ticker-group cache, the date filters, and the entire ranker/EVEngine path
above keep working unchanged.

## Part A — Connector deep-read path

### A1. Slice manifest (not hardcoded names)

Add a declarative manifest keyed like `_FILES`, listing the deep gz + delisted
panels that extend each monolith. A manifest (vs. inline names) lets the
`iv_surface` shards and any future shard register without touching `_load`:

```python
# engine/data_connector.py  (illustrative)
_DEEP_SLICES = {
    "ohlcv":     ["deep/sp500_ohlcv__1994_2018.csv.gz",
                  "deep/sp500_ohlcv__delisted.csv.gz"],
    "vol_iv":    ["deep/sp500_vol_iv_full__1994_2012.csv.gz",
                  "deep/sp500_vol_iv_full__2012_2018.csv.gz",
                  "deep/sp500_vol_iv__delisted.csv.gz"],
    "liquidity": ["deep/sp500_liquidity__1994_2015.csv.gz",
                  "deep/sp500_liquidity__delisted.csv.gz"],
}
```

A missing slice file → **log + skip that slice** (degrade to the monolith), never
raise — matching the connector's existing "missing CSV → empty frame, don't
crash" contract. This is what makes the feature safe to ship before the deep gz
are guaranteed present in every environment (sandbox vs. laptop).

### A2. Opt-in, not default-on

Deep-read is gated by a flag (env `SWE_DEEP_HISTORY=1` or
`MarketDataConnector(deep_history=True)`), **default OFF**. Rationale:

- The current monolith fast-path (tests, the live `as_of≈today` scan, the
  existing 2018+ regression backtests) must stay light and byte-identical until
  R1's re-baseline lands. Flipping the default is itself a re-baseline event.
- Schema parity is proven (deep/delisted columns == monolith columns; verified
  on the bytes), so turning the flag on is a pure superset of rows.

### A3. Assembly recipe (per key, inside `_load`)

```
df_recent = read monolith (existing path)
if deep_history and key in _DEEP_SLICES:
    parts = [df_recent] + [read each present slice]      # gzip via compression=
    df = concat(parts)
    df = normalize tickers (map over uniques — existing pattern)
    df = parse dates (existing)
    df = dedupe on (ticker, date) keep precedence: recent > deep-current > delisted
    df = sort by (ticker, date)
else:
    df = df_recent (+ normalize/parse as today)
cache[key] = df
```

Dedup precedence (A4) is why the concat order is **recent first**, then
`drop_duplicates(["ticker","date"], keep="first")`.

### A4. Ticker normalization + dedup precedence (the correctness crux)

Verified on the bytes:

- **Current** panels store `A UN [Equity]` → `normalize_ticker` → `A`.
- **Delisted** panels + the **membership** file + `_delisted_universe.csv` all
  store the **Bloomberg PIT code** (`0111145D UN`, `1323Q US`, `SVFI US`) →
  `normalize_ticker` → opaque stub (`0111145D`, `1323Q`, `SVFI`). These stubs
  are *internally consistent across all three files*, so the survivorship join
  (Part B) keys on them cleanly; human-readable names come from
  `_delisted_universe.csv` / `delisted_status.csv`.
- **Collision set:** the ~90 relistings/code-changes of *held* names (e.g. an old
  `WMT UN` row that normalizes to the same `WMT` as the live monolith). These are
  the only (ticker, date) overlaps. **`keep="first"` with `recent > deep-current
  > delisted`** means the live monolith row always wins, the deep-current row
  fills pre-2018 gaps, and a stale delisted row never shadows a live one. The 926
  true delistings are pure PIT-code keys → no overlap at all.

The OHLCV rotation contract survives assembly unchanged: every OHLCV piece
(monolith, `__1994_2018`, `__delisted`) is stored in the **same rotated layout**
(`open=row-max=PX_HIGH`, `low=row-min=PX_LOW`); concat preserves it; the
post-`_load` rename + the sampled `_validate_ohlcv_invariants` check still fire.
(The deep + delisted panels independently re-gated 0-violation at pull time.)

### A5. Memory / performance

Assembled frames are large (≈ **5.5 M OHLCV / 5.7 M vol_iv / 5.8 M liquidity**
rows incl. delisted). A full deep sweep that loads all three ≈ 2–3 GB resident.
Mitigations, in order of preference:

1. **Lazy by key, cached for the connector lifetime** (the existing model) — only
   the keys actually accessed are assembled. A `prob_profit`/EV scan that touches
   OHLCV + vol_iv pays for two, not three.
2. **Parquet sidecar cache.** First deep `_load` writes an assembled
   `data_processed/deep_cache/<key>.parquet`; subsequent loads read the parquet
   (≈ 10× faster than re-decompressing + concatenating gz). Invalidate on source
   mtime/hash. This is the single biggest perf lever for repeated backtests.
3. **Column projection** — `usecols` per accessor so vol_iv's 8 cols / iv_surface's
   27 cols aren't all materialized when only a few are needed.
4. **Per-ticker pruning for backtests** — when the harness already knows the PIT
   universe (Part B), pass it down so `_load` can filter rows to those tickers
   before the groupby index is built. (Optional; keeps the simple global cache as
   the default.)

The `_ticker_groups` `id(df)`-keyed cache (already in the connector) keeps a
full-universe sweep at one groupby pass per assembled frame — unchanged.

### A6. The IV moneyness/skew surface (`iv_surface__*`)

The 3 shards (2005→2026, 27 wide cols, current names only) are **not read by any
connector method today**. They are the real put-side skew source post-2004 (the
`hist_put/call_imp_vol` columns are zero-skew after 2004 — see roadmap §1d).
Design: a **new** accessor `get_iv_surface(ticker, as_of)` reading a 4th manifest
entry (`"iv_surface": [the 3 shards]`), returning the tenor×moneyness grid.
This is **additive** (new method, no existing caller) and feeds skew-aware strike
selection / the dormant SVI tooling (`DECISIONS.md` D9) — schedule it *after* the
core OHLCV/IV/liquidity deep-read, not as a blocker.

## Part B — Survivorship-aware backtesting

### B1. The machinery already half-exists

`data/consolidated_loader.py` already implements PIT membership:
`load_index_membership()` + `get_universe_as_of(as_of)` with correct
**"snapshot at or before `as_of`"** semantics and `normalize_ticker` keys. The
design **reuses** this rather than reinventing. Two facts to honor:

- `percentage_weight` is the **all-zeros sentinel** (`≈ -2.4e-14`) → the
  `min_weight` argument is a **no-op**; select by membership presence only.
- `consolidated_loader` is the *feature pipeline*; the **EV path uses
  `MarketDataConnector`**. So the harness wires them together (B2), it does not
  route the connector through consolidated_loader.

### B2. Harness change (in `backtests/`, not `engine/`)

Today `run_backtest(..., tickers=<fixed list>)` takes a **hardcoded modern
universe** and never consults membership → survivorship-biased by construction.
The PIT harness:

1. Load membership once (`ConsolidatedDataLoader.load_index_membership()`).
2. On each rebalance date `d`, set the candidate universe to
   `get_universe_as_of(d)` (PIT names, incl. delisted PIT-code stubs).
3. Feed that per-day list into `runner.rank_candidates_by_ev(tickers=…, as_of=d)`
   — the connector (deep-read on) serves each name's history, current or dead.
4. `assert_data_window_available` must check the **assembled** OHLCV span, not the
   monolith's (today it reads only `sp500_ohlcv.csv` → would reject any pre-2018
   start). Point it at the connector's min/max, or pass the deep floor.

No `engine/` change: the driver already routes 100% through
`rank_candidates_by_ev` / `rank_covered_calls_by_ev` (verified in
`backtests/regression/_common.py`).

### B3. Feasible windows (the gates)

- OHLCV deep floor 1994-01-03 (current) / 1990 (delisted); the **504-day
  survivorship gate** in the ranker ⇒ price-feasible backtests from **~1996**.
- vol_iv usable from 1994 (post-sentinel-null); **IV-dependent EV feasible
  ~1996**. The `iv_surface` skew is 2005+.
- A name that **goes to zero** (Lehman last bar 2008-09-12 @ $3.65; the panel
  carries it) flows through `get_ohlcv` like any other — the forward-replay P&L in
  the driver marks the assigned stock at the crashing spot, exactly the
  survivorship signal R6 must demonstrate.

### B4. Honesty note for early eras

Pre-2008 deep history is still **today's index projected backward** for the
*current* names plus whatever delisted names membership names — coverage tapers
(IV: 263/510 of today's names in 1994). The delisted panels fix the *omission*
of dead names, but per-name field completeness thins going back. Treat pre-2008
as **context-grade**, not a clean tradeable universe; the membership-driven
universe is the right set, but flag the coverage taper in any early-era result.

## Part C — Theta option chains in the cost model

### C1. Today

`rank_candidates_by_ev` **synthesizes the premium** via BSM:
`wheel_runner.py:1291` — *"Synthetic fair-value premium (mid). Real chains will
differ."* A chain (`conn.get_option_chain` / `conn.get_options`, when present) is
used **only** for open-interest lookup + the chain-quality gate, not for the
premium. `engine/theta_connector.py` exists (v3 REST, `_normalise_theta_symbol`).

### C2. Design — preferred → fallback cost source

A `PremiumSource` resolution, highest-fidelity first, all feeding the **same**
`EVEngine.evaluate` cost model (no §2 bypass; this only changes the *premium /
spread inputs*, never the verdict logic):

1. **Real Theta chain** (desktop, where covered): bid/ask → mid premium + true
   half-spread + OI at the target strike.
2. **BSM(iv) + `sp500_bid_ask.csv`**: synthetic mid (current behavior) but with a
   **real underlying** spread overlay from the new bid/ask sibling, and skew from
   `iv_surface` when the strike is off-ATM.
3. **BSM(iv) only** (current fallback): unchanged.

Provenance (`premium_source` ∈ {`theta_chain`, `bsm_bidask`, `bsm`}) rides on the
ev_row as a diagnostic so backtests can audit which fidelity tier priced each
trade. The chain-quality gate (already present) stays a **downgrade-only**
reviewer.

### C3. §2 + scope

This touches `wheel_runner` premium sourcing → it is a **decision-layer input
change** (re-baseline on adoption) and is **R4** in the roadmap — *planned, not
executed*. It must not weaken the downgrade-only contract: a richer/worse spread
can make a candidate worse (proceed→review/skip) but never rescue a negative-EV
trade. Theta historical option coverage is the survivor-biased larder
(`docs/THETA_LARDER_SCOPE.md`); delisted-name chains are **not** available, so
deep-history backtests (R6) price via tiers 2–3, and only recent/live runs get
tier 1.

## Part D — The two audit code fixes

### D1. Credit-rating dead-read — **DONE (R0a)**

`engine/data_connector.py.get_credit_risk()` returns the S&P rating under the
friendly key **`sp_rating`** (it maps raw `rtg_sp_lt_lc_issuer_credit →
sp_rating`). `wheel_runner.py:503` read the **raw** name → always missed →
`credit_rating` was silently `""` for every ticker (`sp500_credit_risk.csv`
wasted). Fixed to `credit.get("sp_rating", "")`.

**§2 / re-baseline:** safe, no re-baseline. `credit_rating` flows only into the
**legacy heuristic** `_calculate_wheel_score()` (`fund_score += 10` for A/B
ratings) — used by `screen_candidates()` (engine_api `/screen` + the module
demo), **never** `rank_candidates_by_ev` / `EVEngine` — plus the memo
(`trade_memo.py:482`) and API (`engine_api.py:976`) display. The four regression
snapshots are driven by the EV path, so they don't move. (It *does* change the
legacy screener's ordering for A/B-rated names — covered by the
`screen_candidates` unit tests, run on this branch.)

### D2. Sector-cap source — **PLANNED, not done (R0b)**

The audit asks to source sectors from the pulled `gics_sector_name` instead of
the hardcoded `DEFAULT_SECTOR_MAP` (`risk_manager.py:1579`), so names absent from
the map stop collapsing into one `"Unknown"` bucket. **This is not a one-liner**:

- `wheel_runner.py:1761/2679/3296` tag each ev_row `"sector"` via
  `DEFAULT_SECTOR_MAP` **deliberately**, with a comment that it must match *"the
  SAME source the sector_cap gate uses"* (`SectorExposureManager`). So the
  diagnostic and the **R9/R10 gate** must change **in lockstep** — otherwise the
  shown label diverges from the gated label.
- The connector already has the truth (`get_fundamentals → gics_sector_name`;
  `analyze_ticker` already loads it into `analysis.sector`). The fix routes the
  R9/R10 `SectorExposureManager` (via
  `portfolio_risk_gates.check_sector_cap`) to a connector-backed sector lookup,
  and points the ev_row tag at the same source.
- **§2:** R9/R10 are downgrade-only soft-warns (and dormant by default — the
  caps are armed on no default path today, per `PROJECT_STATE.md` §3). Changing
  the sector source can only change *which* candidates the cap downgrades/blocks
  when armed — never a rescue — but it **does move R9/R10 firing → re-baseline**,
  which is exactly the "consequential, touches the decision layer" class STEP 3
  defers. Recommend folding R0b's re-baseline into R1's single tagged re-baseline.

## Part E — Fingerprint extension (R5, do first)

`backtests/regression/_common.py:ohlcv_sha256()` pins **OHLCV only**. Add
`vol_iv` + `treasury` hashes to the snapshot `fingerprint` (and the deep gz
hashes once R2 lands) so a future data refresh **trips the guard** instead of
silently moving locked claims. Cheap, isolated to `backtests/`, no §2 surface —
land it before R1.

## Test / validation plan

- **R0a:** `pytest tests/test_wheel_runner_coverage.py tests/test_launch_blockers.py
  tests/test_tv_api.py -q` (the `screen_candidates`/`analyze_ticker` surface) +
  a new unit asserting `credit_rating` is populated from `sp_rating`.
- **R2:** unit tests — assembled span == expected (1994→2026); rotation invariant
  holds post-assembly; a known delisted PIT-code key (`Lehman`) returns its last
  bar; default-OFF path is byte-identical to today; missing-slice degrades.
- **R3:** the PIT universe on a 2008 date includes Lehman/WaMu and excludes
  post-2008 IPOs; `get_universe_as_of` count ≈ 500.
- **R6 smoke:** a 2007-2009 survivorship backtest where ≥1 assigned name goes to
  ~0; assert the loss flows into realized P&L (no silent drop).
- **Re-baseline (R1):** regenerate S27/S32/S34/S35 against main's engine + merged
  data via the `--update-snapshot` workflow, one tagged commit; then full `pytest`.

_Grounded in `engine/data_connector.py`, `engine/wheel_runner.py`,
`backtests/regression/_common.py`, `data/consolidated_loader.py`, and the
on-bytes inventory (`DATA_LAYER_ACTIVATION_ROADMAP.md` §1), all @ `main efc491c`
/ refresh `6bb3399` / deep `e7818f4`._
