# Next data session — turnkey execution runbook

> _Consolidates the open data queue (#339, #355, #354, #357) into one ordered,
> end-to-end execution plan so a single logged-in Bloomberg Terminal session
> clears it in one pass. Generated 2026-06-08 against `main @ 67b57fc`
> (the new HEAD after the #363 IV gate + #364 preflight guard landed). Re-run
> the seam/thin-history audit after any future refresh — newly-added
> constituents tend to land recent-only and add rows to these tables._

**Prerequisite:** a **logged-in Bloomberg Terminal** with `xbbg` reachable
(the Cowork sandbox carries only the committed CSVs — none of the Phase-A pulls
are possible there; see `docs/DATA_POLICY.md` §7). Phases B and C need **no
Bloomberg** and can be done anywhere from a clean clone of `main`.

---

## 0. The one distinction that drives the whole plan

Most of the queue is **NOT** Bloomberg-gated. Only the items in **Phase A** need
the Terminal; everything in **Phase B/C** is reconstructable from data already
in git (per `docs/CASY_BACKFILL_SPEC.md`, verified on the bytes 2026-06-07).

| Truly Bloomberg-gated (Phase A) | Git-reconstructable / local (Phase B–C) |
|---|---|
| CASY pre-2026 `ohlcv`/`vol_iv`/`liquidity`/`earnings` (#339, #355) | BK↔BNY entity collapse (#339) |
| 10 other blue-chip OHLCV backfills (#355) | `sp500_dividends.csv` union: refresh ∪ main's `CTRA/LW/MTCH/PAYC` (#339) |
| Refresh of the 3 script-producible files to the new frontier (optional) | Fragment integration + de-dup into the monoliths |
| #354 dated (PIT) fundamentals panel — larger data-model change | `UNIVERSE_100` re-derivation (#339) |
| #357 W10 `rate_1m` pre-2001 coverage — optional, low-pri | 4-snapshot re-baseline + marker re-run (#339) |

> If a Terminal is genuinely unavailable, **Phase B alone is still real
> forward progress on #339** (BK↔BNY collapse + dividends union are pure git),
> but it forces a *double* `UNIVERSE_100` re-derive + re-baseline (once without
> CASY, once after) — so the spec deliberately batches B after A. Do B-without-A
> only if the Terminal will be unavailable for a long time.

---

## Phase A — Bloomberg pulls (Terminal required)

Pull **fragment CSVs (the affected tickers' rows only)** — do **not** edit the
big monoliths on the pull box. Push/send the fragments; integration + de-dup is
Phase B, done here on a reviewed branch. Match each connector file's exact
schema and ticker format.

### A1. CASY — the authoritative spec already exists
`docs/CASY_BACKFILL_SPEC.md` is the canonical, on-the-bytes spec for CASY's
**4 files** (`ohlcv`, `vol_iv`, `liquidity`, `earnings`), with exact `xbbg`
snippets, the `CASY UW Equity` ticker, the `2018-01-02 → 2026-06-04` range, the
schema-per-file gotchas (e.g. liquidity's 3rd column is `shares_out` =
`EQY_SH_OUT`, not `bid_ask_spread`), and its validation gates (overlap to the
cent on the 52 existing `2026-03-23→` rows; verify the IV field against an
existing name first). **Follow that doc for A1 verbatim.**

### A2. The other 10 blue-chip OHLCV backfills (#355)
Same fragment-pull pattern as CASY's `ohlcv` step (`scripts/pull_ohlcv.py`'s
method: split-adjusted `blp.bdh` of `PX_OPEN/HIGH/LOW/LAST/VOLUME`, long format,
schema `date,ticker,open,high,low,close,volume`, ticker `"<TICKER> <exch> Equity"`).
Range `2018-01-02 → 2026-06-04` (overlap the existing tail rows for validation).
These carry `< 504` OHLCV bars on `main` so the ranker's `min_history_days=504`
gate drops them, though their real history is far longer:

| ticker | first bar (main) | bars | real history |
|---|---|---|---|
| WMT | 2025-12-09 | 122 | public since 1972 |
| KMB | 2025-05-30 | 255 | decades |
| CPB | 2024-08-19 | 450 | decades |
| DPZ | 2025-01-02 | 356 | IPO 2004 |
| PLTR | 2024-11-26 | 380 | IPO 2020 |
| VEEV | 2026-03-23 | 52 | IPO 2013 |
| COHR | 2026-03-23 | 52 | since 1987 (II-VI) |
| LITE | 2026-03-23 | 52 | IPO 2015 |
| SATS | 2026-03-23 | 52 | since 2008 |
| VRT | 2026-03-23 | 52 | since 2020 |

(CASY is the 11th name in #355's table — covered by A1.) **Also check `vol_iv`
coverage** for any name that is *also* IV-thin and backfill it the same way
(CASY's `vol_iv` step in the spec is the template); the OHLCV gate is the
headline, but a name needs IV to be fully tradeable.
**Do NOT** backfill the genuinely-recent names (BNY/FDXF/SNDK/SW/PSKY/Q) — those
are real new constituents, pinned by graceful-degradation tests, not defects.

### A3 (optional). Refresh the 3 script-producible files to the new frontier
Only `sp500_ohlcv.csv`, `sp500_liquidity.csv` (xbbg → Terminal), and
`treasury_yields.csv` (yfinance, no Terminal) have in-repo producers. If you
also want the frontier moved past `2026-06-04`:
- `scripts/pull_ohlcv.py` — **edit the hardcoded `end_date="2026-03-20"` at line 19** first.
- `scripts/pull_liquidity.py` — **edit `end_date="2026-03-20"` at line 26** first.
- `python scripts/pull_treasury_yields_yf.py --incremental` (already current; no Terminal).
The other 6 connector files (incl. the core `sp500_vol_iv_full.csv`) have **no
in-repo producer** — see `docs/bloomberg_refresh_runbook.md`. **If the frontier
moves, bump `EXPECTED_FRONTIER` (Phase C4).**

### A4 (optional, low-pri — #354 / #357)
- **#354 PIT fundamentals:** the real fix needs a **dated** fundamentals panel
  (today's `sp500_fundamentals.csv` is a single dateless 2026 snapshot, so every
  historical backtest reads 2026 values — lookahead). That is a larger
  data-model change (a per-date fundamentals history) *and* a separate
  **trio PR** to add `as_of` to `get_fundamentals`/`get_credit_risk` and thread
  it from `wheel_runner` (a decision-layer change, §2 review). Scope this on its
  own; it is not part of the one-pass data session.
- **#357 W10:** `treasury_yields.csv` `rate_1m` is NaN pre-2001 (23.4%). Only
  pull if pre-2001 1-month coverage is actually wanted; otherwise no action
  (the integrity band-pin already passes).

---

## Phase B — git-reconstructable integration (no Bloomberg, reviewed branch)

Per `docs/CASY_BACKFILL_SPEC.md` §"After you push/send the fragments":

1. **Integrate fragments** — fold the CASY + 10 blue-chip fragments into the
   four monoliths (`sp500_ohlcv.csv`, `sp500_vol_iv_full.csv`,
   `sp500_liquidity.csv`, `sp500_earnings.csv`), **de-duping** the existing
   recent rows against the overlap window.
2. **BK↔BNY collapse** — fold `BNY` into `BK`'s continuous history so the
   connector sees **one** entity (BK OHLCV runs full `2018→2026-03-20`; `BNY`
   re-tickers the rest; `BNY UN` dividends already carry BK's full history). This
   frees the phantom universe slot. *(Pure git — no Bloomberg.)*
3. **Dividends union** — `sp500_dividends.csv` = refresh ∪ main's
   `CTRA UN / LW UN / MTCH UW / PAYC UN` rows (the refresh dropped their
   2018–2024 history; `MTCH` was the one #339 originally missed). Current **and**
   complete. *(Pure git.)*
4. **Re-derive `UNIVERSE_100`** — after B2/B3, regenerate the constant in
   `backtests/regression/universes.py` from
   `MarketDataConnector().get_universe()[:100]` (expect `CMG`/`CMI` to **return**
   as `BNY` collapses into `BK`, and CASY to become a legitimate in-window
   member). `test_universes_match_connector` enforces the derivation matches the
   connector — run it green before proceeding.

---

## Phase C — re-baseline + verify

5. **Expect the drift guard to go red first (this is the signal, not a bug).**
   `tests/test_backtest_regression.py::test_snapshot_data_fingerprint_matches_current`
   (the #340 guard) compares each snapshot's pinned `connector_data_sha256`
   against the live connector set and **fails the moment the data changes** —
   forcing an explicit re-baseline before the multi-hour markers run. Diagnose
   first per `TESTING.md` §"Backtest regression — re-baseline workflow".
6. **Re-baseline all four snapshots** (~4 h total). A dividends change moves the
   covered-call realized cash on *all four* (not just S34) — proven in R1 — so
   regenerate every one:
   ```
   python -m backtests.regression.s27_ivpit_24t_100k   --update-snapshot
   python -m backtests.regression.s32_friction_24t_1m  --update-snapshot
   python -m backtests.regression.s34_universe_100t_1m  --update-snapshot
   python -m backtests.regression.s35_oos_24t_100k     --update-snapshot
   ```
   Amend each snapshot's note preserving the original numbers; file a
   snapshot-update record (see `TESTING.md` step 4).
7. **Confirm the markers** against the new baselines (~2.5 h):
   ```
   pytest tests/test_backtest_regression.py -m backtest_regression
   ```
   All four must pass (byte-identical to the regenerated snapshots).
8. **Bump `EXPECTED_FRONTIER`** in `tests/test_preflight_environment.py` to the
   new OHLCV frontier — **only if A3 moved it** — in the same commit as the
   refresh (the guard-rot rule from `docs/DATA_POLICY.md` §5). CASY/blue-chip
   backfills extend *history*, not the frontier, so they alone do **not** require
   a bump.
9. **Flip the xfail trackers** as each lands (`xfail(strict=True)` → green):
   - `tests/test_data_to_engine.py::test_blue_chip_history_is_complete[<ticker>]`
     — one per backfilled name (#355).
   - `tests/test_data_to_engine.py::test_fundamentals_credit_are_point_in_time`
     — only when the #354 PIT accessor lands (separate trio PR).
10. **Clear S34's ⚠️ provisional flag** (set in PR #338) and run the **§2 review
    panel** on the connector/universe change — it shifts the backtest universe
    but `EVEngine.evaluate` is untouched (no §2 bypass; this is a data/universe
    change, not a decision-layer one).

---

## Validation gates (from `docs/CASY_BACKFILL_SPEC.md`)

- **CASY OHLCV overlap** (`2026-03-23 → 2026-06-04`) matches the 52 existing
  committed rows **to the cent**.
- **vol_iv implied-vol field** verified against an existing name (e.g. `AAPL UW`)
  over a recent month before trusting any backfilled IV (Bloomberg has **no
  put/call skew** — `hist_put_imp_vol == hist_call_imp_vol` exactly; see memory
  `bloomberg-iv-no-skew`).
- **Seam audit** post-integration shows **zero** recent-only (0-in-window) names
  left in `UNIVERSE_100`.
- `test_universes_match_connector` **green**.

---

## Cross-references

- `docs/CASY_BACKFILL_SPEC.md` — authoritative CASY pull spec (Phase A1) + the
  reconstructable-vs-Bloomberg breakdown.
- `docs/bloomberg_refresh_runbook.md` — per-file producer reality (which of the 9
  connector files have a script vs. need recovered BQL).
- `docs/DATA_POLICY.md` §5 — refresh procedures + the `EXPECTED_FRONTIER`
  bump-on-refresh rule.
- `TESTING.md` §"Backtest regression — re-baseline workflow" — the canonical
  re-baseline procedure + the snapshot drift guard.
- `docs/worklog/r1-data-refresh-rebaseline-r1-bloomberg-data-refresh-s27-s32-s34-s35-re-bas.md`
  — R1's "Unresolved / handoff" (the dividends-history drop + provisional S34).
- Issues: **#339** (BK↔BNY + CASY + dividends + re-baseline), **#355** (11
  blue-chip backfills), **#354** (PIT fundamentals lookahead), **#357** (W10/W11
  low-pri hygiene).
