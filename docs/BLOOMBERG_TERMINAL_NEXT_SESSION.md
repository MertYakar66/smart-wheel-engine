# Bloomberg Terminal — next-session runbook

_What to run on a **logged-in Bloomberg Terminal** to advance the data layer.
Authored 2026-06-27, after Phase 1 of the wiring campaign landed (#354 carry-q,
#372 GICS, #369 fallback-IV, #378 IV-staleness+rate; all on `main`). This is the
**operator action list**; for the exhaustive field-level checklist see
[`BLOOMBERG_PULL_LIST.md`](BLOOMBERG_PULL_LIST.md), and for what is already on
`main`/disk (so you don't re-pull) see [`DATA_INVENTORY.md`](DATA_INVENTORY.md)._

> **Why a Terminal is required:** the Bloomberg pullers use `xbbg` (`from xbbg
> import blp`) which only works against a running, entitled Terminal. The
> `*_yf.py` pullers (yfinance) and everything under `data/bloomberg/broad_pull/`
> (already on `main`) do **not** need the Terminal.

---

## 0. Prerequisites (once per session)

1. Bloomberg Terminal open and logged in; `=BLP` Excel add-in or `xbbg` reachable.
2. Repo venv active: `.venv\Scripts\python.exe` (has `xbbg`, pandas).
3. Confirm connectivity: `python -c "from xbbg import blp; print(blp.bdp('AAPL US Equity','PX_LAST'))"` returns a price.
4. **FLDS-verify every mnemonic** in the Terminal (`FLDS <field>`) before a large pull — Bloomberg renames fields.
5. Work on a branch; the pulled CSVs land in `data/bloomberg/` (gitignored large files stay local; committed CSVs are the served monoliths).

---

## 1. PRIORITY 1 — Phase 0A frontier bump (the immediate next step)

**Goal:** advance the served price/spot frontier `2026-06-04 → <today>` so the
engine ranks against current data, and pay the coupled re-baseline once.

> **Sequencing (already satisfied):** #378 (IV-staleness gate) is **on `main`**,
> so the moment OHLCV advances past the IV monolith, stale-IV names fail closed
> to the fundamentals fallback instead of mispricing BSM. You may bump the
> frontier safely.

### 1a. Refresh the served OHLCV monolith
- File/puller: `scripts/pull_ohlcv.py` (`xbbg.blp.bdh`, fields `PX_OPEN/HIGH/LOW/LAST/VOLUME`, `SPX Index` members).
- **Edit `end_date`** (currently hardcoded `"2026-03-20"` → set to the new frontier date) and, if you want the full window, leave `start_date="2018-01-01"`.
- Run it; it overwrites `data/bloomberg/sp500_ohlcv.csv`.
- **Split-adjusted** by Bloomberg — never mix with Theta raw. Re-run the seam/scale audit after (`pytest tests/test_data_integrity_bloomberg.py -q`); the BKNG/CVNA 2026-03-23 split-seam pins should stay green (or flip to green on a clean re-pull — lift the `KNOWN_SCALE_BREAKS` exclusions then).

### 1b. Refresh the served ATM-IV monolith (closes the staleness gap)
- File: `data/bloomberg/sp500_vol_iv_full.csv` — **no in-repo producer script** (this is the W36 gap). Pull `hist_put_imp_vol` / `hist_call_imp_vol` / `volatility_{30,60,90,260}d` for `SPX Index` members to the new frontier via BQL/`blp.bdh`, append to the monolith.
- If you skip this, #378's IV-staleness gate will (correctly) drop names whose IV now lags spot by >30 days — conservative, but you lose those candidates until the IV is refreshed.

### 1c. Refresh the other script-producible served files (optional, same session)
- Treasury curve is already current (1994→2026); liquidity / vol-indices can be extended via `scripts/pull_liquidity.py` / `scripts/pull_vol_indices.py` if you want them at the new frontier.

### 1d. Post-pull — re-baseline + frontier bump (code, after the pull)
This is the coupled tail the agent runs once the new CSVs are committed:
1. Re-pin the 4 regression snapshots: `python -m backtests.regression.<sNN> --update-snapshot` for `s27_ivpit_24t_100k`, `s32_friction_24t_1m`, `s34_universe_100t_1m`, `s35_oos_24t_100k` (S34 ≈ 2 h). Verify input-SHAs change **only** by the intended files.
2. Bump `EXPECTED_FRONTIER` in the preflight env-guard and re-derive `UNIVERSE_100`.
3. Re-pick the W16/W30 earnings-window test names **only if the frontier moved them**.
4. `pytest tests/ -m "not backtest_regression"` green; ruff clean.

---

## 2. PRIORITY 2 — capability unlocks (what makes dormant engine features live)

Pull these to light up engine paths that are built but starved of data. Listed
by **engine ROI**, highest first.

### 2a. ⭐ Real option premiums (the skew/VRP unlock) — **Theta, not Bloomberg**
- **Why first:** the skew surface (Phase 2) is **structurally inert on EV until
  this lands.** Today the candidate premium is *synthetic* (BSM from the same IV
  used for fair value), so `edge_vs_fair ≡ 0` — a richer (skew-aware) IV moves
  the price *level* but not the *edge*. A real market-mid premium to compare a
  skew-aware fair value against is what creates VRP edge.
- **Source:** Theta EOD option chains (already held: `data_processed/theta/option_history*`, ~390M rows, OI + OHLC). The EOD **mid** (from option OHLC) is the real premium; greeks/IV are 404/not-entitled → back-solve. No new pull needed if the larder covers the window — **this is wiring work, not a Terminal pull** (see §3).
- Per-strike OI/greeks/smile is **not on Bloomberg** (OMON manual only) — always Theta.
- **Status — data half DONE (no Terminal needed):** `scripts/produce_option_premiums.py`
  distils the larder → gitignored `data_processed/option_premium/<T>.parquet` (real EOD `mid`,
  PIT date axis, DTE belt), served by `MarketDataConnector.get_option_premium*`. Validated on
  AAPL/MSFT/NVDA: coverage **2016 → 2026-06**, ~12 s/ticker. Regenerate locally with
  `python scripts/produce_option_premiums.py --tickers all --workers 4` (gitignored ⇒ no
  re-baseline). **Remaining = the EV-moving ranker wiring** (swap `ShortOptionTrade.premium`
  from synthetic-BSM to the served mid at the three ranker sites): CEREMONY-tier (trio +
  lane-claim + §2-panel), owns the re-baseline. That is the actual "skew/VRP unlock," not this rail.

### 2b. Macro-event calendar — **already on `main`** (`broad_pull/macro_calendar`)
- FDTR/CPI/NFP/PCE/GDP/ISM release dates + actual/survey/importance. **No pull** — wire it into a market-wide `event_gate` lockout (remove-only). High value for a vol-selling book around FOMC/CPI.

### 2c. IV-surface breadth (skew) — extend, don't acquire
- `data/bloomberg/broad_pull/iv_surface/sp500_iv_surface.csv.gz` is **on `main`** (1.94M rows, 509 names, 2010→2026, real 5×5 moneyness×tenor grid `iv_{30,60,90,180,365}d_{90,95,100,105,110}`). A deep 5×5 archive (2005→2026) is on local disk too.
- A Terminal pull would only **add the {80,120} wings** (true ~25Δ) and the recent tail. Defer until §2a makes skew EV-relevant.

### 2d. Dated/PIT fundamentals — partially on `main`
- The dated **dividend-yield** panel (`broad_pull/dividend_pit`) is on `main` and wired (#354). The broader PIT fundamentals/credit-ratings panels (W28) still need a dated pull (`BEST_PERIOD_END_DT` / `RATING_CHANGE_DT` alongside the values) before the credit reviewer and PIT-fundamentals paths can be made point-in-time.

### 2e. Short interest / borrow — **already on `main`** (`broad_pull/short_interest`, 134,035 rows)
- Wire into R10 / a borrow-cost overlay. No pull.

---

## 3. What is already on `main`/disk — do NOT re-pull (wire instead)

- **27-dataset broad pull** at `data/bloomberg/broad_pull/` (iv_surface, dividend_pit, macro_calendar, short_interest, snapshot_bdp/GICS, …) — committed, dormant. The dominant remaining work is **wiring**, not acquisition.
- **Corporate actions** — on `main`, 52,442 rows (`sp500_corporate_actions.csv`); only the `event_gate` wiring remains.
- **Treasury curve** — 1994→2026, done.
- **Deep archive** (gitignored, local) — OHLCV/vol-IV/liquidity/5×5 IV-surface 1994/2005→2026.
- **Theta larder** (gitignored, local) — ~390M EOD option rows (OI+OHLC), 2016→2026.

---

## 4. One-glance order of operations

1. **Terminal session:** §1a OHLCV bump + §1b IV-monolith refresh (→ closes the staleness gap) [+ §1c optional, + §2d/§2c if doing a full pull].
2. **Agent (post-pull, code):** §1d re-baseline + frontier bump + verify.
3. **Agent (no Terminal needed, highest ROI):** wire §2a real premiums (Theta) → unlocks skew; then §2b macro gate, §2e short-interest, corporate-actions gate — all from data already on `main`/disk.

> **Bottom line:** the single Terminal action that unblocks the most is the
> **§1 frontier bump (OHLCV + IV monolith)**. The single highest-ROI *non-Terminal*
> action is **§2a real premiums (Theta)** — without it, skew stays EV-inert.
