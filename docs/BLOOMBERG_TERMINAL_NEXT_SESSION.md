# Bloomberg Terminal — next-session runbook

_**Refreshed 2026-07-21** (supersedes the 2026-06-27 version, which targeted a
`2026-06-04` frontier that has since advanced). This is the **current, verified**
operator action list for a logged-in Bloomberg Terminal — every command, path,
field mnemonic, and file:line below was checked against the repo at refresh time.
Worked-example run date is `2026-07-21`; **substitute the day you actually run
it** everywhere it appears. For the exhaustive field-level checklist see
[`BLOOMBERG_PULL_LIST.md`](BLOOMBERG_PULL_LIST.md); for what is already on
`main`/disk (so you don't re-pull) see [`DATA_INVENTORY.md`](DATA_INVENTORY.md)._

> **Why a Terminal is required:** the Bloomberg pullers use `xbbg` (`from xbbg
> import blp`), which only works against a running, entitled Terminal. The
> `*_yf.py` pullers (yfinance) and everything already committed under
> `data/bloomberg/broad_pull/` do **not** need the Terminal. Theta-sourced data
> (real option premiums, per-strike chains/greeks/OI, VIX futures) **cannot** be
> refreshed at a Bloomberg seat at all — see §6.


**For:** university Bloomberg Terminal (xbbg / Excel BDH+BQL). **No Theta at the lab.**
**Universe:** Universe A = current SPX members (`blp.bds("SPX Index","INDX_MWEIGHT")` — historical windows use *today's* members).
**Worked example run date (`PULL_DATE`):** `2026-07-21`. Replace with the day you actually run it, everywhere it appears.
**Served connector:** `MarketDataConnector` (`engine/data_connector.py`), the only reader on the EV path. `bloomberg_loader.py` / `ConsolidatedBloombergLoader` are NOT served — ignore them.

> **The one trap that ruins a refresh:** the panel pullers ship with `end_date="2026-06-04"` hardcoded, but the CSVs on disk already reach **2026-07-02**. The scripts are forward-gap-aware (they only pull `[existing_max+1 → END]`), so **a bare run with the stale hardcoded end appends nothing**. You MUST set `SWE_PULL_END=2026-07-21` (or edit the line) or the frontier will not move.

---

## 1. TL;DR — one-glance order of operations

Only three pulls actually move the EV ranker; do them first, in this order:

| # | What | Command (from repo root, Terminal logged in) | Why it matters |
|---|---|---|---|
| 1 | **OHLCV frontier** | `SWE_PULL_END=2026-07-21 python scripts/pull_ohlcv.py` | spot + 5y forward distribution; the frontier `EXPECTED_FRONTIER` is pinned to |
| 2 | **IV / vol monolith** | `SWE_PULL_END=2026-07-21 python scripts/pull_vol_iv.py` | IV rank / percentile / VRP; ATM IV fallback |
| 3 | **Earnings-calendar overlay** | `SWE_SNAPSHOT_ASOF=2026-07-21 python scripts/pull_snapshot_bdp.py` | primary forward earnings lockout (100% coverage) |
| 4 | **Re-baseline + gate** | bump the 5 stale-clone pins + regen the 4 backtest fingerprints, then run the integrity + preflight + smoke tests (§7) | a data change is *expected* to trip these; that IS the signal |

Second-order (refresh if you have time): treasury, vix_term_structure, dividends, corporate_actions, fundamentals snapshot, dividend_pit. Everything in §6 is **skippable** (off the served EV path, already current, or physically un-refreshable at the lab).

---

## 2. Prerequisites

1. **Terminal login** — be logged into the Bloomberg Terminal on the same machine (xbbg talks to `blpapi` on `localhost`).
2. **xbbg connectivity check** — a live value proves the session is up:
   ```python
   from xbbg import blp
   print(blp.bdp("SPX Index", "PX_LAST"))          # non-empty -> connected
   print(blp.bds("SPX Index", "INDX_MWEIGHT").head())  # universe resolves
   ```
3. **FLDS-verify the mnemonics before a big pull** (entitlements vary by terminal). On the Terminal type `FLDS <GO>`, or probe from Python:
   ```python
   blp.bdp("AAPL US Equity",
           ["HIST_PUT_IMP_VOL","HIST_CALL_IMP_VOL",
            "VOLATILITY_30D","VOLATILITY_260D","EQY_DVD_YLD_12M"])
   ```
   Known-blocked at typical university tiers (do NOT waste time on these): `EQY_SHORT_INTEREST_PCT_OF_FLOAT`, `EQUITY_SHORT_BORROW_RATE_NET`, `CDS_SPREAD_*`, rating WATCH/OUTLOOK, ESG, `7DAY/14DAY` IV tenors, `BEST_EPS_SD`. Substitutes already wired: `SHORT_INTEREST`+`SHORT_INT_RATIO`, `VOLATILITY_nD`.
4. **Branch — never commit to `main`:**
   ```bash
   git checkout -b refresh/bloomberg-2026-07-21
   ```
5. **Chunking** (panel scripts): default `SWE_PULL_CHUNK=250` is fine for a short forward gap. Use `SWE_PULL_CHUNK=60` only for a fresh multi-year window (dodges BDH size caps).

---

## 3. PRIORITY 1 — core frontier bump (CRITICAL daily monoliths)

All three currently end **2026-07-02** (verified on disk). They are forward-gap-append; the START is only the historical origin and is irrelevant to a routine append (kept for a full rebuild — do **not** truncate; the backtest harness needs 2018 depth, see §7).

| File | Current last-date | Pull method (preferred) | Fields (mnemonics) | Range START → PULL_DATE | Output columns |
|---|---|---|---|---|---|
| `data/bloomberg/sp500_ohlcv.csv` | **2026-07-02** | **`scripts/pull_ohlcv.py`** — set `SWE_PULL_END`, or edit end at **line 67** (`end_date="2026-06-04"`) | `PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME` | `2018-01-01` → `2026-07-21` | `ticker,date,open,high,low,close,volume` **(stored ROTATED: open←PX_HIGH, high←PX_LAST, close←PX_OPEN; the script does this so the connector's inverse rename at `:518-524` lands correct)** |
| `data/bloomberg/sp500_vol_iv_full.csv` | **2026-07-02** | **`scripts/pull_vol_iv.py`** — set `SWE_PULL_END`, or edit **line 53** (`end_date="2026-06-04"`) | `HIST_PUT_IMP_VOL, HIST_CALL_IMP_VOL, VOLATILITY_30D, VOLATILITY_60D, VOLATILITY_90D, VOLATILITY_260D` (`Fill="P"`) | `2018-01-02` → `2026-07-21` | `ticker,date,hist_put_imp_vol,hist_call_imp_vol,volatility_30d,volatility_60d,volatility_90d,volatility_260d` |
| `data/bloomberg/sp500_liquidity.csv` | **2026-07-02** | `scripts/pull_liquidity.py` — set `SWE_PULL_END` (**dormant/off-EV** — `liquidity_score` is derived from market_cap, not this file; refresh for parity, skippable) | `VOLUME_AVG_30D, TURNOVER, EQY_SH_OUT` (`Fill="P"`) | `2015-01-02` → `2026-07-21` | `ticker,date,avg_vol_30d,turnover,shares_out` |

**Copy-paste (preferred — env override, no file edit):**
```bash
export SWE_PULL_END=2026-07-21          # <- your run date; MUST be > 2026-07-02 or nothing appends
python scripts/pull_ohlcv.py            # writes data/bloomberg/sp500_ohlcv.csv (handles rotation + post-write rotation gate)
python scripts/pull_vol_iv.py           # writes data/bloomberg/sp500_vol_iv_full.csv
python scripts/pull_liquidity.py        # optional (dormant)
unset SWE_PULL_END
```

**Alternative (edit-in-place):** `pull_ohlcv.py` line 67 and `pull_vol_iv.py` line 53: change `end_date="2026-06-04",` → `end_date="2026-07-21",`, then run bare.

**Runbook-myth correction (W36 "no producer"):** `sp500_vol_iv_full` is **not** producer-less — `pull_vol_iv.py` is the in-repo producer (its docstring even carries the exact BQL). The manual Excel/BQL routes below are only a Terminal fallback.

**Manual fallback — `sp500_vol_iv_full` (xbbg or Excel), if you skip the script:**
```python
from xbbg import blp
members = blp.bds("SPX Index","INDX_MWEIGHT").index.tolist()   # ["A US Equity", ...]
blp.bdh(members,
        ["HIST_PUT_IMP_VOL","HIST_CALL_IMP_VOL","VOLATILITY_30D","VOLATILITY_60D","VOLATILITY_90D","VOLATILITY_260D"],
        "2018-01-02","2026-07-21", Fill="P")
```
Excel (tickers in column A, drag down):
```
=BDH(A1&" Equity","HIST_PUT_IMP_VOL,HIST_CALL_IMP_VOL,VOLATILITY_30D,VOLATILITY_60D,VOLATILITY_90D,VOLATILITY_260D","2018-01-02","2026-07-21","Fill=P","Dir=V")
```
Equivalent BQL:
```
=BQL.QUERY("get(hist_put_imp_vol, hist_call_imp_vol, volatility_30d, volatility_60d, volatility_90d, volatility_260d) for(members('SPX Index')) with(dates=range(2018-01-02,2026-07-21), fill=prev)")
```

**Manual fallback — OHLCV (Excel), DANGER:** raw `=BDH(A1&" US Equity","PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME","2018-01-01","2026-07-21","Dir=V")` gives *straight* columns; the served monolith needs the **rotated** layout. **Prefer `pull_ohlcv.py`** — it applies the rotation and a post-write rotation gate. If you must do Excel, feed the exports through `scripts/process_bloomberg_exports.py` (§7 step 1); do not hand-map.

---

## 4. PRIORITY 2 — events & macro

"Served?" = read by `MarketDataConnector` on the EV path. Off-path files are listed because the task requests them, but they change no engine output until someone wires them — refresh only if you're also doing wiring.

| File | Served? | Current last-date | Pull method | Fields | Range / cadence |
|---|---|---|---|---|---|
| `sp500_dividends.csv` | **Yes** (ex-div lockout) | ex_date → 2027-03-12 (fwd present) | `python scripts/pull_dividends.py` — **bare run, no date to edit** | `EQY_DVD_HIST_ALL` (Declared/Ex/Record/Payable/Amount/Freq/Type) | full history; dedup `(ticker,ex_date)` keep-last |
| `sp500_corporate_actions.csv` | **Yes** (corp-action lockout + split-adjust) | effective → 2027-03-12 | `python scripts/pull_corporate_actions.py` — **bare run** | maps `EQY_DVD_HIST_ALL` → announce/effective/type/ratio/amount | backfills to ~1962; already current, low value |
| `vix_term_structure.csv` | **Yes** (regime mult + R11 `vix_level`) | 2026-07-02 | `SWE_PULL_END=2026-07-21 python scripts/pull_vix_term_structure.py` | `PX_LAST` for `VIX / VIX3M / VIX6M Index` | `date,vix,vix_3m,vix_6m`; VIX is in **POINTS** |
| `treasury_yields.csv` | **Yes** (BSM `r`) | 2026-07-02 | `SWE_PULL_END=2026-07-21 python scripts/pull_treasury_yields.py` | `PX_LAST` for `USGG1M/3M/6M/2YR/5YR/10YR/30YR + SOFRRATE` | **OPTIONAL** — runbooks mark DONE; already current. `PX_LAST` is already PERCENT, write UNCHANGED (D20) |
| `broad_pull/macro_calendar/sp500_macro_calendar.csv` | **Yes** (FOMC/CPI/NFP lockout) | release → 2027-12-08 | **No script** — BQL `scripts/bloomberg_bql_pulls.md §1` | `eco_release_dt, eco_release_event, eco_importance, eco_country` | **already forward-dated to 2027 — skip unless empty** |
| `sp500_earnings.csv` | Yes (thin historical) | announce → 2028-01-19 | **NO producer.** Forward earnings served by `snapshot_bdp` (§5) — refresh that instead | — | do not chase this file |
| `sp500_macro.csv` | **No** (ConsolidatedLoader only) | 2026-06-04 | `SWE_PULL_END=2026-07-21 python scripts/pull_macro.py` | `PX_OPEN/HIGH/LOW/LAST` × `USGG10YR, USGG2YR, SPX, DXY, XAU, CL1` | off served path — cosmetic |
| `sp500_index_membership.csv` | **No** (ConsolidatedLoader only) | (quarterly, ~2026-04-01) | `SWE_MEM_END=2026-07-01 python scripts/pull_index_membership.py` | `INDX_MWEIGHT_HIST` (+`NAME`) | quarterly grid; off served path |
| `sp500_sector_etfs.csv` | **No** | 2026-06-05 | `python scripts/pull_sector_etfs.py` — **bare run** (default end = today) | `PX_OPEN/HIGH/LOW/LAST/VOLUME` × 11 XL* ETFs | off served path |
| `broad_pull/short_interest/sp500_short_interest.csv` | **No** (dormant) | 2026-06-15 | `SWE_PULL_START=2026-06-01 SWE_PULL_END=2026-07-21 python scripts/pull_short_interest.py` | `SHORT_INTEREST, SHORT_INT_RATIO` | append; off served path |

**Copy-paste for the served P2 set:**
```bash
python scripts/pull_dividends.py
python scripts/pull_corporate_actions.py
SWE_PULL_END=2026-07-21 python scripts/pull_vix_term_structure.py
# optional (already current): SWE_PULL_END=2026-07-21 python scripts/pull_treasury_yields.py
```

---

## 5. PRIORITY 3 — fundamentals / estimates / snapshots

| File | Served? | Current | Pull method | Fields | Cadence note |
|---|---|---|---|---|---|
| `sp500_fundamentals.csv` | **Yes** (screen_universe, R9 sector, IV fallback, carry-q) | snapshot (1 row/tkr) | `python scripts/pull_snapshots.py` — BDP snapshot, **bare run** (`SWE_SNAP_SMOKE=1` to preview 3 tkrs) | `30day_impvol_100.0%mny_df, volatility_30d, eqy_dvd_yld_12m, gics_sector_name, beta_raw_overridable, cur_mkt_cap, pe_ratio, best_pe_ratio, free_cash_flow_yield, return_com_eqy, tot_debt_to_tot_eqy, gics_industry_group_name` | **snapshot = current values; headers must ship LOWERCASE** (connector does not lowercase). `pull_snapshots.py` already pulls `VOLATILITY_30D`. |
| `broad_pull/per_name/sp500_snapshot_bdp.csv` | **Yes** (primary earnings lockout) | asof=2026-07-03; next_earnings→2026-10-28 | `SWE_SNAPSHOT_ASOF=2026-07-21 python scripts/pull_snapshot_bdp.py` — **append-only new vintage** | `EXPECTED_REPORT_DT` → `next_earnings_dt`, GICS, ratings | **bump `EXPECTED_EARNINGS_CALENDAR_ASOF` in the SAME commit** (§7-4) |
| `broad_pull/dividend_pit/sp500_dividend_yield_pit.csv` | **Yes** (EV-moving PIT carry-q) | 2026-06-30 | `SWE_PULL_START=2026-06-01 SWE_PULL_END=2026-07-21 python scripts/pull_dividend_yield_pit.py` | `EQY_DVD_YLD_12M, EQY_DVD_YLD_IND, DVD_SH_12M` (`Per="M"`) | **monthly** append; dedup `(date,ticker)` |
| `sp500_credit_risk.csv` | No (dead read, off EV) | snapshot | `python scripts/pull_snapshots.py` (same run as fundamentals) | `rtg_sp_lt_lc_issuer_credit, altman_z_score, interest_coverage_ratio` | memo/legacy only |
| `sp500_earnings_estimates.csv` | No (not in `_FILES`) | snapshot | `python scripts/pull_earnings_estimates.py` — bare run | `BEST_EPS*, BEST_SALES*, EXPECTED_REPORT_DT` | off served path |
| `sp500_historical_fundamentals.csv` | No (not in `_FILES`) | 2026-03-20 | **EDIT-ONLY** `scripts/pull_historical_fundamentals.py` **line 29** `end_date="2026-03-20"` → `"2026-07-21"` (no env override exists), `Per="Q"` | `PE_RATIO, IS_EPS, SALES_REV_TURN, EBITDA, BOOK_VAL_PER_SH` | **quarterly — Q2-2026 may not post until August; a July pull may show a partial quarter.** Off served path |

**Copy-paste for the served P3 set:**
```bash
python scripts/pull_snapshots.py                                  # sp500_fundamentals.csv (+ credit_risk)
SWE_SNAPSHOT_ASOF=2026-07-21 python scripts/pull_snapshot_bdp.py  # earnings overlay (new vintage)
SWE_PULL_START=2026-06-01 SWE_PULL_END=2026-07-21 python scripts/pull_dividend_yield_pit.py
```

---

## 6. DO NOT re-pull (waste-of-time / impossible-at-lab)

**Physically un-refreshable at a Bloomberg Terminal (Theta-only — the lab has no Theta):**
- `data_processed/option_premium/*.parquet` — real-EOD option premiums (gitignored, from the Theta larder). Absent → engine uses synthetic-BSM fallback. **Cannot be produced from Bloomberg.**
- Live per-strike option chains, per-strike OI/greeks, per-strike IV-surface snapshots (`data_processed/theta/*`), greeks *history* (v3 404/not-entitled).
- `data_processed/vol_indices.parquet` (VVIX/SKEW/MOVE/…) — `pull_vol_indices.py` is Theta/yfinance, **not xbbg**.
- VIX futures UX1–UX8 — tier-blocked on **both** providers.

**Already on `main` / current — leave alone:**
- `treasury_yields.csv` — DONE, full tenor+SOFR, current to 2026-07-02 (listed optional in §4).
- `sp500_corporate_actions.csv` — 52k+ rows to 2027-03-12, already forward-current.
- The committed `broad_pull/` tree beyond the three consumed panels (`iv_surface`, `options_sentiment`, `vol_term_rv`, `macro_*`, `estimates`, `fundamentals_q`, …) — Bloomberg-pulled but **dormant/unconsumed**; refreshing changes no EV output.

**Deep history — do NOT acquire (opt-in, off by default):**
- `data/bloomberg/deep/*.csv.gz` (OHLCV/vol-IV/liquidity 1994–2018 + `__delisted` survivorship, 5×5 IV-surface 2005–2026). Read only when `SWE_DEEP_HISTORY=1`. **Do not truncate the live monoliths' 2018 start** to "save space" — the backtest harness needs it (§7).

**Unconsumed parallels — refreshing is a no-op for the engine:**
- `sp500_fundamentals_yf.csv`, `sp500_earnings_yf.csv`, `treasury_yields_yf.csv` (yfinance mirrors; connector reads the Bloomberg files).
- `sp500_vol_dvd.csv`, `sp500_vix_full.csv`, `sp500_analyst.csv` — read only by the **non-served** `ConsolidatedBloombergLoader`, never by `MarketDataConnector` (see §8).

---

## 7. Post-pull — process, validate, re-baseline

**Minimum-history rule:** keep both monoliths starting **2018-01-02** — this is an *append forward*, never a re-pull with a later start. The live EV path needs 5y (the `lookback_years=5.0` empirical/block-bootstrap forward distribution); the full 2018 depth is forced by `sim200k_reliability.py::WINDOWS` (earliest window 2020-02-03 + 504-bar HMM warmup + 5y lookback). Truncating below 2018-01-02 breaks the COVID-2020 / 2022-bear regression windows.

**A. (Excel route only) shape raw exports → engine schema:**
```bash
python scripts/process_bloomberg_exports.py --input <export_dir> --output data/bloomberg
```
Skip if you used the `pull_*.py` producers (they write the monolith schema directly).

**B. Integrity / seam / scale gate (primary):**
```bash
pytest tests/test_data_integrity_bloomberg.py -v
pytest tests/test_bloomberg_loader.py tests/test_broad_pull_loaders.py tests/test_data_to_engine.py -v
```
If `test_no_unexplained_split_scale_breaks` goes red (a split-splice landed at the pull boundary):
```bash
python scripts/fix_ohlcv_split_scale_439.py     # back-adjusts pre-seam history, then re-run B
```

**C. Bump the 5 stale-clone pins — in the SAME commit** (they false-pass on old data if not bumped). All currently `2026-07-02` / `2026-07-03` → set to your run date's last bar (`2026-07-21`, or the actual last trading day pulled):

| Constant | File : line | New value |
|---|---|---|
| `EXPECTED_FRONTIER` | `tests/test_preflight_environment.py:48` | `pd.Timestamp("2026-07-21")` |
| `EXPECTED_FRONTIER` | `tests/test_data_connector.py:757` | `pd.Timestamp("2026-07-21")` |
| `FRONTIER` | `tests/test_data_to_engine.py:39` | `"2026-07-21"` |
| `FRONTIER` | `tests/test_data_integrity_bloomberg.py:36` | `pd.Timestamp("2026-07-21")` |
| `EXPECTED_EARNINGS_CALENDAR_ASOF` | `tests/test_preflight_environment.py:58` | `pd.Timestamp("2026-07-21")` (= new `snapshot_bdp` asof) |

```bash
pytest tests/test_preflight_environment.py -v
SWE_LIVE_PREFLIGHT=1 pytest tests/test_earnings_calendar_overlay.py -v
```
(`_OHLCV_FRONTIER_STALE_DAYS=7` at `data_connector.py:877` is self-clearing after a fresh pull — no edit.)

**D. Feature-store backfill + smoke:**
```bash
python scripts/backfill_features.py            # or --limit 20 to smoke
python scripts/feature_smoke_test.py           # exit 0 = pass (--section ev for ranker only)
```
Plus the CLAUDE.md 5-ticker bring-up:
```python
from engine.wheel_runner import WheelRunner
WheelRunner().rank_candidates_by_ev(["AAPL","MSFT","JPM","XOM","UNH"], top_n=10, min_ev_dollars=-1e9, include_diagnostic_fields=True)
```

**E. Re-baseline the 4 backtest fingerprints** — a data change is *expected* to break `test_snapshot_data_fingerprint_matches_current` (SHA of every `_FILES` input + the two broad_pull panels). After confirming the drift is your deliberate refresh, regenerate each (these four are the only `backtests/regression/` modules that carry `--update-snapshot`):
```bash
python -m backtests.regression.s27_ivpit_24t_100k    --update-snapshot
python -m backtests.regression.s32_friction_24t_1m   --update-snapshot
python -m backtests.regression.s34_universe_100t_1m  --update-snapshot   # S34 ≈ 2 h
python -m backtests.regression.s35_oos_24t_100k      --update-snapshot
```
(There is no `param_oos_regime_*` regression module — the parameter-OOS runners `scripts/run_parameter_oos*.py` are separate and are NOT snapshot-fingerprint tests, so nothing there needs regen on a data refresh.)
(`UNIVERSE_100` at `backtests/regression/universes.py:53` is NOT date-coupled — do not edit on a routine refresh.)

**F. Full per-PR / launch-blocker gate for any decision-layer touch:**
```bash
pytest tests/ -m "not backtest_regression" --cov-fail-under=80
```

**Gzip staging rule (GitHub 100 MiB/blob):** any raw per-name CSV > ~100 MB must be committed **gzipped** (`.csv.gz`, pandas `compression='infer'`) and the raw `.csv` gitignored — **stage `.csv.gz`, never the raw CSV**. Levers: WIDE layout (~4× smaller) + 2-dp IV rounding. Applies to `iv_surface`, `vol_term_rv`, `options_sentiment`. The P1/P2/P3 monoliths above are all well under the limit.

**Then:** open the PR from `refresh/bloomberg-2026-07-21` (never push to `main`).

---

## 8. Open questions / verify at the Terminal

1. **`sp500_vol_dvd.csv` (ends 2026-03-20, the one un-refreshed daily monolith):** **confirmed NOT read by the served `MarketDataConnector`** (absent from `_FILES`; only `ConsolidatedBloombergLoader.load_vol_dvd()` touches it, which is off the EV path). **Do not refresh it** unless you first wire `ConsolidatedBloombergLoader` into the served path. Answer to the standing question: it is safe to leave stale.
2. **Frontier target:** 2026-07-21 is a Tuesday. Set `SWE_PULL_END=2026-07-21`; the panel appends whatever trading bars actually exist `≤` that date (likely through Mon 2026-07-20 for a morning run). Bump the §7-C pins to the **actual** last bar the pull produced, not necessarily the literal run date.
3. **FLDS-check before the big pulls** (entitlements differ by university seat): `HIST_PUT_IMP_VOL`, `HIST_CALL_IMP_VOL`, `VOLATILITY_260D`, `30DAY_IMPVOL_100.0%MNY_DF`, `EQY_DVD_YLD_12M`, `EQY_SH_OUT`, `EXPECTED_REPORT_DT`. If any return all-NaN, fall back to the wired substitutes (`VOLATILITY_nD`, `SHORT_INTEREST`+`SHORT_INT_RATIO`) rather than shipping a NaN column.
4. **Fundamentals headers must ship LOWERCASE** — the connector does not lowercase `sp500_fundamentals.csv` mnemonics; verify `pull_snapshots.py` output headers are lowercase (e.g. `gics_sector_name`, not `GICS_SECTOR_NAME`) before committing.
5. **`sp500_historical_fundamentals.py` is the only edit-only script** (line 29 hardcoded end, no env override) — and it is **off the served path**, so it is optional; skip it under time pressure. If pulled in July, expect a partial Q2 (fundamentals may not post until August).
