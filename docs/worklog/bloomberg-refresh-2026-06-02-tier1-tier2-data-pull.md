---
id: bloomberg-refresh-2026-06-02
title: Bloomberg Tier-1 + regime-context data refresh to 2026-06-02
kind: docs
status: in-flight
terminal: lab
pr:
decisions: []
date: 2026-06-02
headline: Delta-refreshed OHLCV/IV/liquidity to 2026-06-02 and added 7 context datasets via the Bloomberg Desktop API (xbbg); deep historical backfill deferred (metered cap)
surface: [scripts/pull_ohlcv.py, scripts/pull_liquidity.py, scripts/pull_vol_iv.py, data/bloomberg/]
---

## Goal

Refresh the stale Bloomberg data layer from a transient university lab terminal
(Bloomberg Terminal present; none of the repo's data persists on the box). The
committed core (`sp500_ohlcv.csv`, `sp500_liquidity.csv`, `sp500_vol_iv_full.csv`)
ended **2026-03-20**, ~50 trading days stale as of **2026-06-02**. Also fill the
Tier-2 regime/vol-context gaps the BQL doc (`scripts/bloomberg_bql_pulls.md`)
lists. Data refresh only — **no engine logic touched**.

## What we tried / approach

1. **STEP 0 — gate on the Desktop API first.** Lab terminals often allow the GUI
   but not the Desktop API. Verified: `bbcomm.exe`/`wintrv.exe` running, port
   **8194 open**, then the decisive test `blp.bdp("SPX Index","PX_LAST")` →
   **7610.67** (AAPL 315.12). Real entitled data → proceeded.
2. **Environment.** No usable Python on PATH (only the MS-Store stub); used
   `C:\Anaconda3\python.exe` (3.13.9) to build a fresh venv and
   `pip install blpapi (Bloomberg index) xbbg pandas`.
   Installed `blpapi 3.26.4.2`, `xbbg 1.2.5`, `pandas 3.0.3`.
3. **Frugal delta strategy.** The metered Desktop API can't take a full
   503-name × multi-year re-pull without risking the lab cap, so we pulled only
   the missing window (**2026-03-23 → 2026-06-02**, first trading day after the
   stale cutoff) and merged/deduped onto the existing CSVs.
4. Tiny smoke tests (3 tickers, no-write) before every full 503-name run.

## What worked

**Tier-1 (delta-merged to 2026-06-02), committed + pushed:**

| File | Before → After (rows) | Range | Notes |
|---|---|---|---|
| `data/bloomberg/sp500_ohlcv.csv` | 988,809 → **1,013,914** (+25,105) | 2018-01-02 → 2026-06-02 | ticker `AAPL UW Equity` |
| `data/bloomberg/sp500_liquidity.csv` | 1,362,737 → **1,387,842** (+25,105) | 2015-01-02 → 2026-06-02 | schema fixed: `shares_out` preserved |
| `data/bloomberg/sp500_vol_iv_full.csv` | 1,361,615 → **1,386,715** (+25,100) | 2015-01-02 → 2026-06-02 | new puller `pull_vol_iv.py` |

**Tier-2 regime/vol context, committed + pushed (new files, full history):**

| File | Rows | Series | Range |
|---|---|---|---|
| `data/bloomberg/vix_futures_curve.csv` | 20,111 | UX1–UX7 | 2015 → 2026-06-02 |
| `data/bloomberg/vol_indices.csv` | 22,968 | SKEW/VVIX/VIX9D/VXN/RVX/VXEEM/OVX/GVZ | 2015 → 2026-06-02 |
| `data/bloomberg/spx_correlation.csv` | 8,605 | COR1M/COR3M/COR6M | 2015 → 2026-06-02 |
| `data/bloomberg/rates_fx_vol.csv` | 8,716 | MOVE/CVIX/JPMVXYG7 | 2015 → 2026-06-02 |

**Tier-2, pulled + validated but NOT YET committed (held by operator; live only on the transient box):**

| File | Rows | Range | Caveat |
|---|---|---|---|
| `data/bloomberg/sp500_short_interest.csv` | 38,099 | 2020-01-31 → 2026-05-29 | monthly cadence; `borrow_rate_net` empty (not entitled); `float_pct` derived `EQY_FLOAT/shares_out×100` |
| `data/bloomberg/sp500_corporate_actions.csv` | 17,717 | 2015-01-02 → 2027-03-12 | dividends + **stock splits (103)** + special cash (133) + spinoffs (68); ex-date = assignment-relevant |
| `data/bloomberg/sp500_macro_calendar.csv` | 282 | 2025-01-02 → 2027-12-08 | 8 indicators (FOMC/CPI/NFP/GDP/PMI/claims/retail/confidence); `importance` empty |

**Scripts:** `pull_ohlcv.py` + `pull_liquidity.py` updated for `xbbg>=1.2`
long-format output + refresh-aware delta-merge + `end_date=2026-06-02`;
`pull_liquidity.py` schema fixed (`date,avg_vol_30d,turnover,shares_out,ticker`,
`EQY_SH_OUT` no longer dropped). `pull_vol_iv.py` is **new** (mirrors
`pull_ohlcv.py`). All three honour `SWE_PULL_LIMIT` / `SWE_PULL_NO_WRITE` knobs
for cheap smoke-testing.

§9 dealer-GEX skipped (needs a special BQL subscription, per the brief).

## What didn't / gotchas (the part that saves the next agent)

- **`xbbg 1.2.5` returns long-format `narwhals` DataFrames** (`ticker,date,field,value`),
  NOT the wide pandas MultiIndex the old scripts assumed. The committed
  `pull_ohlcv.py`/`pull_liquidity.py` used `df.stack(level=0)` — that breaks now.
  Fix: `.to_native()` then `pivot_table(index=['date','ticker'], columns='field')`.
  Also the `INDX_MWEIGHT` member column is now **"Member Ticker and Exchange Code"**
  (title case), not the old snake_case.
- **Windows console encoding:** narwhals' `repr` uses box-drawing chars that
  crash cp1252 — set `PYTHONUTF8=1`.
- **Desktop API ≠ BQL.** `scripts/bloomberg_bql_pulls.md` is written as Excel
  `=BQL.QUERY(...)`. The Desktop API (blpapi/xbbg) has no BQL; px_last series
  (§4/§5/§7/§12) translate cleanly to `bdh`, but special-field queries do not.
- **Fields genuinely unavailable at this entitlement (NOT fabricated):**
  - `EQUITY_SHORT_BORROW_RATE_NET` — NaN in both snapshot and history → `borrow_rate_net` column is empty.
  - `EQY_FLOAT_PCT` — unavailable; `float_pct` instead **derived** from the available `EQY_FLOAT` and `EQY_SH_OUT`.
  - Macro `importance` (`eco_importance`) — not exposed by the Desktop API → empty column.
  - Macro **calendar history** — `ECO_FUTURE_RELEASE_DATE_LIST` only gives a ~2-year rolling window (2025→2027). Full history needs the BQL pull.
  - `INDPRO Index` is an invalid Bloomberg ticker (BAD_SEC) — dropped from the macro list.
- One invalid field in a `bdh` call can **null the entire batch** — only
  request fields verified valid.
- **GitHub flagged the 3 big CSVs (60–79 MB) as over the 50 MB recommendation**
  (warnings only; push accepted, no LFS required).
- The push succeeded (remote has the SHA) but the **local upstream-tracking ref
  didn't persist** — future pushes should name `origin <branch>` explicitly.

## How we fixed it

Rewrote the two pullers to handle the narwhals long format and become
refresh-aware (read existing CSV → pull only `max_date+1 → END_DATE` → merge +
dedupe on `(date,ticker)` keep-last → sort → write); preserved committed schemas
and per-file ticker formats exactly (`AAPL UW Equity` for OHLCV; `AAPL UW` for
liquidity/IV). Tier-2 px_last series pulled via `bdh PX_LAST`; short interest via
`bdh ... Per="M"` (semi-monthly data — monthly is frugal and adequate);
corporate actions via `bds EQY_DVD_HIST_ALL` (a full corp-actions feed: cash +
splits + specials + spinoffs); macro calendar via `bds ECO_FUTURE_RELEASE_DATE_LIST`.

## Evidence

- Branch **`data/bloomberg-refresh-2026-06-02`**, commit **`cbf7239`**
  (verified on remote via `git ls-remote` — SHA matches local HEAD). NOT merged.
- STEP 3 validation: every committed/pulled CSV passed header == committed/spec
  schema, date range reaches 2026-06-02 (daily series), all-NaN check clean
  except the two documented entitlement gaps (`borrow_rate_net`, `importance`).
- Metered spend this session ≈ **~800k–1M data points**, **zero throttling/limit
  errors** — so the lab cap is comfortably above that, but a full deep backfill
  (below) is 20–40× larger and was deferred.
- Reproduce: Bloomberg Terminal running + logged in, then
  `python scripts/pull_ohlcv.py` / `pull_liquidity.py` / `pull_vol_iv.py`
  (each auto-detects the existing CSV and pulls only the delta).

## Unresolved / handoff

**1. Three Tier-2 files are uncommitted and live only on the transient lab box**
(`sp500_short_interest.csv`, `sp500_corporate_actions.csv`,
`sp500_macro_calendar.csv`). They will be lost when the machine is reclaimed
unless committed + pushed. Held per operator instruction.

**2. Deep historical backfill (deferred — needs a Bloomberg terminal).** Operator
wants maximum depth ("as far as Bloomberg goes"). Discovered earliest-available
dates (yearly probe):

| Series | Earliest available |
|---|---|
| OHLCV (per name) | IPO — IBM/KO **1968**, GE 1971, XOM/JPM 1980, AAPL 1982, MSFT 1986; SPX index 1962 |
| Realized vol (`VOLATILITY_30D/260D`) | ~1980–1986 |
| Implied vol (`HIST_PUT/CALL_IMP_VOL`) | **1994** (hard floor) |
| Liquidity | `EQY_SH_OUT` ~1981, `TURNOVER` 1996 |
| Short interest | ~1990–1991 |
| Index/vol context | MOVE 1988, VIX/SKEW 1990, JPMVXYG7 1992, CVIX/VXN 2001, UX/RVX 2004, VVIX/COR 2006, OVX 2007, GVZ 2008, VIX9D/VXEEM 2011 |
| Macro calendar | only ~2yr rolling via Desktop API → needs **BQL** |

Plan: backward-extension pulls (early `START_DATE_FULL` → existing min date),
oldest-first, **committing each panel incrementally** (transient box + cap).
Per-name equity panels are ~20–30M data points and will likely hit the metered
cap mid-pull — pull until the cap, commit what landed, report coverage. Cheap
deep wins first (index context to 1988–2011, corp actions full history, SI to
1990 ≈ <1.1M points total).

**3. BQL-only items** (manual Excel pull from home): macro calendar history +
`importance`, securities-lending `borrow_rate_net`. See
`scripts/bloomberg_bql_pulls.md` and `docs/bloomberg_refresh_runbook.md`.
