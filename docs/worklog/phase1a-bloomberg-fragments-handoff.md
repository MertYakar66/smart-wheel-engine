---
id: phase1a-bloomberg-fragments
title: Phase-1A Bloomberg fragments handoff — CASY + blue-chip backfills + #354 PIT panel (staged on branch)
kind: data
status: completed
---

**Purpose.** Record where the supervised-session Bloomberg-gated pulls live so
whoever runs **Phase 1B** (`docs/NEXT_DATA_SESSION_RUNBOOK.md`) from a fresh clone
knows where to fetch them. The raw fragments are **intermediate inputs**, deliberately
**not** PR'd to `main` — Phase 1B integrates them into the monoliths and *that*
integration PR (monolith diff + re-baseline coupling) is the real merge vehicle.

**Where.** Branch **`claude/phase1a-casy-bloomberg-pull` @ `bbc9c91`** (pushed to origin).
Staging-only: the entire branch diff vs `main` touches `staging/` only — monoliths,
the decision trio, tests, scripts, backtests all **byte-untouched**.

**Manifest (4 commits).**

| commit | path | contents |
|---|---|---|
| `3d45b4c` | `staging/casy/` | CASY UW 4 files — `casy_ohlcv.csv` / `casy_vol_iv.csv` / `casy_liquidity.csv` (2117 rows each, 2018-01-02→2026-06-04) + `casy_earnings.csv` (148 quarterly, **dates+period only**) + `pull_casy.py` + `PULL_NOTES.md` |
| `0856969` | `staging/blue_chips/` | OHLCV backfill batch 1 — WMT/KMB/CPB/DPZ/PLTR + `pull_backfill.py` + `PULL_NOTES.md` |
| `8eabfef` | `staging/blue_chips/` | OHLCV+vol_iv backfill batch 2 (IV-thin) — VEEV/COHR/LITE/SATS/VRT |
| `bbc9c91` | `staging/fundamentals_pit/` | **#354** monthly PIT fundamentals panel — `sp500_fundamentals_pit.csv` (72,937 rows, 503 names, 2015-01-02→2026-05-31) + `pull_fundamentals_pit.py` + `PULL_NOTES.md` |

**Pull environment (reusable).** Fresh repo `.venv`; `blpapi 3.26.5.1` (from Bloomberg's
index, **not** public PyPI), `xbbg 1.3.0`, `pandas==2.3.3`. **xbbg 1.3.0 returns a
narwhals *tidy* frame** `[ticker,date,field,value]`, not the legacy wide-MultiIndex the
old `scripts/pull_*.py` assume — the staged pull scripts adapt; the repo does not pin xbbg.

**Validation / vintage (full detail in each `staging/*/PULL_NOTES.md`).** Raw/deterministic
fields match the committed vintage **to the cent** (prices for CASY + batch-2; batch-1
open/close; realized vols; shares_out; turnover) — and the OHLCV column scramble
(`open←PX_HIGH/high←PX_LAST/low←PX_LOW/close←PX_OPEN`) plus the IV field
(`30DAY_IMPVOL_100.0%MNY_DF`, ATM no-skew) are confirmed. Revision-sensitive aggregates
(vol_iv IV surface, `avg_vol_30d`, batch-1 high/low) carry the **current** Bloomberg
revision vintage on the to-be-replaced recent rows; adjustment basis confirmed consistent
(open/close exact). The Phase-3 re-baseline absorbs these. Bonus: the fresh pull corrects a
stale 2026-03-20 triple-witching volume in several blue-chips.

**Two flags to record before Phase 1B / the #354 trio PR.**

1. **#357 `rate_1m` pre-2001 → WON'T-FIX (not a gap).** The committed `rate_1m` starts
   exactly **2001-07-31**, which is the genuine inception of the US Treasury 1-month
   constant-maturity series — it did not exist before then. The pre-2001 NaN is **correct**,
   not missing data; "filling" it would mean splicing a different proxy series, and it is
   entirely pre-backtest-window with no engine consumer. Closed. *(Distinct from #357's
   **dividend epsilon-clamp** half — still the Phase-1B pure-git item.)*
2. **#354 carry-field coverage is 69.2%.** Before the #354 `as_of` accessor lands (the
   separate trio PR), confirm the ~31% `eqy_dvd_yld_12m` NaN is **"legitimately no dividend
   / not-yet-listed"** vs **"missing data"** — that field feeds the BSM carry `q`, so a
   silent-missing would distort EV. (Expected to be non-payers + names not listed in
   early-window months, but verify.)

**Next (no Bloomberg — runs later from any clone).** Phase 1B fetches this branch and
integrates the fragments into the monoliths (de-dup the overlap), does BK↔BNY collapse,
dividends union + epsilon-clamp, `UNIVERSE_100` re-derive; then Phase 2 (E) fixes and the
Phase 3 re-baseline. The #354 panel awaits its connector `as_of` accessor (trio/§2 PR).
