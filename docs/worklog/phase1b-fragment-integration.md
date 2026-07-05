---
id: phase1b-integration
title: Phase-1B fragment integration — CASY + blue-chips into the monoliths + BK↔BNY collapse + dividends clamp + UNIVERSE_100
kind: data
status: completed
---

**Purpose.** Execute `docs/NEXT_DATA_SESSION_RUNBOOK.md` Phase 1B (no Bloomberg): fold the
Phase-1A fragments (branch `claude/phase1a-casy-bloomberg-pull`) into the connector monoliths,
collapse BK↔BNY, clamp the dividend epsilon-negatives, and re-derive `UNIVERSE_100`. **Data +
`universes.py` only — decision trio byte-untouched. Held for review; coupled to the Phase-3
re-baseline (do NOT merge alone).**

**§2 panel.** `EVEngine.evaluate` is untouched by this data shift: `git diff main..HEAD --
engine/ev_engine.py engine/wheel_runner.py engine/candidate_dossier.py` is **empty**. The change
shifts the backtest universe (data) only; the downgrade-only reviewer contract and the
`EVEngine.evaluate` invariant are not exercised. No §2 bypass.

## What changed (working-tree, branch `claude/phase1b-fragment-integration` off main @ 83eacdd)
| file | before → after | change |
|---|---|---|
| `sp500_ohlcv.csv` | 1,014,920 → 1,035,361 rows, 511 → 510 tickers | 11 names backfilled to full history; BNY→BK; VRT NaN-OHLCV dropped |
| `sp500_vol_iv_full.csv` | 1,037,278 → 1,049,504 | CASY + 5 IV-thin backfilled; BNY→BK |
| `sp500_liquidity.csv` | 1,388,848 → 1,390,913 | CASY backfilled; BNY→BK |
| `sp500_fundamentals.csv` | 503 → 503 | BNY→BK reticker (1 row) |
| `sp500_credit_risk.csv` | 503 → 503 | BNY→BK reticker (1 row) |
| `sp500_earnings.csv` | 49,379 → 49,527 | CASY +148 (year/period + announcement_date; EPS/time blank) |
| `sp500_dividends.csv` | 50,230 rows | 82 epsilon-negative `dividend_amount` → 0.0 |
| `backtests/regression/universes.py` | UNIVERSE_100 | re-derived: −BNY, +CMG |

Backfilled OHLCV (full history): WMT/KMB/CPB/DPZ/CASY/LITE/SATS/VEEV = 2117 bars, COHR 2039,
PLTR 1426 (2020 IPO), VRT 1918 (after dropping 29 pre-2020 NaN-OHLCV rows). vol_iv backfilled for
CASY + the 5 IV-thin names. Integration via `staging/integrate_phase1b.py` (`dtype=str` so
unchanged rows are byte-identical).

## Decisions (deviations from the literal runbook, with rationale)
1. **Batch-1 (WMT/KMB/CPB/DPZ/PLTR) stored under the existing `UW` label, not `UN`.** The
   Phase-1A fragments were re-pulled under the correct NYSE `UN` source ticker (the original
   `UW`/NASDAQ pull was truncated — see the phase1a correction `f117fe0`), but on `main` these
   names are mislabeled `UW` *consistently across all 7 files* and the connector normalizes to
   bare (`WMT`). So the #355 goal is the missing HISTORY, not the exchange code. Storing the full
   history under the established `UW` label fixes the data with **zero cross-file inconsistency**
   (`test_fundamentals_credit_subset_of_ohlcv` stays green). The cosmetic `UW`→`UN` relabel across
   all files is a separable follow-up (NOT done here).
2. **Dividends union was already satisfied on `main`** — CTRA (1990→2026), LW (2016→2025),
   MTCH (2024→2026), PAYC (2023→2026) all carry full available history; the truncating "refresh"
   the runbook anticipated is not on this `main`. So Phase-1B step 3 reduced to the epsilon-clamp.
3. **BK↔BNY collapse covers the 5 connector-read files** that carried BNY: ohlcv/vol_iv/
   fundamentals/liquidity/credit_risk (BNY→BK; `BK UN Equity` now continuous 2018→2026-06-04,
   contiguous across the 2026-03-23 seam, 0 dups). Non-connector files (analyst, corporate_actions,
   index_membership, historical_fundamentals, institutional, iv_snapshot_today) still carry BNY —
   out of scope (not read by the connector; no test enforces them; survivorship/actions history).
4. **VRT**: dropped 29 pre-2020 (pre-SPAC-merger) NaN-OHLCV rows from the fragment at integration
   (the fragment records the raw pull; the monolith stays clean).

## Validation gates (all pass / behave as designed)
- **CASY OHLCV overlap (2026-03-23→06-04) vs ORIGINAL committed**: open/high/low/close maxabsdiff
  **0.000000**; volume +19 on one frontier-day row (documented post-close finalization).
- **vol_iv no-skew**: all 1,043,217 both-non-null rows `hist_put_imp_vol == hist_call_imp_vol`
  exactly (incl. the new history).
- **Seam audit** (connector normalization): **0** recent-only (0-in-window) names in UNIVERSE_100.
- `test_universes_match_connector` **green**; `test_fundamentals_credit_subset_of_ohlcv`,
  `test_ohlcv_nan_price_rows_are_the_known_four`, `test_ohlcv_volume_nonnegative` **green**.
- **W14 served-IV band green** after the IV change.
- **#340 fingerprint guard RED on all 4 snapshots** — the designed signal that a re-baseline is
  owed. NOT re-baselined here (Phase 3).

## Expected RED — deferred to Phase 3/4 (test updates; outside the data+universes.py rail)
- `test_blue_chip_history_is_complete[<11 names>]` — `XPASS(strict)`: the backfill worked; flip the
  xfail markers in Phase-3 step 9.
- `test_seam_membership_split_is_structural`, `test_seam_continuity_and_reticker` — the intended
  BK↔BNY collapse changed the seam; update in Phase 3/4.
- `test_thin_names_degrade_gracefully` — the backfilled names now clear the 504-bar gate, so the
  `ALL_THIN` list is stale; update in Phase 3/4.
- The full-universe 480/31 split (`test_full_universe_no_silent_drops_and_split`) re-pins in Phase 4.

## NOT done (rails)
No trio/engine changes, no #354 PIT-panel integration (stays staged), no Phase-2 (E) fixes, no
Phase-3 re-baseline. Held for review; lands coupled with the re-baseline, not alone on main.
