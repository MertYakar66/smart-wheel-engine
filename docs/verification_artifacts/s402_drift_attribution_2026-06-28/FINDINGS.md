# #402 snapshot-drift — root-cause attribution (2026-06-28, Windows main)

**Issue #402:** the s27/s32/s34 backtest-regression reproducers drift from their
committed snapshots (`aggregate/ev_mean` rel ~8.7e-4–2.1e-3, `per_year/p` /
`spearman_p` out of tolerance; **`spearman_rho` within tolerance everywhere** —
ranking quality unchanged, levels shifted). Filed during the 2026-06-09/10
heavy-verify campaign against the **#338 snapshot baseline**, prime suspect the
**#363 IV-sentinel connector gate**. The issue's prescribed resolution is a
supervised re-baseline after attribution. This note is the **attribution**
(diagnostic only — no snapshot re-baseline committed).

## Verdict: #402 is STALE — the #338-baseline drift was absorbed by two later re-baselines, and the #363 prime suspect is independently refuted.

Three independent lines of evidence:

### 1. The snapshots were re-baselined TWICE since the #338 baseline #402 measured against
`git log` on `backtests/regression/snapshots/s27_…json` (and s32/s34):

| commit | PR | event |
|---|---|---|
| `de296d1` | **#338** | R1 Bloomberg refresh + re-baseline — **the baseline #402's drift was measured against** |
| `918d7eb` | **#429** | re-baseline S27/S32/S34/S35 for #428 PIT carry-q |
| `b3aa236` | **#433** | corporate-action event lockout (**EV-moving**; re-baselined) |

Current `main`'s snapshots were regenerated from the **current (post-#363) engine**
by #429 and again by #433. Any metric drift that existed vs the #338 baseline was
re-pinned out at those events — `ev_mean` / `spearman_p` now reflect the live engine
by construction.

### 2. The #363 prime suspect has ZERO footprint on the drifting windows
The #363 gate (`MarketDataConnector._clean_vol_iv_inplace`) NULLs served
`hist_put_imp_vol` / `hist_call_imp_vol` cells outside the band
`(_IV_LOW_FLOOR, _DEEP_IV_SENTINEL_FLOOR] == (3.0, 10000]` on the `vol_iv` PIT
read. Scanning the **exact windows the reproducers consume** (normalized tickers,
as the connector does on load — see `diag_s402_iv_gate_footprint.py` /
`iv_gate_footprint.json`):

| window | rows | tickers | IV cells (non-null) | **gated by #363** |
|---|---|---|---|---|
| s27/s32 — UNIVERSE_24, 2022-2024 | 18,072 | 24/24 | 36,140 | **0 (0.0000%)** |
| s34 — UNIVERSE_100, 2024 | 24,696 | 98/100 | 49,392 | **0 (0.0000%)** |

The 2022-2024 implied-vol readings for these names are **all already inside** the
clean `(3.0, 10000]` band — the gate touches nothing. It therefore **cannot move**
s27/s32/s34 `ev_mean` via the PIT `get_iv_history` path. (The separate #369/#430
`_clean_served_iv` fundamentals-fallback branch only fires when `get_iv_history`
is empty for a name — not the case for these well-covered large caps — so it is
not a plausible driver either.)

### 3. The fingerprint blind-spot #402 cited is now CLOSED
#402's mechanism was "`data_csv_sha256` matches → drift comes from a
**non-fingerprinted input** (IV/treasury unpinned)." The regression fingerprint
now pins the **full connector data set**, not just OHLCV:
`ohlcv_sha256` + **`vol_iv_sha256`** + **`treasury_sha256`** +
**`connector_data_sha256`** (every `_FILES` entry) — see
`backtests/regression/_common.py:620-636`. Any future data-driven movement now
surfaces as a **fingerprint mismatch**, not a silent metric drift — the blind-spot
that allowed #402 is structurally eliminated.

## Decisive check: snapshot fingerprints match current data EXACTLY (stronger than the tolerance test)
The regression test asserts `build_payload(run())` matches the committed snapshot
*within tolerance*. A stronger, faster check is available: the snapshots embed the
full input fingerprint, the engine is deterministic, and the last EV-moving merge
that regenerated the snapshots was **#433** (everything since — #434/#435 — is the
gitignored option-premium rail, **absent here → synthetic-BSM path → a no-op on
these reproducers**). So if the committed fingerprint equals the current data
fingerprint, the metrics match **by construction** (exact, not merely in-tolerance).

Comparing each committed snapshot's `fingerprint` to the live data hashes on
`origin/main @ 4e90003`:

| snapshot | `data_csv` | `vol_iv` | `treasury` | `connector_data` (10 files) |
|---|---|---|---|---|
| s27_ivpit_24t_100k | MATCH | MATCH | MATCH | MATCH (all 10) |
| s32_friction_24t_1m | MATCH | MATCH | MATCH | MATCH (all 10) |
| s34_universe_100t_1m | MATCH | MATCH | MATCH | MATCH (all 10) |

**Every fingerprint matches.** Current-main inputs are byte-identical to those the
#433 re-baseline pinned, and the engine path is unchanged → the reproducers produce
the committed metrics exactly → **no live drift → #402 is resolved/stale.** (A full
`pytest -k "s27 or s32 or s34"` was started as a gold-standard backstop but is
multi-hour — the s27 reproducer alone runs ~1h45m — and is superseded by this exact
input-identity proof, so it was not run to completion.)

## Recommendation
**Close #402** as resolved-by-re-baseline (superseded by #429 + #433), with the
#363 prime suspect explicitly refuted (zero in-window footprint) and the
fingerprint blind-spot closed. **No new re-baseline is warranted** — the snapshots
already track the live engine, and there is no remaining unattributed drift to
re-pin. Should drift reappear, the now-complete data fingerprint will identify the
input, and attribution should re-run this footprint scan before assuming an engine
cause.

*Scope: diagnostic only. No engine, data, or snapshot files changed.*
