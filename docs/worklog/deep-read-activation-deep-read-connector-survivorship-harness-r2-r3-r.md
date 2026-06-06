---
id: deep-read-activation
title: "Deep-read connector + survivorship harness (R2+R3+R6), default-OFF"
kind: feature
status: complete
terminal: desktop
pr:
decisions: []
date: 2026-06-06
headline: Implemented docs/DATA_LAYER_DEEP_READ_DESIGN.md — the connector assembles monolith ∪ deep ∪ delisted below the get_* accessors (R2, default-OFF), a backtests/ survivorship harness selects the PIT universe and values delisted names at their delisting price (R3), and a 2008 backtest proves Lehman's loss is realized not silently dropped (R6). Trio untouched; default-OFF byte-identical; deep path gated on SWE_DEEP_HISTORY / deep_history=True.
surface: [engine/data_connector.py, backtests/survivorship.py, backtests/regression/_common.py, tests/test_deep_read_connector.py, tests/test_survivorship_harness.py, tests/test_survivorship_r6_lehman.py]
---

## Goal
Make the engine USE the survivorship-free 1990-2026 data layer without breaking
CLAUDE.md §2 and without flipping any default (adoption = re-baseline event the
architect owns). Three sequential commits on one branch: R2 connector, R3 harness,
R6 proof.

## What we did
**R2 (engine/data_connector.py) — default-OFF deep-read.** `_DEEP_SLICES` manifest
(ohlcv/vol_iv/liquidity -> their deep + delisted gz). `_load_assembled()` (reached
only when `deep_history` is ON and the key is in the manifest) concats the recent
monolith FIRST, then each present slice, normalizes tickers + parses dates over the
combined frame, de-dups `(ticker,date)` keep-first (precedence recent > deep-current
> delisted), sorts. Missing slice -> log + skip. The OFF path is the original
`_load`, untouched. Gate: `MarketDataConnector(deep_history=...)` / env
`SWE_DEEP_HISTORY`, **default OFF**. Assembly is BELOW the `get_*` accessors so the
ranker/EVEngine see longer history through unchanged signatures — no trio edit.

**R3 (backtests/survivorship.py) — survivorship harness.** Reuses
`consolidated_loader.get_universe_as_of` for the PIT universe (membership presence
only; `percentage_weight` is the dead all-zeros sentinel). `run_survivorship_backtest`
builds a `deep_history=True` connector, picks the PIT (or curated) universe per
rebalance, and ranks via `rank_candidates_by_ev` (§2-clean, no engine change).
`terminal_spot` is **delisting-aware**: on/after close if trading, else the last
close on/before expiry (the delisting price), else 0.0 — it NEVER returns None for a
name with history, so an assigned delisted position realizes its loss. Extended
`_common.assert_data_window_available` with an additive `extra_floor_paths` so a
pre-2018 start is accepted against the assembled span (default None = monolith-only,
regression path unchanged).

**R6 (test) — the proof.** A 2008 curated backtest (LEHMQ, WAMUQ, AAPL, XOM, JPM):
Lehman flows through the ranker; its realized P&L is non-NaN and a post-delisting put
realizes a real loss (< -$500) at the delisting price — no silent drop.

## What didn't / forks avoided
- Did NOT flip the default to ON (re-baseline event — architect-owned).
- Did NOT overwrite/commit any data file: tests run against a NON-git scratch dir
  (`SWE_DEEP_TEST_DATA`) built from the already-materialized refresh + deep
  worktrees; the code PR is code-only. Deep/assembly tests skip in CI (no deep data).
- The delisting terminal-value choice (last close vs 0) is a harness decision, NOT
  §2 (the ranker is untouched). Chose last-traded close (Lehman 3.65), 0.0 only when
  no history — documented in `terminal_spot`.

## Evidence
- R2: `pytest tests/test_deep_read_connector.py` (SWE_DEEP_TEST_DATA set) -> 10 passed
  (assembly reaches 1994; Lehman last bar 2008-09-12 close~3.65; rotation invariant
  0 violations; default-OFF ignores deep even when present; ON degrades to monolith
  when slices absent). Full suite OFF-path -> 2808 passed, 1 pre-existing f4 fail
  (reproduces on origin/main), no regression.
- R3/R6: `pytest tests/test_survivorship_harness.py tests/test_survivorship_r6_lehman.py`
  (env set) -> 6 passed. PIT 2008 size 500; LEHMQ + WAMUQ in, META/ABNB/GEHC/CEG out.
- ruff clean; check_manifest_coverage OK; lane-claim OK (data_connector + backtests
  are not trio files).

## Unresolved / handoff
Default-OFF on every path; adoption (flip ON + S27/S32/S34/S35 re-baseline) is R1 +
the architect-reviewed re-baseline, done together. R5 (PR #334) pins vol_iv+treasury
in the fingerprint and should land before that re-baseline. Optional deferred:
the A5 parquet sidecar cache and the A6 `get_iv_surface` accessor (additive; not
needed for R2/R3/R6). Connector memory for a full deep sweep ~2-3 GB (lazy by key
mitigates) — see design §A5.
