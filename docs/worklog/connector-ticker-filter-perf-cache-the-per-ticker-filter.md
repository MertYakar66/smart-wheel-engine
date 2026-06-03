---
id: connector-ticker-filter-perf
title: Cache the connector's per-ticker filter (output-identical universe-scan speed-up)
kind: refactor
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-01
headline: A full-universe scan was dominated by the connector re-scanning each data file's object 'ticker' column once per ticker; a lazily-built id(df)-keyed groupby index + a unique-map normalization cut a full scan 62.3s -> 39.1s (~37%) with byte-identical output.
surface:
  - engine/data_connector.py
  - tests/test_data_connector_ticker_filter.py
---

## Goal
The 2026-06-01 V&V campaign's cProfile (see `docs/VNV_CAMPAIGN_2026-06-01.md` §6)
showed a 150-ticker scan spends ~10s in `pandas comp_method_OBJECT_ARRAY` (742
calls) — the connector's `_filter_ticker` doing `df[df["ticker"] == t]`, a full
O(rows) object-column scan **per ticker** — plus 2.4M `normalize_ticker` calls in
`_load`. Make a universe scan cheaper WITHOUT changing any engine output (the
connector feeds the §2 trio, so this must be provably output-identical).

## What we tried
1. **Cache the filtered RESULT per (df, ticker).** Rejected: within one scan
   every (df, ticker) is unique, so a result cache gives zero same-scan benefit.
2. **Categorical `ticker` dtype.** Rejected: a global dtype change with wide blast
   radius (every downstream consumer; `groupby` `observed` semantics) — too risky
   to self-ship on a trio-feeding path.
3. **Lazily-built `{ticker: sub-frame}` groupby index, cached by `id(df)`
   (shipped).** The call sites are all `self._filter_ticker(self._load(KEY), t)`,
   so the frame identity is stable (lives in `self._cache`) — the index is built
   once per data file (one pass) and every later per-ticker lookup is O(1).
4. **`_load`: `.apply(normalize_ticker)` -> map over the ~500 unique raw tickers.**

## What worked
- 62.3s -> **39.1s (~37%)** on a full-universe scan at as_of=2026-03-20;
  survivors=423 unchanged; 5-ticker smoke byte-identical (AAPL 20.38/0.8571,
  MSFT 90.89/0.8286).
- Output-equivalence pinned by `tests/test_data_connector_ticker_filter.py`
  (groupby slice == boolean mask for present/multi/absent tickers; unique-map ==
  apply).

## What didn't
- **`dict(df.groupby("ticker"))` is NOT equivalent to
  `{t: sub for t, sub in df.groupby("ticker")}`.** ruff C416 suggested the
  rewrite; applying it silently broke everything (the 5-ticker scan went empty,
  4 tests failed). A `DataFrameGroupBy` exposes a `.keys` attribute, so `dict()`
  takes the mapping-protocol path instead of iterating `(name, group)` pairs.
  Kept the explicit comprehension with `# noqa: C416` + a comment. Lesson: never
  trust C416 on a groupby.

## How we fixed it
- `engine/data_connector.py`: `self._ticker_groups: dict[int, dict]` in
  `__init__`; `_filter_ticker` (now an instance method — was a staticmethod, no
  external callers) builds + caches the groupby index and returns
  `groups.get(t, df.iloc[0:0])`; `_load` uses the unique-map normalization.

## Evidence
- Before (V&V funnel, unoptimized main): 62.3s full universe.
  After (this branch): 39.1s, survivors=423. ~37% (both with some background
  CPU contention — directional, machine-local).
- `pytest tests/test_data_connector_ticker_filter.py` 5 passed; full suite
  (minus `backtest_regression`) green; ruff clean.

## Unresolved / handoff
- `id(df)` keying assumes the frames stay in `self._cache` (they do — never
  GC'd). If a future caller passes a non-cached frame to `_filter_ticker`, it
  still works (builds a one-off index) but won't be reused.
- Memory: the index roughly doubles each loaded frame's footprint (~150MB for
  OHLCV+IV combined). Fine for a long-lived connector; per-connector + GC'd.
- The HMM E-step (`regime_hmm._forward_backward`, ~8s of a scan) is the next
  hot spot — inherent to the model, not addressed here.
