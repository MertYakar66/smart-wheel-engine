---
id: data-layer-activation
title: Data-layer activation plan: deep-read connector + survivorship harness + verified inventory
kind: research
status: in-flight
terminal: desktop
pr:
decisions: []
date: 2026-06-05
headline: Verified the campaign's survivorship-free 1990-2026 data on the bytes (refresh 6bb3399 / deep e7818f4), confirmed the engine still reads only 2018+ monoliths, and produced a prioritized activation roadmap + a connector deep-read / survivorship-harness design — executing only the safe prep (the credit-rating dead-read fix), deferring every decision-layer-touching step (merge, re-baseline, connector change) for operator review.
surface: [docs/DATA_LAYER_ACTIVATION_ROADMAP.md, docs/DATA_LAYER_DEEP_READ_DESIGN.md, engine/wheel_runner.py]
---

## Goal
Shift from *pulling* Bloomberg data to *using* it. Understand exactly what the
multi-day campaign produced, then PLAN the path from "data sitting on branches"
to "engine running on the full enriched, survivorship-free layer." Produce the
plan + design first; execute only the low-risk, reversible prep; STOP before the
merge / re-baseline / connector change.

## What we tried
1. Oriented: fetched the data branches, read `CLAUDE.md` §2 / `AGENTS.md` /
   `PROJECT_STATE.md` / the campaign worklog `bloomberg-deep-history-2026-06-04.md`
   in full (it already lists the deferred wiring + the 2 known code fixes — not
   re-derived). Found both data branches had **advanced** since the 2026-06-05 QA
   (refresh `3a271c9→6bb3399`, deep `61c2183→e7818f4`).
2. Inventoried on the bytes (not the worklog text): a chunked, memory-safe
   profiler over all 33 refresh CSVs + 13 deep gz/panels recording rows / cols /
   span / tickers / dup-keys.
3. Read the design-critical code first-hand: the connector read path (`_load` →
   `get_*`, the OHLCV rotation rename), the backtest harness
   (`backtests/regression/_common.py`), the existing PIT membership machinery
   (`data/consolidated_loader.get_universe_as_of`), and the premium/chain sourcing
   in `rank_candidates_by_ev`.
4. Sampled the actual stored ticker strings across membership / delisted panels /
   `_delisted_universe` to design the normalization + join correctly.

## What worked
- **Inventory reproduced the campaign claims** and resolved one QA concern:
  treasury now reaches **1994-01-03** (CONCERN-1 closed); vol_iv monolith
  correctly sharded to the 2018 floor; all seams contiguous; assembled current
  spans 1994→2026 for OHLCV/vol_iv/liquidity + ~1,015 delisted names 1990→2026.
- **The deep-read seam is clean:** assembling monolith ∪ deep ∪ delisted inside
  `_load(key)` lands entirely *below* the `get_*` accessors, so the §2 trio sees
  longer series through unchanged signatures — no decision-layer restructure.
- **The survivorship join key is verified:** delisted panels + membership +
  `_delisted_universe` all key on the Bloomberg PIT code (`0111145D UN`), which
  `normalize_ticker` maps to a consistent stub across all three.

## What didn't
- **The "sector-cap source" fix is NOT a one-liner.** The ev_row `"sector"` tag
  (`wheel_runner.py:1769`, three sites) deliberately uses the same hardcoded
  `DEFAULT_SECTOR_MAP` as the R9 sector-cap gate, by documented design (R10 keys
  off ticker, not the sector map). Fixing the source means changing the
  diagnostic AND the gate in lockstep — an R9 firing change that needs a
  re-baseline. That is the "consequential, touches the decision layer" class the
  task says to defer → planned (R0b), not executed.
- **Can't run the regression suite to "measure" the data impact:** ~5h, the
  refresh branch's engine predates the launch campaign, and the suite is already
  RED on main from benign drift — triply confounded. The OHLCV-hash fingerprint
  mismatch (`c3d5443…` pinned vs `7a3e77a4…` refreshed) is the definitive
  re-baseline signal instead.

## How we fixed it
- Shipped **R0a** as its own PR #333 (`claude/fix-credit-rating-deadread`, split
  out so this branch is docs-only): `wheel_runner.py:511` now reads
  `credit.get("sp_rating")` (the key the connector actually returns) instead of
  the raw Bloomberg field name. Off the EV-authoritative path (legacy `_compute_wheel_score` + memo/API
  display only) → §2-safe, no re-baseline.
- Wrote `DATA_LAYER_ACTIVATION_ROADMAP.md` (verified inventory + R0–R7 prioritized
  roadmap with effort/risk/§2-touch/re-baseline per item + the R1 merge hazard)
  and `DATA_LAYER_DEEP_READ_DESIGN.md` (connector assembly seam + slice manifest +
  ticker/dedup precedence + memory/perf + survivorship harness + Theta chain
  fallback + both code fixes + test plan).

## Evidence
- Branch SHAs (origin): main `efc491c`, refresh `6bb3399`, deep `e7818f4`.
- Inventory harness: `C:\tmp\swe-exp\inv\inventory.py` → `inventory_results.json`
  (pandas 2.3.3 / py 3.12.10). Headline rows in the roadmap §1 tables.
- Connector seam: `engine/data_connector.py` `_load` (103-143), `get_ohlcv`
  rotation (235-241); harness `backtests/regression/_common.py` (`tickers=` fixed
  list, `assert_data_window_available` monolith-only, `ohlcv_sha256` OHLCV-only).
- Ticker keys sampled live: delisted `0111145D UN`, membership
  `member_ticker_and_exchange_code=0111145D UN`, `percentage_weight≈-2.4e-14`.
- Full suite (`pytest -m "not backtest_regression"`): **2804 passed, 12 skipped,
  1 failed**. The one failure — `test_f4_rv_widening.py::...calm_regime_5_ticker_
  smoke_preserves_main_baseline` (expects 5 rows, gets 4; JPM drops) — reproduces
  **identically on pristine `origin/main` (efc491c)**, so it is **pre-existing**:
  a date-sensitive no-`as_of` smoke that ages out as wall-clock (now 2026-06-06)
  advances past the stale committed data (ends 2026-03-20) and JPM falls outside
  the staleness window. NOT caused by R0a (which sets a string after candidate
  selection and cannot change row count).
- 7-agent adversarial verification (read-only, raw bytes + source): all 7
  load-bearing claims reproduced. It caught + I corrected 4 accuracy items: the
  heuristic is `_compute_wheel_score` (not `_calculate_wheel_score`); line refs
  shifted +7 after the R0a comment edit (`:511`, sites `1769/2687/3304`); the
  sector source drives **R9** only (R10 keys off ticker); and a plain 3-way merge
  of the refresh branch would NOT auto-revert the engine (the hazard is adopting
  its tree wholesale).

## Unresolved / handoff
Awaiting operator greenlight on the consequential steps: R0b (sector source),
R1 (data-only merge + re-baseline), R2 (connector deep-read), R3 (survivorship
harness), R4 (Theta chains), R5 (fingerprint extension — cheap, do first), R6
(2008 survivorship smoke), R7 (cleanups). Merge hazard: the refresh branch
reverts the engine — take `data/` only. See both new docs for the full plan.
