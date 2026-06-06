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
  (`wheel_runner.py:1761`) deliberately uses the same hardcoded `DEFAULT_SECTOR_MAP`
  as the R9/R10 gate, by documented design. Fixing the source means changing the
  diagnostic AND the gate in lockstep — an R9/R10 firing change that needs a
  re-baseline. That is the "consequential, touches the decision layer" class the
  task says to defer → planned (R0b), not executed.
- **Can't run the regression suite to "measure" the data impact:** ~5h, the
  refresh branch's engine predates the launch campaign, and the suite is already
  RED on main from benign drift — triply confounded. The OHLCV-hash fingerprint
  mismatch (`c3d5443…` pinned vs `7a3e77a4…` refreshed) is the definitive
  re-baseline signal instead.

## How we fixed it
- Shipped **R0a** only: `wheel_runner.py:503` now reads `credit.get("sp_rating")`
  (the key the connector actually returns) instead of the raw Bloomberg field
  name. Off the EV-authoritative path (legacy `_calculate_wheel_score` + memo/API
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

## Unresolved / handoff
Awaiting operator greenlight on the consequential steps: R0b (sector source),
R1 (data-only merge + re-baseline), R2 (connector deep-read), R3 (survivorship
harness), R4 (Theta chains), R5 (fingerprint extension — cheap, do first), R6
(2008 survivorship smoke), R7 (cleanups). Merge hazard: the refresh branch
reverts the engine — take `data/` only. See both new docs for the full plan.
