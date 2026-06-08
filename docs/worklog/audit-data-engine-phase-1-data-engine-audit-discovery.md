---
id: audit-data-engine
title: Phase 1 data + engine audit (discovery)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-07
headline: Reusable data+engine audit pass — 13 ranked findings; data spine sound, frontier-pinned probe clean (480/511 produced, 0 silent drops); 3 HIGH (conditional IV /100 heuristic in-trio, dateless fundamentals/credit lookahead, OHLCV-only fingerprint blind-spot)
surface: [scripts/audit_data_engine.py]
---

## Goal

After the data campaign (Bloomberg refresh, deep history, CASY backfill,
dividends re-pull, OHLCV fingerprint) we never systematically checked that
(a) the data is sound and (b) it flows through into correct engine output.
Phase 1 = build a reusable discovery pass + emit a ranked weakness report
BEFORE writing tests against it. Additive only; decision trio untouched.

## What we tried

`scripts/audit_data_engine.py` — a single reusable pass that logs the
provider, auto-detects the data-supported frontier (not `date.today()`),
inventories every connector CSV, builds a capability map + coverage matrix,
probes the production ranker at the frontier, and emits a markdown findings
doc + JSON sidecar. Every engine probe routes through
`WheelRunner.rank_candidates_by_ev` (no §2 bypass). Recon of ranker
internals (drops attrs, `distribution_source`, the 504-day gate, the IV
heuristic) was done with a 5-agent read-only workflow and every
load-bearing claim re-verified against source.

## What worked

- **Frontier auto-detection** = `min(max(ohlcv.date), max(vol_iv.date))`.
  Resolves to `2026-06-04` on `main`, `2026-03-20` on the
  `claude/suggest-rolls-defensive-surfacing` branch — deterministic on
  either, which is the whole point of not using `today()`.
- **Drops are complete**: the probe reads `frame.attrs["drops"]`
  (`{ticker,gate,reason}`) + checks for tickers that are neither produced
  nor dropped. Full universe: 511 requested → 480 produced, 31 dropped,
  **0 vanished**.
- The OHLCV spine is sound: 0 rename-invariant violations across 1.01M
  rows, 0 non-positive prices, 0 dup `(date,ticker)`.

## What didn't

- The SessionStart "OHLCV 79 days stale / 2026-03-20" warning is
  BRANCH-LOCAL (the primary working tree is on an older branch); `main` is
  current to 2026-06-04. The audit pins to the auto-detected frontier so it
  is correct regardless.
- Two memory-sourced "findings" were CONTRADICTED by the current data and
  corrected to grounded reality rather than shipped: the dividends
  truncation for CTRA/BK/LW/PAYC is NOT present on main (full 2018-24
  history intact), and treasury covers 1994-01-03→present (not "2021-05+").

## How we fixed it

Made every weakness data-driven (computed evidence, not asserted memory).
Scoped the coverage "complete" metric to the core ticker files so it is
meaningful (UNIVERSE_24 = 24/24 clean control). Stored the raw per-ticker
drops in the JSON for forensic re-derivation.

## Evidence

- Run: `python scripts/audit_data_engine.py --universe full --as-of 2026-06-04`
  → `docs/DATA_ENGINE_AUDIT_2026-06-07.md` +
  `docs/verification_artifacts/data_engine_audit_2026-06-07/audit.json`.
- Provider: `MarketDataConnector` (bloomberg, `SWE_DATA_PROVIDER` unset).
- Drops: event 6 / data 8 (departed names hit the 30-day staleness gate) /
  history 17 (thin <504-bar names). Tiers: empirical_non_overlapping 473,
  empirical_overlapping 7. Non-finite outputs: none.
- 2026-03-23 index reconstitution: 8 joiners (BNY/CASY/COHR/FDXF/LITE/SATS/
  VEEV/VRT) absent from earnings; 8 leavers (BK/CTRA/EPAM/HOLX/LW/MOH/MTCH/
  PAYC) absent from fundamentals/credit; BK→BNY re-ticker.

## Unresolved / handoff

Phase 1 is the checkpoint — hold for review before Phase 2 (tests). The 3
HIGH findings: (W1) the conditional `if iv>3.0: iv/100` IV heuristic at
`wheel_runner.py:198/1101/2418/2980` is IN THE TRIO → separate lane-claimed
PR, not this lane; (W2) dateless fundamentals/credit snapshots = structural
lookahead (data-layer fix); (W3) the snapshot fingerprint pins only OHLCV
(extend in `backtests/regression/_common.py`, not the trio). Phase 2:
`tests/test_data_integrity_*.py` (new) + extend
`tests/test_audit_viii_real_data_smoke.py` into `tests/test_data_to_engine_*.py`;
confirmed data defects → `xfail(strict=True)` + issue.
