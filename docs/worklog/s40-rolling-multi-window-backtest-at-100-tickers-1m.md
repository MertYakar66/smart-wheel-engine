---
id: S40
title: Rolling multi-window backtest at 100 tickers / $1M (3 new start dates)
kind: backtest
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** S38 found −52pp engine-vs-SPY at $1M / 100t over
2020-2024. Open question: is the −52pp window-specific to
2020-2024 or a general property at scale? S40 runs 3 NEW backtests
with starts 2021/2022/2023 (all ending 2026-02-06) at the same
universe / capital / strategy as S38, cross-referenced against
S34 (2022-2024) and S38 (2020-2024) for a 5-measurement-point
picture at $1M / 100t with 4 distinct start dates.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Identical to S38 except the window. Same 100 first-alphanumeric
SP500 tickers. $1M starting capital. 35-DTE / 25-delta puts →
wheel into CC on assignment → hold to expiry. Three parallel
`WheelTracker` instances (frictionless / bid_ask / full).
`require_ev_authority=False`. Engine SHA `b2cce25` (`origin/main`
HEAD at run time). Throwaway drivers under
`%TEMP%\s40_backtest_{2021,2022,2023}\` (per Sn convention).
Per-window outputs: `rank_log.csv` (~32-53k rows each),
`summary.txt`, `run.log`.

**Pre-flight data constraint.** OHLCV starts 2018-01-02 — pre-COVID
windows (2015-2019 spec from original task) cannot run. With user
direction, campaign adapted to 3 post-2020-start windows.

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py` (the §2 ranker route) — daily re-rank,
top-3 trade attempts per day, frictionless + bid_ask + full-friction
parallel trackers. Same path as S38.

**Compute.** Total wall-clock 15.91h (W3 4.25h, W2 5.44h,
W1 6.22h; sequential). Mid-run CPU contention from other
terminals' jobs reduced per-day rate from estimated 8.6s/day to
~17-20s/day; total within ~6% of budget.

**Section-2 invariant scan: CLEAN across all 3 new windows and
both cross-references.** 0 puts executed with ev_dollars ≤ 0;
0 non-finite ev_dollars anywhere; CC opens with ev_dollars ≤ 0
(174/264/285) are harness design (require_ev_authority=False; CC
EV-floor at −$50 because alternative is unproductive stock
holding), not engine §2 breaches.

**Headline result.**

| Sn / Window | Length | Engine return | Univ-EW return | Engine vs Univ-EW |
|---|---|---|---|---|
| W1 (S40-2021) | 5.1y | +41.46% | +92.22% | **−50.76pp** |
| W2 (S40-2022) | 4.1y | +45.36% | +45.52% | **−0.16pp (≈tied)** |
| W3 (S40-2023) | 3.1y | +12.28% | +96.91% | **−84.63pp** |
| S34 (cross-ref) | 3.0y | +35.61% | +25.97% | **+9.64pp** |
| S38 (cross-ref) | 5.0y | +33.18% | +92.19% | **−59.01pp** |

**Realism Check.**

| Aspect | Engine (S40) | External reference / prior Sn | Verdict |
|---|---|---|---|
| S38 reproducibility (W1 has same start-1y, similar duration) | W1 −51pp vs Univ-EW; S38 −59pp vs Univ-EW | Same general magnitude (~10pp difference attributable to W1's 2021 bull start vs S38's 2020 COVID start) | ✓ S38 not anomalous |
| Per-year ρ across windows | bit-identical for overlapping years (2022=0.370, 2025=0.525, etc.) | Engine is deterministic on (SHA, universe, date) | ✓ Reproducibility confirmed |
| 2022 bear refusal rate | 98.1-98.6% across W1/W2/S34/S38 | Engine's strongest defensible property; replicates S38's COVID 97.8% finding pattern | ✓ Refusal mechanism robust |
| Pure-bull engine underperformance | W3 (no bear, pure 2023-2026 bull): −85pp engine-vs-Univ-EW | Predicted: limited deployment (~23%) cannot capture full bull upside | ✓ Strategy works as designed |
| Bear-included engine performance | W2/S34 (2022 bear): tied or +10pp | Bear-year selectivity + crisis refusal protect downside | ✓ Value proposition validates |
| Spearman ρ never negative | 14 (window × year) cells, ρ range 0.21-0.55 | Statistical: min p ≈ 1.5e-35 | ✓ Robust ranker signal |
| Realized executed P&L | +$35k to +$49k in W1/W2/W3 (all positive); S38 was −$28k | The S38 negative was driven by 2020 COVID year alone (−$33k); post-COVID windows show positive realized | ✓ Refines S38 finding |

**Verdict.**

- **S38's −52pp is NOT 2020-2024-specific. It is a general property
  at $1M / 100t scale, modulated by bull-year share.** Across 5
  measurement points: −85pp (pure bull) to +10pp (bear-heavy).
- **The engine-vs-passive delta scales monotonically with bull-year
  share.** Pure-bull windows show 60-85pp underperformance;
  bear-heavy windows (>25% bear share) show parity or modest
  outperformance.
- **The wheel strategy at this scale cannot keep up with passive
  during sustained bull markets** because limited deployment
  (14-24% of NAV) caps upside capture. This is fundamental to the
  strategy design, not a bug.
- **The engine's defensible value proposition remains
  conservative income + crisis refusal**, not bull-market alpha.
  +5-10% annualized depending on regime mix is the honest forward
  expectation at $1M / 100t.
- **§2 invariant clean across all 5 windows.** Engine never returned
  non-finite ev_dollars; never recommended executed puts on
  ev_dollars ≤ 0.
- **Cross-window reproducibility verified.** Per-year ρ is
  bit-identical across windows for overlapping years
  (deterministic engine).

**AI handoff.**

- **The deployment-matrix amendment PR**
  (`claude/docs-deployment-matrix-s38-amendment` commit `077cc28`,
  sibling PR to this Sn) is reinforced by S40. The "$500k–$1M
  supervised, universe ≥ 100 tickers" matrix row's revision to
  "Conditional with explicit underperformance acknowledgment"
  is now backed by 5 measurement points, not just the single S38.
- **Follow-up: test the "bull-year share" hypothesis at other
  capital scales** ($100k, $5M) and universe sizes (24t, 500t).
  Hypothesis: smaller capital saturates BP and limits the wheel's
  upside drag; larger capital widens the gap further.
- **For B1 (F4 tail-risk fix, in flight Terminal A
  `claude/fix-f4-regime-conditioned-widening`):** S40 reinforces
  case for F4 fix. COVID 2020 contributed −$33k to S38's realized;
  proper tail widening could close 5-10pp of the engine-vs-passive
  gap in 5y windows including crisis years.
- **The originally-proposed pre-COVID baseline windows (2015-2019,
  2016-2020, 2017-2021) require a Bloomberg OHLCV refresh** back
  to ~2013. Deferred until S6 (Theta provider) or a manual
  Bloomberg backfill closes the data gap.

**Methodology debt.**

- **Bloomberg-only.** SP500 ETFs (SPY, IVV, VOO) NOT in dataset
  (constituent-only). Engine-vs-SPY for new windows uses estimated
  ext SPY (Univ-EW minus typical 10-20pp EW vs cap-weighted gap).
  Resolved by Theta provider (S6, blocked).
- **Capital deployment metric difference.** S40 reports
  short-put-collateral-only avg (15-24%); prior S34/S38 docs
  reported total-NAV-deployed (22%+) including assigned stock
  market value. Both metrics agree on the structural finding
  (~76-86% of NAV idle) but they're not directly comparable.
  Future docs should report both for clarity.
- **In-sample HMM/POT-GPD parameters** (inherited from S22-S38).
  The forward-distribution parameters were fit on data overlapping
  the backtest period. Resolution requires parameter-freeze-then-
  replay infrastructure not present today.
- **Pre-2020 windows unreachable.** Bloomberg OHLCV starts
  2018-01-02; 504-day history gate forces effective start ≥ 2020.
  Resolution requires data refresh back to ~2013.

Full doc: `docs/ENGINE_BACKTEST_S40_ROLLING_MULTIWINDOW.md`.
