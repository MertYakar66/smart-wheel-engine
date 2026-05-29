---
id: S43
title: Rolling 5-window backtest with post-#260 engine
kind: backtest
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Test whether S38's "engine underperforms a same-universe
EW hold by ~62pp over 2020-2024" generalises across rolling 5-year
windows OR is window-specific. Same setup as S38 except dates; engine
is now post-F4-fix (PR #260 realised-vol-ratio widening) + R9 sector
cap (PR #255) + R10 single-name cap (PR #256), so this also
implicitly tests whether (a) the F4 widening changes 2022 outcomes
and (b) the new gates would have changed capacity / concentration
outcomes vs S38's pre-fix engine.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Identical to S34 / S38 except the windows. Same 100 tickers
(first alphanumeric SP500 names — `UNIVERSE_100`). $1M starting
capital. 35-DTE / 25-delta puts → wheel into CC on assignment →
hold to expiry. Three parallel `WheelTracker` instances
(frictionless / bid_ask / full). `require_ev_authority=False`.
Post-#260 engine on `origin/main` (commit `56d8e5c`). Harness
under `backtests/regression/s43_rolling_multiwindow.py` (committed
on the S43 branch).

**Data-coverage finding.** OHLCV CSV on disk starts 2018-01-02 (not
2014 as the task spec assumed). With `enforce_history_gate=True` +
`min_history_days=504`, the survivorship gate rejects every
candidate until ~2020-01-02. So the user's W1=2015-2019 / W2=2016-2020 /
W3=2017-2021 are infeasible (backtest can't start before OHLCV).
Adapted to four runnable windows: W1=2018-2022 (gate-truncated
effective ~3y), W2=2019-2023 (gate-truncated effective ~4y), W3=2020-2024
(direct S38 re-run, the Δ-vs-#260 deliverable), W4=2021-2025 (NEW
clean 5y forward-anchored).

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py` (the §2 ranker route) — daily re-rank,
top-3 trade attempts per day per friction-level tracker, shared SP
rank call across friction levels per
`backtests/regression/_common.py::run_backtest_multi_friction`.
Routes through `EVEngine.evaluate` end-to-end; no engine mocking;
zero §2 bypass.

**Status.** Done. **Verdict: S38's 50-60pp engine-vs-passive
underperformance GENERALISES across all four rolling 5-year windows
on post-#260.** Engine NEVER beats a same-universe EW baseline.
Range −51pp (W3, the post-#260 S38 re-run) to −104pp (W2, the
2019-2023 window where Univ-EW gained +128%). Spearman ρ is
window-INVARIANT at 0.356-0.378. Per-year ρ is POSITIVE in 16 of
16 (window × year) cells measured (no negatives). **F4 fix (PR
#260) is signal-preserving on the W3 ⟷ S38 comparison** (Δρ
−0.002; 2022 mean realised Δ −$2.70). The +$103k NAV delta on the
same 2020-2024 window is driven by harness execution-selection
differences (516 W3 puts vs 305 S38 puts), not by the F4 widening.

**Findings:**

- **(F1 — Window dependence GENERALISES on post-#260).** Engine vs
  Univ-EW: W1 −60.23pp, W2 −103.63pp, W3 −51.52pp, W4 −55.04pp.
  S38 pre-#260 restated against Univ-EW = −61.84pp. All five
  negative. **Engine never beats a same-universe EW baseline.**
- **(F2 — ρ is window-invariant)**. Full-window ρ clusters within
  0.022 (0.356 to 0.378) across four windows. Per-year ρ positive
  in all 16 measured cells.
- **(F3 — F4 fix signal-preserving on W3 ⟷ S38)**. Δρ −0.002;
  Δ2022-mean-realised −$2.70. The post-#260 widening doesn't
  materially change EV ranking on this window. The +$103k NAV
  delta is harness mechanics on the same signal.
- **(F4 — Top-ticker membership rotates per window)**. BKNG: S38
  +$31k (carry) → W2/W3 −$28k (worst loser) → W4 outside top-5.
  AZO: stable winner W1-W3 → +$38k (W4 top winner). ADBE: W1-W3
  winner → W4 worst loser (−$21k). **No ticker carries the
  engine's dollar outcome stably across windows.**
- **(F5 — §2 invariant intact across 184,602 rows)**. Zero
  non-finite `ev_dollars` across all four windows × three friction
  levels. R1a guard (PR #204) holds.
- **(F6 — R10 would-fire 3.7-4.5% of executed opens)**. 19/465
  (W2), 19/516 (W3), 23/509 (W4); W1 reconstructed 35/752. Max
  single-name exposure reached 20-25% of NAV in W2/W3 vs the 10%
  R10 cap. R10 is materially impactful damage-bounding when wired
  live.
- **(F7 — 2025 ρ = 0.524 is the second-highest year measured)**.
  Only 2020 COVID at 0.538 higher. 2025 mean realised +$172/trade
  is the highest of any year. Signal persists through the forward-
  anchored window.

**Implications for `docs/PRODUCTION_READINESS.md` §1 / §5:**
- B1 (F4 tail-risk fix): the RV-ratio widening is signal-preserving
  but does NOT measurably better-bound named F4 cases (BKNG was W3's
  WORST loser despite the widening). R10 single-name cap (PR #256)
  is the operative damage-bound on those names.
- **Deployment matrix should now cite four more (capital × universe
  × window) cells:** W1 −60pp, W2 −104pp, W3 −52pp, W4 −55pp at
  $1M / 100 tickers. Engine never beats same-universe-EW at this
  scale in 5y rolling windows.

**Methodology caveats:**
- W1's `tracker_state.json` is missing (W1 launched before the
  harness was extended with the tracker dump). Concentration / R10
  audit / deployment time-series for W1 are approximate from
  rank_log replay. W2/W3/W4 have exact tracker_state.
- Three of the user's five task-spec windows (2015-2019, 2016-2020,
  2017-2021) are infeasible due to OHLCV coverage starting
  2018-01-02. Documented in the writeup §0.
- Refusal-rate framing differs from S38's: my reported numbers are
  engine-side EV-≤-0 only; S38's 97.8% COVID figure also included
  the harness's secondary BP / already-held / per-day-cap filters.
  Cross-checked: W2's effective COVID open rate is ~1.6% (refusal
  ~98.4%), matching S38's 97.8% framing.

**Compute pacing.** Initially launched W1 alone (5h06m wall-clock
with Terminal B's S40 contending). After W1 done, launched W2 with
~5h estimate. When W2 was 60% through with rate slowing as
positions accumulated (going from 0.47 day/s → 0.10 day/s), I
deviated from "sequential" and launched W3 + W4 in parallel — the
dev box has 16+ cores and each Python process uses only 1. Parallel
saved ~3-4 hours of wall-clock. Deviation announced on board #113.
Total campaign wall-clock 11h47m start-to-finish.

Full doc: `docs/ENGINE_BACKTEST_S43_ROLLING_MULTIWINDOW.md`. Harness
+ scan + analyze + reconstruct under `backtests/regression/s43_*.py`.
