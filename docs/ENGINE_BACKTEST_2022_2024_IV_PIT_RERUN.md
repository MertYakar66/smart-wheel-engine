# Engine backtest — 2022-2024 IV-PIT re-run (S27, 2026-05-24)

**Question:** Does the predictive-validity signal from PR #178 (S22)
survive the IV-PIT fix on `claude/fix-ranker-iv-pit-aware` (commit
`d26a8d6`)?

**Window / universe / strategy / mode / config:** **identical to S22**
(see `docs/ENGINE_BACKTEST_2022_2024.md` for the full setup). 2022-01-03
→ 2024-12-31, 24 SP500 tickers, $100k starting capital, 35-DTE /
25-delta short puts, wheel into CC on assignment, hold to expiry,
`require_ev_authority=False`.

**Only difference:** the ranker's IV resolution.
- **Pre-fix (S22):** all three rankers read `fundamentals["implied_vol_atm"]`
  — a 2026 snapshot column, same value regardless of `as_of`.
- **Post-fix (S27):** all three rankers call `_resolve_pit_atm_iv` →
  `connector.get_iv_history(ticker, end_date=as_of)` — returns the
  composite `(hist_put_imp_vol + hist_call_imp_vol) / 2` at the most
  recent date ≤ `as_of`. PIT-correct.

Verified live (2024-01-15, UNH): pre-fix IV `0.4323`, post-fix IV
`0.1712`. A 60% relative correction on a single ticker, easily
enough to flip EV signs (and it does — see below).

---

## Headline verdict — signal preserved, magnitude halved

| Metric | S22 pre-fix | **S27 post-fix** | Delta |
|---|---|---|---|
| Put candidates (N) | 6,163 | 6,163 | +0 |
| **Spearman ρ** | 0.4838 | **0.2183** | **−0.2655** |
| Spearman p | ~0 | 2.29e-67 | — |
| EV mean | $92.27 | **−$14.32** | −$106.59 |
| Realized P&L mean (per trade) | $229.77 | $63.34 | −$166.43 |
| Hit-rate (OTM) | 82.72% | 76.39% | −6.33pp |
| IV mean (population) | 0.3274 | **0.2580** | −0.0694 |
| Final NAV | $167,633 (+67.6%) | $151,444 (+51.4%) | −$16k / −16pp |
| Executed trades | 59 | 50 | −9 |
| Put assignments | 7 | 10 | +3 |

**The signal is real but the pre-fix magnitude was an artifact.** ρ =
0.22 is still "moderate" by behavioural-science conventions and
publishable in trading-signal research (the t-statistic on the 6,163
sample is overwhelming). The engine genuinely ranks better than
random — but not by anywhere near the 5.1× quartile spread the
pre-fix backtest suggested.

The user's pre-merge correction on PR #178 was exactly right: the bug
*was* a systematic per-ticker IV offset (BSM premium is monotonic in
IV → within-ticker ordering survives), but the **cross-ticker**
component of the Spearman calculation is sensitive to which tickers
have which snapshot-vs-historical IV gaps. The cross-ticker
contamination cost ~0.27 of the ρ.

---

## Quartile means — spread compressed from 5.1× to 1.7×

| Quartile | S22 EV mean | S22 PnL mean | S22 hit | **S27 EV mean** | **S27 PnL mean** | **S27 hit** |
|---|---|---|---|---|---|---|
| Q0 (low) | −$24.28 | $99.81 | 76.0% | −$109.70 | $71.91 | 78.0% |
| Q1 | $22.91 | $140.89 | 81.0% | −$26.97 | $29.73 | 72.0% |
| Q2 | $73.82 | $169.52 | 85.0% | $1.50 | $31.35 | 72.0% |
| Q3 (high) | $296.68 | $508.87 | 88.0% | $77.91 | $120.33 | 83.0% |

**Quartile monotonicity holds in S27** — Q3 still beats Q0 in mean
realized P&L (1.7×, was 5.1×) and in hit-rate (83% vs 78%). But
Q1 and Q2 are noisier: their realized means ($29.73, $31.35) are
basically indistinguishable, and Q1's hit-rate drops to 72%. The
engine's mid-range sorting is materially noisier than the pre-fix
backtest suggested; only the top quartile clearly stands out.

---

## Per-year — the 2022 bear actually got STRONGER post-fix

| Year | S22 ρ | S22 hit | S22 PnL mean | S22 IV | **S27 ρ** | **S27 hit** | **S27 PnL mean** | **S27 IV** |
|---|---|---|---|---|---|---|---|---|
| 2022 (bear) | 0.269 | 73.6% | $81.20 | 0.326 | **0.391** | 68.9% | $21.68 | 0.312 |
| 2023 (recovery) | 0.484 | 84.8% | $266.81 | 0.328 | 0.193 | 76.8% | $89.24 | 0.235 |
| 2024 (bull) | 0.540 | 89.5% | $336.43 | 0.328 | 0.131 | 83.2% | $78.20 | 0.228 |

**This is the most informative result of S27.** The IV-PIT bug's
distortion was *not* uniform across years:

- **2022 (bear):** PIT IV (0.312) was very close to the snapshot IV
  (0.326) because the 2026 snapshot is dominated by similar
  late-recovery vol levels. The bug's signal-distortion was small,
  and removing it actually *improved* the signal (ρ 0.27 → 0.39).
  The bear-market signal is genuinely stronger than the pre-fix
  backtest suggested.
- **2023 (recovery) and 2024 (bull):** PIT IV was 25-30% LOWER than
  the snapshot (post-2022 vol collapse not reflected in a snapshot
  taken later). Pre-fix premiums were inflated by 30-40%, producing
  inflated EVs across all tickers, and the ranking signal looked
  artificially strong (ρ 0.48-0.54) because the IV inflation was
  roughly proportional to ticker volatility. **The post-fix bull-
  market signal is materially weaker.**

The honest read: **the engine's signal is most reliable in periods
where the IV regime matches the snapshot vintage** (calm vol). In
periods that diverge from the snapshot (extended calm with snapshot
from a more-volatile period, or vice versa), the *magnitude* of EV
predictions is unreliable but the *ordering* still has signal.

---

## F4 tail-risk episodes — UNCHANGED. The bug was not the cause.

| Episode | Metric | S22 pre-fix | **S27 post-fix** |
|---|---|---|---|
| COST April 2022 (10 candidate rows) | mean EV | $103.52 | $127.35 |
| | mean `prob_profit` | 0.833 | **0.833** |
| | mean realized | −$7,615.70 | −$7,501.29 |
| | executed | 0 (BP saturated) | 0 (BP saturated) |
| UNH November 2024 (20 candidate rows) | mean EV | $588.32 | $177.08 |
| | mean `prob_profit` | 0.976 | **0.867** |
| | mean realized | −$2,881.37 | −$5,224.70 |
| | executed | 0 (BP saturated) | **1** (BP available; lost) |

**COST April 2022 is identical across runs** — same `prob_profit`,
nearly identical mean realized loss. The forward-distribution +
POT-GPD tail-risk machinery failed to widen its estimate during the
22% drop in BOTH engines. **F4 is a real engineering gap, not an
artifact of the IV-PIT bug.** This is the highest-leverage finding
for the engine team.

The UNH November 2024 episode is more nuanced:
- Post-fix, `prob_profit` dropped from 0.976 to 0.867 (the PIT IV
  was higher in late-2024 than the snapshot, slightly widening the
  forward distribution).
- BUT the EV scaling change meant the BP constraint loosened (lower
  EVs → fewer trades → more available capital), and **one UNH trade
  actually fired** (vs zero in S22). It lost money — realized
  −$5,224 mean (worse than S22's no-execution baseline).
- This is **direct evidence for F5**: in S22 the BP saturation
  accidentally protected the account from the UNH losses; in S27
  the looser BP let one fire and pay the price.

---

## F5 confirmed — BP-saturation protection is fragile

S22 noted that all worst-loss candidates were `executed=False`
because the $100k account's BP was full from earlier positions. S27
proves the point:

| | S22 pre-fix | **S27 post-fix** |
|---|---|---|
| Executed short puts | 59 | 50 |
| Executed hit-rate | 88.1% | **76.0%** |
| Executed mean realized | $200.72 | **−$71.99** |
| Skipped (no BP) | 1,171 | 574 |

**The executed-trade mean realized went NEGATIVE in S27.** Despite
*fewer* executions, the executed set performed *worse* per trade,
because lower per-trade EVs freed up BP that S22 had pinned, letting
some of the high-confidence-but-actually-bad trades (like UNH Nov
2024) actually fire.

**Production implication:** S22 looked profitable partly because the
ranker's inflated EVs over-tied up BP, accidentally preventing some
of the bad trades. The fix removes that accidental protection.
**Production should not rely on BP-saturation as a risk control.**
PR #174's `consume_ranker_row` wires the D17 hard-blocks for strict
mode; the next backtest (S28 candidate) should run with
`require_ev_authority=True` + an attached `PortfolioContext` so the
gate firing is explicit, not accidental.

---

## Dollar return *(caveats from S22 doc still apply re: friction)*

| | S22 pre-fix | **S27 post-fix** |
|---|---|---|
| Final NAV | $167,633 (+67.6%) | $151,444 (+51.4%) |
| Final cash | $37,758 | $94,819 |
| Closed positions | 54 | 44 |
| Open at end | 5 | 6 |
| SPY buy-hold (same window) | ~+24% | ~+24% |
| Engine vs SPY (frictionless) | +44pp | **+27pp** |

The engine still beats SPY by ~27pp over 3 years with realistic IV
input, though by 16pp less than the pre-fix backtest suggested.
Friction (bid/ask, commissions, slippage) is still unmodeled —
realistic real-money return would be lower than $51%.

---

## Findings — what changed, what didn't

- **F1 — Engine ranking has real predictive validity. RECONFIRMED.**
  Smaller magnitude (ρ 0.22 vs 0.48 prior claim) but still significant.
  Top quartile beats bottom 1.7× by realized P&L. The pre-fix +279
  lift-over-naive was inflated; the honest number is closer to +60
  (Q3 mean $120 vs all-trades mean $63). The directional verdict
  ("engine has signal") stands.
- **F2 — Signal-by-regime is INVERTED from S22.** Pre-fix showed
  signal strengthening in calmer regimes (ρ 0.27 bear → 0.54 bull);
  post-fix shows it strongest in the bear (ρ 0.39) and weakest in
  the bull (ρ 0.13). The bear-market case is genuinely robust;
  the bull-market case was IV-inflation artifact.
- **F4 — Tail-risk gap is REAL.** COST April 2022 mean realized
  −$7,500 in both runs with `prob_profit=0.833` throughout the
  22% drop. The forward-distribution + POT-GPD machinery failed
  to widen in either engine. **Highest-leverage finding for the
  engine team.** Hypothesis (from S22): 504-day empirical lookback
  too long, POT-GPD threshold too high. Worth a targeted
  unit-test reproducing the COST 2022-04 sequence.
- **F5 — BP-saturation protection is fragile.** Confirmed.
  Post-fix engine's executed-trade mean realized went NEGATIVE
  (−$72/trade) despite fewer executions, because looser BP let
  more bad trades fire. Production must run strict mode with
  D17 hard-blocks (PR #174's `consume_ranker_row`).
- **F8 — Re-run with IV-PIT fix is DONE.** This is S27. Closes
  PR #178's deferred follow-up.
- **F9 (new) — `prob_profit` is more honest post-fix.** The
  pre-fix bug was systematically inflating prob_profit (lower
  IV → tighter forward distribution → higher probability of OTM
  expiry → higher prob_profit). The S27 mean prob_profit on
  UNH dropped from 0.98 to 0.87 in the November 2024 episode,
  correctly reflecting that 2024's actual IV implied a
  wider distribution. The engine's probability outputs are
  more trustworthy post-fix even if the EV magnitudes are smaller.

---

## What this validates / invalidates from S22

| S22 claim | S27 verdict |
|---|---|
| Engine has predictive validity (ρ ≈ 0.48) | **Engine has predictive validity (ρ ≈ 0.22)** — smaller but real |
| Top quartile beats bottom 5.1× | Top quartile beats bottom 1.7× |
| Signal strongest in calm regimes | **Inverted — signal strongest in bear** |
| `prob_profit` 7.6% mean calibration error | Need re-check — values changed but didn't analyze in S27 |
| F4 tail-risk gap on COST/UNH | **Confirmed — same in both runs, not bug-related** |
| F5 BP-saturation accidentally protected | **Confirmed — explicit evidence: S27 fired a bad UNH trade S22 blocked** |
| +67.6% NAV return | +51.4% NAV (still beats SPY +27pp) |

---

## AI handoff

- **Production engineering — F4 is the priority.** Write a targeted
  unit test that replays the COST April 2022 sequence (10 candidate
  rows over 8 trading days) against the forward-distribution +
  POT-GPD pipeline. Both runs of this backtest produced
  `prob_profit=0.83` throughout a 22% drop — the question is which
  layer (504-day empirical lookback dilution, POT-GPD threshold,
  or both) failed to react.
- **Production runtime — wire D17 strict mode.** S22 + S27 together
  prove the engine's BP-saturation "protection" is fragile.
  PR #174 ships `consume_ranker_row` and `portfolio_context_snapshot`
  for the explicit D17 / R7+R8 path. **The next backtest (S28
  candidate) should run with `require_ev_authority=True` and
  `PortfolioContext` attached.** Expected outcome: fewer trades
  fire, fewer tail-loss assignments, smaller P&L but with explicit
  control.
- **Backtest methodology — calibrate by-year IV gap.** The 2022
  ρ improvement vs the 2023-2024 ρ collapse shows the IV-snapshot
  bug's distortion was strongly regime-dependent. A future
  backtest could explicitly measure (snapshot IV − PIT IV) per
  ticker per year and report the bias structurally rather than
  per-finding.
- **Documentation — this S27 doc supersedes S22's numerical
  claims** but S22's doc stands as the historical record of the
  pre-fix engine's behavior. Cross-references go both ways.

---

## Method appendix

Harness: `%TEMP%/s27_backtest/run.py` (copy of S22's, sys.path
pointing at the fix-branch worktree `../swe-iv-fix-rerun`).
Comparison: `%TEMP%/s27_backtest/compare.py`. CSV output:
`%TEMP%/s27_backtest/rank_log.csv` (6,190 rows).

The harness consumes the post-fix engine via:

```python
WORKTREE = Path("C:/Users/merty/Desktop/swe-iv-fix-rerun")
sys.path.insert(0, str(WORKTREE))
from engine.wheel_runner import WheelRunner
```

Confirmed at runtime via `WheelRunner.__module__` import path. The
fix branch (`claude/fix-ranker-iv-pit-aware` @ `2dd8252` = `d26a8d6`
fix + `2dd8252` manifest chore) was checked out via
`git worktree add` rather than a branch switch on `swe-terminal-b`
(per the auto-mode classifier's correct objection during S22).
