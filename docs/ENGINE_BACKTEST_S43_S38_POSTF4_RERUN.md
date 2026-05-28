# Engine backtest — S43: S38 re-run on post-F4 engine (2026-05-28)

**Question:** *Does PR #260's F4 realized-vol-ratio widening close
any of S38's −52pp engine-vs-passive gap at $1M / 100t / 2020-2024?
The S40 multi-window doc explicitly flagged this as the natural
follow-up; S41 (PR #267) tested the same question at the 24t / 2022-2024
scale and found PR #260 alone is a slight dollar NEGATIVE on S27.
Does the longer window's inclusion of COVID 2020 change the story?*

**Headline answer:** **NO.** F4 fix has **near-zero impact** on the
S38 setup. Engine return moves from +33.18% to +33.74% (+0.56pp);
realized total moves from −\$28,647 to −\$32,729 (−\$4,082 — slightly
WORSE); Spearman ρ moves from 0.358 to 0.354 (−0.0037, −1.0%);
executed puts move from 305 to 307 (+0.7%). The −52pp engine-vs-passive
gap is **NOT closed by F4 widening**, and the 5-10pp closure hypothesis
from S40's AI-handoff is **falsified**. The S40 finding stands:
the gap at $1M/100t scale is structural to the strategy's limited
deployment (15-23% NAV), not to a missing tail-risk widening
mechanism on the engine side.

**Engine SHA at run:** `origin/main` @ `56d8e5c` (post-PR #260 F4
realized-vol-ratio widening + PR #262 R10 single-name cap).
**Baseline (pre-F4):** S38's original `%TEMP%\s38_backtest\rank_log.csv`
generated against engine SHA `b2cce25` (pre-#260). Both runs are
fully reproducible from on-disk artifacts.

---

## 1. Setup (identical to S38 except engine SHA)

- 100 first-alphanumeric SP500 tickers (same universe as S34 / S38 / S40)
- $1M starting capital
- 35-DTE / 25-delta short puts, wheel into CC on assignment, hold to expiry
- `require_ev_authority=False`
- Three parallel `WheelTracker` instances per friction level
- Window: 2020-01-02 → 2024-12-31 (1,258 trading days)
- Engine SHA: `56d8e5c` (post-F4)
- Harness: `%TEMP%\s38_postf4_backtest\run.py` (throwaway, not committed)
- Compute: **6.31h wall-clock** (18 sec/day rate; mid-range vs S40's 17-20s/day)

## 2. Headline results (full friction)

### 2.1 Per-friction-level NAV

| Friction | Pre-F4 (b2cce25) | Post-F4 (56d8e5c) | Δ NAV | Δ return |
|---|---|---|---|---|
| Frictionless | $1,348,704 | $1,354,550 | +$5,846 | +0.58pp |
| bid_ask | $1,332,071 | $1,337,663 | +$5,592 | +0.56pp |
| **full** | **$1,331,764** | **$1,337,350** | **+$5,586** | **+0.56pp** |

Friction drag is consistent (~1.5pp NAV across the 5y window).

### 2.2 Aggregate metrics (full friction)

| Metric | Pre-F4 | Post-F4 | Δ | Δ% |
|---|---|---|---|---|
| n_candidates | 17,192 | 17,192 | 0 | — |
| n_executed_puts | 305 | 307 | +2 | +0.7% |
| n_cc_opened | 168 | 174 | +6 | +3.6% |
| Put assignments | 69 | 72 | +3 | +4.3% |
| Realized puts total | −\$27,894 | −\$30,041 | −\$2,148 | +7.7% |
| Realized CCs total | −\$754 | −\$2,688 | −\$1,934 | +257% |
| **Realized grand total** | **−\$28,647** | **−\$32,729** | **−\$4,082** | **+14.2%** |
| Mean realized per executed put | −\$91.45 | −\$97.85 | −\$6.40 | +7.0% |
| Hit-rate (realized > 0) | 82.6% | 82.1% | −0.5pp | — |
| **Spearman ρ** | **0.3576** | **0.3539** | **−0.0037** | **−1.0%** |

**Reading:** F4 fix produced a tiny, mostly-neutral shift on the
S38 setup. The slight NAV improvement (+\$5,586) is offset by
slightly worse realized executed P&L (−\$4,082). The engine
opened 2 more short puts and 6 more CCs — F4 widening did NOT
make the engine materially more selective at this scale. ρ
degraded by 1.0% (vs S41's 3.3% degradation on S27).

### 2.3 Engine-vs-passive (recovery of S40's headline)

S40's published Univ-EW (equal-weighted 100-ticker, price-only)
for the 2020-01-02 → 2024-12-31 window was +92.19%. Post-F4 result:

| | Pre-F4 | Post-F4 | Δ |
|---|---|---|---|
| Engine return (full friction) | +33.18% | +33.74% | +0.56pp |
| Univ-EW return | +92.19% | +92.19% | (unchanged — passive benchmark) |
| **Engine vs Univ-EW** | **−59.01pp** | **−58.45pp** | +0.56pp closer |

**The F4 fix closes 0.56pp of the −59pp engine-vs-Univ-EW gap at
$1M/100t/2020-2024. That is not material.** The S40 AI-handoff
hypothesis predicted 5-10pp closure; the observed closure is **1.1%
to 5.6% of the predicted range**. Hypothesis falsified.

## 3. Per-year breakdown

### 3.1 Per-year ρ and executed mean realized

| Year | Pre-F4 ρ | Post-F4 ρ | Δρ | Pre-F4 exec | Post-F4 exec | Pre-F4 mean exec | Post-F4 mean exec | Δ mean exec |
|---|---|---|---|---|---|---|---|---|
| 2020 | 0.548 | 0.533 | −0.015 | 78 | **79** | −\$430.61 | −\$420.91 | **+\$9.70** |
| 2021 | 0.211 | 0.214 | +0.003 | 63 | 63 | −\$140.15 | −\$140.15 | $0.00 |
| 2022 | 0.370 | 0.367 | −0.003 | 47 | 48 | +\$180.18 | +\$139.08 | **−\$41.10** |
| 2023 | 0.309 | 0.307 | −0.002 | 71 | 71 | +\$72.26 | +\$72.57 | +\$0.31 |
| 2024 | 0.312 | 0.305 | −0.007 | 46 | 46 | +\$20.10 | +\$4.60 | −\$15.50 |

**Critical observations:**

- **2020 (the COVID test):** The hypothesis was that F4 widening
  would fire during COVID's elevated vol regime and refuse more
  candidates, shifting realized positive. **Observed:** F4 took
  ONE MORE trade in 2020 (78 → 79), and 2020 mean realized improved
  by only +\$9.70 per trade (−\$430.61 → −\$420.91). **The COVID
  effect was the smallest possible**: F4 widening DID fire on some
  2020 dates (you can infer from the slight numerical shift) but
  not enough to materially change the year's realized P&L.
- **2022 (the bear test):** Worse — mean realized DROPPED by \$41 per
  executed trade (+\$180.18 → +\$139.08). This is consistent with
  S41's finding that 2022 mean realized dropped 88% (\$1.72 → \$0.21
  on S27 24t). Same effect at 100t scale: the post-F4 reshuffling
  of top-10-per-day produces slightly worse realized in the bear
  year.
- **2024 (the bull test):** Mean realized dropped from +\$20 to
  +\$5. The fix's lagged signal occasionally fires on 2024 dates
  (S41's calibration showed 11% fire rate in 2024) and the
  reshuffling happens to produce slightly worse picks.
- **2023 and 2021:** Essentially unchanged (deterministic engine
  produces same output for same inputs; F4 didn't fire on these
  years' dates much).

**No year flipped negative ρ. No year flipped positive mean realized
into negative or vice-versa. The fix is signal-preserving at the
per-year level, consistent with S41's finding.**

### 3.2 Per-year mean across ALL candidates (refusal-mechanism view)

| Year | Pre-F4 mean (all 3,400/year) | Post-F4 mean (all) | Δ |
|---|---|---|---|
| 2020 | −\$15.74 | −\$17.53 | −\$1.79 |
| 2021 | +\$149.49 | +\$152.21 | +\$2.72 |
| 2022 | −\$118.28 | −\$119.89 | −\$1.61 |
| 2023 | +\$93.38 | +\$93.59 | +\$0.21 |
| 2024 | +\$101.99 | +\$99.95 | −\$2.04 |

**Reading:** the "what if I'd blindly executed every candidate"
mean shifts by ≤ \$3 per year. F4 widening barely changes the
candidate-level realized distribution.

## 4. Refusal-rate comparison (the F4-hypothesis-targeted view)

### 4.1 COVID period (2020-02-15 → 2020-05-15)

This is the core test. F4 widening should fire most aggressively
during COVID (rv30/rv252 elevated by the late-Feb to mid-May
trajectory). If the hypothesis is right, post-F4 refusal rate
should be HIGHER and post-F4 mean-realized-if-blindly-executed
should be LOWER (because the refused trades would have lost more).

| Metric | Pre-F4 | Post-F4 | Δ |
|---|---|---|---|
| n_candidates | 847 | 847 | 0 |
| n_executed | 19 | 18 | **−1** |
| Refusal rate | 97.76% | 97.87% | +0.11pp |
| Mean realized if blindly executed | −\$254 | −\$258 | −\$4 |

**Result:** F4 widening took **1 fewer trade** in the 12-week COVID
window. The refusal rate barely moved. The mean realized of all
COVID candidates barely moved.

**This is the falsification.** The F4 hypothesis predicted material
COVID refusal-rate increase + material loss-avoidance. Observed:
\$254 → \$258 = $4/trade improvement in the would-have-been
refusal pool. Multiplied by the 829 refused candidates, that's
**~\$3,316 of incremental loss avoidance**, which is much smaller
than the original S38 doc's "$215k of avoided losses on refused
candidates" framing implied was at stake.

The reason the COVID effect is so small: per S41's §1c live probe,
rv30/rv252 takes ~30 days to "catch up" to an unfolding vol event.
The COVID drawdown started 2020-02-21 and was sharp; by the time
rv30/rv252 crossed the 1.30 threshold, most of the candidates the
engine would have refused via the threshold had ALREADY been
refused by the 97% baseline refusal mechanism (HMM regime
classifier + R5 EV gate + harness MIN_PROCEED_EV=10). The F4
threshold catches a small marginal subset on top of the existing
refusal layer.

### 4.2 2022 bear year (2022-01-01 → 2022-12-31)

| Metric | Pre-F4 | Post-F4 | Δ |
|---|---|---|---|
| n_candidates | 3,390 | 3,390 | 0 |
| n_executed | 47 | 48 | **+1** |
| Refusal rate | 98.61% | 98.58% | −0.03pp |
| Mean realized if blindly executed | −\$118.28 | −\$119.89 | −\$1.61 |

**Result:** F4 widening took **ONE MORE trade** in 2022 (47 → 48)
— the OPPOSITE of the hypothesis. This is consistent with S41's
finding that the rv30/rv252 ratio in 2022 (per the per-year fire
rate table) was the most active region for F4 firing, but the
reshuffling effect produced slightly different top-10-per-day
sets that happened to include one MORE trade.

Mean realized per blindly-executed candidate is essentially
unchanged (−\$118 → −\$120).

## 5. Section-2 invariant scan

| Window | PUTS exec & ev_dollars≤0 | PUTS exec & non-finite | Any non-finite anywhere | CC exec & ev≤0 (info) |
|---|---|---|---|---|
| **S38 pre-F4** | **0** ✓ | **0** ✓ | **0** ✓ | 258 (harness design) |
| **S38 post-F4** | **0** ✓ | **0** ✓ | **0** ✓ | 270 (harness design) |

**§2 invariant CLEAN both pre- and post-F4.** F4 fix preserves
the §2 contract perfectly. The CC negative-EV opens (270 vs 258)
are harness design (`require_ev_authority=False`, CC EV-floor at
−\$50), not engine §2 breaches.

## 6. Findings

**F1 — F4 fix has near-zero impact on S38 at the 5y/100t scale.**
Engine return moves +0.56pp; realized P&L gets slightly worse;
ρ degrades minimally (−1.0%); executed-trade count moves +0.7%.
The fix is signal-preserving but not value-creating at this
configuration.

**F2 — The S40 AI-handoff hypothesis is FALSIFIED.** Predicted 5-10pp
closure of the −59pp engine-vs-Univ-EW gap; observed 0.56pp closure
(1.1% to 5.6% of the predicted range). **The −52pp / −59pp gap is
NOT a function of missing tail-risk widening; it is structural to
the strategy's limited deployment.**

**F3 — COVID was specifically tested and the effect is minimal.**
Pre-F4 COVID refusal 97.76%, post-F4 97.87% (1 fewer trade taken
out of 847 candidates). The would-have-been-refused mean realized
shifted by only −\$4 per trade. Total incremental loss avoidance
in the COVID window is ~\$3,316 — much smaller than the original
S38 doc's "$215k of avoided losses on refused candidates" framing
implied could be at stake from improved tail widening.

**F4 — Per-year ρ degradation is universal but mild.** Every year
shows post-F4 ρ within 0.015 of pre-F4 ρ (worst case 2020:
0.548 → 0.533). No year flipped negative. No year experienced
ρ collapse. Consistent with S41's "signal preservation" framing.

**F5 — 2022 mean realized per executed trade dropped \$41**
(+\$180 → +\$139). Consistent with S41's S27 24t finding (2022
mean realized dropped \$1.72 → \$0.21 = −88% on a smaller universe).
This is the "post-F4 top-10-per-day reshuffling produces slightly
worse picks in the bear year" pattern S41 identified.

**F6 — Section-2 invariant CLEAN.** Both pre- and post-F4 runs
have 0 PUT executions on ev_dollars ≤ 0 or non-finite EV. F4 fix
preserves §2.

**F7 — F4 fix is NOT a deployment-matrix-changer at $1M/100t.**
PROD_READINESS §5 row "$500k-$1M supervised, universe ≥ 100
tickers" was amended (PR #263) to "Conditional with explicit
underperformance acknowledgment" citing S38's −52pp. Post-F4 result
is −51.5pp engine-vs-SPY-external (still well within the −52pp
framing's error bars). The matrix verdict does NOT change with
the F4 fix.

## 7. Comparison to S41 (PR #267)

S41 (Terminal A, 24t × $100k × 2022-2024 on S27 reproducer)
reported PR #260 alone causes:
- ρ −3.3% (0.18815 → 0.18186)
- NAV −12.1% (\$127,694 → \$112,223)
- executed_trades −22% (51 → 40)

My S43 (100t × $1M × 2020-2024 on S38 setup) reports PR #260 causes:
- ρ −1.0% (0.3576 → 0.3539)
- NAV **+0.4%** (\$1,331,764 → \$1,337,350)
- executed_trades **+0.7%** (305 → 307)

**The effect changes sign on NAV and on executed_trades when the
configuration moves from (24t × \$100k × 2022-2024) to
(100t × \$1M × 2020-2024).** Possible explanations:
1. **Universe size dilutes per-trade impact.** S27's 24-ticker
   universe has high concentration in F4-prone names (COST, MSFT,
   etc); the 100-ticker universe spreads exposure across more
   names where F4 firing rate is more balanced.
2. **Window length averages out per-year noise.** S27's 3-year
   window has F4 effects concentrated in the 2022 bear; S43's 5-year
   window includes 2 calm-regime years (2023, 2024) where F4 doesn't
   fire much, averaging the per-year impact toward zero.
3. **\$1M capital opens more candidates per day** (S40 measured
   23.5% deployment with 126 BP-binding skips in W3; S38 had
   16% avg deployment and 0 BP skips). Larger capital means each
   F4-refused candidate has less marginal impact.

**Both S41 and S43 agree on the structural finding:** **PR #260
alone is not a dollar-improver.** The size and sign of the
non-improvement varies by configuration; the qualitative direction
("F4 fix is the frequency guard, not the magnitude guard")
holds.

## 8. Implications for the deployment matrix

`docs/PRODUCTION_READINESS.md` §5 (per PR #263 amendment) currently
reads:

> **\$500k-\$1M supervised, universe ≥ 100 tickers — ⚠ Conditional
> with explicit underperformance acknowledgment.** S34 +11.6pp,
> S38 −52pp — span of −52pp to +11.6pp across measured multi-year
> windows.

**No change required.** S43's post-F4 result (−51.5pp vs SPY ext) is
within the same range. The amendment text already captures the
underperformance honestly; F4 fix does not move the needle enough
to require a re-revision.

The deployment-bundle framing from S41 (and S43 reinforces): the
combination of PR #260 (F4 RV widening, frequency guard) + PR #262
(R10 single-name notional cap, magnitude guard) is what closes
PROD_READINESS §3 B1, not either alone. S43 didn't test R10 because
the harness uses `require_ev_authority=False` and R10 is
conditional on PortfolioContext + strict mode.

## 9. Method appendix

**Engine version under test:** `origin/main` @ `56d8e5c` (post-F4).
**Baseline:** the previously-published S38 run on engine `b2cce25`
(pre-F4); rank_log preserved at `%TEMP%\s38_backtest\rank_log.csv`
from the S38 campaign.

**Data provider:** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
OHLCV `data/bloomberg/sp500_ohlcv.csv` (2018-01-02 → 2026-03-20).
IV `data/bloomberg/sp500_vol_iv_full.csv` (2015-01-02 → 2026-03-20).

**Harness pattern:** S38's `%TEMP%\s38_backtest\run.py` cloned to
`%TEMP%\s38_postf4_backtest\run.py` via PowerShell substitution
(WORK_DIR only — START_DATE, END_DATE, UNIVERSE unchanged).
Throwaway script not committed (per the established Sn convention).

**Analyzer:** `%TEMP%\s38_postf4_compare.py` produces the pre/post
deltas reported in §2-§5. Throwaway, not committed.

**Compute:** 6.31h wall-clock (18 sec/day rate for 1,258 trading days).
Mid-range vs S40's 17-20s/day rate; CPU contention was lower at
this run time than during S40.

**Hit-rate definition note:** this doc reports
`(realized_pnl > 0).mean()` (= 82.1%-82.6%); the previously-published
S38 doc reported `(exit_reason == "otm_expire") / n_executed` (= 77.0%).
Both are valid; the broader definition includes assigned puts that
netted positive via large premium.

---

## AI handoff

- **S43 falsifies the S40 AI-handoff hypothesis** that F4 widening
  would close 5-10pp of the −52pp engine-vs-passive gap. The gap
  is structural to the strategy's limited deployment at \$1M/100t,
  not to the engine's tail-risk widening capability.
- **The dollar-alpha range from the deployment-matrix amendment
  (PR #263) doesn't need revision.** S43's −51.5pp is within the
  −52pp framing.
- **The F4 + R10 deployment-bundle framing from S41 + S43 is the
  honest closure of PROD_READINESS §3 B1.** F4 alone is signal-
  preserving but not value-creating; R10 alone caps notional but
  doesn't widen the engine's distribution; together they form
  defense-in-depth on (a) refusal frequency in vol-cluster regimes
  and (b) single-trade loss magnitude.
- **For future research:** test R10 in strict mode on the S38 setup.
  Set up a PortfolioContext on each tracker step that aggregates
  per-name exposure; verify R10 blocks AAPL / BKNG / AZO when
  any single name approaches 10% NAV. This would close the
  parallel S43 question of "does R10 actually constrain anything
  at \$1M/100t scale?" Likely answer: rarely, since the engine's
  natural deployment is wide; R10 mostly fires when the engine
  ranks AAPL or BKNG heavily for several consecutive days.
- **The Univ-EW benchmark is a more conservative passive comparison
  than external SPY.** S40 reported Univ-EW +92.19% vs external
  SPY ~+85% on the same window. S43 reinforces using Univ-EW as
  the apples-to-apples benchmark for engine-vs-passive analysis.
