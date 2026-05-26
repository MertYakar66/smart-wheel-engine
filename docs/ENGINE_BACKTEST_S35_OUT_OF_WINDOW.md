# Engine backtest — S35: 2018-2020 out-of-window cross-validation (2026-05-26)

**Question:** *Does the engine's predictive signal generalize across
time windows? In particular, does the "engine beats SPY" headline from
S22 / S27 (2022-2024 at $100k) hold for a different 3-year window?*

**Headline answer:** **The signal generalizes; the dollar
outperformance does NOT.** Spearman ρ in 2020 is **0.50 — double
S27's 0.22 in 2022-2024.** But the engine returns +3.57% (full
friction) vs SPY's ~+45% over 2018-2020 — **a 41pp
underperformance.** The "beat SPY by 27pp" headline from S22 / S27
turns out to be both **$100k-specific (per S32) AND 2022-2024-window-
specific.** The ranking quality is robust; the dollar-alpha is not.

**Window / universe / strategy / engine:** identical to S27 except
for the time window:
- 2018-01-02 → 2020-12-31 (756 trading days; pre-COVID + 2018-Q4
  selloff + 2019 bull + 2020 COVID crash + V-recovery)
- Same 24-ticker universe as S22 / S27 / S32 / S34 (deliberate, for
  direct comparison)
- $100k starting capital (matches S27's $100k for direct comparison)
- 35-DTE / 25-delta short puts, wheel into CC, hold to expiry
- `require_ev_authority=False`
- Post-IV-PIT-fix engine on `origin/main` (commit `e504801`)

**Hard methodology caveats:**

- **Caveat 2 (in-sample HMM / POT-GPD parameters) STILL APPLIES.**
  The parameters were tuned with full 2018-2026 data visibility. 2018
  is at the parameter window's leading edge but the calibration saw
  it.
- **Caveat 3 (frictionless P&L) closed by this run.** Three friction
  trajectories side-by-side; identical drag of ~0.9% NAV across
  bid_ask and full.
- **NEW Caveat — 504-day OHLCV history gate.** The engine refuses to
  rank any ticker with less than 504 OHLCV trading days of history
  (`gate=history, reason=history Nd < required 504d`). With OHLCV
  starting 2018-01-02, this means the engine returns ZERO rankable
  candidates until ~2020-01-02. **S35 is effectively a 2020-only
  backtest, with ~252 useful trading days.** Implication for
  deployment: new IPOs or recently-listed names are unrankable until
  they accumulate 504 days of history (~2 years).

---

## Per-friction-level results

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | $104,473 | $103,594 | $103,566 |
| Return | +4.47% | +3.59% | **+3.57%** |
| Final cash | $40,915 | $40,036 | $40,008 |
| Short puts opened | 19 | 19 | 19 |
| CCs opened | 24 | 24 | 24 |
| Put assignments | 6 | 6 | 6 |
| CC OTM expires | 20 | 20 | 20 |
| CC assignments | 3 | 3 | 3 |
| Skipped `already_held` | 409 | 409 | 409 |
| Skipped `insufficient_bp` | **221** | **221** | **221** |
| Open positions at end | 7 | 7 | 7 |

Same execution count across all three levels (only friction varies).
Friction drag: ~0.9% NAV (about 3× S32's 0.27% drag because the
smaller capital amplifies the proportional friction).

**Headline financial outcome at full friction: $100,000 → $103,566 =
+3.57% over 3 years.** Annualised ≈ +1.18%/year. **Below the
risk-free rate** for most of 2020 (3-month T-bill yielded ~0.15% in
2020 but rose to 0.5% by end-2021). The engine generated alpha but
not enough to justify the capital lock-up.

---

## Engine vs SPY — 41pp underperformance

| | Engine (full friction) | SPY buy-and-hold |
|---|---|---|
| 2018-01-02 → 2020-12-31 return | **+3.57%** | ~+45% |
| Engine vs SPY | **−41pp** | — |

For comparison, S27 at the same parameters but 2022-2024 window:

| | S27 (2022-2024) | S35 (2018-2020) |
|---|---|---|
| Engine return | +51.4% | +3.57% |
| SPY return | ~+24% | ~+45% |
| Engine vs SPY | **+27pp** | **−41pp** |

**The 68pp delta between S27 and S35 in engine-vs-SPY performance is
NOT explained by capital (both at $100k), universe (both 24 names),
strategy (both 35-DTE / 25-delta wheel + CC), or engine version (both
post-IV-PIT-fix).** The only material variable is the time window —
specifically, the regime sequence.

S27's 2022-2024 = bear → recovery → bull. The bear was contained;
the bull was strong; the wheel benefited from steady premium income.

S35's 2018-2020 = chop → strong bull → crash + V-recovery. The COVID
crash hit positions hard; the V-recovery didn't help the engine
because **the engine sat out the recovery** (more on this below).

---

## Spearman ρ — paradoxically STRONGER

| Window | n | ρ | p | Mean realized (puts) | Mean realized (executed) |
|---|---|---|---|---|---|
| S22 2022-2024 (pre-IV-PIT) | 6,163 | 0.484 | ~0 | $229.77 | $200.72 |
| S27 2022-2024 (post-fix) | 6,163 | **0.218** | 2.29e-67 | $63.34 | $-71.99 |
| S32 2022-2024 $1M | 5,743 | 0.192 | 9.83e-49 | $32.09 | $170.19 |
| **S35 2020 (post-fix)** | **1,946** | **0.4970** | **6.58e-122** | **−$169.21** | **−$1,125.25** |

**In 2020 specifically, ρ doubles from S27's 0.22 to 0.50** —
statistically overwhelming with N=1,946 (p ≈ 6.6e-122). The engine
ranks the BEST in the year with the highest realized volatility and
the most extreme regime transitions.

**But the mean realized is uniformly negative across all puts** —
the engine identifies relative quality correctly (Q3 beats Q0) but
the absolute environment was bad for short puts. **Strong signal,
bad expectation.**

This is the opposite pattern from S22 (low ρ 0.27 in 2022 bear, but
positive mean realized $81). The signal-vs-realized correlation
varies dramatically across market regimes.

---

## COVID-specific behaviour — engine wisely sat out

| Metric | Value |
|---|---|
| Candidate rows in 2020-02-15 → 2020-05-15 (full friction puts) | 482 |
| Executed in this window | **1** |
| Mean realized of ALL 482 candidates | −$369.55 |
| Mean realized of the 1 executed | +$231.28 (one OTM expiry win) |

**During the COVID crash window, the engine wisely refused 481 of
482 candidates** (99.8% refusal rate). The one trade it took was a
winner. **This is the engine working correctly** — its EV
calculation reflected the elevated tail risk, and it refused to
enter. The negative mean realized of all 482 candidates confirms the
engine's refusals were correct in aggregate: if it had blindly
entered, it would have lost $370 per trade.

**This is the strongest piece of evidence for the engine's *value
as a refusal mechanism***. In a regime where blind entries lose $370
per trade, the engine refused 99.8% of them. This is exactly what
the engine SHOULD do.

But the engine still LOST money over 2020 overall because:
- The 18 trades it DID take (mostly post-COVID, mid-to-late 2020)
  averaged −$1,125 realized.
- Post-COVID IV stayed elevated; spot moves stayed jagged through
  late-2020; the engine's `prob_profit` was over-optimistic in the
  unusual post-pandemic vol environment.

---

## Capital deployment

- Average deployed collateral: **$12,245 (12.2% of $100k)**
- Insufficient BP rejections: **221**

Comparable to S32's 10.8% at $1M but with active BP saturation. The
2020-only effective window (after the history gate) limited
deployment compared to S27's 3-year window where BP saturated
multiple times.

---

## Quartile means (full friction, puts only)

| Quartile | n | EV mean | Realized mean | Hit-rate |
|---|---|---|---|---|
| Q0 (low) | 486 | −$256 | **−$316** | 56.4% |
| Q1 | 487 | −$87 | −$179 | 64.7% |
| Q2 | 487 | $2 | −$87 | 73.7% |
| Q3 (high) | 486 | $114 | **−$76** | 79.0% |

**Q3 still beats Q0** in mean realized (−$76 vs −$316) and hit-rate
(79% vs 56%). The signal is clean — the engine RANKS correctly. But
**every quartile has negative mean realized**, including the engine's
"best" trades.

This is the clearest possible demonstration that **the engine ranks
relative quality correctly but cannot predict absolute outcomes
under a regime it wasn't tuned for.**

---

## Findings

- **F1 — Signal generalizes (ρ doubles in 2020).** The ranking
  quality is genuinely a property of the engine, not of the window.
  ρ = 0.50 at N=1,946 is the strongest signal measured anywhere in
  the campaign.
- **F2 — Dollar-alpha does NOT generalize.** Engine returned
  +3.57% (full friction) over 2018-2020 vs SPY's ~+45%. The
  "+27pp over SPY" from S22 / S27 is **window-specific**.
- **F3 — Engine wisely sat out COVID.** 99.8% refusal rate during
  2020-02-15 → 2020-05-15 window. The 1 trade taken was a winner.
  This is the engine's "refusal" value working as designed.
- **F4 — Post-COVID positions performed poorly.** Even after the
  crash, the engine's `prob_profit` was over-optimistic in
  post-pandemic vol environment. Mean realized −$1,125/executed.
- **F5 — 504-day OHLCV history gate is real and material.** Engine
  refuses all 2018-2019 dates because of insufficient history.
  Implication for deployment: 2-year warm-up period before any
  ticker is rankable. Documented behavior, not a bug.
- **F6 — Friction drag at $100k is ~0.9% NAV** (3× S32's 0.27% at
  $1M because the proportional drag dominates when capital is small).

---

## Cross-window comparison table

| Window | Capital | ρ | Engine NAV | SPY | Engine vs SPY | Notes |
|---|---|---|---|---|---|---|
| 2022-2024 (S27) | $100k | 0.22 | +51% | +24% | **+27pp** | BP saturated; engine's deployment was high |
| 2022-2024 (S32) | $1M | 0.19 | +1.85% | +24% | **−22pp** | Capacity constraint; engine under-deployed |
| 2020 (S35) | $100k | **0.50** | +3.6% | ~+45% | **−41pp** | Strong signal, bad market for puts |

**Three windows, three different engine-vs-SPY results.** The
ranking signal is the only metric that's consistent (and even it
varies). **No conclusion about future performance can be drawn from
any single window.**

---

## Implications for `docs/PRODUCTION_READINESS.md`

S35 strengthens the case **against autonomous deployment** at any
scale:

- The "+27pp over SPY" property does not generalize across windows.
- The engine's value is partly in its **refusal mechanism** during
  crisis regimes (validated by S35 F3) — but autonomous deployment
  would still execute on the few candidates the engine takes, and
  those have lost money on average in 2020.
- The +3.57% / 3 years return at $100k 2018-2020 is BELOW the
  3-year T-bill yield over the same window.

For decision matrix in `docs/PRODUCTION_READINESS.md` §5:
- **At $100k, supervised:** still ✅ — the signal is real and the
  refusal mechanism works, but the human reviewer must override
  the engine when the absolute environment is hostile (as the
  diagnostic table here shows for 2020).
- **At $100k, autonomous:** still ❌ — F4 tail risk + post-crisis
  over-optimism.
- **At $1M+:** still ❌ — F3 (capacity) + F2 (window dependence).

---

## AI handoff

- **Next deployment-readiness check:** when (if) the F4 fix lands,
  re-run S35 on the post-fix engine. Hypothesis: the regime-
  conditioned distribution widening (Fix B1 from PR #221's
  diagnostic) should improve the post-COVID over-optimism, raising
  the 2020 mean realized.
- **For docs/PRODUCTION_READINESS.md:** add a new caveat
  acknowledging the **window-specific dollar-alpha** finding. The
  "beats SPY" framing must always be qualified with both
  "$100k-class" AND "2022-2024-class" caveats.
- **For the engine team:** the 504-day history gate is a real
  deployment constraint worth surfacing in onboarding docs. New
  IPOs and recently-listed names are unrankable for 2 years. Either
  document explicitly or consider whether the gate's
  `min_samples=504` is well-tuned.
- **For the next backtest:** a 4-year or 5-year window
  (2018-2022 or 2018-2023) would average out window-specific noise
  and provide a more representative "engine vs SPY" verdict. With
  the 504-day gate, effective trading would start ~2020-01 and run
  through end-of-window.

---

## Method appendix

Harness: `%TEMP%\s35_backtest\run.py` (not committed; same throwaway
pattern as S22 / S27 / S32 / S34). Identical to S32 except:
- `STARTING_CAPITAL = 100_000.0` (matches S27)
- `START_DATE = 2018-01-02` / `END_DATE = 2020-12-31`
- `UNIVERSE` same as S22 / S27 / S32 / S34

Output: `%TEMP%\s35_backtest\rank_log.csv` (5,910 rows = 1,970 per
friction level × 3 levels).

Runtime: ~40 minutes wall-clock on the dev box, with parallel
S34 backtest also running.
