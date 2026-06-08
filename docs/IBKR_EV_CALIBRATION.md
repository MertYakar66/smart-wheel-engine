# Phase 3 — EV calibration against the operator's real trades

Does the engine's **point-in-time** prediction match what actually happened on
the operator's real wheel options — cash-secured puts (CSPs) **and** covered
calls? `scripts/ibkr_ev_calibration.py` answers this by replaying every short
option the operator sold-to-open against the engine **as it would have run at
entry**, then comparing to the realized hold-to-expiry outcome. Both legs are
calibrated: the short-put leg, and the short-call (covered-call) leg in isolation
— which is exactly what the engine's call-leg `prob_profit` models.

**Read-only / observational (CLAUDE.md §2/§3).** This is analysis. It *uses*
`EVEngine.evaluate` (the authoritative evaluator) at each trade's exact strike —
it never bypasses it, never issues an EV-authority token, never converts anything
to a tradeable verdict, and does not modify the decision-layer trio
(`ev_engine` / `wheel_runner` / `candidate_dossier`). Real account data stays
gitignored.

## Method
For each SELL-put-to-open fill (from the IBKR Flex export):
- **PIT inputs at entry** (reusing the ranker's own machinery, no reimplementation):
  spot = OHLCV close ≤ entry; ATM IV = `_resolve_pit_atm_iv` (`get_iv_history(end_date=entry)`);
  as-of risk-free; the empirical forward distribution
  (`best_available_forward_distribution(ohlcv, horizon_days=dte, as_of=entry)` →
  `realized_vol_widened_log_returns`).
- **Engine prediction** at the operator's **exact strike / DTE / actual opening
  credit** via `EVEngine.evaluate`. `prob_profit` / `prob_assignment` / `mean_pnl`
  (ev_raw) are computed from the forward distribution *before* regime scaling, so
  `regime_multiplier=1.0` does not affect them.
- **Realized outcome** = hold-to-expiry, from the underlying's actual close on the
  option's expiry date (Bloomberg): short-put P&L = (premium − max(0, K − S_exp))·100;
  win = P&L > 0; assigned = S_exp < K. (This is the quantity the engine's
  probabilities model — used even for puts the operator closed early.)

PIT discipline was independently verified: **0/175 rows** where the recorded spot
≠ the connector's last close ≤ entry. As a second, harness-independent check, the
PIT spot, expiry close, and realized P&L were re-derived directly from raw OHLCV
for **all 456 legs** — **0 mismatches** on each (spot / expiry / P&L), and the
headline stats recompute identically. (This reproduces the numbers on current
data — the connector reads `data/bloomberg/` at 2026-06-04, not a stale clone.)

## Funnel
949 short-option opens → **456 calibrated** (**175 puts + 281 covered calls**).
Dropped: 426 out-of-universe (CLS, TSM ADR, CCO, CNQ, ENB, BN, TOU — outside the
S&P-500 mandate), 37 non-positive DTE (same-day 0DTE / data gaps), 21 expiry beyond
OHLCV (2026-06-05+), 4 no PIT spot, 2 thin forward dist, **2 moneyness/scale gate**
(see data-quality finding), 1 no IV. 23 underlyings.

## Results — short-put (CSP) leg (175 CSPs, Mar 2025 – Jun 2026)
- **Engine is slightly conservative overall:** observed win-rate **0.857** vs mean
  predicted prob_profit **0.823**; observed assignment **0.183** vs predicted **0.216**.
- **prob_profit:** Brier **0.122**, ECE **0.078**.
- **prob_assignment:** Brier **0.152**, ECE **0.081** (slightly over-predicts → conservative).

### prob_profit reliability (predicted bin → observed win-rate [95% Wilson CI], n)
| bin | mean pred | observed | 95% CI | n |
|---|---|---|---|---|
| [0.6,0.7) | 0.671 | 0.600 | 0.313–0.832 | 10 |
| [0.7,0.8) | 0.765 | 0.833 | 0.720–0.907 | 60 |
| [0.8,0.9) | 0.852 | 0.921 | 0.838–0.963 | 76 |
| **[0.9,1.0)** | **0.936** | **0.821** | **0.644–0.921** | **28** |

Two biases: the **0.7–0.9 band (the bulk, n=136) is *under*-confident** (observed
> predicted), while the **top [0.9,1.0) bin is *over*-confident** — predicted 0.936
sits **above** the 95% CI upper bound (0.921).

### ev_raw (predicted mean P&L) vs realized
Pearson **0.42**, Spearman **0.11** (weak per-trade rank discrimination).
*(Before the scale gate, a single corrupted trade inflated Pearson to 0.9966 —
Spearman exposed it. Pearson is reported only with that caveat.)*

### EV-sign gate value (modest)
| predicted ev_raw | n | win-rate | mean realized | median |
|---|---|---|---|---|
| > 0 | 83 | 0.892 | +$332 | +$244 |
| ≤ 0 | 92 | 0.826 | +$43 | +$175 |

Positive-EV CSPs realized better, but both buckets were net-positive over this
(largely bull) period — the gate ranks better-from-worse, it does not cleanly
separate winners from losers on this sample.

## Results — covered-call (short-call) leg (281 calls)
- **Engine is *under*-confident on covered calls:** observed win-rate **0.840** vs
  mean predicted prob_profit **0.760**; observed assignment **0.192** vs predicted
  **0.274** — it systematically **over**-predicts the call finishing ITM (being
  called away). prob_profit Brier **0.130**, ECE **0.094**.
- **EV-sign gate is *stronger* on calls:** predicted ev_raw > 0 → win **0.985**,
  mean **+$212** (n=65); ≤ 0 → win 0.796, mean **−$96** (n=216). ev_raw rank
  signal is still weak (Spearman 0.07).

### call prob_profit reliability (predicted bin → observed [95% CI], n)
| bin | mean pred | observed | 95% CI | n |
|---|---|---|---|---|
| [0.5,0.6) | 0.562 | 0.714 | 0.500–0.862 | 21 |
| [0.6,0.7) | 0.656 | 0.785 | 0.670–0.867 | 65 |
| [0.7,0.8) | 0.748 | 0.879 | 0.792–0.933 | 83 |
| [0.8,0.9) | 0.840 | 0.833 | 0.713–0.910 | 54 |
| [0.9,1.0) | 0.945 | 0.981 | 0.899–0.997 | 52 |

The 0.5–0.8 band (n=169) is materially **under-confident** (observed well above
predicted, mostly outside the CI) — the engine over-states call-assignment risk.

## Combined (456 legs)
Observed win-rate **0.847** vs predicted **0.785**; prob_profit Brier **0.127**,
ECE **0.075**; ev_raw Spearman **0.13**.

## The two weaknesses (found + gone through)
The puts and calls miscalibrate in **opposite** directions, and both trace to the
same root cause — the empirical forward distribution does not match the realized
short-window moves over this regime:
- **Puts: top-bin *over*-confidence** (the dangerous one — detailed below).
- **Calls: broad *under*-confidence** — the engine over-predicts the stock
  rallying through the strike, so it under-rates covered calls (predicted 0.76,
  realized 0.84). Less dangerous (it leaves premium on the table rather than
  taking hidden risk), but it means the engine would *skip* covered calls that
  would have won. Both point at forward-distribution tail/shape calibration over
  short horizons.

### Put top-bin prob_profit over-confidence
The engine's **most confident** short puts disappoint. In the [0.90,1.0) bin the
engine predicted **93.6%** profit but only **82.1%** realized (n=28). All 5
failures were **genuine assignments** (the put finished ITM), and they do **not**
cluster in the April-2026 crash — they span Oct '25, Feb '26, May '26, i.e. a
*persistent* miscalibration:

| ticker | entry → expiry | K | spot → S_exp | pred prob_profit | realized |
|---|---|---|---|---|---|
| ORCL | 2025-12-16 → 2026-02-20 | 175 | 188.65 → **148.08 (−21%)** | 0.9375 | **−$1,853** |
| META | 2025-10-09 → 2025-10-10 | 715 | 733.51 → 705.30 | 0.9149 | −$847 |
| NVDA | 2025-10-09 → 2025-10-10 | 185 | 192.57 → 183.16 | 0.9329 | −$142 |
| TSLA | 2026-05-14 → 2026-05-15 | 425 | 443.30 → 422.24 | 0.9130 | −$137 |
| GOOGL | 2026-02-02 → 2026-02-06 | 327.5 | 343.69 → 322.86 | 0.9395 | −$74 |

**Root cause:** the empirical forward distribution under-weights individual-name
**left tails** — the high-confidence trades that fail do so because the underlying
made a larger adverse move than the historical-return distribution anticipated
(ORCL −21% over the hold). **Impact:** this is the most dangerous miscalibration
because (a) a trader sizes *up* on "94% safe" trades, and (b) the top bin holds
the largest dollar losses (ORCL −$1,853). This independently corroborates the
prior internal PIT finding of top-bin over-confidence — now confirmed on **real,
out-of-sample, real-money** trades.

**Why R11 does not cover this (quantified).** The engine already carries **R11**,
an elevated-vol top-bin size-down (`candidate_dossier.py`: downgrade *proceed →
review* when `prob_profit > 0.90` **and** `vix_level > 25.0`). R11 validates the
*direction* — top-bin confidence is unreliable — but it is gated on **crisis VIX**,
and these failures are a **calm-regime** phenomenon. Pulling the actual VIX level
at each entry (`vix_term_structure.csv`): all 5 failures occurred at VIX **16.3–
17.3** (ORCL's −$1,853 at VIX **16.5**), and across all 28 top-bin puts VIX-at-entry
ranged only **15.3–25.8**. **R11 would fire on just 2 of the 28 top-bin trades and
catch 0 of the 5 realized failures.** Its VIX>25 gate is structurally blind to this
miss — so R11 is not the fix here.

> **⚠️ Guardrail — do NOT "fix" this by widening R11's VIX gate.** "R11 catches
> 0/5" is **not** an argument to lower `R11_VIX_THRESHOLD`. R11's crisis-only scope
> (VIX>25) is **deliberate and validated** — leave-one-crisis-out survives at 25
> while ≥27.5 fails the 2022 fold (heavy-verify I11). Making R11 fire in calm vol
> would size down *every* high-confidence trade regardless of regime — an
> unvalidated over-correction that stretches a narrow crisis rule into a general
> one. The finding here is that a **separate, calm-regime** over-confidence exists
> that R11 was **never meant to cover**; the only §2-safe response is the
> observational recalibration overlay (path 1 below) — **not** a wider R11, and
> **not** forward-distribution widening (path 2, S27 ρ risk).

**Remediation (supervised — not shipped here).** Two paths, both needing
validation, *not* an autonomous engine change:
1. *Observational recalibration overlay* (isotonic/Platt map from predicted →
   empirical prob_profit) surfaced in the viewer/analysis. **The better-targeted
   path** for a calm-regime miss that R11's VIX gate cannot see. Safe (no trio
   change) but must be fit out-of-sample (175 in-sample points overfit).
2. *Forward-distribution left-tail widening* in the engine. **High risk:** prior
   tail-widening attempts inverted the S27 backtest ρ and were rolled back twice
   (see `docs/F4_TAIL_RISK_DIAGNOSTIC.md`). Any such change must clear the
   S27 ρ ≥ +0.15 gate and the regression baselines before merge.

## Data-quality findings (engine-affecting — for the data layer)
Surfaced while verifying this calibration; the data itself is the data layer's to
fix (the connector reads `data/bloomberg/sp500_ohlcv.csv`). A scan of all 511
OHLCV series found **3 scale-corrupted names** that the engine would **mis-rank
live** (none are real stock splits — these mega-priced names famously do not split):

1. **NFLX — uniformly ~10× low.** The connector returns a close range of
   **16.64–133.91** vs the real ~$166–1,339. A 2025 NFLX put struck at 1075 shows
   a Bloomberg "spot" of ~110. In this harness it produced a garbage moneyness
   (prob_profit 0.0, realized −$95,831) that **single-handedly** inflated the EV
   Pearson r (0.9966 → true Spearman 0.11) and flipped the EV-sign split — caught
   by the moneyness gate (drop strike/spot ∉ [0.5, 1.5]).
2. **BKNG — scale break at the 2026-03-23 reconstitution seam:** 4293.38 → 168.75
   (~25×). 3. **CVNA — same seam:** 291.36 → 58.40 (~5×).

The raw file stores Bloomberg-format identifiers (`BKNG UW Equity`, `CVNA UN
Equity`, …) that the connector normalizes; the corruption rides through to
`get_ohlcv('NFLX'/'BKNG'/'CVNA')`. This cross-references the known 2026-03-23 seam
(see the data-engine audit) and pins the specific affected tickers. **Recommend a
data-layer re-pull of NFLX / BKNG / CVNA**; until then the engine's EV for those
names is unreliable. (The other 508 names show no scale discontinuity.)

## Limitations
- **Universe-restricted** to S&P-500 names with Bloomberg data — the operator's
  dominant position (CLS) and TSM ADR / Canadian sleeve are excluded, so this
  calibrates the engine's *mandate*, not the whole book.
- **n=456** (175 puts + 281 calls); per-bin counts are small (put top bin n=28) —
  findings are reported with Wilson CIs; the put top-bin over-confidence and the
  call 0.6–0.8 under-confidence are significant (predictions outside the CI), the
  rest are suggestive.
- **In-sample / single regime** — Mar 2025–Jun 2026 was largely a bull market with
  one sharp April-2026 drawdown; calibration may differ in other regimes.
- **dividend_yield** from the dateless fundamentals snapshot (slow-moving; matches
  engine-as-deployed; minor lookahead).
- **Hold-to-expiry counterfactual** is used even for puts the operator closed
  early — that is the quantity the engine's probabilities model.
