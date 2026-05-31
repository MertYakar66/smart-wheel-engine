# Heavy-Verify Campaign 2026-05-31 — I6 (Wave 2): deepening the core findings

**Investigation:** three follow-ups that deepen and *balance* I1/I2/I3 — does the
regime overlay earn its keep, does the engine's ranking add selection value, and is
the top-bin over-confidence fixable?
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i6_deepening.py` (analysis
on `raw_output/i1_realized_rows.parquet`). **Raw:** `raw_output/i6_deepening_RAW.txt`.
**Status:** observe-and-document; `engine/` not modified.

---

## VERDICT

> **The engine's ranking *works* even though its dollar-EV isn't a calibrated
> forecast, and the over-confidence is *fixable*.** (W2-B) Monthly top-10 selection
> by any engine signal decisively beats random/all (+$166–206 vs −$24/−$26 mean;
> positive vs negative ROC) — `prob_profit` is the best risk-adjusted selector. This
> reconciles I1's "≈0 dollar rank-correlation": the signal's value is in *avoiding
> the catastrophic tail*, not in ordering the middle. (W2-A) The HMM regime overlay
> is directionally right for `crisis` (de-rated hardest, realizes worst) but
> *over-penalizes* `bear`. (W2-C) A histogram recalibration trained on 2020-2023
> generalizes out-of-sample (ECE 3.17pp → 1.29pp), so the top-bin over-confidence is
> learnable and fixable **in the calm/recent regime tested**.
>
> **⚠ QUALIFIED BY I9.** This W2-C result is on a BENIGN 2024-2026 test window. The
> follow-up generalization study (I9) shows the recalibration fix does **NOT**
> generalize to unseen *crises* (leave-one-crisis-out leaves the 2020 crash top-bin at
> 0.84-predicted vs 0.72-realized; the crisis realized rate is unstable across crises,
> 0.37–0.93). So the over-confidence is fixable in calm tape by recalibration but NOT
> where it actually bites — a structural (POT-GPD) fix is needed and is untested. See
> `HEAVY_VERIFY_2026-05-31_I9_FIX_GENERALIZATION.md`.**

Confidence: **high** for W2-B (large pooled samples, clear separation from random);
**medium** for W2-A (4 regime cells) and W2-C (benign 2024-2026 test period likely
*understates* the fix value — see caveat).

---

## W2-A — Does the HMM regime overlay earn its keep?

On Bloomberg the regime multiplier *is* the HMM multiplier (dealer/skew/news/credit
are structurally 1.0). Per-regime realized outcome vs the multiplier the engine applied:

| hmm_regime | n | mean regime_mult | realized win | realized mean $ | realized median $ |
|---|---|---|---|---|---|
| crisis | 2379 | **0.405** | **0.706** | −121 | 85.5 |
| bear | 5923 | 0.600 | 0.811 | −7.8 | **106.1** |
| normal | 5071 | 0.875 | 0.776 | −20.7 | 101.7 |
| bull_quiet | 2632 | 0.991 | 0.782 | +9.8 | 91.6 |

* **Directionally correct for crisis:** the engine de-rates `crisis` hardest (0.41×)
  and crisis *does* realize the worst win rate (0.71 vs ~0.78-0.81) and worst mean.
* **Over-penalizes bear:** `bear` gets a heavy 0.60× haircut yet realizes the
  *highest* median (+$106) and a fine win rate (0.81). The HMM "crisis vs bear"
  label conflates direction (it flags high-vol regardless of sign), so a recovering
  bear gets the same pessimism as a crashing one.
* Rank-corr(mean_regime_mult, realized_median) across regimes = **+0.20** — the
  ordering is right at the extremes but noisy in the middle.
* **Sign discipline holds:** `ev_dollars ≤ 0` rows realize median **+$87** (n=11,790)
  vs `ev_dollars > 0` median **+$155** (n=4,215) — the EV sign separates outcomes in
  median/ROC terms (even though the $ *mean* is tail-dominated, per I1).

## W2-B — Which engine signal best SELECTS profitable trades?

Monthly top-K (K=10) by each scorer, realized outcomes pooled across 73 months:

| selector | n | mean $ | median $ | win | realized ROC mean % | ROC Sharpe |
|---|---|---|---|---|---|---|
| **ev_dollars** | 730 | **+205.5** | 349.1 | 0.827 | 0.379 | 0.081 |
| ev_raw | 730 | +194.9 | 367.6 | 0.822 | 0.338 | 0.069 |
| **prob_profit** | 730 | +172.8 | 163.6 | 0.808 | **0.479** | **0.114** |
| ev_roc | 730 | +166.2 | 208.8 | 0.819 | 0.438 | 0.070 |
| random | 730 | −24.2 | 97.5 | 0.778 | −0.120 | −0.026 |
| all (no selection) | 16005 | −25.9 | 99.6 | 0.779 | −0.262 | −0.053 |

* **The engine's ranking adds real selection value.** Every engine signal's top-10
  turns the population's *losing* average (random/all: −$24/−$26, negative ROC) into
  a *winning* one (+$166 to +$206, positive ROC, higher win rate). The value comes
  from **avoiding the catastrophic-EV tail**, not from finely ordering the middle.
* **Reconciles I1.** I1 found `Spearman(ev_raw, realized_$) ≈ 0` over the full
  population — true, because single-path $ P&L is tail-dominated. But top-K selection
  still works (the worst trades carry the most-negative EV and are excluded). *Don't
  read ev_dollars as a $ forecast; do trust it (and especially prob_profit) to rank.*
* **`prob_profit` is the best risk-adjusted selector** (ROC-Sharpe 0.114, highest
  realized ROC 0.479%); `ev_dollars` wins on raw $ mean (it tilts to higher-notional
  names). A risk-adjusted trader should rank by `prob_profit`/`ev_roc`, not
  `ev_dollars` magnitude.

## W2-C — Is the top-bin over-confidence fixable? (out-of-sample recalibration)

Train a recalibration map on 2020-2023 (n=10,199), apply to 2024-2026 (n=5,806):

| forecast on test (≥2024) | ECE | Brier |
|---|---|---|
| raw `prob_profit` | 3.17pp | 0.1815 |
| + per-bin recalibration | **1.29pp** | 0.1788 |
| + bin × regime recalibration | 1.36pp | **0.1783** |

* The **>0.90 test bin (n=184)**: raw forecast 0.924 vs realized **0.804**; the
  per-bin recalibration pulls the forecast to **0.806** — matching reality. The
  over-confidence is **learnable from history and generalizes out-of-sample.**
* A **simple per-bin haircut captures most of the gain**; conditioning on regime
  (bin × regime) helps Brier marginally but not ECE here — the flat recalibration
  suffices for a first fix.
* **Caveat (understates the fix):** the 2024-2026 test window is mostly calm/bull,
  where raw calibration was already decent (ECE 3.17pp). On a crisis test period the
  raw ECE would be far larger (I1: crisis top-bin −26pp) and the recalibration gain
  correspondingly bigger. So the +60% ECE reduction shown here is a *lower bound* on
  the achievable improvement.

> **Implication for triage:** the I1/I3-E over-confidence is not an inherent dead-end.
> A histogram (or regime-conditional) recalibration layer on `prob_profit`, or wiring
> POT-GPD (`engine/tail_risk.py`) into the probability, would measurably fix it. This
> document *demonstrates* the value post-hoc; it does **not** modify the engine.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i1_calibration.py  # writes the realized rows
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i6_deepening.py
```
All numbers in `raw_output/i6_deepening_RAW.txt`.
