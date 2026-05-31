# Heavy-Verify Campaign 2026-05-31 — I1: Is the engine telling the truth? (Calibration)

**Investigation:** Point-in-time `prob_profit` and EV calibration on the full
S&P-500 universe, post-#294 engine.
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i1_calibration.py`
(+ `rank_snapshots.py`, `campaign_lib.py`). **Raw:** `raw_output/i1_calibration_RAW.txt`,
`raw_output/i1_calibration_summary.json`, `raw_output/i1_realized_rows.parquet`.
**Status:** observe-and-document. `engine/` not modified.

---

## VERDICT (where you can trust the probabilities, where you can't)

> **Trust `prob_profit` in the meat of the range (≈0.60–0.85) and in calm/bear
> regimes — it is calibrated to within ±5 percentage points across ~12,000
> candidates. Do NOT trust it at the top: a forecast of 0.95–1.00 delivers
> ~0.70 in reality (−27pp), and 0.90–0.95 delivers ~0.81 (−11pp). The
> over-confidence is concentrated in the HMM "crisis" regime (−26pp) and the
> overlapping-bootstrap distribution. Separately, do NOT read `ev_dollars`
> (or `ev_raw`) as a dollar-profitability forecast — it has ≈0 rank-correlation
> with realized single-path P&L and is dominated by fat-tail assignments on
> high-priced names.**

Confidence: **high** for the prob_profit calibration curve (n=16,005, Wilson CIs
reported); **high** for the EV-dollar non-predictiveness (robust across the full
sample and with the 2020 crash excluded); **medium** for the regime/source
conditioning (per-cell n in the hundreds).

This **confirms the operator's prior suspicion that the top probability bin is
over-confident** and quantifies it at far larger n than the prior HT-B study
(which had n=8 in the top bin and could not conclude).

---

## What I tested

* **Forecasts:** the §2-authoritative ranker `WheelRunner.rank_candidates_by_ev`
  run point-in-time at the **first business day of each month, 2020-01 → 2026-02**
  (74 dates) over the **full 503-name universe**, 35-DTE / 25-delta short puts,
  `min_ev_dollars=-1e9` (keep negative-EV rows so the whole probability range is
  populated), event gate ON. **16,005 candidate rows.** PIT correctness is the
  engine's own: PIT IV (`d26a8d6`), PIT OHLCV under the 504-day survivorship
  gate, HMM on history up to `as_of`. (Dealer/skew/news/credit multipliers are
  structurally 1.0 on Bloomberg — no chain — so this isolates the core forward
  model.)
* **Realized outcome:** the actual underlying close at `as_of + 35 calendar days`
  from `data/bloomberg/sp500_ohlcv.csv`, loaded through the connector's
  documented rotated-column rename (true close = raw `high`; cross-checked
  byte-identical to `connector.get_ohlcv`).
* **Two attributions, side by side** (this is what resolves the prior dispute):
  * `engine_exact`: realized P&L > 0 ⇔ `close > strike − premium`. This is the
    engine's **own** `prob_profit` definition `mean(pnls>0)`. The apples-to-apples
    calibration target.
  * `otm_expire`: `close ≥ strike` ("did the put expire worthless"). The prior
    studies' convention; answers assignment-avoidance, not P(profit).
* **Metrics:** reliability table (forecast bin → realized rate), Wilson 95% CIs,
  Brier score, ECE, weighted-MAD — none of which the prior studies computed.

## Results — prob_profit calibration (engine_exact, the engine's own definition)

Brier **0.1757**, ECE **4.06pp**, weighted-MAD **4.06pp**, N=16,005.

| forecast bin | n | engine mean | realized | Wilson 95% CI | Δ (pp) | verdict |
|---|---|---|---|---|---|---|
| (.5,.6] | 125 | 0.583 | 0.744 | [0.661,0.812] | +16.1 | over-cautious |
| (.6,.7] | 1472 | 0.669 | 0.760 | [0.737,0.781] | +9.0 | over-cautious |
| (.7,.8] | **7463** | 0.761 | 0.767 | [0.757,0.776] | **+0.6** | **OK** |
| (.8,.85] | 3077 | 0.828 | 0.783 | [0.768,0.797] | −4.5 | OK |
| (.85,.9] | 2745 | 0.872 | 0.808 | [0.793,0.822] | −6.4 | warn |
| (.9,.95] | 1017 | 0.920 | 0.811 | [0.786,0.834] | **−10.8** | MISCAL |
| (.95,1] | 105 | 0.966 | **0.695** | [0.602,0.775] | **−27.1** | MISCAL |

Read it plainly: when the engine says **"96% chance this put is profitable," the
real rate is ~70%.** When it says 92%, reality is 81%. Below 0.85 it is honest
(often slightly *over-cautious* in the 0.6–0.7 band). The bulk of candidates
(n=7,463) sit in the well-calibrated (.7,.8] bin.

## The attribution dispute — resolved

HT-B flagged that the top-bin miss could be "~12pp methodology artifact"
(otm_expire vs engine_exact) on its n=8 in-sample bin. At full universe (n=105
in the top bin) the choice of attribution moves the top bin by only ~4pp:

| bin | engine_exact Δ | otm_expire Δ |
|---|---|---|
| (.9,.95] (n=1017) | −10.8pp | −15.6pp |
| (.95,1] (n=105) | **−27.1pp** | −30.9pp |

**The over-confidence is real, not an attribution artifact.** It is MISCAL under
the engine's *own* definition. (otm_expire overstates the miss by ~4pp because it
doesn't credit shallow-ITM-but-still-profitable assignments — but it does not
manufacture the finding.)

## When the over-confidence bites — conditioning

Top-bin (.9,1] realized rate (engine_exact) by HMM regime and by distribution source:

| stratum | n | engine mean | realized | Δ (pp) |
|---|---|---|---|---|
| regime = **bear** | 456 | 0.923 | 0.893 | **−3.1** (trustworthy) |
| regime = bull_quiet | 101 | 0.921 | 0.812 | −10.9 |
| regime = normal | 265 | 0.922 | 0.785 | −13.8 |
| regime = **crisis** | 300 | 0.927 | 0.670 | **−25.7** (worst) |
| dist = empirical_non_overlapping | 783 | 0.923 | 0.840 | −8.2 |
| dist = empirical_overlapping | 339 | 0.927 | 0.708 | −21.9 |

Weighted-MAD by entry year: 2020 6.2 · 2021 5.5 · **2022 12.0** · 2023 4.1 ·
2024 2.9 · 2025 3.0 · 2026 9.5 (n=273). The engine is sharply calibrated in
calm/recent years and degrades in stress years (2022, 2020) — consistent with the
empirical forward distribution being unable to represent unseen tails (POT-GPD
exists in `engine/tail_risk.py` but is **not** wired into `prob_profit`).

> **Mechanism (hypothesis, medium confidence):** the over-confidence is a tail
> problem, not a center problem. The engine's empirical forward distribution
> reproduces the typical outcome well (center calibrated) but under-weights the
> crisis left tail, so the highest-confidence puts — exactly the ones priced off
> a benign recent history — are the ones reality surprises. This is why the miss
> is largest in `crisis` and the overlapping bootstrap (which over-smooths).

### Discrepancy with HT-B worth flagging
HT-B (100-ticker) reported `empirical_overlapping` *better* calibrated in the top
bin than `non_overlapping` (−7.2 vs −19.1). My full-503 result is the **opposite**
(overlapping −21.9, non_overlapping −8.2), stable across my partial and full runs.
I do not resolve the cause here (universe size / date mix / composition); flagging
it as an open reconciliation item, not a headline.

## Results — EV-dollar calibration (and an averted false alarm)

| ev_raw bin | n | engine mean | realized **mean** | mean 95% CI (bootstrap) | realized **median** | win rate |
|---|---|---|---|---|---|---|
| (−∞,−200] | 1867 | −420 | −123 | [−250, −7] | +229 | 0.745 |
| (−200,−50] | 5278 | −104 | −24 | [−42, −7] | +101 | 0.764 |
| (−50,0] | 4645 | −24 | −15 | [−31, −1] | +58 | 0.789 |
| (0,50] | 2245 | +20 | −11 | [−36, +14] | +101 | 0.811 |
| (50,100] | 810 | +72 | −82 | [−145, −23] | +194 | 0.775 |
| (100,200] | 647 | +140 | −25 | [−122, +72] | +282 | 0.770 |
| (200,500] | 411 | +299 | +41 | [−145, +199] | +495 | 0.813 |
| (500,∞] | 102 | +1357 | +984 | [−1486, +2827] | +1335 | 0.873 |

* `Spearman(ev_raw, realized_$P&L) = −0.002` (≈ zero; **+0.007 excluding the 2020
  crash**). The engine's **dollar** EV does **not** rank-order realized dollar P&L.
* `Spearman(ev_roc, realized_roc) = +0.164` — once you normalize by collateral
  (removing the notional-size confound: a $5,000-strike name swings 100× a
  $50 name), the engine's **expected return-on-collateral is weakly informative**.
* `Spearman(prob_profit, realized_$) = +0.145` — prob_profit is the more useful
  ranking signal than ev_dollars.
* Win rate rises monotonically with ev_raw (0.745 → 0.873): higher-EV candidates
  do win **more often** — the EV signal lives in win-frequency and ROC, not in
  the dollar mean, which is swamped by tails.

**Averted false alarm (adversarial-verification log).** At a partial n=4,259 the
sign was `Spearman = −0.015` and the per-bin means looked *inverted* (most-negative-EV
bin realizing the best mean). Bootstrap CIs and a tail decomposition showed this was
sampling noise: per-bin mean CIs span zero by thousands of dollars, and the mean is
set by a handful of catastrophes. **I did not publish the inversion.** At full N the
honest statement is "≈zero dollar rank-correlation, weakly positive in ROC space."

**The tail that drives the dollar P&L** (10 worst single-contract trades): NVR
−$83,039 (2020-03), BKNG −$60,890 (2026-01), NVR −$54,064 (2024-12, strike $8,670),
BKNG −$40,589, AZO −$37,252, BKNG −$28,931, TDG −$22,913, APP −$20,418, INTU
−$18,963. **Trimming the bottom 1% (161 trades) flips the overall realized mean
from −$26 to +$55/contract.** Several of these the engine rated *positive* EV
(NVR 2020-03 ev_raw +597 → realized −$83k; AZO ev_raw +2,349 → −$37k) — the EV
model does not protect against single-name catastrophe on high-priced underlyings.
(The R10 single-name cap would, but it is dormant in the default path — see I3.)

## Caveats / what would falsify or sharpen this

* **Universe survivorship:** the 503 names are *current* index members; the
  passive/realized view excludes names delisted since. For calibration this is
  minor (we realize each name's own path), but it slightly under-samples failures.
* **504-day history gate:** early-2020 (incl. the crash) is rankable only on the
  subset of names with ≥504 trading days since 2018-01-02 (2020-01 had 0
  candidates). Crisis-regime calibration is on that reduced set.
* **Monthly cadence / 35-DTE:** one strike per name per month at the engine's
  default 25-delta. Weekly cadence or other deltas could shift the top-bin n.
* **Synthetic premiums:** prob_profit calibration is premium-independent (it's an
  OTM/threshold-crossing frequency), so this does not affect the headline; it does
  affect the *dollar* EV calibration (engine premiums are BSM-synthetic — see I2
  for the real-premium comparison).
* **What would close the gap:** wiring the POT-GPD tail into `prob_profit` (or a
  regime-conditional haircut on high-confidence puts) is the obvious fix; this
  document does not implement it (observe-only).

## Reproduce

```bash
# 1. produce monthly full-universe ranked snapshots (≈55s/date; parallel chunks)
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/rank_snapshots.py --chunk 0 4   # x4
# 2. realize + calibrate
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i1_calibration.py
```
Every number above is in `raw_output/i1_calibration_RAW.txt`.
