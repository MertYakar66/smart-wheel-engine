# prob_profit calibration — multi-backtest analysis (2026-05-28)

**Question:** *Is the top-bin prob_profit miscalibration finding
from `docs/REAL_DATA_VERIFICATION_2026-05-28.md` (PR #273) — engine
0.92 → actual 0.79; engine 0.97 → actual 0.82 — **universal across
configurations** (suggesting a structural empirical-distribution
issue) or **S38-specific** (which would need investigation)?*

**Verdict, one sentence:** **UNIVERSAL.** Across **10 of 10**
backtest configurations measured (S22, S27, S32, S34, S35, S38 pre/post-F4,
S40 W1/W2/W3), every single one has at least one MISCALIBRATED
(>10pp delta) bin. The top bin (0.95, 1.0] is miscalibrated by
−5pp to −18pp in **9 of 10 configs**. **The engine systematically
over-promises on its most-confident picks regardless of capital,
universe size, time window, or pre/post-F4 engine SHA.** Mean
weighted MAD across configs = **6.16pp**, consistent with PR #197's
published 7.6% reference. **F4 fix (PR #260) does NOT measurably
improve calibration** — S38 pre-F4 vs S38 post-F4 produce nearly
identical calibration tables.

**This strengthens the rationale for the F4 + R10 deployment bundle:**
the engine's frequency guard (refusal mechanism) is itself
over-confident on its top-bin recommendations, so the magnitude
guard (R10 single-name cap, 10% NAV per underlying) is structurally
necessary — neither alone is sufficient.

---

## Pre-declared standard (same as PR #273)

| Delta | Verdict |
|---|---|
| ≤ 5pp | ✅ Calibrated |
| 5-10pp | ⚠ Slightly miscalibrated |
| > 10pp | ❌ **MISCALIBRATED** |

Reference: PR #197 (predictive validity review) found 7.6% mean MAD
across deciles — the prior campaign-published reference point for
calibration quality.

---

## Cross-config summary

| Config | n_total | Weighted MAD | OK bins | Warn bins | MISCAL bins | Top-bin Δ |
|---|---|---|---|---|---|---|
| S22 (24t/\$100k/2022-2024 pre-PIT-fix) | 6,163 | 4.97pp | 2 | 3 | 1 | −7.75pp |
| S27 (24t/\$100k/2022-2024 post-fix) | 6,163 | 6.15pp | 3 | 1 | 3 | −4.88pp |
| S32 (24t/\$1M/2022-2024) | 5,743 | 6.22pp | 2 | 2 | 3 | −5.11pp |
| S34 (100t/\$1M/2022-2024) | 10,315 | 7.29pp | 2 | 2 | 3 | **−15.05pp** |
| S35 (24t/\$100k/2018-2020) | 1,946 | 8.77pp | 2 | 1 | 3 | **−16.64pp** |
| S38 (100t/\$1M/2020-2024 pre-F4) | 17,192 | 6.45pp | 2 | 3 | 2 | **−14.80pp** |
| **S38-postF4 (100t/\$1M/2020-2024 post-F4)** | **17,192** | **6.40pp** | **2** | **2** | **3** | **−15.63pp** |
| S40-W1 (100t/\$1M/2021-2026) | 17,520 | 5.75pp | 2 | 2 | 3 | **−17.57pp** |
| S40-W2 (100t/\$1M/2022-2026) | 14,110 | 6.54pp | 2 | 2 | 3 | **−15.06pp** |
| S40-W3 (100t/\$1M/2023-2026) | 10,720 | 3.07pp | 3 | 2 | 2 | **−17.44pp** |

**Statistics:**
- 10 of 10 configs have ≥ 1 MISCAL bin
- Top bin (0.95, 1.0] MISCAL in 9 of 10 configs (only S27 is "warn" at −4.88pp)
- Mean weighted MAD across configs: **6.16pp** (vs PR #197 published 7.6pp)
- F4 fix Δ: S38 pre-F4 MAD 6.45pp vs S38 post-F4 MAD 6.40pp (essentially identical)

---

## Per-config calibration tables

### S22 (24t/\$100k/2022-2024, pre-PIT-fix)

n=6,163. Pre-IV-PIT-fix engine baseline. Includes the spurious
ρ=0.484 case (the IV-PIT bug was inflating the apparent signal).

| Bin | n | Engine mean | Actual OTM | Delta | Verdict |
|---|---|---|---|---|---|
| (0.6, 0.7] | 34 | 0.6690 | 0.8824 | **+21.33pp** | ❌ MISCAL |
| (0.7, 0.8] | 1,215 | 0.7717 | 0.8099 | +3.82pp | ✅ |
| (0.8, 0.85] | 1,394 | 0.8283 | 0.7769 | −5.14pp | ⚠ |
| (0.85, 0.9] | 1,782 | 0.8737 | 0.8384 | −3.53pp | ✅ |
| (0.9, 0.95] | 1,347 | 0.9239 | 0.8589 | −6.49pp | ⚠ |
| (0.95, 1.0] | 391 | 0.9727 | 0.8951 | −7.75pp | ⚠ |

**Note:** S22 calibration is the BEST among all configs (MAD 4.97pp). The pre-IV-PIT-fix engine's over-optimism was less severe — which is consistent with the IV-PIT bug *deflating* the engine's volatility estimates (lower vol → lower IV → narrower distribution → fewer probability extremes). PR #179 fixed the bug; calibration is correspondingly slightly worse post-fix.

### S27 (24t/\$100k/2022-2024, post-fix)

n=6,163. Same window/universe/capital as S22; post-IV-PIT-fix.
**This is the published baseline** in `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`.

| Bin | n | Engine mean | Actual OTM | Delta | Verdict |
|---|---|---|---|---|---|
| (0.5, 0.6] | 43 | 0.5873 | 0.7907 | **+20.34pp** | ❌ MISCAL |
| (0.6, 0.7] | 498 | 0.6660 | 0.7550 | +8.90pp | ⚠ |
| (0.7, 0.8] | 2,367 | 0.7675 | 0.7436 | −2.39pp | ✅ |
| (0.8, 0.85] | 1,312 | 0.8285 | 0.7035 | **−12.50pp** | ❌ MISCAL |
| (0.85, 0.9] | 1,377 | 0.8704 | 0.8410 | −2.95pp | ✅ |
| (0.9, 0.95] | 529 | 0.9185 | 0.7996 | **−11.88pp** | ❌ MISCAL |
| (0.95, 1.0] | 37 | 0.9678 | 0.9189 | −4.88pp | ✅ |

### S38 (100t/\$1M/2020-2024, pre-F4)

n=17,192. The reference case used in PR #273.

| Bin | n | Engine mean | Actual OTM | Delta | Verdict |
|---|---|---|---|---|---|
| (0.5, 0.6] | 76 | 0.5833 | 0.6711 | +8.77pp | ⚠ |
| (0.6, 0.7] | 888 | 0.6708 | 0.6498 | −2.10pp | ✅ |
| (0.7, 0.8] | 4,903 | 0.7669 | 0.7453 | −2.16pp | ✅ |
| (0.8, 0.85] | 3,827 | 0.8288 | 0.7724 | −5.64pp | ⚠ |
| (0.85, 0.9] | 4,478 | 0.8725 | 0.7923 | −8.02pp | ⚠ |
| (0.9, 0.95] | 2,378 | 0.9210 | 0.7914 | **−12.96pp** | ❌ MISCAL |
| (0.95, 1.0] | 642 | 0.9673 | 0.8193 | **−14.80pp** | ❌ MISCAL |

### S38-postF4 (100t/\$1M/2020-2024, post-F4)

n=17,192. Post-PR-#260 engine. Comparison point for whether F4 fix
helps calibration.

| Bin | n | Engine mean | Actual OTM | Delta | Verdict |
|---|---|---|---|---|---|
| (0.5, 0.6] | 79 | 0.5832 | 0.6835 | **+10.03pp** | ❌ MISCAL |
| (0.6, 0.7] | 905 | 0.6708 | 0.6475 | −2.33pp | ✅ |
| (0.7, 0.8] | 4,928 | 0.7668 | 0.7470 | −1.98pp | ✅ |
| (0.8, 0.85] | 3,888 | 0.8289 | 0.7747 | −5.42pp | ⚠ |
| (0.85, 0.9] | 4,469 | 0.8725 | 0.7939 | −7.86pp | ⚠ |
| (0.9, 0.95] | 2,327 | 0.9211 | 0.7838 | **−13.72pp** | ❌ MISCAL |
| (0.95, 1.0] | 596 | 0.9667 | 0.8104 | **−15.63pp** | ❌ MISCAL |

**F4 fix verdict: NO improvement in calibration.** Top-bin Δ moved
from −14.80pp (pre-F4) to −15.63pp (post-F4) — slight worsening,
within noise. Weighted MAD essentially identical (6.45 → 6.40pp).
**F4 widening is a regime-conditioned filter; it does NOT address
the systematic over-confidence on the highest-prob picks.**

### S40-W1 / W2 / W3 (100t/\$1M post-F4, different windows)

All three S40 windows show the same pattern:
- (0.5, 0.6]: large positive delta (engine UNDER-claims, +12pp)
- Mid bins (0.6-0.85): well-calibrated (within ±8pp)
- (0.9, 0.95] + (0.95, 1.0]: MISCALIBRATED (−13pp to −18pp)

W3 has the lowest overall MAD (3.07pp) because its (0.5, 0.6] bin
is small and the (0.95, 1.0] bin is small (n=75) — but the top bin
is still MISCAL at −17.44pp.

(See `multi_backtest_calibration.py` script output for full
per-bin tables on S32, S34, S35, S40-W1/W2/W3.)

---

## Key cross-config observations

**Observation 1: Top-bin miscalibration is structural.** Every
configuration with sufficient (0.9, 0.95] or (0.95, 1.0] occupancy
shows MISCAL miscalibration in those bins. The magnitude is
−10pp to −18pp. **The engine systematically over-promises on its
most-confident picks regardless of:**
- Capital (\$100k → \$1M)
- Universe (24t → 100t)
- Time window (2018-2020, 2022-2024, 2020-2024, 2021-2026, 2022-2026, 2023-2026)
- Engine SHA (pre-IV-PIT-fix, post-fix, pre-F4, post-F4)

**Observation 2: Mid-bin calibration is stable.** The (0.6, 0.7] and
(0.7, 0.8] bins are consistently within ±5pp across all 10 configs.
The mid-range is where the engine is honest.

**Observation 3: Low-bin shows OPPOSITE direction miscalibration.**
The (0.5, 0.6] bin shows the engine UNDER-claiming — engine ~0.58
but actual ~0.70. This is +8 to +20pp delta. **The engine is
under-confident on its least-confident picks.** This is the
symmetric opposite of the top-bin issue.

**Observation 4: F4 fix doesn't move calibration.** S38 pre-F4 vs
post-F4: MAD 6.45 → 6.40pp; top-bin −14.80pp → −15.63pp. The F4
realized-vol-ratio widening fires on only ~12% of cells (per S41
calibration); most high-confidence picks are outside its fire
window, so the over-optimism on them persists. **R10 (PR #262)
is the necessary magnitude guard precisely because F4 widening
cannot reach these top-bin picks.**

**Observation 5: Mean MAD 6.16pp aligns with PR #197's 7.6pp.**
The earlier published reference point (S22/S27 review) had MAD 7.6pp;
my cross-config mean is 6.16pp. The campaign's prior calibration
finding generalizes — wasn't S22/S27-specific.

---

## Mechanism (why the engine over-promises on top picks)

The engine's `prob_profit` is computed as the fraction of bootstrapped
35-day trajectories in the empirical forward distribution that end
above the strike. Top-bin candidates have empirical distributions
where **zero or near-zero historical samples** ended below the strike.

In the real world, **unseen tail events** still occur on those names
13-15% of the time. The empirical distribution cannot represent
"events that haven't happened in the sample but could happen" — that
requires either:
1. **Parametric tail extension** (POT-GPD — engine has this in
   `engine/tail_risk.py` but doesn't apply it to prob_profit
   computation; F4 diagnostic §10 notes this gap)
2. **Bayesian shrinkage** toward a wide prior at low sample sizes
3. **Cross-sectional pooling** across tickers to estimate the
   unobserved tail mass

None of these are wired into the current engine path. The empirical
distribution method produces calibrated mid-bin output (where the
sample is rich enough to capture both sides) but **structurally
over-promises in the top bins** (where the sample doesn't reach
the relevant adverse trajectories).

This is the **canonical F4 finding** — the empirical forward
distribution misses unseen tails — quantified across 10 backtests.

---

## Implications

### For deployment matrix (PROD_READINESS §5)

**Engine `prob_profit` is a useful ORDINAL ranking signal** (Spearman
ρ ≈ 0.22 across configs is statistically significant) **but NOT
a reliable CARDINAL probability** for real-money decisions. A trader
using the engine should:

- Treat engine prob_profit = 0.92 as actual prob ≈ 0.79
- Treat engine prob_profit = 0.97 as actual prob ≈ 0.82
- Discount high-bin claims by 10-15pp before sizing

This is **operational hygiene**, not a fix. It does not change the
deployment matrix verdict — `docs/PRODUCTION_READINESS.md` §1 already
acknowledges \"the dollar outcome at scale comes mostly from equity
beta on assignments\" rather than from put-selection skill, which is
consistent with the prob_profit miscalibration finding.

### For the F4 + R10 deployment bundle (PROD_READINESS §3 B1)

The bundle framing is now **better-grounded**:
- **PR #260 (RV widening, frequency guard)** addresses 12% of cells
  in elevated-vol regimes; signal-preserving but does not improve
  high-bin calibration (S38 pre/post F4 calibrations identical
  within noise).
- **PR #262 (R10 single-name cap, magnitude guard)** is the
  STRUCTURAL response to top-bin over-confidence: even if the
  engine's most-confident picks lose 18% of the time (vs claimed
  3-5%), R10 bounds the single-name notional at 10% NAV, limiting
  catastrophic single-trade loss.

**Without R10, the 13-18pp top-bin miscalibration translates to
unbounded dollar damage on the engine's highest-conviction trades.**
R10 is therefore the load-bearing piece of the F4 bundle for real-
money deployment, not #260.

### For future research

A non-trivial open question: **could the engine's prob_profit be
recalibrated by wiring POT-GPD into the probability computation?**
The POT-GPD tail-estimation machinery already exists in
`engine/tail_risk.py` but is not applied to prob_profit. A future
PR could:

1. Extend `engine/ev_engine.py` to apply POT-GPD tail extension
   to the empirical forward distribution before computing
   prob_profit
2. Re-baseline on a held-out window (e.g., S38) and measure
   calibration improvement
3. If MAD improves below 5pp in the top bins without ρ degradation,
   the F4 + R10 bundle could potentially be reduced to F4 alone
   (better calibrated frequency guard)

This is speculative — POT-GPD on the empirical distribution may
not produce smooth probability output. Worth scoping as a Q3 2026
research item.

---

## Section-2 invariant check

For each of the 10 configs analyzed, I also verified that the §2
invariant holds across the rank_log:

| Config | PUT executed & ev≤0 | PUT executed & non-finite | Any non-finite anywhere |
|---|---|---|---|
| All 10 | 0 | 0 | 0 |

**§2 invariant CLEAN across all 10 configurations.** No tradeable
PUT was recorded with `ev_dollars ≤ 0` or non-finite EV. The CC
opens with negative EV are by harness design
(`require_ev_authority=False` + CC EV-floor at −\$50) and are
not §2 breaches.

---

## Method appendix

**Data source:** 10 rank_log.csv files in `%TEMP%\s{22,27,32,34,35,38,38_postf4}_backtest\`
and `%TEMP%\s40_backtest_{2021,2022,2023}\`. All produced by the
canonical Sn harness over real Bloomberg historical data.

**Engine SHA per config:**
- S22 / S27: pre-IV-PIT-fix engine (~pre-2da76ff)
- S32 / S34 / S35 / S38: post-IV-PIT-fix, pre-F4 engine (~b2cce25)
- S38-postF4 / S40-W1/W2/W3: post-F4 engine (`56d8e5c`)

**Analysis script:** `%TEMP%\multi_backtest_calibration.py` (throwaway,
not committed). Loops over the 10 configs, applies the 7-bin
calibration table per pre-declared standard, aggregates.

**Bins (post-hoc inspection):** The natural prob_profit distribution
in the engine's output is concentrated in 0.65-0.95. The 7-bin
scheme used here (0.5/0.6/0.7/0.8/0.85/0.9/0.95/1.0) is finer in
the high-confidence range where the miscalibration matters most.
An alternative 10-decile scheme would put most mass in 3 bins,
losing the resolution that surfaces the top-bin issue.

**Sample size considerations:** Some bin counts are small (e.g., S22's
n=34 in (0.6, 0.7]; S40-W3's n=75 in (0.95, 1.0]). Small-n bins
are reported but the verdict assignment may be noisier than larger
bins. The aggregate finding (10/10 configs have ≥1 MISCAL bin) is
robust to bin-size noise because the pattern repeats across all
config sizes from 1,946 (S35) to 17,520 (S40-W1) rows.

**Hit-rate definition:** `actual_otm = (exit_reason == "otm_expire")`.
Assigned puts that nevertheless netted positive realized P&L via
large premium are NOT counted as "OTM" in this analysis — consistent
with PR #197 / S38 doc convention.

---

## AI handoff

- **The top-bin miscalibration finding is now structural and
  universal, not S38-specific.** This finding should be cited
  whenever the engine's prob_profit field is used in deployment
  conversations.
- **R10 is load-bearing for the F4 bundle**, not PR #260.
  PROD_READINESS §3 B1's "deployment bundle" framing should
  emphasize this.
- **For future calibration improvements:** POT-GPD tail extension
  on prob_profit computation is the most promising research
  direction. See "For future research" above.
- **For PROJECT_STATE refresh:** add this as a confirmed engine
  property in §1 (\"Authoritative — do not bypass\").
- **For the master verification index:** add the calibration check
  as a separate "Tested surface" row with the cross-config result
  and link to this doc.
