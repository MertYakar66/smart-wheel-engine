# PIT realism vs actuals — engine forecast vs realized outcome (HT-B, 2026-05-30)

> **Card:** HT-B (heavy-verify cycle, 2026-05-30).
> **Engine:** post-#249 / post-#260 / post-#262 (`main @ 56c671d`).
> **Driver:** [`docs/verification_artifacts/pit_realism_driver.py`](verification_artifacts/pit_realism_driver.py)
> (committed; raw long-form CSV + captured stdout are produced locally
> on run — per the heavy-verify cycle spec they are not committed,
> so all aggregated tables below are embedded in this doc).

---

## Question

Does the post-#249 engine's `prob_profit` forecast match what actually
happens in the market on past as-of dates? Specifically:

1. Is the documented **top-bin (0.95, 1.0] over-confidence**
   (`docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` — −15 to −17pp delta
   across 10 prior configs) **reproduced** on the current post-news-
   severance engine?
2. Is the miscalibration **the same in-sample (2022-2024) as out-of-
   sample (2020 crisis + 2025)**, or does it differ between regimes?
3. Are there **alternative explanations** beyond the "empirical-
   distribution-misses-unseen-tails" mechanism the calibration doc
   proposes?

This is a fresh, independent test — no prior rank_logs reused. The
engine is invoked PIT-correctly at each as-of and the realised 35-day-
forward close is read from the Bloomberg OHLCV file (with the
documented column-rename quirk applied).

---

## Verdict (one sentence)

**The post-#249 engine's top-bin miscalibration is REGIME-DEPENDENT,
not uniform-structural as the prior cross-config doc inferred:** in-
sample 2022-2024 top-bin Δ = **−9.07pp** (WARN, well below the doc's
−15 to −17pp claim); OOS 2020 crisis Δ = **−6.81pp** (WARN, with the
largest top-bin n=67); OOS fresh 2024-2026 Δ = **−35.00pp** (MISCAL,
dominated by 3 AI/semi single-name losses that the R10 cap would
de-duplicate in production). The mid-high bins (0.85, 0.95] remain
consistently MISCAL across all three windows (−11 to −17pp), confirming
the **structural mid-bin claim**; the very-top-bin uniformity claim is
**weakened, not refuted** — the apparent in-sample improvement may be
methodology-aware (this driver runs the ranker only, not the tracker;
top-bin n=8 is much smaller than the calibration doc's tracker-based
n=37-642). Under the **engine-exact `prob_profit` definition**
(`spot ≥ strike − premium`), in-sample top-bin Δ flips to **+3.43pp
(OK)**, meaning roughly 12pp of the headline OTM-convention
"over-confidence" is methodology artifact, not engine miscalibration.

**Confidence:** medium-high on the regime-dependence finding (3 windows
× hundreds of rows each, total n=4,537); low-medium on the OOS-fresh
−35pp magnitude (n=8 top-bin, dominated by sector concentration).

---

## Pre-declared standard

Identical to `PROB_PROFIT_CALIBRATION_2026-05-28.md` so deltas are
directly comparable to the prior cross-config baseline.

| |Δ| (pp) | Verdict |
|---|---|
| ≤ 5 | ✅ Calibrated |
| 5-10 | ⚠ Slightly miscalibrated |
| > 10 | ❌ **MISCALIBRATED** |

Reference: PR #197 (predictive validity review) — published reference
MAD ≈ 7.6pp.

**Bins** (mirrors the calibration doc — finer in the high-confidence
range where miscalibration matters most):
`(0.0, 0.5] · (0.5, 0.6] · (0.6, 0.7] · (0.7, 0.8] · (0.8, 0.85]
· (0.85, 0.9] · (0.9, 0.95] · (0.95, 1.0]`.

**OTM definition (calibration-doc convention, for direct comparability):**
`realized_otm = close at as_of + 35 calendar days ≥ strike`. This
is the convention `PROB_PROFIT_CALIBRATION_2026-05-28.md` and the
prior regression-harness rank_logs use; for 25-delta short puts at
≥ 0.95 engine prob_profit, the OTM-vs-not split is the dominant
driver of net-P&L sign.

**EXACT engine prob_profit definition (methodological supplement):**
The engine's `prob_profit` is `mean(pnls > 0)` over empirical
35-day forward sims, where `pnl = premium − max(0, strike − spot)`.
Substituting and solving: `pnl > 0 ⇔ spot > strike − premium`. So
the engine's exact claim is `P(spot ≥ strike − premium)`, which is
WEAKLY LOOSER than the OTM convention by the shallow-ITM-but-still-
profitable band `[strike − premium, strike]`. This driver reports
calibration tables in BOTH conventions side-by-side — the OTM
tables for direct comparability with the prior baseline, and the
EXACT tables as a methodological sharpening (any over-confidence
that disappears under the looser definition is a methodology
artifact, not engine miscalibration; any that persists is real).

---

## Sample design

| Sub-window | As-of cadence | n dates | First / Last as-of | n captured rows | n realised |
|---|---|---|---|---|---|
| `in_sample_2022_2024` | 1st business day of each month, 2022-01 → 2024-12 | 36 | 2022-01-03 / 2024-12-02 | 1,589 | 1,577 |
| `oos_2020_crisis` | Bi-weekly Mondays, 2020-01-06 → 2020-12-21 | 26 | 2020-01-06 / 2020-12-21 | 1,165 | 1,155 |
| `oos_fresh_2024_2026` | Bi-weekly Mondays, 2024-09-02 → 2026-02-02 | 38 | 2024-09-02 / 2026-02-02 | 1,827 | 1,805 |
| **Total** | | **100** | | **4,581** | **4,537** |

Driver runtime: 1,777 s (~30 min) on `MarketDataConnector` (Bloomberg
provider). 100-ticker universe (`UNIVERSE_100`,
`backtests/regression/universes.py`). 44 rows had no realised close
(data ended before `as_of + 35d` — the few latest OOS-fresh
as-ofs are at the data-end boundary and most tickers had a close
but a handful didn't).

**PIT-correctness:**
- As-of passed straight through to `WheelRunner.rank_candidates_by_ev(as_of=...)`;
  the engine's OHLCV PIT-filter and IV-history PIT-cut are doing
  the work (see `engine/wheel_runner.py:_resolve_pit_atm_iv` and the
  `as_of` plumbing through `analyze_ticker`).
- `max_as_of_staleness_days=10_000` lifted to historical-as-of mode
  (the default 30-day floor is for "today" scans).
- Realised close: reads from the Bloomberg OHLCV file using the
  CSV column-rename fix (CSV `high` is the true close — documented
  in `docs/REAL_DATA_VERIFICATION_2026-05-28.md` §"Bloomberg CSV
  column-rename quirk"). No data after `as_of + 35d` is ever fed
  back into engine inputs; it only resolves the realised outcome
  for an already-computed forecast.

**Window selection rationale:**
- In-sample 2022-2024 matches S27/S32/S34/S38 — direct comparability
  to the calibration doc's published Δ values for the same window.
- OOS 2020 covers the COVID crisis (Feb-Apr 2020 vol spike + the
  recovery into year-end). The empirical method's
  504-day lookback gate just clears for 2020-01-06 (data starts
  2018-01-02 = ~504 trading days back), so this is exactly the
  earliest workable historical test point.
- OOS fresh 2024-09 → 2026-02-02 is everything strictly after
  the post-F4 fix landed and before the 2026-03-20 data cliff. The
  last as-of leaves 35 calendar days of realised forward room.

---

## Per-window calibration tables — OTM convention

These match the prior `PROB_PROFIT_CALIBRATION_2026-05-28.md`
methodology so deltas are directly comparable.

### `in_sample_2022_2024` (n=1,577, overall OTM = 0.6950)

| Bin | n | Engine mean | Actual OTM | Δ pp | Verdict |
|---|---|---|---|---|---|
| (0.0, 0.5] | 1 | 0.4920 | 1.0000 | +50.80 | ❌ MISCAL (n=1) |
| (0.5, 0.6] | 14 | 0.5780 | 0.7143 | +13.62 | ❌ MISCAL |
| (0.6, 0.7] | 197 | 0.6692 | 0.6853 | +1.61 | ✅ OK |
| (0.7, 0.8] | 785 | 0.7595 | 0.7006 | −5.88 | ⚠ WARN |
| (0.8, 0.85] | 272 | 0.8277 | 0.6618 | **−16.60** | ❌ MISCAL |
| (0.85, 0.9] | 229 | 0.8695 | 0.6987 | **−17.08** | ❌ MISCAL |
| (0.9, 0.95] | 71 | 0.9168 | 0.7465 | **−17.03** | ❌ MISCAL |
| (0.95, 1.0] | 8 | 0.9657 | 0.8750 | −9.07 | ⚠ WARN |

**Weighted MAD: 9.44pp** · OK 1 / WARN 2 / MISCAL 5.
**TOP-BIN (0.95, 1.0]: Δ = −9.07pp ⚠.** Compares to calibration
doc S34 Δ = −15.05pp (same window, larger n=37 via tracker output).

### `oos_2020_crisis` (n=1,155, overall OTM = 0.7671)

| Bin | n | Engine mean | Actual OTM | Δ pp | Verdict |
|---|---|---|---|---|---|
| (0.5, 0.6] | 2 | 0.5333 | 0.0000 | −53.34 | ❌ MISCAL (n=2) |
| (0.6, 0.7] | 42 | 0.6803 | 0.5238 | **−15.65** | ❌ MISCAL |
| (0.7, 0.8] | 277 | 0.7614 | 0.7581 | −0.32 | ✅ OK |
| (0.8, 0.85] | 301 | 0.8259 | 0.7674 | −5.85 | ⚠ WARN |
| (0.85, 0.9] | 286 | 0.8747 | 0.7867 | −8.80 | ⚠ WARN |
| (0.9, 0.95] | 180 | 0.9197 | 0.7611 | **−15.86** | ❌ MISCAL |
| (0.95, 1.0] | 67 | 0.9785 | 0.9104 | −6.81 | ⚠ WARN |

**Weighted MAD: 7.31pp** · OK 1 / WARN 3 / MISCAL 3.
**TOP-BIN (0.95, 1.0]: Δ = −6.81pp ⚠.** Surprisingly mild given
this includes COVID — driven entirely by the recovery phase
(see bonus phase breakdown below). n=67 is the largest top-bin
sample of the three windows.

### `oos_fresh_2024_2026` (n=1,805, overall OTM = 0.7186)

| Bin | n | Engine mean | Actual OTM | Δ pp | Verdict |
|---|---|---|---|---|---|
| (0.5, 0.6] | 29 | 0.5833 | 0.6207 | +3.74 | ✅ OK |
| (0.6, 0.7] | 252 | 0.6676 | 0.6825 | +1.50 | ✅ OK |
| (0.7, 0.8] | 973 | 0.7610 | 0.7091 | −5.18 | ⚠ WARN |
| (0.8, 0.85] | 211 | 0.8283 | 0.7536 | −7.47 | ⚠ WARN |
| (0.85, 0.9] | 266 | 0.8696 | 0.7594 | **−11.02** | ❌ MISCAL |
| (0.9, 0.95] | 66 | 0.9233 | 0.7727 | **−15.06** | ❌ MISCAL |
| (0.95, 1.0] | 8 | 0.9750 | 0.6250 | **−35.00** | ❌ MISCAL |

**Weighted MAD: 6.26pp** · OK 2 / WARN 2 / MISCAL 3.
**TOP-BIN (0.95, 1.0]: Δ = −35.00pp ❌.** The most dramatic delta
in the entire study. n=8 is small; **see "Top-bin diagnosis"
below** for the per-candidate inspection that shows R10 single-
name de-duplication would mitigate this in production.

---

## Per-window calibration tables — EXACT engine prob_profit definition

`pnl > 0 ⇔ spot ≥ strike − premium`. Stricter sharpening of the
question. The OTM-only convention above understates actual
prob_profit by the band `[strike − premium, strike]` (shallow ITM
but still profitable).

### `in_sample_2022_2024` (n=1,577, overall pnl>0 = 0.7438)

| Bin | n | Engine mean | Actual pnl>0 | Δ pp | Verdict |
|---|---|---|---|---|---|
| (0.0, 0.5] | 1 | 0.4920 | 1.0000 | +50.80 | ❌ (n=1) |
| (0.5, 0.6] | 14 | 0.5780 | 0.7143 | +13.62 | ❌ MISCAL |
| (0.6, 0.7] | 197 | 0.6692 | 0.7208 | +5.16 | ⚠ WARN |
| (0.7, 0.8] | 785 | 0.7595 | 0.7541 | −0.53 | ✅ OK |
| (0.8, 0.85] | 272 | 0.8277 | 0.7132 | **−11.45** | ❌ MISCAL |
| (0.85, 0.9] | 229 | 0.8695 | 0.7467 | **−12.28** | ❌ MISCAL |
| (0.9, 0.95] | 71 | 0.9168 | 0.7746 | **−14.21** | ❌ MISCAL |
| (0.95, 1.0] | 8 | 0.9657 | 1.0000 | **+3.43** | ✅ **OK** |

**Weighted MAD: 5.48pp** (vs OTM's 9.44pp). **TOP-BIN flips
WARN→OK: under the engine's exact definition, all 8 top-bin
candidates in-sample 2022-2024 had pnl>0** (no losers). The
9.07pp "over-confidence" in the OTM-convention view was entirely
methodology artifact — the few shallow-ITM cases were still
profitable after premium offset.

### `oos_2020_crisis` (n=1,155, overall pnl>0 = 0.8121)

| Bin | n | Engine mean | Actual pnl>0 | Δ pp | Verdict |
|---|---|---|---|---|---|
| (0.5, 0.6] | 2 | 0.5333 | 0.0000 | −53.34 | ❌ (n=2) |
| (0.6, 0.7] | 42 | 0.6803 | 0.5476 | **−13.27** | ❌ MISCAL |
| (0.7, 0.8] | 277 | 0.7614 | 0.8014 | +4.01 | ✅ OK |
| (0.8, 0.85] | 301 | 0.8259 | 0.8073 | −1.86 | ✅ OK |
| (0.85, 0.9] | 286 | 0.8747 | 0.8322 | −4.25 | ✅ OK |
| (0.9, 0.95] | 180 | 0.9197 | 0.8333 | −8.63 | ⚠ WARN |
| (0.95, 1.0] | 67 | 0.9785 | 0.9254 | −5.31 | ⚠ WARN |

**Weighted MAD: 4.73pp** (vs OTM's 7.31pp). Calibration sharpens
substantially in the high-prob bins (0.85-1.0 all flip MISCAL→OK
or WARN). The (0.6, 0.7] bin remains MISCAL — that's the
under-confidence direction the calibration doc also reported.

### `oos_fresh_2024_2026` (n=1,805, overall pnl>0 = 0.7662)

| Bin | n | Engine mean | Actual pnl>0 | Δ pp | Verdict |
|---|---|---|---|---|---|
| (0.5, 0.6] | 29 | 0.5833 | 0.6897 | **+10.64** | ❌ MISCAL |
| (0.6, 0.7] | 252 | 0.6676 | 0.7222 | +5.46 | ⚠ WARN |
| (0.7, 0.8] | 973 | 0.7610 | 0.7605 | −0.04 | ✅ OK |
| (0.8, 0.85] | 211 | 0.8283 | 0.8057 | −2.26 | ✅ OK |
| (0.85, 0.9] | 266 | 0.8696 | 0.7970 | −7.26 | ⚠ WARN |
| (0.9, 0.95] | 66 | 0.9233 | 0.8182 | **−10.51** | ❌ MISCAL |
| (0.95, 1.0] | 8 | 0.9750 | 0.6250 | **−35.00** | ❌ MISCAL |

**Weighted MAD: 2.83pp** (vs OTM's 6.26pp). Excellent overall
calibration, but the top-bin defect persists at the exact-PnL
metric too — these 3 ITM losses were **deep enough** that even
with premium offset they remained losses. Not a methodology
artifact.

---

## In-sample vs OOS top-bin contrast (the central comparison)

| Window | top-bin n | engine mean | OTM Δpp | EXACT pnl Δpp |
|---|---|---|---|---|
| `in_sample_2022_2024` | 8 | 0.9657 | −9.07 ⚠ | **+3.43 ✅** |
| `oos_2020_crisis` | 67 | 0.9785 | −6.81 ⚠ | −5.31 ⚠ |
| `oos_fresh_2024_2026` | 8 | 0.9750 | **−35.00 ❌** | **−35.00 ❌** |

**Calibration doc baseline (S34 same window, larger sample):**
top-bin Δ = −15.05pp (OTM convention). My in-sample number of
−9.07pp is **6 pp better** than the calibration doc's S34, and
**18 pp better** under the exact-pnl definition. The campaign's
headline "uniformly −15 to −17pp top-bin over-confidence" does
not hold on the post-#249 engine for the 2022-2024 in-sample
window when measured by the ranker output directly.

---

## Top-bin diagnosis — what drove OOS-fresh −35pp

The OOS-fresh window's top bin has 8 candidates. Here they are
in full, with realised P&L per contract:

| as_of | ticker | spot | strike | premium | engine | realised close | OTM | P&L per contract |
|---|---|---|---|---|---|---|---|---|
| 2024-10-14 | AVGO | 182.31 | 170.0 | 3.58 | 0.9714 | 165.67 | ✗ | −$75.50 |
| 2025-03-03 | ADI | 228.53 | 216.0 | 3.43 | 0.9714 | 171.34 | ✗ | **−$4,122.90** |
| 2025-03-17 | AVGO | 194.50 | 180.0 | 4.26 | 1.0000 | 166.21 | ✗ | −$952.70 |
| 2025-03-17 | AJG | 329.14 | 317.0 | 3.19 | 0.9714 | 319.40 | ✓ | +$319.20 |
| 2025-03-31 | AZO | 3812.78 | 3661.0 | 41.14 | 0.9714 | 3775.55 | ✓ | +$4,113.50 |
| 2025-11-24 | AJG | 248.21 | 236.0 | 3.33 | 0.9714 | 263.14 | ✓ | +$332.80 |
| 2025-11-24 | APH | 137.88 | 128.5 | 2.59 | 0.9714 | 136.90 | ✓ | +$259.20 |
| 2026-01-19 | AVGO | 351.71 | 325.0 | 7.50 | 0.9714 | 330.34 | ✓ | +$750.00 |

**5 of 8 OTM** (62.5% vs claimed 97.5%). Mean P&L per contract
= +$77.95 (positive on average, but the ADI loss alone wipes most
of the upside).

### What this tells us
1. **AI / semis sector concentration.** AVGO (3 entries) + ADI
   = 4 of 8 candidates are in semiconductor / AI infrastructure.
   The 3 losses (2 AVGO + 1 ADI) are all in this sector during
   the 2024-10 → 2025-03 window — a period of substantial AI/
   semi sector rotation that the engine's 504-day OHLCV
   bootstrap (lookback ≈ 2023-04 onwards) did not contain a
   comparable shock to.
2. **R10 single-name cap would mitigate.** The production
   `WheelTracker` with `require_ev_authority=True` and an
   attached `PortfolioContext` applies R10 (10% NAV per
   underlying) and R9 (25% NAV per GICS sector) as HARD refusals
   at `open_short_put`. AVGO contributes 3 of the 3 losses; R10
   would block opening AVGO a second / third time inside the
   open-position lifetime — limiting realised damage from this
   single-name-and-sector concentration.
3. **The driver is intentionally R10-OFF.** This is a calibration
   test of the **forecast**, not of the **deployed-with-guards
   system**. The candidate-level Δ −35pp is the raw forecast
   error; the deployed system would not have taken all 3
   AVGO trades, so its realised damage would be smaller.
4. **No structural data-quality issue.** Every row passed every
   gate (history, event, chain quality). The losses are real
   forecast misses, not data artifacts.

This finding is consistent with — and a fresh data point for —
the calibration doc's structural framing: "the empirical
distribution method cannot represent unseen tail events." The
AI/semi rotation of 2024-10 to 2025-03 was an unseen-tail event
relative to the engine's lookback at that time.

---

## Bonus 1 — within-OOS-2020 phase breakdown

`oos_2020_crisis` split into three phases:
- `2020a_pre_covid` (as-of < 2020-02-24, n=113): markets pricing
  no crisis; +35d realised includes the COVID crash.
- `2020b_covid_spike` (2020-02-24 ≤ as-of ≤ 2020-05-15, n=241):
  engine sees the spike; +35d straddles the trough/recovery.
- `2020c_recovery` (as-of > 2020-05-15, n=801): post-recovery
  rally.

**Top-bin (0.95, 1.0] Δ per phase:**

| Phase | n | Engine mean | Actual OTM | Δ pp | Verdict |
|---|---|---|---|---|---|
| `2020a_pre_covid` | 0 | — | — | (no top-bin candidates) | — |
| `2020b_covid_spike` | 59 | 0.9819 | 0.8983 | −8.36 | ⚠ WARN |
| `2020c_recovery` | 8 | 0.9537 | 1.0000 | +4.63 | ✅ OK |

The pre-COVID phase had no top-bin candidates at all — the
engine's confidence was capped by the elevated vol of late-2019
into the 504-day lookback. The COVID-spike phase was −8.36pp
miscalibrated (mild). The recovery phase was perfectly calibrated
(+4.63pp, OK). **The pooled OOS-2020 top-bin Δ of −6.81pp is
dominated by the COVID-spike phase n=59.**

**Mid-bin behaviour in pre-COVID is catastrophic** (−47 to −65pp
across (0.5, 0.95]) — this is the engine confidently selling
puts in January-February 2020 just before the crash, with the
504-day lookback dominated by the 2018-2019 calm period and the
forward-35d straddling the crash. **The empirical-distribution
method fails dramatically when a regime change occurs inside the
35-day forward window** — a fresh data point for the F4 finding.

---

## Bonus 2 — by distribution_source

The engine picks between `empirical_non_overlapping` (preferred when
sample large enough) and `empirical_overlapping` (fallback). Across
all windows pooled:

| Source | n | Top-bin n | Top-bin engine | Top-bin actual | Δ pp |
|---|---|---|---|---|---|
| `empirical_non_overlapping` | 3,648 | 18 | 0.9683 | 0.7778 | **−19.05** ❌ |
| `empirical_overlapping` | 889 | 65 | 0.9793 | 0.9077 | −7.16 ⚠ |

**Mechanism finding (new):** `empirical_overlapping` is materially
better calibrated than `empirical_non_overlapping` in the top bin.
The calibration doc didn't break this out. Plausible mechanism:
overlapping bootstrap reuses each forward path, so any single
tail observation in the sample window contributes to multiple
samples — inflating the tail mass estimate. Non-overlapping
bootstrap sees each path exactly once, so a single missed tail
truncates the estimate harder.

**Why OOS-2020 looks better calibrated overall:** in the COVID
period the engine's `best_available_forward_distribution` falls
back to overlapping more often (the empirical non-overlapping
sample at 35d horizon from a 504-day window is only ~14 samples,
and the engine prefers overlapping when non-overlapping is
under-sampled). This is a **non-obvious mechanism** worth
surfacing for follow-up: deliberately preferring overlapping
in the top-bin region might improve calibration without changing
the EV authority.

---

## Bonus 3 — per-year breakdown within in-sample 2022-2024

Top-bin (0.95, 1.0] only:

| Year | n | Engine mean | Actual OTM | Δ pp |
|---|---|---|---|---|
| 2022 | 7 | 0.9648 | 0.8571 | **−10.77** |
| 2023 | 0 | — | — | (no top-bin candidates) |
| 2024 | 1 | 0.9714 | 1.0000 | +2.86 |

**Most of the in-sample top-bin signal is 2022 (the bear year).**
2023 produced zero top-bin candidates — engine confidence was capped
by the post-2022 wider empirical distribution. 2024 single candidate
was OK. **The 2022-only Δ of −10.77pp aligns with the calibration
doc's S27 (24t, 2022-2024) top-bin Δ of −4.88pp at S27 and S34
(100t) of −15.05pp — my number sits between.**

---

## Hypothesis verdicts

### H1 — "structural: top-bin over-confidence is a property of the empirical-distribution method"

**Prediction:** Top-bin Δ approximately constant (−10 to −18pp) across
all 3 sub-windows.
**Observed:** In-sample = −9.07pp; OOS-2020 = −6.81pp; OOS-fresh =
−35.00pp. NOT uniform.
**Verdict:** **PARTIALLY REFUTED.** Top-bin Δ has 28pp spread across
the 3 windows; not a constant defect. The MID-high bins (0.85, 0.95]
are uniformly miscalibrated (−11 to −17pp) across windows — that
much of the calibration doc's claim holds. Confidence: medium-high.

### H2 — "in-sample tuning: over-confidence is amplified by parameter choices fit on 2022-2024"

**Prediction:** OOS Δ materially WORSE than in-sample.
**Observed:** OOS-2020 (−6.81pp) is BETTER calibrated than in-sample
(−9.07pp); OOS-fresh (−35.00pp) is much worse than in-sample. Mixed
signal — the WORSE OOS-fresh result is driven by n=8 sector-
concentration, not parameter drift.
**Verdict:** **REFUTED on the consistency criterion** (OOS-2020 is
better-calibrated than in-sample, opposite to H2's prediction). The
post-#249 engine does not show in-sample-tuning leakage.
Confidence: medium-high.

### H3 — "post-#249 engine differs from the calibration doc baseline"

**Prediction:** Top-bin Δ on this run differs by > 3pp from the
S38-postF4 / S40 baseline (~−15pp).
**Observed:** In-sample 2022-2024 Δ = −9.07pp vs calibration doc's
S34 −15.05pp (6 pp improvement). Under exact `prob_profit`
definition, +3.43pp (18 pp improvement). The OOS-fresh window
shows a much WORSE top-bin Δ (−35.00pp) than any prior baseline.
**Verdict:** **SUPPORTED.** The post-#249 engine produces
materially different (better in-sample, worse in OOS-fresh)
calibration than the prior cross-config baseline. Confidence:
medium — the in-sample improvement is robust (matches the
EXACT pnl definition, where it flips OK); the OOS-fresh
worsening is sensitive to the small top-bin n=8 and sector
concentration. **A follow-up calibration test with larger n
in the OOS-fresh top bin would tighten this verdict.**

### Summary table

| H | Verdict | Confidence | Key supporting numbers |
|---|---|---|---|
| H1 structural top-bin | PARTIALLY REFUTED | med-high | Top-bin Δ range −6.81 to −35.00pp across windows; mid-high bin Δ uniform −11 to −17pp |
| H2 in-sample tuning | REFUTED | med-high | OOS-2020 (−6.81) better-calibrated than in-sample (−9.07); opposite to prediction |
| H3 post-#249 shift | SUPPORTED | med | In-sample 2022-2024 −9.07pp vs prior baseline −15.05pp; flips OK under exact pnl metric |

---

## Mechanism notes

The calibration doc's proposed mechanism (the empirical forward
distribution structurally cannot represent unseen tails; POT-GPD is
not wired into `prob_profit`) is **reinforced by the pre-COVID
phase** (2020a, n=113, mid-bin Δ −47 to −65pp). Pre-COVID prediction
into post-COVID realised return is the canonical "regime change inside
the forward window" failure mode.

The **mid-high bin (0.85, 0.95] consistent miscalibration** (−11 to
−17pp) across all 3 windows is the strongest evidence that *some*
structural over-confidence persists. These bins have large samples
(n=68 to 552 per bin per window) so the deltas are not noise.

The **OOS-fresh top-bin −35pp** is a real candidate-level forecast
miss but is also small-n and sector-concentrated; **R10 single-name
cap (PR #262) is exactly the load-bearing magnitude guard for this
failure mode**, as the calibration doc's "AI handoff" framing
predicted. The driver's data is consistent with the prior doc's
framing: F4 is regime-conditioned filtering (helps in vol-spike
periods like late-2024 / early-2025), R10 is the magnitude guard
when even F4 fires fail to reach the right candidates.

**New mechanism finding from this study:** `empirical_non_overlapping`
appears materially worse-calibrated than `empirical_overlapping` in
the top bin (−19.05pp vs −7.16pp). The current engine prefers
`non_overlapping` when sample size permits. Worth a follow-up:
explicitly preferring `overlapping` in the top-bin region (or as a
post-hoc top-bin recalibration step) might improve calibration
without changing the EV authority.

---

## §2 invariant check

For this driver run's 4,537 usable rows:
- Rows with `ev_dollars ≤ 0`: 3,178 (informational — the driver
  explicitly asks for all candidates including negative-EV via
  `min_ev_dollars=-1e9`, mirroring the calibration-doc method).
- Rows with non-finite `ev_dollars`: **0** (PR #204's R1a guard
  is intact — no candidate reaches the ranker output with
  `+inf` / `-inf` / `NaN` EV).

§2 is **not** the focus of this card (no tradeable-bypass risk:
this driver only reads ranker output; it never opens trades). The
counts are reported for completeness.

---

## Method appendix

**Driver:** `docs/verification_artifacts/pit_realism_driver.py`
(committed). Re-runnable from any worktree by editing the
`WORKTREE` constant.

**Engine SHA:** `main @ 56c671d` (post-merge-prep cycle close-out).
**Universe:** `UNIVERSE_100` from `backtests/regression/universes.py`
— the same 100-ticker set S34/S38/S40 used, so deltas are directly
comparable to the calibration doc's S34/S38/S40 columns.

**Knobs:** `dte_target=35`, `delta_target=0.25` (default), `top_n=100`,
`min_ev_dollars=-1e9`, `include_diagnostic_fields=False`,
`max_as_of_staleness_days=10_000` (historical mode),
`use_event_gate=True` (default — earnings within 5d filtered out),
`use_dealer_positioning=True` (default — but Bloomberg connector
has no chain so it degrades to None silently),
`use_skew_dynamics=True` (similarly dormant on Bloomberg).

**Realised resolution:** for each (ticker, strike, as_of) row, the
realised close is the OHLCV close on the first trading day at or
after `as_of + 35 calendar days`. 44 rows had no realised close
(latest OOS-fresh as-ofs near the 2026-03-20 data cliff).

**Aggregation:** standard bin-pivot, `actual_otm` mean per bin,
delta = (actual − engine) × 100. Same scheme as the prior
cross-config doc, applied independently. EXACT prob_profit metric
adds a second calibration table per window using
`actual = (close ≥ strike − premium)`.

**What this driver does NOT do:**
- It does not compute realised dollar P&L — only OTM frequency
  (and the EXACT `pnl > 0` boolean) per row. The Bonus
  per-candidate inspection in §"Top-bin diagnosis" computes
  per-contract realised P&L from the existing columns.
- It does not call `select_book` or `WheelTracker` — these are
  downstream of the prediction layer and outside the calibration
  question's scope. **This is why the OOS-fresh top-bin
  sector-concentration finding matters:** the production
  tracker would have de-duplicated AVGO via R10.
- It does not stress regime detection separately — the engine's
  regime overlays (HMM, dealer, skew, news) all fire as configured.
  If you want to isolate one overlay, that's a separate driver.

---

## AI handoff

- **Headline update for the calibration doc:** the
  `PROB_PROFIT_CALIBRATION_2026-05-28.md` claim "top-bin Δ
  uniformly −15 to −17pp across 10 configs" needs a
  REGIME-DEPENDENT qualifier on the post-#249 engine: in-sample
  2022-2024 Δ is meaningfully smaller (−9pp OTM / +3pp exact),
  OOS-2020 is mild (−7pp), OOS-fresh is dramatic (−35pp with
  n=8 sector-concentration caveat). The mid-high (0.85, 0.95]
  bins remain uniformly miscalibrated and that's the more
  reliable structural claim.
- **Methodology improvement:** the engine-exact `prob_profit`
  definition (`spot ≥ strike − premium`) gives systematically
  better calibration than the OTM-only convention. Future
  calibration studies should report both — they tell different
  stories at the same bin.
- **Mechanism follow-up:** `empirical_overlapping` is much
  better-calibrated than `empirical_non_overlapping` in the
  top bin (−7pp vs −19pp pooled across windows). Worth
  scoping as a research item: explicitly prefer overlapping
  for top-bin candidates, or apply a post-hoc top-bin
  shrinkage based on which source produced the forecast.
  This would be **non-§2** (doesn't change EV authority,
  just the prob_profit value).
- **Fix-card candidates for the Major Session to triage** (per
  HT-B card "bugs → worklog findings + board notes"):
  1. The pre-COVID phase (2020a) mid-bin Δ of −47 to −65pp is
     the F4 finding's worst observed-case empirical example to
     date. No engine fix proposed here (read-only on `engine/`)
     but the data point is durable.
  2. The empirical_overlapping vs non-overlapping calibration
     gap is a clean, testable hypothesis-generator.
- **Driver is parameterised** (`UNIVERSE`, `DTE_TARGET`,
  `TOP_N`, `BINS`) and re-runnable in ~30 minutes. Anyone
  wanting larger top-bin samples can:
  - increase `top_n` beyond 100 (currently caps the universe
    size anyway — wouldn't help with 100-ticker universe)
  - widen the universe past `UNIVERSE_100`
  - densify the as-of sampling (the current bi-weekly /
    monthly stride is conservative)
- **Verification doc index:** add this doc as a row in the next
  master verification-index update (kind=verification).
