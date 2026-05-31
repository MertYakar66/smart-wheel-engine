# Heavy-Verification Campaign — 2026-05-31

**Operator question:** *"Where can I trust this engine with real capital, and where
can't I?"* — tested adversarially, point-in-time-correctly, on real S&P-500 data,
on the engine **as it is today** (post-#294).

**Mandate:** observe-and-document, NOT fix. `engine/` was **never modified**
(`git diff origin/main -- engine/` is empty for the whole campaign). Any bug is a
written finding for separate triage. Every quantitative claim traces to a value
computed by the campaign's own code on real data; findings were stress-tested
against competing hypotheses before being published (one alarming "EV inversion"
was traced to sampling noise and withheld).

Branch `claude/heavy-verify-campaign-2026-05-31`. Drivers + raw output:
`docs/verification_artifacts/campaign_2026-05-31/`.

---

## TL;DR — the one-paragraph answer

> **Trust the engine as a calibrated *probability* estimator in the normal range and
> as a *defensive* return engine in down/sideways/high-rate markets — distrust its
> high-confidence probabilities, distrust its dollar-EV as a profitability forecast,
> and distrust its behavior at crisis onset.** Concretely: `prob_profit` is honest
> from ~0.6–0.85 (±5pp over 16,005 trades) but over-promises at the top (says 96%,
> delivers 70%), worst in crisis. Net of real spreads, T-bill carry, and dividends,
> the wheel **beats buy-and-hold by ~27pp in the 2022 bear and ~10pp in the 2020
> crash with a quarter of the drawdown**, roughly matches it in 2025, and **trails by
> 19–26pp in strong bulls** (capped upside + cash drag). The §2 firewall is intact —
> a negative-EV trade cannot be made tradeable. The two things a PM must know before
> wiring capital: (1) the portfolio risk caps are **off by default**, and (2) the
> engine is **procyclical at crash entry** — it recommended selling puts straight
> into the March-2020 bottom.

---

## Where you CAN trust it

| Area | Evidence | Investigation |
|---|---|---|
| **`prob_profit` in the 0.6–0.85 range** — calibrated within ±5pp across ~12,000 candidates; the bulk (.7,.8] bin, n=7,463, is +0.6pp | reliability table, Wilson CIs | I1 |
| **Calibration in calm/recent markets** — weighted-MAD ~3pp in 2023-2025; ~3pp in `bear` regime | per-year/regime stratification | I1 |
| **Downside protection / bear & crash alpha** — +26.8pp vs passive in 2022, +9.9pp in the 2020 window, ¼ the drawdown | frictioned multi-regime backtest w/ real Theta fills | I2 |
| **The §2 invariant** — no negative-EV candidate can become tradeable; reviewers are downgrade-only; dealer/regime multipliers are sign-preserving | 6 adversarial attacks, all held; lead-verified | I4 |
| **The negative-EV hard-block & downgrade-only review contract** — no rescue path (negative/inf/NaN → blocked even with a perfect chart) | dossier probes | I3-C, I4 |
| **EV semantics are sound** — `ev_dollars` correctly reflects probability-weighted tail risk (MU CC −$812 despite $1,334 premium) | re-verified on current engine | I5-A |
| **The ranking adds real selection value** — monthly top-10 by any engine signal beats random/all (+$166–206 vs −$26 mean; positive vs negative ROC); `prob_profit` is the best risk-adjusted selector | top-K selection vs random | I6-B |
| **The over-confidence is fixable** — a recalibration trained on 2020-2023 generalizes (out-of-sample ECE 3.17→1.29pp; >0.90 forecast 0.924→0.806 = realized) | out-of-sample recalibration | I6-C |

## Where you CANNOT trust it (ranked by materiality)

| # | Finding | Magnitude | Investigation |
|---|---|---|---|
| 1 | **Procyclical at crisis onset.** The empirical forward distribution lags the regime, so at the 2020 crash entry the engine flagged **89% of candidates positive-EV**; they realized **−$1,305/contract at 82% assignment**. It recommends selling puts into the buzzsaw, then self-corrects ~4 weeks later. | −$1,305/contract; 23% win | **I3-E** (+ cross-validates I1) |
| 2 | **Top-bin probability over-confidence.** `prob_profit` (.95,1] says 96.5%, reality is **69.5%** (−27pp, n=105, CI excludes forecast); (.9,.95] is −11pp. Worst in `crisis` (−26pp). | −11 to −27pp | **I1** |
| 3 | **Risk caps are dormant by default.** Sector 25% / single-name 10% / delta / Kelly all sit behind `require_ev_authority=False`; **174 of 73 monthly books breach them** (Healthcare to 47% NAV). No ranker path arms them. | ~half of months breach | **I3-A** |
| 4 | **Dollar-EV is not a profitability forecast.** `Spearman(ev_raw, realized_$) ≈ 0`; the mean is tail-dominated (bottom-1% trim flips it). Use `prob_profit`/ROC for ranking, not `ev_dollars` magnitude. | ρ=−0.002 ($), +0.16 (ROC) | **I1** |
| 5 | **Bull-market underperformance.** −19pp (2021), −26pp (2023-24) vs passive; ~half is cash drag, half is capped upside. | −19 to −26pp | **I2** |
| 6 | **Concentration / catastrophe tail.** A single high-notional assignment (NVR, BKNG, AZO) loses −$40k to −$83k/contract; the dollar-P&L mean is set by the bottom 1%. R10 would cap it but is dormant (see #3). | −$83k worst single | I1, I2 |
| 7 | **Earnings-gate look-ahead** (conservative direction — cannot inflate returns, but candidate counts aren't strictly PIT). | bounded | I3-D |

## The unifying mechanism

Findings #1 and #2 are the same root cause seen from two angles: **the empirical
forward distribution under-models the left tail at regime transitions.** It is built
from trailing returns, so at crisis onset it still "remembers" the prior calm/bull
regime — making the highest-confidence puts (priced off benign history) exactly the
ones reality surprises, and making the engine procyclical. POT-GPD tail machinery
exists (`engine/tail_risk.py`) but is **not wired into `prob_profit`**. The HMM
regime de-rate helps (0.31× in the 2020 crash) but does not fully offset it. This is
the highest-value fix target — flagged, not implemented (observe-only).

## How to allocate (if you do)

* **Use it for what it's good at:** defensive carry — bear/sideways/high-rate. Size
  as a complement to long beta, not a replacement (it gives up 19-26pp in bulls).
* **Arm `require_ev_authority=True`** in any live tracker, or enforce concentration
  caps outside the engine — the defaults do not.
* **Haircut high-confidence signals:** treat `prob_profit > 0.90` as ~0.80 and
  `> 0.95` as ~0.70; be especially wary when `hmm_regime == "crisis"` or
  `distribution_source == "empirical_overlapping"`.
* **Don't deploy at crisis onset** on the engine's say-so; it lags the regime ~4
  weeks. Don't read `ev_dollars` magnitude as a dollar forecast.
* **Credit the carry:** in a 4-5% rate world the T-bill yield on collateral is a
  first-order (+3 to +8pp) part of the return — make sure your accounting includes it.

## Investigations

| # | Title | Verdict | Doc |
|---|---|---|---|
| I1 | Calibration — is it telling the truth? | Calibrated in the middle; over-confident at the top (esp. crisis) | `HEAVY_VERIFY_2026-05-31_I1_CALIBRATION.md` |
| I2 | Net-of-reality P&L vs passive | Defensive winner (bear/crash), bull laggard; rf-carry is first-order | `HEAVY_VERIFY_2026-05-31_I2_NET_OF_REALITY_PNL.md` |
| I3 | Stress / discipline | Sound rules; caps dormant by default; procyclical at crash entry | `HEAVY_VERIFY_2026-05-31_I3_STRESS_DISCIPLINE.md` |
| I4 | §2 invariant adversarial probe | HELD across 6 attacks; firewall intact | `HEAVY_VERIFY_2026-05-31_I4_SECTION2_INVARIANT.md` |
| I5 | Re-verify prior claims | All hold; #294 did not touch the EV path | `HEAVY_VERIFY_2026-05-31_I5_PRIOR_CLAIMS.md` |
| I6 | Wave-2 deepening (regime overlay / selection value / fixability) | Ranking *works* (top-K beats random); regime overlay over-penalizes bear; over-confidence is recalibratable | `HEAVY_VERIFY_2026-05-31_I6_DEEPENING.md` |

## Method & honesty notes

* **PIT correctness** is the engine's own: PIT IV (`d26a8d6`), PIT OHLCV under the
  504-day survivorship gate, HMM on history up to `as_of`. Real Theta bid/ask used
  for fills where covered (≈55-70%), modeled fallback elsewhere (counted).
* **Data shape that bounds the work:** OHLCV 2018-01-02→2026-03-20 (the 504-day gate
  means early-2020 is rankable only on a reduced universe); Theta real spreads dense
  2018-2022, sparser 2023-2026; benchmark uses the current 503 members (survivorship
  biases it *up*, i.e. harder for the wheel to beat); Bloomberg IV has no skew (I5-C).
* **Adversarial discipline:** the alarming "EV-dollar inversion" at n=4,259 was traced
  to sampling noise via bootstrap CIs + tail decomposition and **withheld**; the
  published statement is the verified "≈zero rank-correlation, tail-dominated."
* **Reproduce everything:** `docs/verification_artifacts/campaign_2026-05-31/` — each
  `iN_*.py` driver prints to a matching `raw_output/iN_*_RAW.txt`. Start with
  `rank_snapshots.py` (the shared monthly ranked snapshots), then `i1`..`i5`.
