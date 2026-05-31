# Heavy-Verify Campaign 2026-05-31 — I11: R11 risk-budget parameter study (RESULTS)

**Investigation:** pick the parameters of the candidate **R11 risk-budget reviewer**
(a §2-clean, downgrade-only "cap top-bin `prob_profit>0.90` size when VIX level is
elevated" rule) by measuring its leave-one-crisis-out cost/benefit on real data.
**Spec:** `HEAVY_VERIFY_2026-05-31_I11_RISK_BUDGET_STUDY_SPEC.md`.
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i11_vix_budget_study.py`.
**Raw:** `raw_output/i11_vix_budget_study_RAW.txt`. **Status:** observe-only; `engine/` not modified.

---

## VERDICT

> **NOT a null result — the conditional rule is supported. A VIX-*level* > 25 size-down
> of the top bin is favorably asymmetric in EVERY well-powered crisis fold** (2020 crash:
> +$86,366 net, averting the −$1,305/contract onset tail; 2022 bear: +$3,550 net), so
> the **conditional R11 form** (downgrade `prob_profit>0.90` when VIX `close > 25`) is the
> recommendation — not the null-result unconditional haircut. **VIX>25 is the
> robust-not-optimal pick**: {20, 22.5, 25} all survive every fold; **θ≥27.5 FAILS the
> 2022 fold**. Two honest caveats below: (1) the top-bin 2022 cost is *smaller* than the
> spec assumed (a spec-correction), (2) VIX-level is a *market-wide-vol* budget — it
> MISSES idiosyncratic single-name tails in calm markets (§6 residual gap), which the
> now-armed R10 single-name cap (#303) partly covers. R11 and R10 are complementary.**

Confidence: **high** on the 2020/2022 cells (n=358 / 270 top-bin trades, well-powered);
**medium** on the threshold's exact value (coarse grid by design — robust-not-optimal).
R11 itself remains a **separate gated card** (decision-layer lane-claim + §2 second-read).

---

## §4/§5 — LOCO cost/benefit, per crisis × coarse θ

Top-bin (`prob_profit>0.90`, n=1,122). A trade is *downgraded* if VIX(`as_of`) > θ;
`net_book = −Σ(realized over downgraded)` (>0 ⇒ sizing them out helped on net).

| θ (VIX>) | 2020 net | 2022 net | survives every fold? | agg averted:forgone |
|---|---|---|---|---|
| 20 | +$81,758 | +$6,252 | ✅ | 1.74 |
| 22.5 | +$83,833 | +$3,328 | ✅ | 1.76 |
| **25** | **+$86,366** | **+$3,550** | **✅** | **1.81** |
| 27.5 | +$103,239 | **−$33,400** | ❌ (2022) | 1.90 |
| 30 | +$116,741 | **−$33,400** | ❌ (2022) | 2.32 |

The crisis-aggregate ratio rises with θ (because 2020's averted tail grows), but that's
misleading: the **per-fold** view shows θ≥27.5 turns the 2022 fold negative. Robustness =
survives *every* fold (§5) → **{20, 22.5, 25}**; the cleanest round-number "elevated-vol"
line is **VIX > 25**.

## §4.4 — the 2022 cell (a correction to the spec's framing)

The spec anticipated 2022 as a pure *false-positive cost* — "the rule sizes down a year
that netted +$5k." That +$5k was the **full book**; the **top bin** (what R11 acts on)
**netted −$14,445 in 2022**. So at θ≤25 the size-down *helps* 2022 too (averts $56,900 >
forgoes $53,350 → net +$3,550). The rule is therefore **more favorable than the spec
assumed** — both the 2020 crash *and* the 2022 bear top-bins benefit. (Only at θ≥27.5
does 2022 flip negative, because that selectively downgrades the high-VIX 2022 months,
which happened to be the profitable ones — an argument *against* a high threshold.)

VIX context: top-bin VIX-on-`as_of` median by year — 2020 **29**, 2021 23, 2022 **27**,
2023 18, 2024 16, 2025 18. 2022 is elevated enough that VIX>25 fires through most of it
(no detection miss — consistent with I10 P3).

## Why VIX-level succeeds where I10's signals failed

I10 killed `rv_ratio` (peaks at the 2020 *recovery*, 2.69 > onset 1.69) and the other
*ratio/drawdown/accel* features. VIX *level* orders onset vs recovery correctly (2020
median 29, far above the calm-year ~16-18) and, crucially, the rule doesn't *need* to
separate onset from recovery: as a **risk budget** it sizes down whenever vol is elevated,
and the 2020 averted-tail so dominates that the rule is net-favorable even though it also
forgoes the 2020 recovery premium. That is the I10→I11 thesis: *you can't detect the
transition, so manage exposure regardless* — and the measured asymmetry confirms it.

## §6 — honesty scan: the residual gap (low VIX + fat tail)

Scanning all top-bin trades (window 2020-02 → 2026-02) for **VIX ≤ θ AND bottom-decile
realized** (≤ −$625):

| θ | low-VIX + fat-tail misses | mean realized | backwardation flags | by year |
|---|---|---|---|---|
| 25 | 30 | −$1,611 | 27% | 2022:8, 2025:11, 2024:5, 2021:3, 2020:2, 2023:1 |
| 30 | 47 | −$2,060 | 17% | 2022:25, 2025:11, 2024:5, … |

**Named, not papered over:** VIX-level is a **market-wide-vol** budget. It misses
**idiosyncratic single-name tails in calm markets** (most in 2024-2025) — top-bin trades
that bled with VIX ≤ 25. Term-structure backwardation (the §2 secondary) flags only
~17-27% of them, so it is **not** a sufficient catch either. This is the genuine residual
gap. The *worst* of these (high-notional NVR/AZO/BKNG-style single-contract catastrophes)
are bounded by the **now-armed R10 single-name cap (PR #303)** — so **R10 (idiosyncratic
size) and R11 (market-wide vol) are complementary tail defenses**. Smaller calm-market
idiosyncratic bleeds remain uncovered by either; no signal in this study catches them.

## Recommendation → R11 form (for the separate gated card)

- **Conditional R11 (supported):** downgrade top-bin (`prob_profit>0.90`) `proceed→review`
  when VIX `close > 25`. §2-clean (downgrade-only; never touches `ev_raw`/`ev_dollars`).
- **Payload (computed, not hardcoded — spec §8):** carry the matched-cell realized loss
  (here the crisis top-bin: ~−$1,305/contract at ~0.57 realized win, recomputed from the
  rows at build time) so review is anchored to the loss number, not vibes.
- **Pair with R10** (already merged, #303) for idiosyncratic single-name tails; R11 alone
  does not cover calm-market single-name bleeds (§6).
- R11 build is a decision-layer change → lane-claim + §2 second-read; gated on this result.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i1_calibration.py
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i11_vix_budget_study.py
```
All numbers in `raw_output/i11_vix_budget_study_RAW.txt`.
