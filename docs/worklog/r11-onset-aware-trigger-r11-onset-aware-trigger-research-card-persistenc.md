---
id: r11-onset-aware-trigger
title: R11 onset-aware trigger — research card (persistence vs VIX-level)
kind: research
status: done
terminal: ultracode
pr:
decisions: [D23]
date: 2026-06-01
headline: DONE 2026-06-29 — hypothesis REFUTED. Persistence-N (VIX>25 for N consecutive days) does NOT beat the level trigger: all whole-book diffs within noise (paired daily-return |t|≤0.5, both windows), and consecutive-N retains only 63-65% of the 2022 averted loss (never ≥80%) because 2022's elevation was choppy (run≥20 covers 24% of its >25 days) vs 2020 sustained (61%) — it fires backwards from the premise. KEEP R11 as-is. Evidence: docs/HEAVY_VERIFY_2026-06-29_R11_REFINEMENT.md. Only untested variant = "≥N of last M days" (non-consecutive), but whole-book is noise-dominated so unlikely material.
surface: []
---

## Goal

Improve R11 so it keeps its genuine protection (the 2022-style sustained
grind-down) without its measured downside (forgoing the 2020 V-recovery). This
card is the research follow-up flagged in D23's post-ship validation note; it is
NOT a committed change and touches no code yet.

## Motivation (from the R11 dollar-impact backtest)

`docs/verification_artifacts/r11_dollar_impact_2026-06-01/R11_DOLLAR_IMPACT_FINDINGS.md`
showed R11's current `vix_level > 25` trigger is regime-blind in a specific way:

- **2022 sustained bear (VIX 25-35 for months): R11 helps** — averts −$166-269k
  of CSP-leg loss, ~50% assignment. The grind-down has no quick recovery to forgo.
- **2020 sharp V-crash (VIX>35 briefly): R11 backfires** — the *level* trigger
  fires AFTER the spike, blocking trades that then ride the V-recovery (forwent
  +$26.5k W3). It cannot tell "spike that will recover" from "grind that won't."

I10 already showed real-time onset detection is hard (`rv_ratio` peaks at the
2020 *recovery*, not bear-onset). So the improvement must NOT require predicting
the regime — it must use only what's observable at entry.

## Hypothesis

A **persistence condition** distinguishes the two crisis shapes WITHOUT
forecasting: fire R11 only when `vix_level > 25` has held for **N consecutive
trading days** (candidate N: 5, 10, 20).

- 2022's months-long elevation clears any reasonable N → R11 still fires → keeps
  the averted loss.
- 2020's spike is brief and V-shaped → by the time N days of >25 accumulate, vol
  is already falling (recovery) → R11 stops firing on the recovery entries →
  drops the forgone gain.

This is a pure look-back on the VIX series (PIT-safe, no look-ahead), so it's
§2-clean and downgrade-only just like R11 today.

## Test plan (re-uses existing harness)

1. Extend `r11_dollar_impact_driver.py` with a third arm: `active_persist_N`
   (R11 gate AND "VIX>25 for ≥N prior trading days"). The driver already threads
   the daily VIX; add a rolling-count condition. No engine change for the study
   (replicated gate, as in the shipped driver).
2. Run W3 (2020-2024) + W4 (2021-2025) for N ∈ {5,10,20}; report Δ NAV / Sharpe
   and the per-regime blocked-set counterfactual vs both `suppressed` and the
   current `active`.
3. **Success criterion:** a persistence N that retains ≥~80% of the 2022 averted
   loss while cutting the 2020 forgone gain to near zero, and whose whole-book Δ
   is no worse than current `active` (ideally distinguishable-positive — which
   current `active` is not).
4. If a winning N exists, THEN (and only then) propose a D-entry to add the
   persistence condition to the live R11 — a §2-surface change requiring the
   usual second-read.

## Unresolved / open

- Whether ANY N beats the current level trigger on the whole book, or whether
  R11's net-neutral-to-the-book result is structural (the wheel recovery leg
  dominates regardless). The backtest's null whole-book result leaves this open.
- This is two overlapping windows; a real answer wants a walk-forward incl.
  2018-2022 and any future elevated-vol episode.
- Status: **DONE 2026-06-29 — hypothesis refuted.** (Was: proposed.)

## Result (2026-06-29)

Ran the test plan exactly (`r11_persist_driver.py`, arms suppressed/active/
persist{5,10,20}, W3 + W4, $1M/100t, full friction). **No persistence N clears
the bar; keep R11 as-is.**

- **Whole-book Δ is within noise for every arm** — paired daily-return |t| ≤ 0.5
  (vs ~2 for significance), both windows. persist10's nominal W4 win (+$70.8k vs
  active +$62.8k) is t = +0.14 vs active and *negative* in W3 (sign-flips). The
  card's success criterion (c) — a persistence N whose whole-book Δ is no worse
  than active, ideally distinguishable-positive — is **not met by any N**, and
  the "Unresolved" question above resolves to: **the null whole-book result is
  structural** (the wheel recovery leg dominates).
- **Criterion (a) fails in both windows:** consecutive-N retains only **63-65%**
  of the 2022 averted loss (persist5/10), or 0.7% (persist20) — never ≥80%.
- **Mechanism:** 2022's VIX>25 was *choppy* (run≥20 covers 24% of its 135
  elevated days) vs 2020 *sustained* (61% of 91). So "VIX>25 for N consecutive
  days" fires *backwards* from the premise (more in 2020, less in 2022).
- **Only untested variant:** a **non-consecutive** condition ("VIX>25 on ≥N of
  the last M days") would target choppy-2022 better — but since the whole book is
  noise-dominated, it is unlikely to be material. Not pursued.

Full write-up + artifacts: `docs/HEAVY_VERIFY_2026-06-29_R11_REFINEMENT.md`,
`docs/verification_artifacts/r11_refinement_2026-06-29/`. R11 unchanged (no §2
surface touched). Reconciles #442 (no static band carve-out either).
