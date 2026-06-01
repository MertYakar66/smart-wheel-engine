---
id: r11-dollar-impact
title: R11 dollar-impact backtest — post-ship validation (#306/#307)
kind: backtest
status: complete
terminal: ultracode
pr:
decisions: []
date: 2026-06-01
headline: R11 is targeted insurance for the 2022-style sustained grind-down (reliably averts ~$165-269k of CSP-leg loss, ~50% assignment) — but it lowers Sharpe in BOTH windows, net-COSTS the book in a window with a sharp V-recovery crash (W3 2020-2024 −$37.6k/−3.76pp), HELPS where the 2022 bear dominates (W4 2021-2025 +$21.7k/+2.17pp), and its per-contract "averted loss" over-states full-wheel value because blocking entry forecloses the wheel's recovery leg.
surface:
  - docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_dollar_impact_driver.py
  - docs/verification_artifacts/r11_dollar_impact_2026-06-01/R11_DOLLAR_IMPACT_FINDINGS.md
---

## Goal

Measure R11 (elevated-vol top-bin size-down, shipped #306/#307 on the I11
leave-one-crisis-out evidence) in a full-scale dollar backtest — never done
before. Run the engine TWO ways over the same windows: `suppressed` (pre-R11
baseline) vs `active` (R11 on), and report the DIFFERENCE per-window /
per-regime in dollars + Sharpe, with the averted-vs-forgone framing. $1M /
100t / full friction, S38/S43/S44 setup family. PIT-correct.

## What we tried

- **Discovered R11 is not in the backtest's ranker route.** R11 lives in the
  `EnginePhaseReviewer` dossier chain; the canonical
  `_common.run_backtest_multi_friction` routes through `rank_candidates_by_ev`
  and opens on EV>0 — never invoking the reviewer. So the canonical harness
  could not be used verbatim; R11 is dormant in it.
- Built a new driver (modelled on `r10_strict_driver.py`) running two arms over
  one shared daily rank. The active arm applies R11's EXACT gate
  (`pp>0.90 ∧ vix(as_of)>25 ∧ ev>$10`, constants imported from
  `engine.candidate_dossier`, VIX from `get_vix_regime`) to the put-open
  decision and backfills the quota; the suppressed arm is the literal pre-R11
  open policy. Replicated rather than called the reviewer to avoid R2
  (chart-missing) swamping R11 — and to keep the baseline = documented S38/S43.
- Validated on a 24t COVID-quarter pilot (R11 fires in March 2020; arms
  diverge; §2 clean; full analyze pipeline works) before launching the heavy
  runs. W3 = 2020-01-02→2024-12-31 (all 4 regimes); W4 = 2021-01-04→2025-12-31.

## What worked

The decomposition cleanly separates R11's two opposite effects:

- **Sustained grind-down bear (2022): R11 AVERTS large loss** — blocked 2022
  top-bin CSPs counterfactually lose −$268,804 (W3) / −$165,688 (W4), ~50%
  assignment, ≈ −$1,300 to −$2,100/contract. Works as designed, both windows.
- **Sharp V-crash (2020): R11 BACKFIRES** — its VIX>25 *level* trigger fires
  AFTER the spike, blocking trades that ride the V-recovery (forwent +$26,525
  W3). Calm-regime VIX blips forgo more (+$67,405 W3).
- **Whole-book sign flips by window:** W3 −$37,590 / −3.76pp (hurts); W4
  +$21,733 / +2.17pp (helps).
- **Sharpe falls in BOTH windows** (−0.010 W3, −0.054 W4): R11 thins the book.
- **The CSP-leg "averted" number over-states full-wheel value:** W3 R11 "averts"
  −$173,583 on the blocked CSPs in isolation, yet the book is −$37,590 — the
  suppressed arm wheels its extra assignments into the recovery (the
  counterfactual can't see the second leg).

## What didn't

- The held-to-expiry blocked-set counterfactual and the whole-book Δ NAV
  DISAGREE IN SIGN (W3: −$173,583 "averted" vs −$37,590 book worse). Initially
  reads as a bug; it is real — the counterfactual prices the CSP leg in
  isolation; the book wheels assignments into the recovery. Documented as the
  central nuance, not papered over.
- The per-contract "+$86k averted 2020" (I11) does not reproduce at the
  full-wheel book level — at the book the 2020 contribution is a FORGONE gain.
  R11's value concentrates in 2022, not 2020.

## How we fixed it

Reported both metrics side by side and decomposed by regime + VIX-at-entry
bucket so the reader sees WHERE R11 helps (25-35 grind-down band, mostly 2022)
vs hurts (>35 spike band, mostly 2020) — rather than a single net number that
hides the mechanism.

## Evidence

- Driver: `docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_dollar_impact_driver.py`
  (committed 8752d3f). Findings: `R11_DOLLAR_IMPACT_FINDINGS.md` (same folder).
  Raw companions: `r11_w{3,4}_analysis_RAW.txt`, `r11_w{3,4}_summary.json`.
- W3 2020-2024: supp NAV $1,370,771 (+37.08%) / active $1,333,180 (+33.32%) /
  **Δ −$37,590 / −3.76pp / Sharpe −0.010**. 399 blocked, counterfactual
  −$173,583 (21.1% assigned). §2: 521/490 opened, 0 nonpositive-EV.
- W4 2021-2025: supp NAV $1,458,145 (+45.81%) / active $1,479,878 (+47.99%) /
  **Δ +$21,733 / +2.17pp / Sharpe −0.054**. 145 blocked, counterfactual
  −$152,052 (44.1% assigned). §2: 477/418 opened, 0 nonpositive-EV.
- Pilot (24t, 2020-02→05): R11 fired 21× in the COVID spike, forwent +$13,074
  (0% assignment) — first hint that R11 fires post-spike.

## Unresolved / handoff

- **A regime-discriminating trigger is the natural follow-up:** VIX *level* AND
  term-structure / realised-vol slope might keep the 2022 averted loss while
  dropping the 2020 forgone gain (R11 currently can't tell spike-with-recovery
  from grind-down). Worth its own study before any R11 tightening.
- This is two overlapping windows, not a distribution — the divergent sign is
  driven by W3's 2020 V-crash. A walk-forward over more windows (incl.
  2018-2022) would firm up the "helps in sustained bears / hurts in V-crashes"
  claim.
- R11 stays §2-safe and downgrade-only; nothing here motivates a code change,
  only an honest reframing of its value ("insurance paid in Sharpe + forgone
  recovery premium," not "free crisis alpha").
