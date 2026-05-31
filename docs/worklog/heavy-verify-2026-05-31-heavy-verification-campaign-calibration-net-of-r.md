---
id: heavy-verify-2026-05-31
title: Heavy-verification campaign: calibration, net-of-reality P&L, stress, §2, prior-claims
kind: verification
status: complete
terminal: Major Session
pr:
decisions: []
date: 2026-05-31
headline: prob_profit calibrated in the middle / over-confident at the top (esp. crisis, -27pp); wheel beats passive by +27pp in 2022 bear / -26pp in bulls; §2 firewall held; risk caps dormant by default; engine procyclical at crash entry.
surface: []
---

## Goal
Answer the operator's allocation question — "where can I trust this engine with real
capital, and where can't I?" — adversarially and point-in-time-correctly on the
post-#294 engine. Five investigations (I1 calibration, I2 net-of-reality P&L, I3
stress/discipline, I4 §2 invariant, I5 prior-claims). Observe-and-document only;
`engine/` never modified (`git diff origin/main -- engine/` empty throughout).

## What we tried
1. Surveyed the prior heavy-verify cycle (HT-A/B/C/D) so we'd build on it, not
   duplicate. Found the EV-authority path was byte-identical post-#294 (D19/D21
   reverted; CSP/committee fixes off the decision path).
2. Built shared scaffolding (`campaign_lib.py`): connector-faithful OHLCV (rotated-
   column rename + fail-loud invariant), real Theta bid/ask loader, forward-outcome
   realizer, PIT-cap/equal-weight index proxy, PIT risk-free + dividend helpers.
3. Produced monthly full-503-universe ranked snapshots (`rank_snapshots.py`,
   4-way parallel) — the shared expensive computation consumed by I1 + I2.
4. I1 calibration; I2 frictioned wheel sim w/ real fills + rf-on-collateral +
   dividend-adjusted benchmark; I3/I4 via parallel subagents (lead re-verified the
   material ones); I5 quick re-verify.

## What worked
- The shared-snapshot design (rank once, consume by I1 + I2) saved ~half the compute.
- Adversarial discipline caught a false alarm: at n=4,259 the EV-dollar/realized
  correlation looked *inverted* (−0.015); bootstrap CIs + tail decomposition showed
  it was sampling noise → at full n=16,005 it's ≈0 (tail-dominated). Withheld the
  inversion; published the verified statement.
- Two independent investigations (I1 crisis-regime −26pp over-confidence, I3-E crash
  procyclicality) converged on the same root cause: empirical forward distribution
  lags the regime at transitions.

## What didn't
- Cross-run friction differencing (slippage 1.0 minus 0.0) is confounded by path
  divergence (recent_2025 showed *negative* friction — impossible). Replaced with a
  within-path half-spread+commission counter (`friction_paid`).
- First cut of the passive benchmark used *current* market cap to weight *historical*
  returns → +44.7% "2021" (look-ahead overweighting NVDA et al.). Fixed to PIT cap
  (shares = cur_cap/last_px, weight = shares × price_at_start) → +31.5%, realistic.
- Theta `open_interest` column only exists in the 2023-2026 pull (`include_oi=True`);
  guarded the loader. `data_processed/theta/` is gitignored → lives only in the
  primary clone; the lib searches both roots.

## How we fixed it
Findings + reproducible drivers under `docs/HEAVY_VERIFY_2026-05-31_*.md` and
`docs/verification_artifacts/campaign_2026-05-31/`. PRs opened for operator review,
not merged (per the campaign handoff decision).

## Evidence
- I1: `i1_calibration.py` over 74 monthly snapshots → 16,005 realized rows. Top bin
  (.95,1] fc 0.965 vs realized 0.695 (engine_exact), Δ −27pp, Wilson [0.60,0.78].
  Spearman(ev_raw, realized_$)=−0.002; (ev_roc, realized_roc)=+0.16.
- I2: `i2_pnl.py` 5 regimes × 2 slippage. 2022 bear +4.5% vs index −22.4% (+26.8pp);
  2020 crash +9.1% vs −0.9%; 2023-24 +40.2% vs +66.5% (−26pp); rf added +8.4pp in
  2023-24. Within-path friction ~0.5-1.5%/yr.
- I3: gate dormancy (`require_ev_authority=False` default), 174/73-month cap breaches;
  crash procyclicality (2020-03-02: 89% positive-EV → realized −$1,305/contract, 82%
  assigned) — lead-reproduced.
- I4: 6 adversarial attacks, §2 HELD; lead-verified `git diff` empty + sign-preservation.
- I5: MU CC −$812 (was −$1,058 pre-PIT-IV); IV no-skew 100.0000%; PIT-IV fix live.
- All raw output: `docs/verification_artifacts/campaign_2026-05-31/raw_output/*`.

## Unresolved / handoff
- **Highest-value fix target** (NOT done — observe-only): the empirical forward
  distribution under-models the crisis left tail → top-bin over-confidence (I1) +
  crash procyclicality (I3-E). Candidate fixes: wire POT-GPD (`engine/tail_risk.py`)
  into `prob_profit`, or a regime-conditional high-confidence haircut. Needs §2-safe
  design + a fresh calibration re-baseline.
- Risk caps (R9/R10/delta/Kelly) are dormant unless `require_ev_authority=True`
  (I3-A) — decide whether to arm them in the ranker/book path or enforce externally.
- Earnings-gate PIT look-ahead (I3-D) — non-PIT realized-date calendar; conservative
  direction (cannot inflate returns) but candidate counts aren't strictly PIT.
- HT-B vs this campaign disagree on which `distribution_source` is better-calibrated
  in the top bin — open reconciliation (universe-size / date-mix).
