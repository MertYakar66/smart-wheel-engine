---
id: data-tests-realism-scale
title: Data-test PR-7 realism-at-scale + vix-R11 + liquidity (W33/W35/W34)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Phase-2 round-2 PR-7 of the data-layer test audit. Adds W33 (full-universe BANDING at scale — extends the finite-only slow pin so an out-of-band tail/post-seam row is caught, not just non-finite), W35 (VIX content band — EV-decision-relevant via R11, the elevated-vol downgrade reviewer; pins vix in POINTS so a decimal-scale flip can't silently disable the R11 vix>25 downgrade), W34 (liquidity avg_vol_30d non-neg+finite, off-EV/LOW). Gaps from the round-2 recon; W35 reconciles round-1's C3. Test-only; trio/data untouched.
surface: [tests/test_data_to_engine.py, tests/test_data_integrity_bloomberg.py]
---

## Goal
<!-- What we set out to do, and why. -->

Round-2 recon surfaced three realism/peripheral gaps. PR-7 closes them.

## What we tried
<!-- Approaches, in the order we tried them. -->

- **W33** (`tests/test_data_to_engine.py`) — EXTEND the slow
  `test_full_universe_no_silent_drops_and_split` (it pinned counts + FINITE-at-scale
  only) with a per-row BAND loop: 0<iv<3, prob∈[0,1], premium>0, strike<spot, valid
  tier — across all 480 produced. One full-universe rank, no duplicate.
- **W35** (`tests/test_data_integrity_bloomberg.py`) `test_vix_content_band_protects_r11`
  — vix positive/finite, band [5,150], median in the tens (catches a percent/decimal
  flip vs the R11=25 threshold), keys unique, no future bars.
- **W34** (`tests/test_data_integrity_bloomberg.py`) `test_liquidity_avg_vol_nonneg_and_finite`
  — avg_vol_30d non-negative + finite (zeros/nulls allowed).

## What worked

All pass; fast suite 63 passed / 13 xfailed; slow W33 passes (36.95s, 480 produced).

## What didn't
<!-- The dead ends + WHY. -->

The round-1 wiring agent said "vix is OFF the EV verdict" (true for the ranker's
ev_dollars — the regime mult is per-ticker OHLCV HMM, not VIX). The round-2
completeness critic caught that vix IS EV-decision-relevant via **R11**: I verified
at source — candidate_dossier.py:546-553 reads `dossier.vix_level` (← get_vix_regime,
wheel_runner.py:3490) and downgrades proceed→review when `vix_level > 25.0 AND
prob_profit > threshold`. So a wrong-scale vix silently breaks an EV-authoritative
downgrade reviewer — hence W35 is MEDIUM (not the INFO the first cut implied), and the
median-in-tens assertion specifically protects the R11 threshold. Capability
correction **C3** refined accordingly.

## How we fixed it
<!-- The approach that shipped. -->

Test-only. W33 folded into the existing slow test (no extra full-universe rank). W35
pins vix as POINTS to protect R11. W34 allows the ~1,123 legitimate zeros + 10 nulls
(non-neg + finite, not strictly >0) — it's off the EV path (LOW).

## Evidence
<!-- Exact commands run, numbers. -->

Worktree off `origin/main 37258f9` (post PR-6 merge), provider `MarketDataConnector`.

- vix: nulls 0, nonpos 0, min 9.14, median **17.61**, max 82.69; liquidity avg_vol_30d
  neg 0 / inf 0 / min 0.0; full-universe **480 produced, 0 band+finite violations**.
- `pytest -m "not slow"` → 63 passed / 13 xfailed; slow `test_full_universe_no_silent_drops_and_split` → 1 passed (36.95s). `ruff` clean.

## Unresolved / handoff
<!-- What's still open. -->

- **HOLD for review.** Next: PR-8 (W36 cross-file vol_iv↔ohlcv date consistency,
  W37 data_integration rate-accessor divergence) + (E) tracking issues for the
  engine-side parts (IV-staleness gate; rate-fallback divergence). Then land the audit
  doc updated with W29–W37 + the C3 (vix→R11) refinement.
- W35's threshold band [5,150] is generous (VIX all-time intraday ~89); revisit only
  if a legit print exceeds it.
