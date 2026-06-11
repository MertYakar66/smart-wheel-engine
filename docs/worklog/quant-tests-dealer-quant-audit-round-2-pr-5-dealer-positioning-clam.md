---
id: quant-tests-dealer
title: "Quant audit round 2 PR-5: dealer-positioning clamp + regime invariants (W60-W62)"
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Pins the §2-critical dealer [0.70,1.05] clamp under ANY confidence (incl inf/nan — they resolve to 1.0, no leak) + asymmetry, the _classify_regime boundaries (gex==0 neutral, near-flip override), and analyze() flip-distance consistency
surface: [tests, quant]
---

## Goal
Quant audit round 2, PR-5 — the §2-critical dealer overlay
(`engine/dealer_positioning.py`). The `dealer_regime_multiplier` is the only
final-`ev_dollars` scalar clamped asymmetrically to [0.70, 1.05] (CLAUDE.md §2/§3
forbids altering it). Pin the clamp + classification. Tests-only.

## What we tried
Probed the clamp + classifier on `a2eea4e` before writing.

## What worked
- **W60 the §2 clamp (HIGH):** for every regime x confidence in
  {-5, 0, 0.5, 1, 5, inf, nan}, the multiplier stays in [0.70, 1.05] and finite;
  `None` -> 1.0; asymmetry pinned (long-gamma boost caps +0.05, short-gamma cut to
  0.70, cut > boost). The output bound was only emergent from the internal
  `max(0, min(1, conf))` (dealer_positioning.py:750) — never proven for out-of-[0,1].
- **W61 _classify_regime boundaries:** gex==0 -> 'neutral' (the long/short boundary),
  sign branches, and the near-flip override (within `flip_neighborhood_pct` of the
  flip beats the GEX sign; just outside, the sign governs again).
- **W62 flip-distance consistency:** through `analyze()`, `flip_distance_pct ==
  (flip_level - spot)/spot` when a flip is found.

## What didn't
- **No NaN/inf leak after all.** I expected `min(1.0, nan)` to leak a NaN multiplier
  (like the realized_vol / HMM cases) — but the probe showed Python's `min(1.0, nan)`
  returns 1.0, so conf resolves to 1.0 and the clamp holds. So W60 is a clean PASS,
  no (E) — the clamp is genuinely robust. (Verified by probe, not assumed.)
- **Wall-ordering deferred.** put_wall <= spot <= call_wall is a real invariant, but a
  simple chain yields None walls (detection needs OI concentration past a threshold);
  pinning it needs a reverse-engineered wall fixture -> follow-up, noted in the file.

## How we fixed it
`tests/test_dealer_positioning_invariants.py` (new): 32 tests (28 = 4 regimes x 7
confidences, + asymmetry/classify/flip), all passing. ruff-clean.

## Evidence
- Probe (py -3.12) on `a2eea4e`: clamp table all in [0.70,1.05] (incl inf/nan);
  classify boundaries; flip_level 98.73 / flip_distance_pct consistent. No trio/data edits.

## Unresolved / handoff
- Wall-ordering (needs a wall-producing OI fixture) + R6 at-the-put-wall boundary
  (reviewer rule -> PR-7).
- Remaining round-2 PRs: PR-6 pricing/Greeks/gates (binomial Greek units HIGH),
  PR-7 reviewers, + the deferred vol-surface (SVI/Spline, dormant D9 + (E) Spline-0.20).
