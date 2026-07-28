---
id: trader500k-traceability-2026-07-28
title: TRADER-500K traceability closure + parameter-OOS reconciliation
kind: docs
status: complete
terminal:
pr:
decisions: []
date: 2026-07-28
headline: Committed scripts/analyze_trader500k_pool.py reproduces the report's §3 headline from the bundles (pooled +0.0205 [−0.0215,+0.0631]); three drifts corrected (~46%→48.7%, quiet-bull −0.12→−0.11, W05_rail +2.2%→+2.1%); §7 records the parameter-OOS reconciliation (CONSISTENT — same population-width curve at opposite ends, matched pair −0.2315 vs −0.2332).
surface: []
---

## Goal

PR #520's report headlined figures (pooled Spearman ≈ +0.02, its cluster CI,
the regime split) that existed in NO committed artifact — they were derived in
a sandbox from `calibration_rows.csv.gz` and the deriving script was never
committed. A read-only reconciliation run (2026-07-28) had also (a) judged the
campaign CONSISTENT with the parameter-OOS 100t campaign (PR #485) and
(b) found three factual drifts in the report. Close all three gaps: make the
headline reproducible, correct the drifts, and cross-link the two campaign
reports so neither headline is ever read without its population qualifier.

## What we tried

Single approach, no dead ends: a pure-pandas/scipy recomputation script over
the committed bundles (no engine imports), gated by the operator's approval
conditions (stop on sign flip, on regime-direction change, or on missing
columns).

## What worked

`scripts/analyze_trader500k_pool.py` reproduces every §3 figure from committed
inputs alone:

- pooled Spearman(ev_dollars, realized_pnl_synth) **+0.0205**, date-clustered
  bootstrap CI95 **[−0.0215, +0.0631]** (doc stated +0.02 [−0.02, +0.06] ✓)
- regime split crisis **+0.1469** / bear **+0.0278** / normal **−0.0133** /
  bull_quiet **−0.1147** (doc stated +0.15/+0.03/−0.01/**−0.12** — last one
  corrected to −0.11)
- prob_profit pooled +0.1726; per-regime +0.1495..+0.1838 (doc "+0.15..+0.18" ✓)
- row accounting resolved the 180,382-vs-180,360-vs-180,451 confusion exactly:
  180,451 committed rows − 69 unresolved-spot = **180,382** analyzed (the §2
  headline was right); the bin table shows 180,360 because **22** rows carry
  forecasts < 0.5, below its first bin
- event gate **48.65%** (200,188 / 411,485) — doc's "~46%" corrected
- fidelity: per-window recomputed pool rho matches every committed
  `summary.json` to **≤2.2e-07**

## What didn't

Nothing failed. One deliberate scope call: the §1 friction matched pairs, §2
pooled bin table / transition finding / headline Brier–ECE, and §4 matched-pair
figures are still sandbox-only — labelled as such in the report's new §5 item 7
(reproducibility register) rather than silently left, per the run's invariant.
Also note: the task card's "~221–261 candidates/rank-week" was slightly off at
the low end — committed counts give **~212–261** (W01 = 15,474/73 = 212.0);
the docs state the artifact-derived range.

## How we fixed it

- New `scripts/analyze_trader500k_pool.py` → byte-stable
  `docs/verification_artifacts/trader500k/CROSS_WINDOW_SUMMARY.json`
  (seed 12345, n_boot 2000, input SHA-256s recorded; rails never pooled).
- `docs/TRADER_500K_RELIABILITY_2026-07.md`: §3 sourced to the script + values
  corrected (quiet-bull −0.11) + full-menu framing made explicit (~212–261
  names/rank-week, NOT the top-20 execution slice) + executed-slice positivity
  in all 11 windows (+0.25..+0.71) stated; headline verdict rows corrected
  (48.7% event gate, −0.11); §2 row-accounting note; §4 W05_rail +2.1%
  (misround) + provenance note; §5 item 7 reproducibility register; new §7
  reconciliation section (tier curve, matched pair, verdict CONSISTENT).
- `docs/PARAMETER_OOS.md` §7.4: short cross-reference back (scope-limited).
- FILE_MANIFEST rows for the two new files; report/doc rows refreshed.

## Evidence

`python scripts/analyze_trader500k_pool.py` (Python312, pandas 2.3.3 /
scipy 1.17.0 / numpy 2.4.2):

```
rows: concat=180451  finite ev/pnl pairs=180382  reliability-valid=180382  (of which prob<0.5: 22)
pooled Spearman(ev_dollars, realized): +0.0205  CI95 [-0.0215, +0.0631]  (se 0.0217, 337 date clusters, n_boot 2000)
pooled Spearman(prob_profit, realized): +0.1726  CI95 [+0.1443, +0.1988]
crisis n=27972 ev +0.1469 | bear n=65524 ev +0.0278 | normal n=58202 ev -0.0133 | bull_quiet n=28421 ev -0.1147
event gate: 200188 / 411485 name-weeks = 48.65%
per-window recompute vs committed summary.json (max |delta|): 2.222e-07
```

Reconciliation evidence (committed on main): A snapshot
`backtests/regression/snapshots/param_oos_regime_100t.json`
`top_n_tiers.holdout` (+0.597/+0.473/+0.371/+0.228/+0.102/−0.077) and
`walk_forward_folds[3]` ρ −0.2315 (n=12,976, 2023-06→2024-06) vs B
`W08/summary.json` full-menu pool −0.2332 (2023-07→2024-12) — the matched pair.
Executed-slice rho positive in all 11 core windows (+0.247 W04 .. +0.707 W11)
per `per_friction.full.trades.spearman_ev_realized_executed`.

## Unresolved / handoff

- Standing follow-up (report §5 item 7): commit derivations for the still
  sandbox-only figures — §1 friction matched pairs, §2 pooled bin table +
  transition finding + headline Brier/ECE, §4 matched-pair premium/calibration
  deltas.
- The report's per-window high-vol/calm-bull window labelling remains
  interpretive (no committed labelling); §3 now names the windows explicitly so
  the numbers themselves are traceable.
- The date-clustered CI is block_len=1 (per the report's limitation 5); a
  contiguous-block upgrade à la `backtests/parameter_oos.py` §7.0 would widen
  it honestly — measurement-harness follow-up, not a defect.
