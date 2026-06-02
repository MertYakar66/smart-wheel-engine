---
id: prob-profit-ci
title: Surface prob_profit small-sample uncertainty (n_scenarios + Wilson 95% CI) on EVResult + ranker
kind: feature
status: in-flight
terminal: UltraCode
pr:
decisions: []
date: 2026-06-01
headline: prob_profit is a k/N binomial frequency over a small forward-scenario set (N~30-35 on the empirical non-overlapping path) but was reported to 4 decimals with no N and no interval — false precision (true 95% CI ~20pp wide; 30/35=0.857 -> Wilson [0.706,0.937]). Added ADDITIVE EVResult fields n_scenarios + prob_profit_ci_low/high (Wilson 95%) and ranker columns; prob_profit is unchanged. Reliability-honesty about PRECISION, not the gated recalibration. Trio (ev_engine + wheel_runner), additive -> lane-claim + independent §2 read.
surface:
  - engine/ev_engine.py
  - engine/wheel_runner.py
  - tests/test_prob_profit_ci.py
---

## Goal
<!-- What we set out to do, and why. -->

Operator mandate: make the engine's outputs **reliable and realistic**. A
quant-lens audit on trunk (origin/main 576e019) ranked `prob_profit`
false-precision as the top reliability defect. `prob_profit =
float(np.mean(pnls > 0))` (`ev_engine.py:393`) is a `k / N` binomial
frequency over the forward-scenario set, and on the default
`empirical_non_overlapping` path N is only ~30-35. It was reported to 4
decimals everywhere with no N and no interval — implying ~0.01pp precision
on a quantity whose true 95% interval is ~20 percentage points wide. Live
on trunk at as_of=2026-03-20: AAPL `prob_profit=0.8571` is literally
`30/35`; Wilson 95% = `[0.706, 0.937]` (~23pp). N was computed but buried
in `EVResult.metadata['n_scenarios']` and never surfaced to the ranker
frame / API / memo.

## What we tried
<!-- Approaches, in the order we tried them. -->

1. **Surface the sampling uncertainty (shipped):** add the sample size N
   and a Wilson 95% CI as additive fields.
2. **Calibration band from the I1 reliability table (rejected — see below).**

## What worked

Additive `EVResult.n_scenarios` + `prob_profit_ci_low/high` (Wilson 95%),
threaded to the ranker frame as core columns. `prob_profit` unchanged. The
Wilson lower bound is itself sobering on the over-confident bins (AAPL
lower bound 0.706 ≈ the documented top-bin realized rate), so it delivers
much of the honesty value with unimpeachable, regime-independent math.

## What didn't
<!-- The dead ends + WHY. This is the part that saves the next agent. -->

- **Calibration band (the audit critic's #1): attach a per-candidate
  `prob_profit_calibrated_lo/hi` from the validated I1 reliability table
  (which quantified the −27pp top-bin over-confidence).** Higher
  paper-impact, but REJECTED for now: it bakes a *single backtest's*
  reliability table into the live engine, and the campaign's own I9
  leave-one-crisis-out test proved that recalibration **does not
  generalize** (crisis realized 0.37–0.93). A band asserting "your 0.94
  realizes 0.70" inherits exactly that non-generalization and is
  **recalibration-adjacent** (the gated EV-authority work). The Wilson CI
  makes no fragile forward claim — it is the correct sampling-uncertainty
  interval, true in every regime. Escalated the band to the operator as a
  higher-impact-but-debatable follow-up (their §2 call), not self-authorized.

## How we fixed it
<!-- The approach that shipped. -->

`engine/ev_engine.py`:
- New module helper `_wilson_score_interval(k, n, z=1.96)` (Wilson score
  interval; clamped to [0,1]; (nan,nan) when n<=0). Chosen over Wald
  because it stays in [0,1] and behaves at small N / extreme p.
- At the `prob_profit` computation: compute `n_scenarios = len(pnls)` and
  the Wilson CI from `(count(pnls>0), n_scenarios)`.
- Three new ADDITIVE `EVResult` fields: `n_scenarios`,
  `prob_profit_ci_low`, `prob_profit_ci_high` (defaults `0` / `nan`, so the
  event-lockout short-circuit path is correct with no edit).
- Docstring note on `heavy_tail`: it is False-by-default and the POT-GPD
  fit only runs at N>=200, so `heavy_tail=False` / NaN `tail_xi` at the
  small-N path mean "not evaluated", not "thin tail confirmed" — read with
  `n_scenarios` (the RA-3 provenance freebie from the audit).

`engine/wheel_runner.py` (`rank_candidates_by_ev`): emit `n_scenarios`,
`prob_profit_ci_low`, `prob_profit_ci_high` in the **core** row dict
(present even when `include_diagnostic_fields=False` — the honesty
annotation always travels with `prob_profit`). The put ranker assembles
its frame via `pd.DataFrame(rows)` (no pinned column list), so adding row
keys adds columns with no schema-list edit.

§2: strictly additive. `prob_profit`, `ev_raw`, `ev_dollars`, the verdict,
and the D16 EV-authority token are all unchanged — the token canonicalizes
a fixed field set (`ticker/strike/premium/dte/ev_dollars/prob_profit/
distribution_source`, `wheel_tracker.py:406-412`) that does not include the
new fields, and the backtest-regression fingerprint is executed_trades +
data_csv_sha256 (not the frame schema). Touches the trio (ev_engine +
wheel_runner) so it carries a lane-claim + needs the independent §2 read,
but it is observability only.

## Evidence
<!-- Exact commands run, numbers, links to raw artifacts. -->

- Audit (worktree @ 576e019): AAPL `prob_profit=0.8571`==30/35, Wilson
  [0.706,0.937]; MSFT 0.8286==29/35, Wilson [0.673,0.919]; N=35
  (empirical_non_overlapping).
- `pytest tests/test_prob_profit_ci.py tests/test_audit_invariants.py
  tests/test_audit_viii_unit_invariants.py tests/test_audit_viii_e2e.py
  tests/test_dossier_invariant.py tests/test_ranker_tracker_wire.py`
  → **87 passed** (12 new + 75 EV/dossier/ranker invariants — confirms
  prob_profit unchanged / additive).
- Full suite minus the slow backtest-regression snapshots: see PR `Tested:`.
  The snapshots are structurally unaffected (token hash + fingerprint use
  fixed field sets, not the frame schema).
- `ruff check` + `ruff format --check` clean on all three files.

## Unresolved / handoff
<!-- What's still open; what the next agent should look at next. -->

- **Calibration band (operator's call):** the higher-impact (but
  recalibration-adjacent / I9-non-generalizing) bias band — surface for an
  explicit §2 decision before building.
- **Other consumers:** propagate `n_scenarios` + CI into `engine_api.py`
  (`/api/candidates` ships `prob_profit` raw at ~:716) and `trade_memo.py`
  (no uncertainty language today), and have the dashboard render
  `prob_profit` as `0.86 [0.71, 0.94] (N=35)`. Off-trio follow-ups.
- **CC / strangle rankers:** mirror the three columns in
  `rank_covered_calls_by_ev` / `rank_strangles_by_ev` (mechanical;
  EVResult already carries the fields for them).
- **RA-2 (earnings-gate PIT look-ahead)** and **RA-3 (tail-fit status string)**
  from the same audit remain as separate additive/off-trio follow-ups.
  (RA-N = reliability-audit finding numbers, distinct from `DECISIONS.md`
  D-numbers — see the tier-gate worklog.)
