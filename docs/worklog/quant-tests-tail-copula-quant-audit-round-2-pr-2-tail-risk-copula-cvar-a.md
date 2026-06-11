---
id: quant-tests-tail-copula
title: "Quant audit round 2 PR-2: tail-risk, copula-CVaR and stress invariants (W44-W49)"
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Pins CVaR>=VaR on the gpd xi<0 branch + the REAL copula simulators (the existing copula tests monkeypatch the math away), MC-stress seed reproducibility (#366 false-green) + cvar95<=var95, and run_scenario's expired-intrinsic branch; t-copula df-bound guard tracked as (E) #384
surface: [tests, quant]
---

## Goal
Quant audit round 2, PR-2 — the risk-math feeding the heavy-tail EV penalty
(`engine/tail_risk.py`), reviewer R8's vol-spike drawdown trigger
(`engine/stress_testing.py`), and the portfolio tail-dependence CVaR
(`engine/portfolio_copula.py`). Tests-only; (E) tracked, not grabbed.

## What we tried
Probed each invariant against real behaviour on `a2eea4e` before writing (the
PR-1 probe caught a subagent overclaim, so this is now standard). The recon was
accurate here — all five (T) invariants held as described.

## What worked
- **W44** `gpd_var_cvar` CVaR>=VaR on the xi<0 thin-tail branch (distinct formula,
  never exercised) + the abs(xi)<1e-8 exponential-branch boundary.
- **W45** copula CVaR>=VaR on the REAL simulators (both Gaussian and t legs). The
  existing `test_portfolio_copula_coverage.py` monkeypatches both simulators, so the
  percentile->tail-mean math that MUST satisfy CVaR>=VaR was never asserted — a real
  false-green closed here.
- **W46** `monte_carlo_stress` seed reproducibility (same-seed identical risk numbers,
  diff-seed different) — the docstring claimed it, only key-presence was checked.
- **W47** `cvar_95 <= var_95` (conditional-tail mean no greater than the 5% quantile).
- **W48** `run_scenario` expired-intrinsic branch: ITM short put -> finite loss,
  OTM -> finite gain (the branch driving R8 drawdown for assigned near-dated books).

## What didn't
n/a — invariants verified by probe (gpd xi=-0.3 cvar 0.091>=var 0.083; copula
g/t cvar>=var; MC same-seed==, cvar -12797<=var -9302; intrinsic ITM pnl -11946).

## How we fixed it
`tests/test_tail_copula_stress_invariants.py` (new): 7 tests, all passing.
ruff-clean (imports auto-sorted).

## Evidence
- Probe output (py -3.12) on `a2eea4e` confirmed every invariant before writing.
- Touches: the new test file + this fragment + FILE_MANIFEST row. No engine/data/trio edits.

## Unresolved / handoff
- **(E) #384** — guard `t_copula_df` (`df<=0` invalid, `df<=2` infinite variance).
  No (T) test added: pinning today's unguarded crash/NaN would lock in the bug;
  add the guard + assertion together under §2-style review. Sibling of (E) #382.
- Remaining round-2 PRs: PR-3 HMM regime (NaN-guard + crisis firing-rate
  discrimination), PR-4 skew/surface, PR-5 dealer `[0.70,1.05]` clamp, PR-6
  pricing/Greeks/gates, PR-7 reviewers — one surface each, held for review.
