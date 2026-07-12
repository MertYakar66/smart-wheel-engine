---
id: validation-v1-tail-exceedance
title: V1 tail-risk exceedance harness — Kupiec/clustered-CI/severity backtesting of pnl quartiles + cvar_5
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-07-12
headline: The engine's own risk quantiles (pnl_p25/p50/p75) and cvar_5 are now formally backtestable — Kupiec POF + date-clustered bootstrap CIs + violation-clustering permutation test + ES-bound breach severity, with a synthetic calibrated-PASS/understated-FAIL contract pinning the harness itself.
surface: [backtests/tail_exceedance.py, scripts/run_tail_exceedance.py, tests/test_tail_exceedance.py, docs/VALIDATION_PHASE_PLAN.md]
---

## Goal

Open the formal validation phase (docs/VALIDATION_PHASE_PLAN.md) with its V1
workstream: turn the engine's per-candidate risk outputs into formally
backtested quantities. The record had strong *anecdotes* — the 2026-06-15
trader stress test found cvar_5 understating single-name blowups 3-4.5x, and
prob_profit top-bin over-confidence is established across 10 configs — but
no exceedance statistics of the kind a desk risk team would demand (Kupiec
proportion-of-failures, violation-independence, expected-shortfall breach
severity). Measurement-only; trio untouched.

## What we tried

- Reusing the committed parameter-OOS fixtures directly: rejected — they
  carry prob_profit/realized_pnl but not the tail block (pnl quartiles,
  cvar_5), so a new capture column set was needed.
- Extending `backtests/parameter_oos.py::build_rank_table` in place:
  rejected — that module is fixture<->snapshot recompute-locked; a sibling
  module with the same conventions is safer.
- A literal Christoffersen (1998) independence test: rejected as
  panel-dishonest — it assumes one P&L series; with ~20 trades per entry
  date sharing one market path, transition counts are mechanically
  clustered. Replaced with a permutation test on the lag-1 autocorrelation
  of the *date-level* violation-rate series, plus date-clustered bootstrap
  CIs on every coverage rate (the `cluster_bootstrap_ci` convention from
  parameter_oos).

## What worked

`backtests/tail_exceedance.py` — capture (mirrors build_rank_table; adds
pnl_p25/p50/p75, cvar_5, cvar_99_evt, tail_widening_factor, n_scenarios,
distribution_source, entry-VIX) + pure-numpy statistics (kupiec_pof,
date_clustered_rate_ci, date_rate_autocorr_test, heterogeneous_coverage_z,
quantile_coverage_report, cvar_breach_report, full_report with a
PASS/WARN/FAIL/INSUFFICIENT ladder). `scripts/run_tail_exceedance.py`
build/analyze driver (24t and 100t canonical configs, frontier-capped end
date). 30 unit tests, engine-free, incl. the load-bearing synthetic
end-to-end pair: same-distribution generator must PASS, 3x-understated tail
must FAIL, conservative model must never FAIL.

## What didn't

- First run of `test_iid_violations_not_flagged` failed at p=0.036 — a
  textbook 5%-level Type-I on a single seeded draw. Fixed by asserting the
  *median* p across five independent data seeds, which a biased statistic
  cannot pass but an honest one can.
- A stray non-ASCII character slipped into a test comment; scrubbed
  (Windows-console/cp1252 discipline).

## How we fixed it

See above; harness shipped with the two fixes. Verdict-ladder asymmetry is
deliberate and documented: coverage *below* nominal (model too fat-tailed)
reports but never FAILs — the safe direction for a short-vol book. Realized
P&L keeps the locked `_forward_replay_realized_pnl` convention (gross of
entry costs + $5 ITM fee) for cross-study comparability; the few-dollar
offset favors the engine on lower-tail tests, so FAILs are conservative
evidence.

## Evidence

- `python3 -m pytest tests/test_tail_exceedance.py -q` -> 30 passed.
- 24t capture/analysis (V1-a): `python scripts/run_tail_exceedance.py full
  --config 24t` (UNIVERSE_24, 2022-01-03 -> frontier-capped, every 5
  bdays); results recorded in docs/VALIDATION_PHASE_PLAN.md section 4 once
  both runs land.
- Pre-registered expectations (falsifiable, written before the first
  analyze): p50/p75 roughly honest; cvar_5 PASS pooled but WARN/FAIL in the
  crisis stratum; breach severity well above 1x; strong violation
  clustering (I3-E procyclicality).

## Unresolved / handoff

- V1-b: the 100t run is the terminal's task card (docs/VALIDATION_PHASE_PLAN.md
  section 4) — `python scripts/run_tail_exceedance.py full --config 100t`.
- Findings doc once both runs are in; any FAIL triaged to engine-finding
  (re-baseline queue) vs harness artifact.
- Follow-on workstreams V2 (parameter freeze-replay) and V3 (plateau sweep)
  are queued in the plan doc.
