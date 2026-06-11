---
id: fix-382-realized-vol-log-guard
title: Close (E) #382: _log_ratio guards raw OHLC operands so ratio estimators never leak +inf / swallow negative bars
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: The OHLC-ratio realised-vol estimators applied _log to a ratio computed BEFORE the non-negativity guard, so a zero denominator leaked +inf and an all-negative bar was swallowed to a normal-looking vol. New _log_ratio guards the RAW operands before the division — byte-identical on valid bars, NaN on any non-positive/non-finite input.
surface: [engine, quant]
---

## Goal
Close (E) #382 from the 2026-06-09 quant-layer test audit round 2 (filed by PR-1,
W40). `engine/realized_vol.py::_log` honours the MODEL_CARDS "non-negativity
enforced" contract (a bad bar -> NaN, never ±inf) only for `close_to_close_vol`,
which logs the RAW close. The four OHLC-ratio estimators (parkinson / garman_klass
/ rogers_satchell / yang_zhang) apply `_log` to a ratio computed BEFORE the guard,
so the guard never sees the raw prices and the contract is not actually met.

## What we tried
Probed the exact failure modes on `1985547` (worktree off live origin/main) before
writing — never trusting the issue text blind. Two confirmed defects:
- **+inf leak:** a zero DENOMINATOR (`low=0`) -> `high/low == +inf`; `_log` tests
  `+inf > 0` (True) and passes it to `np.log(+inf) == +inf`. parkinson/garman_klass
  return +inf.
- **silent swallow:** an all-negative bar -> `-1/-1 == 1` -> `log(1) == 0`, so the
  bad bar contributes a benign 0 term and the estimator returns a normal-looking
  vol (parkinson 0.0759 vs clean 0.0763) — the bad data is invisible.

## What worked
A sibling helper `_log_ratio(num, den)` that applies the non-negativity guard to the
RAW operands (numerator AND denominator) BEFORE the division: it returns NaN whenever
either side is non-positive or non-finite, and otherwise `log(num/den)` bit-for-bit.
Routing the four ratio estimators through it closes both defects with zero change to
valid-bar output. `np.mean` / `np.var(ddof=1)` are retained (NOT the nan-variants) so
a single poisoned bar correctly propagates NaN to the whole-window result — that
propagation IS the contract.

## What didn't
- **np.nanmean/np.nanvar** would re-introduce the swallow (a bad bar would be skipped
  rather than flagged) — rejected.
- **Difference-of-logs** (`_log(num) - _log(den)`) also fixes both defects but is NOT
  bit-identical to `log(num/den)` on valid bars (~1e-16 ULP drift); `_log_ratio`
  preserves the exact division so no served/advisory value moves.

## How we fixed it
`engine/realized_vol.py`: add `_log_ratio`; replace `_log(a / b)` with
`_log_ratio(a, b)` in parkinson_vol / garman_klass_vol / rogers_satchell_vol /
yang_zhang_vol. `close_to_close_vol` is untouched (already correct). The placeholder
`np.where(good, den, 1.0)` denominator only fires where `good` is False (value
discarded) and must be positive nonzero so the not-taken branch never divides by zero.

Not on the EVEngine.evaluate path: `engine/forward_distribution.py` computes its own
inline vol (`np.std(np.diff(np.log(closes)))`) and does NOT import `engine.realized_vol`;
the only consumers are the in-module `realised_vol_bundle` / `vol_risk_premium_bundle`
(advisory VRP display) and `scripts/feature_smoke_test.py`. So this is a defensive
contract fix, not a live EV defect (the connector's OHLCV-positivity integrity tests
also prevent non-positive bars upstream today). No §2 surface touched.

## Evidence
- `pytest tests/test_forward_distribution_invariants.py tests/test_realized_vol.py -q`
  -> 30 passed. The two formerly-`xfail(strict)` leak cases
  (`test_zero_low_should_not_leak_inf[parkinson_vol|garman_klass_vol]`) now pass as
  regular tests (xfail marker removed; assertion tightened from `not isinf` to
  `isnan`). New `test_all_negative_bar_not_swallowed[parkinson_vol|garman_klass_vol|
  rogers_satchell_vol]` pins the swallow fix.
- Clean-bar output byte-identical (parkinson 0.076313, GK 0.083082, RS 0.084857 old
  == fix); `TestVarianceFloor` (all-positive high==low==close) still floors to 0.0.
- `ruff format` + `ruff check` clean.

## Unresolved / handoff
- Sibling (E)s from the same audit round still open: #384 (portfolio_copula t_copula_df
  bounds), #386 (GaussianHMM.fit non-finite guard).
- Benign asymmetry noted (not a defect): parkinson_vol / close_to_close_vol have no
  `max(var, 0.0)` floor, but parkinson's variance is a sum of squares (can't go
  negative) and c2c uses std — so no gap.
