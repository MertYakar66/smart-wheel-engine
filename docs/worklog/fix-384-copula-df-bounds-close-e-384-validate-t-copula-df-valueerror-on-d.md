---
id: fix-384-copula-df-bounds
title: Close (E) #384: validate t_copula_df (ValueError on df<=0/non-finite, warn on infinite-variance df<=2)
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: student_t_copula_simulation / portfolio_cvar_copula took t_copula_df and fed it straight to rng.chisquare(df) and stats.t.cdf(df) with no bound check. _validate_t_copula_df now raises a clear domain ValueError on df<=0 (and non-finite df) and warns on the infinite-variance 0<df<=2 band; df>2 unchanged.
surface: [engine, quant]
---

## Goal
Close (E) #384 from the 2026-06-09 quant-layer test audit round 2 (filed by PR-2,
W49). `engine/portfolio_copula.py` accepted `t_copula_df` (default 5.0) and used it
directly — `rng.chisquare(df)` and `stats.t.cdf(df)` — with no validation.

## What we tried
Probed the actual behaviour on `1985547` before writing (the recon agent for this
issue failed to emit structured output, so the behaviour was re-confirmed by hand):
- `df=5.0` (default): clean, shape (n, N), all finite.
- `df=2.0`: **runs** today — the t-distribution has infinite variance but the
  simulation is mathematically valid (no crash), so the fragile CVaR is produced
  silently.
- `df=0.0` / `df=-1.0`: numpy's `rng.chisquare` raises a terse `ValueError: df <= 0`
  — a real error, but with no hint that the offending knob is `t_copula_df`.

So the issue's "df<=0 -> crash/NaN" is precisely a numpy ValueError; the gap is a
clear message + a guard for the infinite-variance band, not a missing raise.

## What worked
A single validator `_validate_t_copula_df(df)`:
- `df <= 0` or non-finite -> raise a clear domain `ValueError` naming `t_copula_df`,
  BEFORE `rng.chisquare` is reached.
- `0 < df <= 2` -> `warnings.warn(RuntimeWarning, "... infinite variance ...")` and
  proceed (a heavy-tailed-but-valid draw is the operator's prerogative; blocking it
  would be over-strict).
- `df > 2` -> unchanged.

Placed in `student_t_copula_simulation` only. `portfolio_cvar_copula` **delegates**
to it (the t-leg), so the guard covers both with a single source of truth and no
double-warning.

## What didn't
- **Raising on df<=2** — rejected as over-strict: the simulation is mathematically
  valid for any df>0; only the variance is infinite. A warning is the honest signal.
- **Duplicating the check in portfolio_cvar_copula** for a fast-fail before the
  Gaussian leg — rejected: it would double-fire the df<=2 warning, and the wasted
  Gaussian sim on the df<=0 path is a sub-millisecond cost. Single source of truth
  wins.

## How we fixed it
`engine/portfolio_copula.py`: `import warnings`; add `_validate_t_copula_df`; call it
at the top of `student_t_copula_simulation`. `portfolio_cvar_copula` inherits the
guard through delegation.

Severity LOW / defensive: `t_copula_df` is operator-supplied (safe default 5.0) and
`portfolio_cvar_copula` is an advisory tail-dependence report, **not on the default
`EVEngine.evaluate` path**. No §2 surface touched.

## Evidence
- `pytest tests/test_tail_copula_stress_invariants.py -q` -> 15 passed (11 prior + 4
  new). New `TestTCopulaDfGuard`: ValueError on df in {0, -1, nan, inf} on the real
  simulator and through `portfolio_cvar_copula`; RuntimeWarning on df in {1, 2} with a
  well-shaped finite draw; df=5 emits NO warning (`simplefilter("error")` guard).
- `ruff format` + `ruff check` clean.

## Unresolved / handoff
- Sibling (E)s from the same round: #382 (realized_vol _log_ratio — separate PR),
  #386 (GaussianHMM.fit non-finite guard — separate PR).
