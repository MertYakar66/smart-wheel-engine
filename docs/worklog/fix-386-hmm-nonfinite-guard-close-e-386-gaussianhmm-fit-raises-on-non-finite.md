---
id: fix-386-hmm-nonfinite-guard
title: Close (E) #386: GaussianHMM.fit raises on non-finite observations so the neutral-1.0 fallback engages
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: GaussianHMM.fit guarded T<K*3 and near-constant returns but NOT non-finite input — np.nanstd ignores NaN, so a NaN/inf observation passed the gates and produced a NaN position_multiplier. fit now raises ValueError on non-finite input (like the other degenerate guards) so wheel_runner's neutral-1.0 fallback engages instead of leaking NaN into the multiplier/diagnostic/cache.
surface: [engine, quant]
---

## Goal
Close (E) #386 from the 2026-06-09 quant-layer test audit round 2 (filed by PR-3,
W55). `engine/regime_hmm.py::GaussianHMM.fit` had two degenerate-input guards
(`T < K*3` and near-constant `nanstd < 1e-7`, both raising ValueError) but no
non-finite guard.

## What we tried
Probed on `1985547` before writing (recon agent + hand re-confirm):
- A 200-element return series with one NaN (or one inf) -> `fit` returns OK,
  `means_finite=False`, `position_multiplier == NaN`. `np.nanstd` IGNORES NaN, so the
  near-constant guard never sees it, and the M-step has no isfinite filter.
- Traced the §2 path: `wheel_runner.py` hmm.fit -> hmm_regime_mult ->
  combined_regime_mult -> `ShortOptionTrade.regime_multiplier` -> `EVEngine.evaluate`
  reads it and scales `ev_dollars = ev_raw * regime_mult`. BUT ev_engine already
  clamps a non-finite `regime_multiplier` to 1.0 (tagging metadata
  `regime_mult_nonfinite`), so `ev_dollars` is NOT corrupted today. The real damage:
  the `wheel_runner` try/except neutral-1.0 fallback only engages when fit RAISES (it
  does not here), so a NaN leaks into the `hmm_multiplier` diagnostic row and the
  per-ticker regime cache.

## What worked
A non-finite guard at the top of `fit`, placed AFTER the `T<K*3` guard and BEFORE the
`nanstd` near-constant guard (np.nanstd can't detect NaN): `if not
np.isfinite(obs).all(): raise ValueError(...)`. Evaluated after the
`obs = np.asarray(..., dtype=float)` reshape, so it covers 1-D and 2-D inputs and
every feature column. The raise lets `wheel_runner`'s broad `except` set the neutral
1.0 multiplier cleanly, and the cache write (which happens after a successful fit)
never sees the NaN.

## What didn't
- **Relying on the downstream ev_engine clamp** (the status quo) — works for
  `ev_dollars` but leaves the diagnostic + cache poisoned and the documented
  fit-failure contract unmet. The guard belongs upstream at the source.
- **Filtering non-finite rows in the M-step** — rejected: a NaN return means the
  series is corrupt; the honest action is to refuse the fit (mirroring the existing
  degenerate guards), not to silently drop bars.

## How we fixed it
`engine/regime_hmm.py`: add the `np.isfinite(obs).all()` guard. Removed the
`xfail(strict)` (and its now-unneeded `filterwarnings`) from
`tests/test_regime_hmm_invariants.py::TestHmmNonFiniteInput::test_nonfinite_input_should_raise`
so the now-passing test (parametrized [np.nan, np.inf]) is asserted normally.

Determinism is null by construction: the guard only fires on non-finite input, which
`np.diff(np.log(close))` cannot produce from OHLCV-positivity-integrity-tested data;
and even if it did, the EV path result is identical (NaN -> ev_engine clamp 1.0 ==
direct 1.0). The backtest-regression fingerprint (S27/S32/S34/S35) therefore cannot
change — it is excluded from the per-PR lane and left for the supervised re-baseline
session, but no re-baseline is needed for this change.

## Evidence
- `pytest tests/test_regime_hmm_invariants.py tests/test_quant_upgrades.py -q` ->
  38 passed (the 2 formerly-xfail now pass; TestGaussianHMM intact).
- Full suite minus the slow backtest-regression lane (the per-PR CI lane,
  `-m "not backtest_regression"`): 3104 passed, 17 xfailed, 4 deselected, 14 skipped
  in 671s. The only 2 failures are `test_theta_connector` (test_ohlcv_shape,
  test_iv_rank_in_range) — the known Windows-only Theta-connector failures (green on
  ubuntu CI), unrelated to this change (which touches only regime_hmm).
- `ruff format` + `ruff check` clean.

## Unresolved / handoff
- Sibling latent gaps in the same module (out of scope for #386, same defensive
  class): `predict_proba`/`viterbi` don't guard non-finite `observations` directly
  (they receive fit-validated `tail` in production); the near-constant guard only
  checks the primary feature column. Note only — no fix here.
- Sibling (E)s from the same round: #382 (realized_vol), #384 (portfolio_copula).
