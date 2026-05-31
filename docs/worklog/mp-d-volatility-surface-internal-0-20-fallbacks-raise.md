---
id: MP-D
title: Volatility-surface internal 0.20 fallbacks raise SurfaceDataUnavailable
kind: refactor
status: in-flight
terminal: D
pr:
decisions: [D9]
date: 2026-05-30
headline: get_iv/get_skew internal 0.20 fallbacks now raise SurfaceDataUnavailable; same D9 contract as the public require_surface guard, end-to-end
surface: [engine/volatility_surface.py, tests/test_iv_surface_failloud.py]
---

## Goal
Pick up the explicit follow-up from the A2-C1 worklog: convert the **internal**
`return 0.20` fallbacks in `VolatilitySurface.get_iv` / `get_skew` to raise
`SurfaceDataUnavailable`, so the D9 fail-loud contract that #286 set on the
public surface (`require_surface`, `create_empirical_surface`) reaches all the
way down into the surface object's own methods. After this PR, no method on
`VolatilitySurface` can silently quote a fabricated flat 20% IV out the back
when the surface has no data.

## What we tried
- **Caller-graph audit first.** Grep'd every `.get_iv(`/`.get_skew(` call in
  the repo (excluding `archive/`). Production callers of `get_iv` are all
  internal to `volatility_surface.py` (self-recursion in `_interpolate_expiry`,
  `get_term_structure`, `estimate_iv_for_delta`); the only test caller
  (`tests/test_audit_improvements.py`) always passes `expiries[0]` on a
  populated surface (direct SVI path, never hits `_interpolate_expiry`).
  `get_skew` has zero callers outside `tests/test_iv_surface_failloud.py` and
  `scripts/diagnose_iv_surface.py` (which iterates `sorted(surface.svi_params)`
  → the requested expiry is always present → never raises). Conclusion: the
  three fallback sites are reachable only by an empty-surface call or an
  unknown-expiry call — exactly the missing-data cases D9 says must raise.
- **Three sites identified, all in `class VolatilitySurface`:**
  1. `_interpolate_expiry` (top of fn) — `if not self.svi_params: return 0.20`
     fires when `get_iv` is called on an empty surface (any expiry routes here
     because `expiry not in {}` is always True).
  2. `_interpolate_expiry` (bottom fall-through) — `return 0.20` after the
     bracket-finding loop. Practically unreachable (the two edge-case branches
     above already handle expiries outside the bracketed range and the loop
     covers everything in between for a monotone-sorted expiry list), but
     defensible to convert for policy consistency.
  3. `get_skew` — `if expiry not in self.svi_params: return 0.20, 0.20, 0.20`
     fires on a missing-expiry call. Unlike `get_iv`, `get_skew` has no
     interpolation logic to preserve — replacing the return with a raise is the
     simplest faithful read of the card.

## What worked
- All three sites converted to `raise SurfaceDataUnavailable(...)` with
  ticker-aware error messages that mirror `require_surface`'s wording (mention
  `self.underlying`, point at DECISIONS.md D9, suggest `create_constant_surface`
  for an explicit opt-in flat surface).
- Added two new test classes in `tests/test_iv_surface_failloud.py`:
  - `TestGetIvFailLoud` — empty-surface raises; populated-surface known expiry
    returns a value; populated-surface unknown-but-bracketed expiry interpolates
    cleanly (no raise — real SVI data on both sides); populated-surface expiry
    past the last calibrated expiry extrapolates from the last SVI fit.
  - `TestGetSkewFailLoud` — empty-surface raises; populated-surface unknown
    expiry raises.
- Existing `TestDiagnosticCore::test_atm_iv_honoured` (calls `surf.get_skew`
  with known expiries on a populated surface) is the regression pin for the
  happy path — stayed green.

## What didn't
- **Did NOT touch `SplineVolSurface.get_iv`'s** own `return 0.20` at line 543
  on origin/main — that is a separate class (cubic-spline alternative to SVI)
  and the card's `owns` is bound to `get_iv`/`get_skew` on
  `VolatilitySurface`. Card spec was explicit: "if scope spills past
  `get_iv`/`get_skew`, stop and ask the Major Session." I stopped.
- **Did NOT touch `estimate_iv_for_delta`'s** `return spot, 0.20` math
  fallback (fires when `T <= 0`). It's a math-domain fallback (expired
  contract), not a missing-data fallback, and it's not `get_iv` / `get_skew`.
  Out of MP-D scope; flagged in "Unresolved / handoff" below.
- **Did NOT touch `_interpolate_expiry`'s `(iv1 + iv2) / 2` branch** (fires
  when `T_target ≤ 0` or `w_interp ≤ 0`) — that branch uses real SVI-derived
  `iv1`/`iv2`, not a fabricated constant. It's a numeric-degeneracy guard, not
  a missing-data fallback. Same reasoning.

## How we fixed it
- `engine/volatility_surface.py`: three `return 0.20…` lines → three
  `raise SurfaceDataUnavailable(...)` calls. `get_skew`'s docstring updated
  with a `Raises:` section pointing at D9 + `create_constant_surface`.
- `tests/test_iv_surface_failloud.py`: 6 new tests across two new classes
  (4 + 2). Existing 8 tests stayed green.
- No changes to public surface, importable names, or callable signatures —
  just the missing-data return semantics.

## Evidence
- `pytest tests/test_iv_surface_failloud.py tests/test_audit_improvements.py`:
  **39 passed in 5.26s** (14 fail-loud + 25 surface-builder/audit). Both new
  test classes green; all populated-surface regressions held.
- Broader unit suite (`pytest tests/ -x --ignore=tests/test_backtest_regression.py`):
  **2137 passed, 1 failed, 2 xfailed in 205.10s**. The one failure is the
  documented Windows-local Theta-tier flake
  (`tests/test_theta_connector.py::test_ohlcv_shape`, empty DataFrame because
  Theta Terminal isn't running in this sandbox) — see PROJECT_STATE.md §2
  and memory `windows-local-vs-ubuntu-ci.md`. Unrelated to this PR.
- Non-§2 surface (volatility_surface is not the trio) → `decision-layer-claim`
  CI gate does not fire on this PR.

## Unresolved / handoff
- `estimate_iv_for_delta` still has `return spot, 0.20` at the `T <= 0` branch
  — a math-domain fallback, separate from `get_iv`/`get_skew`. If/when a
  follow-up MP-card touches it, the parallel question is whether `T <= 0`
  should raise a domain error or stay the silent 0.20.
- `SplineVolSurface.get_iv` has its own `return 0.20` for the missing-expiry
  case. Same D9 reasoning would apply, but it's a different class and was
  excluded from MP-D's `owns`.
