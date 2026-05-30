---
id: A2-C1
title: iv_surface fail-loud wiring (A2) + bloomberg-CSV tracking decision (C1)
kind: feature
status: in-flight
terminal: Major Session
decisions: [D9]
date: 2026-05-30
headline: SVI tooling wired in fail-loud (require_surface + diagnose_iv_surface); bloomberg CSVs stay tracked as data commits
surface: [engine/volatility_surface.py, scripts/diagnose_iv_surface.py, tests/test_iv_surface_failloud.py]
---

## Goal
Close the two ROADMAP "open question" items the operator selected:
- **A2** — pick the iv_surface missing-data contract for the dormant SVI tooling.
- **C1** — decide whether the regenerated bloomberg yfinance CSVs stay tracked.

## What we tried
- **A2 fact-check first.** The "zero non-test callers as of 2026-04-25" claim in
  D9 / PROJECT_STATE was stale-sounding, so I grepped: the only references to
  the SVI calibration tooling outside `engine/volatility_surface.py` + tests are
  a docstring mention in `skew_dynamics.py`, a dep-map entry in
  `dependency_check.py`, and one call in `scripts/feature_smoke_test.py`.
  `ThetaConnector.get_iv_surface` is a *separate* live data-fetch method, not the
  SVI calibrator. Conclusion: the SVI calibration layer was genuinely dormant.
- **Contract options weighed:** (a) fail-loud, (b) named flat_iv_fallback,
  (c) deprecate. Operator chose **(a) wire in, fail-loud**.
- **Implementation route weighed:** changing `get_iv`/`get_skew`'s long-standing
  `return 0.20` defaults to raise vs. an additive guard. Checked the test suite:
  only `tests/test_audit_improvements.py` exercises the surface, and only on
  *populated* surfaces — nothing depends on the empty-surface 0.20 default.

## What worked
- **Additive guard** (`SurfaceDataUnavailable` + `require_surface(surface, ticker)`)
  consumers call to fail loud — zero behaviour change to the existing
  `get_iv`/`get_skew` happy paths, so zero risk to the green suite.
- **First production caller:** `scripts/diagnose_iv_surface.py` — builds a
  smile-aware surface from a ticker's ATM IV term structure via
  `create_empirical_surface`, reports per-expiry skew + term structure, and
  **exits non-zero** on any uncovered ticker. Numeric core is pure + unit-tested.
- **C1:** keeping the CSVs tracked is the status quo + aligns with the repo's PIT
  audit-value stance, so the decision is a doc record with zero migration risk.

## What didn't
- Changing `get_iv`'s internal `0.20` defaults to raise directly was rejected for
  this pass — it's a behaviour change on a path I can't run locally (no
  numpy/scipy in the Cowork sandbox), so the additive guard is the safer route;
  converting the internals is a documented D9 follow-up.
- The diagnostic's connector path (assembling the ATM term structure from a live
  provider) can't be end-to-end verified here (no Theta) — kept thin/defensive
  and marked operator-first-run-verify, mirroring the EDGAR puller convention.

## How we fixed it
- `engine/volatility_surface.py`: `SurfaceDataUnavailable` + `require_surface`.
- `scripts/diagnose_iv_surface.py`: the fail-loud production caller.
- `tests/test_iv_surface_failloud.py`: pins the contract (require_surface raises
  on empty; create_empirical_surface raises on empty; diagnostic core builds on
  synthetic ATM term structure + fails loud on missing; populated surface works).
- Records: `DECISIONS.md` D9 (resolved), ROADMAP A2 + C1 (done),
  `MODULE_INDEX.md` (live), `PROJECT_STATE.md` §3 (resolved),
  `docs/DATA_POLICY.md` §5 (C1 policy), `CHANGELOG.md`.

## Evidence
- `ruff check` + `ruff format --check` clean on the 3 new/edited files; `ast.parse` OK.
- numpy-dependent tests (`tests/test_iv_surface_failloud.py`) run in CI, not the
  sandbox — same verification path as the #285 lane-gate test.
- Touches no decision-layer file → the `decision-layer-claim` gate does not fire.

## Unresolved / handoff
- Convert the internal `get_iv`/`get_skew` `0.20` defaults to raise directly
  (currently governed by the `require_surface` consumer contract).
- Live-verify the diagnostic's connector term-structure path on a Theta-up box.
