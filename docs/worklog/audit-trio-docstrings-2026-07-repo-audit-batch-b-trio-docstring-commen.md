---
id: audit-trio-docstrings
title: 2026-07 repo audit — Batch B trio docstring/comment micro-fix
kind: docs
status: in-flight
terminal:
pr:
decisions: []
date: 2026-07-07
headline: Three comment/docstring truth-fixes on decision-layer files — broken link, a phantom API in the ev_engine example, and a stale "(default on)" R9 comment. Zero executable change; full suite green.
surface: [engine/ev_engine.py, engine/candidate_dossier.py, engine/wheel_tracker.py]
---

## Goal
Batch B of the 2026-07 repo audit (D-number at merge): correct three
comment/docstring inaccuracies on decision-layer files. **No executable code
changes** — only docstrings and one code comment. Kept separate from Batch A
(PR #487) because these touch the CI-gated trio and need a lane-claim + the full
suite per `CLAUDE.md` §4.

## What worked
- **candidate_dossier.py:272** — the R11 docstring linked to
  `docs/HEAVY_VERIFY_2026-05-31_I11.md`, which does not exist. Repointed to the
  real file `docs/HEAVY_VERIFY_2026-05-31_I11_RISK_BUDGET_STUDY.md` (verified the
  wrong target absent + the right target present).
- **ev_engine.py:72-77** — the "Typical caller" example called
  `feature_store.get_forward_distribution("AAPL", horizon_days=35)`; `feature_store`
  is never imported and `get_forward_distribution` exists nowhere, contradicting
  the module's own "pure function, caller supplies the forward distribution"
  principle. Rewrote to the real API:
  `best_available_forward_distribution(ohlcv, horizon_days, as_of)` from
  `engine.forward_distribution` (signature verified at `forward_distribution.py:315`),
  feeding `EVEngine().evaluate(trade, forward_log_returns=...)`.
- **wheel_tracker.py:2082** — the R9 gate comment said
  "armed by enforce_sector_cap (default on)". Verified the opposite:
  `enforce_sector_cap` defaults **False** (`wheel_tracker.py:302`), and it is armed
  `=True` only by `wheel_runner.make_live_book_tracker()` (`wheel_runner.py:865-866`)
  for production / live books. Corrected the comment. (Operator-greenlit
  decision-layer comment touch — the one `PROJECT_STATE` §3 defers.)

## How we fixed it
Each fix verified against source before editing (link-target existence, real
function signature, actual default value + arming site). Comment/docstring text
only — no control-flow or value change.

## Evidence
- `python -c "import engine.candidate_dossier, engine.ev_engine, engine.wheel_tracker, engine.wheel_runner"` → imports OK.
- `ruff format --check` + `ruff check` on the 3 files → clean.
- `pytest -m "not backtest_regression"` (full suite per CLAUDE.md §4) →
  **3468 passed, 28 skipped, 4 deselected, 16 xfailed, 0 failed** (392.9s).
- §2 invariant untouched: no `EVEngine.evaluate` bypass, downgrade-only contract
  intact, dealer clamp unchanged — the edits are text-only.

## Unresolved / handoff
- None. Separate audit finding (not this PR): `ev_engine.py:123` comment names
  `engine.regime_detector` for the `regime_multiplier` field though the live
  source is the HMM — left for the engine/ Phase-2 block.
