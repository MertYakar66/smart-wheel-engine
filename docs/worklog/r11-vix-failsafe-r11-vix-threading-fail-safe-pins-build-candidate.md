---
id: r11-vix-failsafe
title: R11 VIX-threading fail-safe pins (build_candidate_dossiers)
kind: verification
status: in-flight
terminal: ultracode
pr:
decisions: []
date: 2026-05-31
headline: Pinned the best-effort VIX-threading fail-safe on the live R11 ranking path (build_candidate_dossiers) — 5 connector-degradation cases + 2 anti-vacuity teeth, mutation-tested both directions. Closes backlog theme ② from the #308 worklog.
surface:
  - tests/test_r11_elevated_vol.py
---

## Goal
Close backlog **theme ②** from the `dossier-downgrade-property` (#308) worklog —
flagged there as NEXT PRIORITY because it is a live crash-risk in just-shipped
code, not a hypothetical one. R11 (#306/#307) downgrades a top-bin candidate
`proceed→review` when market-wide VIX > 25. `WheelRunner.build_candidate_dossiers`
threads the PIT VIX *level* into the dossier reviewer via a best-effort
`try/except` around `connector.get_vix_regime(as_of).get("vix")`. VIX is advisory,
so any connector failure must degrade R11 to a no-op (`vix_level=None`) and never
fail the rank. That fail-safe shipped with #306 but had **no test pinning it** — one
refactor that narrows or drops the `except` re-introduces a crash on the live
ranking path. Lock the contract.

## What we tried
1. **Located the fail-safe precisely.** Corrected the #308 backlog note: the
   threading lives in `build_candidate_dossiers` (`engine/wheel_runner.py:3357–3373`),
   **not** `rank_candidates_by_ev` (which never calls `get_vix_regime`).
   `build_candidate_dossiers` ranks first (`:3332`), threads VIX (`:3358`), then
   calls `build_dossiers(..., vix_level=…)` (`:3365`); `build_dossiers` attaches
   `vix_level` to every `CandidateDossier` and R11 reads `dossier.vix_level`.
2. **Surgical fixture over full-data integration.** The existing
   `build_candidate_dossiers` tests (`test_tv_dossier.py`) drive the *real* ranker
   over synthetic OHLCV — but that can't deterministically force
   `prob_profit > 0.90`, so R11's teeth couldn't fire on demand. Instead: construct a
   real `WheelRunner`, set `_connector` directly (the `connector` property is lazy,
   so the real connector is never built), monkeypatch `rank_candidates_by_ev` to a
   canned 1-row R11-eligible frame, and pass a clean-chart provider. This isolates
   *exactly* the threading + degradation logic; the ranker and data layer are tested
   elsewhere.
3. **Home = `tests/test_r11_elevated_vol.py`** (the dedicated R11 file). Already
   manifest-listed, so no `FILE_MANIFEST` churn and no new-file coverage-gate timing
   trap (the trap that bit #308). Its sibling
   `test_build_dossiers_threads_vix_level_and_r11_fires` pins the *dossier*-side
   threading; these new pins cover the `wheel_runner` side.

## What worked
- **7 pins.** Five degradation cases — `get_vix_regime` raises / returns `None`
  (→ `None.get` `AttributeError`) / returns a vix-less `{}` (→ `_v is None`, no
  exception) / returns non-numeric `{"vix": "not-a-number"}` (→ `float()`
  `ValueError`) / connector lacks the method (→ `hasattr` False) — each degrades to
  `vix_level=None` with R11 dormant and the rank completing. Plus two **anti-vacuity
  teeth**: a working `{"vix": 30.0}` + `prob_profit=0.95` fires R11
  (`review`/`elevated_vol_top_bin`), and a working `{"vix": 15.0}` threads the real
  level through but R11's own `> 25` gate holds (`proceed`).
- **The teeth are the key vacuity guard.** They prove R11 is *reachable* in this
  fixture, so the dormant cases are dormant because the connector failed — not
  because R11 was unreachable.

## What didn't
- First draft used a function-local `from engine.wheel_runner import WheelRunner`
  plus a redundant local `import pandas as pd` inside the shared helper. The scope
  auditor flagged both; since the helper is module-level and called 7×, module-level
  imports are cleaner — moved `WheelRunner` up to the import block, dropped the
  redundant `pandas`.

## How we fixed it
Shipped the 7 pins in `tests/test_r11_elevated_vol.py` (test-only — no engine edit,
so no §2 second-read and the lane-claim gate does not fire; it triggers on trio
*source* edits, not tests).

## Evidence
- `pytest tests/test_r11_elevated_vol.py` → **15 passed** (8 existing + 7 new) in
  0.58 s.
- **Mutation test (the teeth proof), both directions, in the worktree then reverted:**
  - Drop the fail-safe (`except Exception: raise`) → the 3 exception-path tests
    **FAIL** (RuntimeError / AttributeError / ValueError propagate); the 4
    non-exception cases correctly still pass (they don't depend on the `except`).
  - Silently kill the threading (`vix_level = None` always) → **both teeth tests
    FAIL** (R11 can't fire; `vix_level != expected`).
  - `git checkout -- engine/wheel_runner.py` after each; `git status` confirmed
    `engine/` clean.
- Full suite (`pytest tests/ -m "not backtest_regression"`): **2697 passed, 3
  failed, 2 xfailed** in 275 s. `2697 = 2690` (#308 baseline) `+ 7` new. The 3
  failures are the known Windows-local-only `tests/test_theta_connector.py` cases
  (no live Theta Terminal on this box; they pass on Ubuntu CI).
- `ruff format` + `ruff check`: clean. Single-file diff (`tests/test_r11_elevated_vol.py`
  only). `check_lane_claim.py`: no decision-layer source touched.
- **Independent 3-agent verification** (read-only, against worktree source @
  `origin/main 26fda24`): re-grounding skeptic **SOUND** (12/12 claims confirmed with
  line-level evidence — every connector stub → exact code path → asserted outcome,
  constants `R11_VIX_THRESHOLD=25.0` / `R11_TOP_BIN_PROB=0.90`, strictly-`>`,
  `_R11_ELIGIBLE_ROW` reaches `proceed` at R5); scope auditor `scope_clean=true`,
  `engine_files_touched=[]`, **NOT vacuous**; completeness critic
  `complete_enough`, no additions.

## Unresolved / handoff
- **Theme ③** (the other open item from #308) remains: the R11 **strangle-path
  coverage gap**. The timing-gated strangle path emits *prefixed*
  `put_prob_profit` / `call_prob_profit` (`wheel_runner.py:3220–3221`), which R11's
  unprefixed `prob_profit` read would miss. A focused test or a key-normalisation
  fix; secondary path, lower priority than theme ② was.
- **Out-of-scope VIX fail-safe siblings** (noted by the completeness critic, NOT part
  of the R11 ranker path, not pinned here): `wheel_runner.py:559` (per-ticker analyze
  path) and `strangle_timing.py:869` (`score_entry_with_iv`). Each has its own
  `try/except`; pin only if they become load-bearing.
