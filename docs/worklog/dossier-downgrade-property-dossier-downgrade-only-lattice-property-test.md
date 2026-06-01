---
id: dossier-downgrade-property
title: Dossier downgrade-only lattice property test
kind: verification
status: in-flight
terminal: ultracode
pr:
decisions: []
date: 2026-05-31
headline: §2 downgrade-only severity lattice stated as one property over R1–R11, plus source-introspection meta-tripwires that force any future rule into the matrix; 3-theme test backlog preserved.
surface:
  - tests/test_dossier_downgrade_property.py
---

## Goal
Mechanize the manual *"did any rule invert severity?"* review that every
decision-layer change currently relies on a human to catch by eye (`CLAUDE.md`
§2). Replace scattered per-rule examples with ONE property over the whole
`EnginePhaseReviewer.review()` chain, and make it self-extending so a future
rule (R12+) added without severity protection trips CI for free.

## What we tried
1. **Enumerated matrix as primary.** `{overlay} × {ev_dollars}` with the exact
   ev probe set from the card (`-50, NaN, +inf, 5, 50, None`), asserting
   `severity(with_overlay) >= severity(without)` per cell. Chosen over a
   pure-Hypothesis design so the core §2 invariant still runs when Hypothesis is
   absent (the suite guards Hypothesis behind `HYPOTHESIS_AVAILABLE`).
2. **Hypothesis only on the ev axis.** Generating overlay *objects* via
   Hypothesis was rejected — too complex/slow (each R7 example builds a 120-row
   returns frame; R9/R10 run real sector math). Instead Hypothesis fuzzes
   `ev_dollars` across the whole real line (incl. nan/±inf) over the *cheap*
   overlays (R6/R8-dealer/R11); the deterministic matrix covers the expensive
   ones exhaustively.
3. **Meta-tripwires via `ast`.** Introspect `review()`'s source to count
   overlay guards, prove no branch hard-returns `"proceed"`, and prove every
   overlay `verdict_reason` maps to a firing scenario here.

## What worked
- **48-cell matrix + firing-cell teeth.** `test_overlay_actually_downgrades_at_proceed`
  asserts each scenario *strictly* downgrades `proceed → review` with the exact
  reason at ev=50, so a scenario that silently stops firing can't make the
  monotonic matrix vacuously true.
- **Duck-typed `MarketStructure` stub for R6.** `review()` only `getattr`s
  `.regime` and `.nearest_put_wall.strike`, so a stub exercises the exact R6
  path without coupling to the real `dealer_positioning.MarketStructure`
  constructor.
- **Reused on-main firing recipes.** R7–R10 configs are lifted verbatim from the
  passing examples in `tests/test_dossier_invariant.py`, so they stay in
  lockstep with what the engine actually scores as a breach.
- **Lattice pinned to the type.** `test_severity_lattice_matches_verdict_literal`
  asserts `SEVERITY.keys() == typing.get_args(Verdict)` so the property can't go
  unsound if a new verdict string is added.

## What didn't
- Trying to separate overlay returns from R2's `chart_context_missing` (the one
  non-overlay rule that also returns `"review"`) by AST **line position** was
  brittle. Settled on a documented `_NON_OVERLAY_REVIEW_REASONS = {"chart_context_missing"}`
  exclusion set — explicit and obvious to maintain.

## How we fixed it
Shipped `tests/test_dossier_downgrade_property.py` (test-only — **no engine
edit**, so no §2 second-read required and the lane-claim gate does not fire; it
triggers on trio *source* edits, not tests). Structure: severity lattice +
`EV_VALUES` probe + 8 `OVERLAY_SCENARIOS` (R6×2, R7, R8×2, R9, R10, R11) →
parametrized monotonic matrix → firing-cell teeth → `cannot_rescue_blocked` →
Hypothesis ev-fuzz → 5 META tripwires.

## Evidence
- `pytest tests/test_dossier_downgrade_property.py` → **64 passed in 1.14s**
  (48 matrix + 8 firing + 1 rescue-blocked + 1 hypothesis + 1 lattice + 5 meta).
- **Tripwire mutation check** (throwaway, fed synthetic mutated `review()`
  sources to the AST helpers): a planted `return "proceed", …` (upgrade), a
  planted R12 with a new reason + extra guard, an unknown verdict literal, and
  an overlay returning `"skip"` — **all four caught**. The real `review()` gives
  6 proceed-guards, no `"proceed"` hard-return, and overlay reasons exactly equal
  the scenario set.
- Grounded against `engine/candidate_dossier.py` @ `origin/main 2495cda`
  (R11 live via #306/#307).
- Full suite (`pytest tests/ -m "not backtest_regression"`): **2690 passed,
  3 failed, 2 xfailed** in 297s. The 3 failures are all
  `tests/test_theta_connector.py` (`472 Client Error` — no live Theta Terminal
  on this Windows box); known Windows-local-only failures unrelated to this
  test-only change (they pass on Ubuntu CI). New file passes 64/64.
- Independent verification (3-agent workflow, read against the worktree source):
  lattice re-derivation matched (verdicts, 6 guards, 8 overlay reasons, no
  `proceed` hard-return — **zero mismatches**); adversarial §2 bug-hunt
  **clean** (no inversion; R11's `except → pp,vix=0.0` is a hold, not an
  upgrade); test audit SOUND — fixed its one real note (the cannot-rescue test
  now attaches the three independent surfaces honestly rather than claiming all
  eight branches at once).
- `ruff format` + `ruff check`: clean. `check_lane_claim.py`: no decision-layer
  source touched → gate not triggered (test-only).

## Unresolved / handoff
This PR is **theme ①** of a 3-theme test direction surfaced while grounding the
R11-era decision layer. The other two are NOT built here — preserved so they
aren't lost:

**Theme ② — `get_vix_regime`-raises fail-safe on the ranker R11 path. → NEXT
PRIORITY (live crash risk in just-shipped code, not a hypothetical future one).**
`wheel_runner.py:3357–3362` threads the PIT VIX level into the dossier reviewer
via `self.connector.get_vix_regime(as_of).get("vix")` inside a broad
`try/except Exception → vix_level = None`. The fail-safe (a raising — or
`None`-returning, which makes `.get` raise `AttributeError` — connector degrades
R11 to a no-op and never blocks the rank) **exists but has no test pinning it**.
One refactor that drops/narrows that `except` re-introduces a crash on the live
ranking path. A test should inject (a) a connector whose `get_vix_regime` raises
and (b) one returning `None`, and assert `rank_candidates_by_ev(...)` completes
with `vix_level → None` (R11 dormant), not propagating.

**Theme ③ — R11 ranker-path coverage gap.** R11
(`candidate_dossier.py:534–566`) fires only when BOTH `vix_level > 25` AND
`ev_row["prob_profit"] > 0.90`. The ranker threads `vix_level`
(`wheel_runner.py:3357–3373`) but emits `prob_profit` under **different keys on
different paths**: unprefixed `"prob_profit"` at `wheel_runner.py:1698` & `2600`,
but **prefixed** `"put_prob_profit"` / `"call_prob_profit"` at `3220–3221`. R11
reads the unprefixed key only. If the rows reaching the vix-threaded
`build_dossiers` carry only the prefixed keys, `ev_row.get("prob_profit", 0.0)`
→ `0.0` → **R11 can never fire live** despite correct VIX threading. Theme ③ =
an end-to-end test asserting R11 actually fires on the ranker path (or a fix +
test that closes the prefix gap). Needs tracing which ranker method feeds the
vix-threaded dossier build — **not resolved in this PR** (flagged, not fixed).

**Corrected facts (preserve — these were stated wrong in earlier triage):**
- `prob_profit = float(np.mean(pnls > 0))` at **`ev_engine.py:393`** — a plain
  empirical hit-rate over the simulated P&L paths. There is **no GPD / POT tail
  term** in `prob_profit`; the GPD/tail tooling feeds `cvar_5` and tail risk
  elsewhere, never `prob_profit`. Any test/doc asserting a GPD term in
  `prob_profit` is wrong.
- The R11 ranker-path gap above (theme ③) is the live integration risk to
  confirm, distinct from the dossier-side R11 logic this PR's matrix already
  pins.

Scope note: this property pins the **dossier-side** §2 invariant. The
**ranker-side** authority (no negative-EV rescue inside
`rank_candidates_by_ev`) is covered separately by `test_authority_hardening.py`
and `test_audit_viii_*`.
