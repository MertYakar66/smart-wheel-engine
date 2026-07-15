# Codebase weakness audit — 2026-07-15 (register)

**Status: IN PROGRESS.** Read-only audit; nothing applied to any engine code.
The decision-layer trio (`engine/ev_engine.py`, `engine/wheel_runner.py`,
`engine/candidate_dossier.py`) is never edited by this audit. Coordination hub:
**issue #493**. Fixes are a separate, governed, consented step — this document
is a findings register; ready-to-approve proposals are drafted separately.

Three parallel streams feed the register:

| Stream | Source | State |
|---|---|---|
| **A — Structural** | Computer 1 · `codebase-memory-mcp` graph + grep | **DONE** (`RESULT C1-001`, #493) |
| **B — Runtime** | Computer 2 · execution / coverage / fuzz | PENDING (`COMMAND C2-001` open) |
| **C — Breadth** | in-session multi-agent workflow · ~12 dimensions, 3-skeptic adversarial verify | RUNNING |

Method: every finding is grep- or verify-confirmed; the consolidated severity
ranking is assembled once all three streams land.

---

## Headline results so far

1. **§2 invariant STRUCTURALLY RE-CONFIRMED (independent).** Stream A Task 3
   traced the decision layer end-to-end: `ev_raw` (`ev_engine.py:431`) is
   computed once and never re-assigned; the dealer multiplier is clamped to
   **[0.70, 1.05]** (`:589`) and folded in *only* via `regime_mult *=
   dealer_mult` (`:590`); `ev_dollars = ev_raw * regime_mult` (`:604`) — so the
   multipliers touch `ev_dollars` only, never `ev_raw`; and `EVResult` is
   constructed at just two sites (`:323`, `:606`), **both inside `def
   evaluate`**, so no verdict is reachable without a fresh `evaluate`. This is a
   fresh, independent structural confirmation of the CLAUDE.md §2 hard invariant
   — **no defect.**

2. **TOOLING FINDING (actionable) — the `codebase-memory-mcp` graph has a
   systematic CALLS-recall gap.** On the pre-registered §2 shakeout the graph
   returned **6 of 10** true `EVEngine.evaluate` callers (missed
   `ev_engine.py:83`, `wheel_runner.py:4033`, `wheel_tracker.py:2620`, `:2994`)
   and, worse, flagged two live functions as dead: the §2-critical R1–R11
   reviewer **`EnginePhaseReviewer.review`** (live at `candidate_dossier.py:678`)
   and **`calculate_assignment_costs`** (`transaction_costs.py:311`, 6 real
   callers). Every graph error is a **miss, never a fabrication** — but a
   blast-radius or dead-code decision taken on the graph alone would have
   under-counted §2 call sites and declared the most important reviewer dead.
   **Operational rule, now empirically proven: the graph is a free first pass;
   grep is the verdict.** Never delete or assert "unused" from the graph alone.
   (Mechanisms: CALLS-edge de-duplication drops repeat call-lines; inline-
   constructor receivers (`EVEngine().evaluate()`) unresolved; typed-local
   receivers (`reviewer.review(...)`) unresolved.)

---

## Stream A — Structural (Computer 1) — DONE

Graph re-indexed to HEAD `22ba653`, `is_worktree:false`, **21,357 nodes /
104,929 edges**. Read-only; nothing committed on that machine. Full detail:
`RESULT C1-001` on #493.

| # | Finding | File:line | Severity | Verdict |
|---|---|---|---|---|
| A1 | §2 invariant holds: multipliers feed `ev_dollars` only; no `EVResult` without `evaluate`; dealer clamp [0.70,1.05] | `ev_engine.py:431/589/590/604` | — (confirmation) | PASS |
| A2 | Graph CALLS-recall gap — 6/10 §2 callers; two live fns flagged dead (`EnginePhaseReviewer.review`, `calculate_assignment_costs`) | `candidate_dossier.py:678`, `transaction_costs.py:311` | **MEDIUM (tooling)** | graph-WRONG, grep-caught |
| A3 | No production function confirmed safe-to-delete. 2 dormant-by-design clusters (`signal_context`, `dependency_check`) match MODULE_INDEX; 7 low-confidence orphan leads, all documented-dormant or off-EV-path (SVI/D9) | `signal_context.py`, `dependency_check.py`, `volatility_surface.py`, `performance_metrics.py`, … | LOW / INFO | graph-agrees-doc |
| A4 | No MODULE_INDEX doc-vs-reality drift — every documented role (dormant / live / D9-parked / D18-severed) matches the code | MODULE_INDEX.md | — | accurate |
| A5 | Import cycles: 2 (`portfolio_risk_gates↔wheel_tracker`, `mcp_client↔tradingview_bridge`), **both runtime-safe by design** (TYPE_CHECKING + lazy/function-local); 0 cycles of length >2 | `portfolio_risk_gates.py:52/292`, `wheel_tracker.py:1871` | INFO | grep-confirmed |

**Net for Stream A:** the decision-layer invariant is independently
re-confirmed; there is no dead code to remove, no doc drift, and no problematic
import cycle. The single actionable item is A2 — a property of our *graph tool*,
not of the engine — captured in Headline #2.

---

## Stream B — Runtime (Computer 2) — PENDING

*(Awaiting `RESULT C2-001` on #493: decision-path coverage map, quant-primitive
fuzz failures, engine stress-run NaN/warnings, determinism A/A.)*

---

## Stream C — Breadth (in-session workflow) — RUNNING

*(Awaiting the multi-agent audit across ~12 dimensions — invariant integrity,
decision-logic correctness, quant/numerical, PIT/look-ahead, silent failures,
test-coverage, data-layer, doc/state drift, interface, secrets, performance —
each finding adversarially verified by 3 refuting skeptics.)*

---

## Consolidated ranking — PENDING

*(Assembled once all three streams land: deduped, ranked by severity ×
blast-radius, each with a ready-to-approve pre-registered proposal. Nothing is
applied to the engine from this document.)*
