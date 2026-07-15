# Codebase weakness audit — 2026-07-15 (register)

**Status: COMPLETE (2026-07-15).** All three streams folded; consolidated
ranking + pre-registered proposals written. Read-only audit; nothing applied to any engine code.
The decision-layer trio (`engine/ev_engine.py`, `engine/wheel_runner.py`,
`engine/candidate_dossier.py`) is never edited by this audit. Coordination hub:
**issue #493**. Fixes are a separate, governed, consented step — this document
is a findings register; ready-to-approve proposals are drafted separately.

Three parallel streams feed the register:

| Stream | Source | State |
|---|---|---|
| **A — Structural** | Computer 1 · `codebase-memory-mcp` graph + grep | **DONE** (`RESULT C1-001`, #493) |
| **B — Runtime** | Computer 2 · execution / coverage / fuzz | **DONE** (`RESULT C2-001`, #493) |
| **C — Breadth** | in-session multi-agent workflow · ~12 dimensions, 3-skeptic adversarial verify | **DONE** (237 agents; 70→46→43 ranked) |

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

3. **Runtime confirms the engine is conservative, deterministic, and hardened
   — with one cross-cutting theme: observability.** Stream B: engine coverage
   **89%** (the authoritative `evaluate` path **96%**, every R1–R11
   primary/downgrade branch test-covered); the 5-ticker A/A run is
   **byte-identical across all 60 columns**; quant-primitive fuzzing found
   **zero genuine defects** (every primitive fails loud on degenerate input or
   propagates NaN by a test-pinned contract that R1a then catches). The
   machine-checked §2 "no R-rule inverts severity" property — which the C2 host
   could not run (`hypothesis` absent) — **was executed in-sandbox and PASSES
   (79 property cases).** The only actionable runtime items are *observability*
   (the ranker refuses via an empty frame with no per-ticker reason, so a
   data-staleness/missing-earnings lockout is indistinguishable from a real
   one — conservative, no bad-trade risk, but opaque) and *test-infra* (get
   `hypothesis` onto the runtime host; one env-sensitive deep-read test).

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

## Stream B — Runtime (Computer 2) — DONE

Ran on HEAD `e06e26a`, connector `MarketDataConnector`, provider `bloomberg`
(logged — no silent switch). Suite: **3463 passed / 34 skipped / 1 failed**
(minus 6 files: 2 need `hypothesis`, absent on that host; 4 slow data/e2e
buckets), **engine total coverage 89%**. Read-only; nothing committed. Full
detail: `RESULT C2-001` on #493.

| # | Finding | File:line | Severity | Verdict |
|---|---|---|---|---|
| B1 | **Silent refusal opacity** — the ranker returns an empty frame with NO per-ticker reason, so a data-staleness / missing-forward-earnings fail-safe lockout is indistinguishable from a real event lockout (4 mega-caps refused at a near-frontier `as_of` because their earnings calendar ends Q1-2026 → `next_earnings=None`, refused with no warn/log) | `wheel_runner.rank_candidates_by_ev` return boundary; event gate | **MEDIUM** | conservative (fails toward not-trading; no bad-trade risk) but opaque |
| B2 | §2 downgrade-lattice property test could not run on the C2 host (`hypothesis` absent) → the machine-checked "no R-rule inverts severity" invariant was unverified there. **RESOLVED in-sandbox: 79 property cases PASS** — invariant holds; residue is a host-env test-infra gap | `tests/test_dossier_downgrade_property.py`, `tests/test_properties.py` | LOW (was MED; host-env only) | verified elsewhere → PASS |
| B3 | 1 failing test — `test_deep_read_connector::test_deep_on_without_slices_degrades_to_monolith`; data-state-dependent (deep slices present on host violate the "no slices" premise); theta/deep-read plumbing, **not** the decision layer | `tests/test_deep_read_connector.py` | MEDIUM (env-sensitive) | not a decision-path defect |
| B4 | Silent refusals elsewhere — thin-history (<504d), missing-IV, bogus ticker all refuse with no warn/log (same opacity theme as B1) | ranker refusal paths | LOW | correct refusals, opaque |
| B5 | `horizon_days=0` raises an incidental numpy "slice step cannot be zero" instead of a purpose-built "horizon ≥ 1" message; never occurs in prod (dte=35) | `forward_distribution.empirical_forward_log_returns` | LOW (cosmetic) | fail-loud but unpolished |

**Reassuring (no defect):** the authoritative `evaluate` path is **96%** covered
and every R1–R11 primary/downgrade branch has a covering test (only benign
notes + defensive `except` handlers uncovered — none can flip a verdict); the
A/A run is **byte-identical across all 60 columns**; quant-primitive fuzzing
found **zero genuine defects** (fail-loud `ValueError` on degenerate
HMM/pricing input, or the deliberate test-pinned NaN-propagation contract in
the pricer that R1a catches downstream); coverage by module: `ev_engine` 96 /
`candidate_dossier` 91 / `dealer_positioning` 90 / `wheel_tracker` 86 /
`wheel_runner` 79 (the short-put path is exercised; the 79% gaps are the
covered-call / strangle / roll / diagnostic branches).

**Net for Stream B:** no correctness defect on the decision path; the engine is
conservative and deterministic. The actionable items are **observability**
(B1/B4 — refusals are opaque at the return boundary) and **test-infra** (B3, and
getting `hypothesis` onto the runtime host — B2's invariant itself already
passes here).

---

## Stream C — Breadth (in-session workflow) — DONE

237 agents, ~3.4h, 12.3M tokens across ~12 dimensions; **70 raw findings → 46
survived 3-skeptic adversarial refutation → 43 ranked** (2 high / 22 medium / 19
low; 11 touch the trio). Every survivor passed ≥2-of-3 refuting verifiers.

**The EV-authority core held.** No finding shows a negative-EV trade being
rescued; the dealer multiplier clamp is intact; R1–R11 remain downgrade-only.
The two HIGH defects both live **outside** the protected trio, in the
risk-metric and interface layers *around* the ranker. The trio-touching findings
are integrity/quality gaps (a dead reviewer, PIT carry-q, sizing), not invariant
breaks. Roughly a third of the register is backtest-only performance or doc
drift with no runtime-correctness impact.

---

## Consolidated ranking — top findings (all streams)

Full 43-row Stream-C list + the 10 pre-registered fix specs are in
**`docs/CODEBASE_AUDIT_2026-07-15_PROPOSALS.md`**. The two HIGH items were
spot-confirmed against source. **Nothing is applied to any code from this
document** — fixes are the operator's to approve, one at a time.

| Rank | Sev | Trio | File:line | Finding |
|---|---|---|---|---|
| **1** | **HIGH** | — | `engine/risk_manager.py:363` | `gamma_dollars` carries a stray `/100` while `delta_dollars` (`:362`) does not → portfolio convexity P&L understated ~100×, gutting short-gamma VaR/CVaR and defeating the R7/R8 soft-warns. **Confirmed in source.** |
| **2** | **HIGH** | — | `engine_api.py:646` | CORS allow-list uses boundary-less `origin.startswith('http://localhost')`, so `http://localhost.evil.com` is echoed into `Access-Control-Allow-Origin` → a browser page can read live IBKR portfolio JSON. **Confirmed in source.** |
| 3 | MED | ✓ | `engine/candidate_dossier.py:366` | R6 dealer-positioning downgrade is **dead in production**: `build_dossiers` never threads `market_structure`, so R6's guard is always False. The §2-documented short-gamma-at-put-wall review can't fire (only the residual dealer multiplier protects it). **Confirmed.** Fails toward less protection — not an invariant break. |
| 4 | MED | ✓ | `engine/wheel_runner.py:3082` | CC + strangle rankers price BSM carry-`q` off the **dateless** fundamentals snapshot (no `as_of`), unlike the puts ranker — dated CC/strangle backtests use the 2026 dividend yield. |
| 5 | MED | ✓ | `engine/wheel_runner.py:899` | Provider selected from `SWE_DATA_PROVIDER` with **no log**; an unrecognized value silently falls back to Bloomberg — the exact silent-provider class CLAUDE.md §4 warns about. |
| 6 | MED | — | `data/bloomberg_loader.py:1100` | `get_current_risk_free_rate` silently returns hard-coded **0.05** on missing/empty rates (the unfixed data-twin of the engine's removed silent-5% bug). |
| 7–9 | MED | — | `engine/stress_testing.py:467/791`, `engine/wheel_tracker.py:2624` | Stress MC t-draws over-dispersed ~29% (unnormalized Student-t); `greeks_scenario_matrix` reports **annual** theta vs the engine's daily; roll-suggester `hold_ev` carries a phantom entry commission. |
| 10 | MED | ✓ | `engine/candidate_dossier.py:526` | Dossier soft-warns R7–R10 size every candidate as **1 contract** (put ranker doesn't emit `contracts`), understating multi-contract concentration. |
| 11–24 | MED | mixed | data-layer / interface / perf | IV-rank PIT-vs-live window divergence, slash-share-class carry-q, several unauthenticated/fail-open API endpoints, path-traversal params, hot-path recompute. |
| 25–43 | LOW | mixed | across | IV clamp/convergence quirks, ddof mix, sign bugs off the EV path, DoS/rate-limit gaps, doc/state drift, FRED key in logs, pickle load. |

**Bottom line:** the ranking invariant is sound; the two most dangerous defects
are in the **risk-metric (VaR gamma)** and **interface (CORS)** layers around it.
Recommended first approvals: proposal #1 (gamma `/100`) and #2 (CORS) — both
non-trio, both with drafted specs and tests.
