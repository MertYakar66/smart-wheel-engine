# Launch Readiness

The contractual checklist that pins what "ready to ship a
decision-layer change" means. Every item here is a **gate** — break
it and the change is unsafe regardless of what other tests pass.

This file is the operational consolidation of:
- `CLAUDE.md` §2 (the hard EV invariant)
- `MODULE_INDEX.md` (authoritative vs reviewer roles)
- `TESTING.md` (launch-blocker subset)
- `DECISIONS.md` D1 (EV is the only ranker)
- `tests/test_launch_blockers.py` and the `test_audit_*` family

---

## 1. The hard invariant (`CLAUDE.md` §2)

> **No tradeable candidate bypasses `EVEngine.evaluate`.**

Reviewers (chart provider, news sentiment, advisor committee, dealer
positioning, TradingView bridge) can downgrade a verdict
(`proceed → review → skip → blocked`). None of them can rescue a
negative-EV trade. The dealer multiplier is clamped to
`[0.70, 1.05]` and only scales `ev_dollars`; it never touches
`ev_raw`.

If you add a new input (new data source, advisor, MCP feed, sentiment
model), wire it as a chained-provider participant or a downgrade-only
reviewer. Never as a code path that converts non-tradeable → tradeable
without a fresh `EVEngine.evaluate` call.

---

## 2. The four authoritative routes

These are the **only** sanctioned paths from raw inputs to a
tradeable verdict:

| Module | Public entry | Pinned by |
|---|---|---|
| `engine/ev_engine.py` | `EVEngine.evaluate` (line 234) | `tests/test_audit_invariants.py`, `tests/test_audit_viii_*.py` |
| `engine/wheel_runner.py` | `WheelRunner.rank_candidates_by_ev` (line 445) | `tests/test_authority_hardening.py`, `tests/test_audit_viii_real_data_smoke.py` |
| `engine/candidate_dossier.py` | `EnginePhaseReviewer` (line 113), rules R1–R6 | `tests/test_dossier_invariant.py` |
| `engine_api.py` | HTTP API on `:8787`; 29 endpoints | `tests/test_tv_api.py`, `tests/test_tv_dossier.py` |

Any change touching a file under `engine/` that affects these routes
**must** run the full launch-blocker subset (§4 below) and the full
suite for `engine/ev_engine.py` or `engine/wheel_runner.py` changes
(invariants are cross-cutting).

---

## 3. Dossier downgrade rules (R1–R6)

`EnginePhaseReviewer` is the gatekeeper that converts an EV verdict
+ chart context into the final disposition. The six rules
(`engine/candidate_dossier.py:113`, pinned by
`tests/test_dossier_invariant.py`):

| Rule | Trigger | Effect |
|---|---|---|
| **R1** | EV is negative | **blocked** (hard stop) |
| **R2** | Chart missing | **review** |
| **R3** | Spot mismatch > 2% between engine-side and chart-side | **skip** |
| **R4** | Phase contradiction (Pine signal disagrees with engine) | **skip** |
| **R5** | EV above threshold | **proceed** |
| **R6** | Short-gamma regime + strike at/above the put wall | **downgrade** |

R1 is the structural realisation of §1 — negative EV ⇒ blocked. The
test is the merge gate.

---

## 4. Launch-blocker test subset

Run **every** time you change anything under `engine/` that could
touch the decision layer. This is the floor before merge:

```bash
pytest tests/test_audit_invariants.py \
       tests/test_dossier_invariant.py \
       tests/test_authority_hardening.py \
       tests/test_audit_viii_unit_invariants.py \
       tests/test_audit_viii_e2e.py \
       tests/test_audit_viii_real_data_smoke.py \
       tests/test_launch_blockers.py -v
```

For changes to `engine/ev_engine.py`, `engine/wheel_runner.py`, or
`engine/candidate_dossier.py`: **also run the full suite**:

```bash
pytest tests/ -v
```

Invariants are cross-cutting. Subsetting is a false economy when the
authority files change.

---

## 5. Pre-merge checklist (decision-layer changes)

- [ ] The hard invariant from §1 still holds — no new code path
      converts a non-tradeable candidate into a tradeable one
      without `EVEngine.evaluate`.
- [ ] If a new reviewer was added, it joins `ChainedChartProvider`
      or `EnginePhaseReviewer` as **downgrade-only**.
- [ ] If a new multiplier was added, it's clamped (`[low, high]`)
      and only scales `ev_dollars`, never `ev_raw`.
- [ ] All R1–R6 dossier rules still pass
      (`tests/test_dossier_invariant.py`).
- [ ] The launch-blocker subset (§4) passes locally.
- [ ] If the change touches percent↔decimal handling (IV, risk-free
      rate, etc.) — `tests/test_audit_viii_unit_invariants.py`
      passes. Audit-VIII shipped 20 tests around this; do not
      regress.
- [ ] If the change touches the TradingView path —
      `tests/test_tv_signals.py::test_pine_parity_constants`
      passes (Pine ↔ engine constant parity).
- [ ] If the change touches `wheel_tracker.py` — the rolled-position
      P&L accumulator tests pass; the three ledgers
      (`realized_pnl`, `transaction_costs`, `stock_basis`) remain
      orthogonal (audit-VIII fix).
- [ ] `PROJECT_STATE.md` is updated if the change shifts what is
      "authoritative" or "deprecated".
- [ ] `CHANGELOG.md` has an entry under the current month.
- [ ] If the change retired a documented behaviour, the
      corresponding entry in `DECISIONS.md` is updated (not deleted)
      with a `**SUPERSEDED by D<N>**` note.

---

## 6. Pre-launch checklist (rare; for going live with new data /
new strategy / new broker integration)

These are the gates beyond the per-PR checklist. Reach for this
list when shipping something genuinely user-facing:

- [ ] `audit.py` smoke tests pass against a running
      `engine_api.py` on `localhost:8787`.
- [ ] `scripts/feature_smoke_test.py` reports the expected pass /
      fail / skip distribution for the current data tier
      (~111 PASS / 0 FAIL / 16 SKIP after the 2026-05-04 refresh per
      `PROJECT_STATE.md` §3.4 — adjust as data tiers shift).
- [ ] On a Theta-up laptop:
      `python scripts/diagnose_candidates.py` (full universe, not the
      Cowork 5-ticker shim) produces a candidate funnel without
      surprises (no zero-trade ticker that should be tradeable).
- [ ] `python scripts/probe_theta_capabilities.py` regenerates
      `data_processed/theta_capabilities.json` and the persistent
      failure set matches the documented tier ceiling
      (currently: BF.B / BRK.B / NVR / DAY — upstream-data gap, not
      a code bug).
- [ ] If broker / OMS surface is being introduced (which is
      out-of-scope per `CLAUDE.md` §4 — get explicit consent first):
      a separate launch-readiness review against the new surface,
      not just this checklist.
- [ ] The dashboard (`dashboard/src/`, Next.js) still builds and
      hits the `:8787` API without errors.

---

## 7. What is *not* a launch gate

Useful to be explicit so reviewers don't over-block:

- **Coverage threshold drift** is informational, not blocking. The
  invariants matter; the line count doesn't (`DECISIONS.md` D10).
- **Doc drift** in `README.md`, `docs/ARCHITECTURE.md`, etc. is
  tracked in `ROADMAP.md` Track B — known and being repaired.
- **Deprecation warnings** in the test output (currently 287 — down
  from 578 — per `PROJECT_STATE.md` §2) are not blockers; they're a
  known cleanup queue.
- **The `iv_surface` integration decision** is open
  (`ROADMAP.md` A2) — shipping any code path that *uses* the SVI
  tooling does require resolving that decision first; everything
  else is unaffected.

---

## 8. Sandbox-vs-laptop launch caveats

Some launch-readiness checks are simply impossible without the Theta
Terminal. Cowork sandbox can:

- Run the full `pytest` suite
- Run `WheelRunner.rank_candidates_by_ev` on a 5-ticker explicit list
- Run `audit.py` against an `engine_api.py` started in the same
  sandbox

Cowork sandbox cannot:

- Run `scripts/diagnose_candidates.py` with the default full universe
  (3-min runtime exceeds the 45s bash timeout — pass an explicit
  short ticker list)
- Pull live Theta data of any kind
- Hit the live MCP-driven TradingView surface

For a true launch-readiness pass, run the full checklist on the
laptop with the Terminal up. Cowork can stage and validate code
correctness; only the laptop can validate live-data correctness.
