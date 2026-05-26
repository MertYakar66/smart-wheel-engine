# Launch Readiness

The contractual checklist that pins what "ready to ship a
decision-layer change" means. Every item here is a **gate** — break
it and the change is unsafe regardless of what other tests pass.

This file is the operational consolidation of:
- `CLAUDE.md` §2 (the hard EV invariant)
- `MODULE_INDEX.md` (authoritative vs reviewer roles)
- `TESTING.md` (launch-blocker subset)
- `DECISIONS.md` D1 (EV is the only ranker), D11 (Theta fail-loud),
  D13 (MCP opt-in), D14 (FILE_MANIFEST coverage CI gate), D15
  (per-terminal worktree + env), D16 (token verdict-bound),
  D17 (portfolio-risk gates on both surfaces)
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

**Operationalised by the EV-authority token (D16).** The hard invariant
is enforced at the `WheelTracker` launch-gate by a two-stage predicate:
`issue_ev_authority_token` raises `EVAuthorityRefused` for any row with
`ev_dollars <= 0`, and `_consume_ev_authority_token` re-checks a fresh
`current_ev_dollars` at fire time — a token that was positive at rank
time but went stale at fire time is rejected (the token is retained
for retry). Both legs (`open_short_put` and `open_covered_call`) flow
through the same predicate. See `DECISIONS.md` D16.

---

## 2. The four authoritative routes

These are the **only** sanctioned paths from raw inputs to a
tradeable verdict:

| Module | Public entry | Pinned by |
|---|---|---|
| `engine/ev_engine.py` | `EVEngine.evaluate` | `tests/test_audit_invariants.py`, `tests/test_audit_viii_*.py` |
| `engine/wheel_runner.py` | `WheelRunner.rank_candidates_by_ev` | `tests/test_authority_hardening.py`, `tests/test_audit_viii_real_data_smoke.py` |
| `engine/candidate_dossier.py` | `EnginePhaseReviewer`, rules R1–R6 | `tests/test_dossier_invariant.py` |
| `engine_api.py` | HTTP API (default `:8787`); full endpoint list in the module-header docstring | `tests/test_tv_api.py`, `tests/test_tv_dossier.py` |

Any change touching a file under `engine/` that affects these routes
**must** run the full launch-blocker subset (§4 below) and the full
suite for `engine/ev_engine.py` or `engine/wheel_runner.py` changes
(invariants are cross-cutting).

---

## 3. Dossier downgrade rules (R1–R8)

`EnginePhaseReviewer` is the gatekeeper that converts an EV verdict
+ chart context into the final disposition. The eight rules (defined
on `EnginePhaseReviewer` in `engine/candidate_dossier.py`, pinned by
`tests/test_dossier_invariant.py`):

| Rule | Trigger | Effect |
|---|---|---|
| **R1** | EV is negative | **blocked** (hard stop) |
| **R2** | Chart missing | **review** |
| **R3** | Spot mismatch > 2% between engine-side and chart-side | **skip** |
| **R4** | *Conditional / reserved.* Phase contradiction (chart `visible_indicators['phase']` disagrees with engine phase) → **skip**. Implemented and unit-tested but **dormant in production**: no current chart provider populates the `phase` field and the ranker emits no `phase` on `ev_row`, so neither operand of the predicate is fed. R4 activates only when a phase-aware chart provider lands (see `docs/TRADINGVIEW_INTEGRATION.md`). Not a live downgrade today. |
| **R5** | EV above threshold | **proceed** |
| **R6** | Short-gamma regime + strike at/above the put wall, or dealer regime near the gamma flip | **downgrade** (`proceed → review`) |
| **R7** | *D17 soft-warn.* Portfolio VaR_95 (30-day horizon) above 5% NAV (`check_var`). Fires only when a `PortfolioContext` is attached and the verdict is currently `proceed`. Missing correlation matrix or returns data → skip (no fire on absent evidence). | **downgrade** (`proceed → review`, `verdict_reason="portfolio_var_breach"`) |
| **R8** | *D17 soft-warn.* One rule, two triggers (mirrors R6). Either the C4 vol-spike scenario shows portfolio drawdown > 8% NAV (`check_stress_scenario` → `"stress_breach"`) OR the candidate's underlying is in `short_gamma_amplifying` regime (`check_dealer_regime` → `"short_gamma_regime"`). Fires only when a `PortfolioContext` is attached and verdict is currently `proceed`. | **downgrade** (`proceed → review`) |

R1 is the structural realisation of §1 — negative EV ⇒ blocked. The
test is the merge gate. R7 + R8 are *soft-warns*: they only downgrade
`proceed → review` and never override R1's `blocked`. See
`DECISIONS.md` D17 for the rationale and the locked defaults.

The corresponding **hard-block** half of D17 lives on the tracker —
`engine/wheel_tracker.py._evaluate_d17_hard_blocks` refuses
position-opening on sector / portfolio-delta / Kelly-size breaches
when `require_ev_authority=True`. The hard-block surface is what
guarantees a book-level cap; R7 + R8 surface the same kind of
evidence on the dossier so an operator reviewing candidates sees the
warning at ranking time, not at firing time.

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
- [ ] **If the change touches `wheel_tracker.py`'s EV-authority
      token surface** — verify both stages of the D16 predicate
      fire (issue refuses non-positive `ev_dollars` and raises
      `EVAuthorityRefused`; consume re-checks a fresh
      `current_ev_dollars` at fire time). Run
      `tests/test_authority_hardening.py` plus
      `tests/test_ev_authority_log_schema.py` (the audit-log
      shape regression).
- [ ] **If the change touches the portfolio-risk gates (D17)** —
      verify the tracker hard-blocks **refuse** the position
      (`tests/test_authority_hardening.py::TestD17HardBlocks`,
      `tests/test_ev_authority_log_schema.py::TestD17EntryShapes`),
      and the dossier soft-warns **downgrade only** —
      R7/R8 may move `proceed → review` and must never upgrade or
      override R1's `blocked` (`tests/test_dossier_invariant.py`).
      Confirm the locked defaults in
      `engine/portfolio_risk_gates.py` (sector 25%, Kelly 0.5,
      VaR 5%, stress 8%, delta $300/$100k NAV) are not edited;
      D17's contract treats them as constants — tuning is a
      follow-on decision, not a remediation PR. Verify the four
      D17 audit-log shapes (`nav_exhausted` / `sector_cap_breach`
      / `portfolio_delta_breach` / `kelly_size_exceeded`) still
      validate against `_VALID_SHAPES`.
- [ ] If the change touches `wheel_tracker.py` more broadly — the
      rolled-position P&L accumulator tests pass; the three
      ledgers (`realized_pnl`, `transaction_costs`, `stock_basis`)
      remain orthogonal (audit-VIII fix).
- [ ] **Any new tracked file has a `FILE_MANIFEST.md` row** — the
      `FILE_MANIFEST Coverage` CI job (`scripts/check_manifest_coverage.py`,
      per `DECISIONS.md` D14) hard-fails the build otherwise. Run
      `python scripts/check_manifest_coverage.py` before push to
      catch this locally.
- [ ] `PROJECT_STATE.md` is updated if the change shifts what is
      "authoritative" or "deprecated".
- [ ] `CHANGELOG.md` has an entry under the current month.
- [ ] If the change retired a documented behaviour, the
      corresponding entry in `DECISIONS.md` is updated (not deleted)
      with a `**SUPERSEDED by D<N>**` note.
- [ ] If you ran the suite from a per-terminal worktree (D15), the
      relevant env was sourced — `source scripts/setup-terminal.sh
      <letter>` so `COVERAGE_FILE` / `PYTEST_CACHE_DIR` don't
      collide with another terminal's run.
- [ ] **If the change touches `engine/ev_engine.py`,
      `engine/wheel_runner.py`, `engine/forward_distribution.py`,
      `engine/dealer_positioning.py`, or `engine/tail_risk.py`** —
      the backtest-regression harness (§10) must run green in
      addition to the launch blockers. Paste the run timestamp + SHA
      into the PR description. The four ledger backtests
      (S27/S32/S34/S35) are downstream of all five files; silent
      drift here means the documented ρ / NAV / hit-rate claims
      become falsehoods on `main`.

---

## 6. Pre-launch checklist (rare; for going live with new data /
new strategy / new broker integration)

These are the gates beyond the per-PR checklist. Reach for this
list when shipping something genuinely user-facing:

- [ ] `audit.py` smoke tests pass against a running `engine_api.py`.
      Default `localhost:8787`; both server and client now honour
      `SWE_API_PORT` (closed D15 Unresolved). Set the env to bind a
      non-default port for multi-instance launches.
- [ ] `scripts/feature_smoke_test.py` reports a pass / fail / skip
      distribution consistent with the current data tier. The
      absolute count moves with each Theta refresh; the live
      reference is `PROJECT_STATE.md` §3.4. Re-baseline that section
      before treating any drift as a regression.
- [ ] On a Theta-up laptop:
      `python scripts/diagnose_candidates.py` (full universe, not the
      Cowork 5-ticker shim) produces a candidate funnel without
      surprises (no zero-trade ticker that should be tradeable).
- [ ] **Theta failures fail loud, not silent (D11).** If a per-symbol
      Theta endpoint times out while the Terminal itself is healthy,
      `engine/theta_connector.py` raises a typed `PerEndpointFailure`
      with a `FailureRecord` — it does **not** silently substitute
      Bloomberg CSVs. Pre-launch checks must surface any
      `PerEndpointFailure` accumulator
      (`connector.get_failures()`) and resolve them per puller; a
      mid-pull mixed-provenance mode is a launch blocker, not
      acceptable degradation. See `DECISIONS.md` D11.
- [ ] `python scripts/probe_theta_capabilities.py` regenerates
      `data_processed/theta_capabilities.json` and the persistent
      failure set matches the documented tier ceiling
      (currently: BF.B / BRK.B / NVR / DAY — upstream-data gap, not
      a code bug).
- [ ] **MCP chart provider mode is explicit, not implicit (D13).**
      The MCP path is opt-in via `SWE_USE_MCP_CHART=1`; default off.
      If you intend to go live with MCP charts on, set the env var
      explicitly in the launch environment and re-run the §4
      launch-blocker subset (the import-guarded
      `test_mcp_provider_*` contract test auto-activates). If MCP
      is off, the chained provider falls through to filesystem
      without silent substitution.
- [ ] If broker / OMS surface is being introduced (which is
      out-of-scope per `CLAUDE.md`'s NEVER list — get explicit
      consent first): a separate launch-readiness review against
      the new surface, not just this checklist.
- [ ] The dashboard (`dashboard/src/`, Next.js) still builds and
      hits the API (default `:8787`) without errors. The dashboard
      is the canonical operator surface for live ranking + dossier
      review.

---

## 7. What is *not* a launch gate

Useful to be explicit so reviewers don't over-block:

- **Coverage threshold drift** is informational, not blocking. The
  invariants matter; the line count doesn't (`DECISIONS.md` D10).
- **Doc drift** in `README.md`, `docs/ARCHITECTURE.md`, etc. is
  tracked in `ROADMAP.md` Track B — known and being repaired.
- **Deprecation warnings** in the test output are not blockers;
  they're a known cleanup queue. Treat any large delta as worth a
  look (the suite report total drifts with every audit); raw count
  doesn't gate merge.
- **The `iv_surface` integration decision** is open
  (`ROADMAP.md` A2) — shipping any code path that *uses* the SVI
  tooling does require resolving that decision first; everything
  else is unaffected.

---

## 8. Sandbox-vs-laptop launch caveats

The full sandbox-vs-laptop capability table lives in
`docs/DATA_POLICY.md` §7.

The launch-readiness corollary: Cowork can stage and validate code
correctness; only the laptop can validate live-data correctness.
Anything that needs the Theta Terminal — `scripts/diagnose_candidates.py`
on the full universe, live chain / Greeks / IV-surface checks, the
MCP-driven TradingView surface — must be re-run on the laptop before
declaring a launch ready.

### Per-terminal env (D15)

Multi-terminal coordination (`docs/PARALLEL_SESSIONS.md`) is N-generic.
Each executor terminal runs from its own worktree
(`../swe-terminal-<x>`) with env loaded by
`source scripts/setup-terminal.sh <letter>` (bash / Git Bash / WSL)
or `. .\scripts\setup-terminal.ps1 <letter>` (PowerShell). The loader
sets six env vars per terminal letter — `SWE_API_PORT`,
`SWE_DATA_PROCESSED_DIR`, `SWE_MODELS_DIR`, `COVERAGE_FILE`,
`PYTEST_CACHE_DIR`, `SWE_DATA_PROVIDER`. `SWE_API_PORT` (PR #158,
honoured by `engine_api.py._resolve_port()` and `audit.py`'s `BASE`),
`COVERAGE_FILE` (coverage.py), `PYTEST_CACHE_DIR` (pytest), and
`SWE_DATA_PROVIDER` (`WheelRunner.connector`) are real today —
each one is read by a live consumer. `SWE_DATA_PROCESSED_DIR` and
`SWE_MODELS_DIR` remain **conventions** until each consumer is
wired up. See `DECISIONS.md` D15.

### Launch-mode switching

Going from Cowork sandbox to laptop is a deliberate env-var flip,
not a code branch. The three switches:

| Mode change | Env var | Sandbox | Laptop |
|---|---|---|---|
| Data provider | `SWE_DATA_PROVIDER` | `bloomberg` (default; `MarketDataConnector` reads tracked CSVs) | `theta` (live `ThetaConnector` against the Theta Terminal on `127.0.0.1:25503`) — see `DECISIONS.md` D6, D7 |
| MCP chart provider | `SWE_USE_MCP_CHART` | unset / `0` (chained provider falls through to filesystem) | `1` (live `tv` CLI subprocess; requires TradingView Desktop + the `tv` shim — `DECISIONS.md` D12, D13) |
| API port | `SWE_API_PORT` | `8787` (default; override via env for multi-instance) | `8787` (default; override via env for multi-instance) |

After flipping any switch: re-run the §4 launch-blocker subset, then
the relevant pre-launch checks in §6. The provider switch in
particular is a recurring bug source — `CLAUDE.md` §4 emphasises
**always log which provider was actually selected**.

---

## 9. Observability surface

`engine/observability.py` ships structured-logging machinery that
the launch-readiness contract should make visible:

- **`TraceContext`** — per-evaluation context carrying the EV row,
  the reviewer's verdict, the multipliers applied, and the final
  `ev_dollars`. The compliance audit identity
  (`ev_dollars = ev_raw × Π(multipliers)`, see PR #149 / S16) is
  reconstructable from a `TraceContext`.
- **`DecisionEntry` + `DecisionJournal`** — append-only journal of
  decision-layer outcomes. The journal is the upstream of any
  external audit / dashboard / compliance consumer.
- **`AuditLogger`** — file-or-stream sink for the journal. Pre-launch
  checks should confirm the sink is configured for the launch
  environment.
- **`trace_operation`** — instrumentation helper used by the runner
  and the dossier reviewer.
- **`WheelTracker._ev_authority_log`** — the D16 + D17 audit log.
  Nine entry shapes total, all pinned by
  `tests/test_ev_authority_log_schema.py` and persisted via
  `to_dict` / `from_dict` (PR #128):
  - **D16 (five shapes):** `action="issue"`, `action="refuse_issue"`
    with `reason="non_positive_ev"`, `action="consume"`, and three
    `action="reject"` variants — `reason="unknown_token"` /
    `"missing_current_ev_dollars"` / `"stale_ev"`.
  - **D17 hard-block (four shapes):** `action="reject"` with
    `reason="nav_exhausted"` (pre-gate floor),
    `reason="sector_cap_breach"`, `reason="portfolio_delta_breach"`,
    `reason="kelly_size_exceeded"`. All four carry the live-NAV
    fingerprint (`nav` + `nav_source`) so an audit consumer can
    grep `nav_source="static_fallback"` to spot gate runs that
    landed on the static cap rather than live mark-to-market.

Pre-launch: confirm the launch environment writes both surfaces
(`DecisionJournal` and `_ev_authority_log`) somewhere recoverable —
S16's compliance walkthrough verdict was **partial**, so the audit
trail is verifiable for survivor rows but the structured-log
operator-facing story is incomplete. Tracked in audit issue #154
(C6).

---

## 10. Backtest regression gate

The four committed ledger backtests
(`docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md` S27,
`docs/ENGINE_BACKTEST_S32_FRICTION.md` S32,
`docs/SOUNDNESS_REVIEW_2026-05-26.md` S34,
`docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` S35) are pinned by
executable assertions in `tests/test_backtest_regression.py` against
snapshots committed under `backtests/regression/snapshots/`.

This is a **per-release** gate, not a per-merge gate. The runtime is
prohibitive for per-PR CI (~4–5 hours total, dominated by S34 at
~3.5 h on a 100-ticker universe), so it lives behind the
`backtest_regression` pytest marker and is excluded from
`.github/workflows/ci.yml`'s `Test Suite` job.

When to run:
- **Before any release tag on `main`.** Required green.
- **Before any PR that touches the five files in the §5 last bullet.**
  Required green; paste the run timestamp + SHA into the PR
  description.
- **Weekly heartbeat** if the dedicated workflow
  (`.github/workflows/backtest-regression.yml`) is wired with a
  schedule. Otherwise: triggered manually via the
  `.claude/commands/backtest-regression.md` skill on the laptop.

On failure: see TESTING.md § "Backtest regression — re-baseline
workflow" — diagnose before regenerating.

S35 is the only backtest whose snapshot was generated against the
**post-PIT-fix** engine (PR4 re-baseline). Snapshots for
S27/S32/S34 were locked against the same post-fix engine; the
documented headline numbers in their respective docs should match
within the per-metric tolerance band.
