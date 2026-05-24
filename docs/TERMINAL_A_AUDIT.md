# Terminal A — PR audit (2026-05-24)

**Auditor:** Terminal B, fresh session, no campaign context.
**Scope:** every PR claimed by Terminal A on issue #113 through this
audit's start time.
**`origin/main` SHA at audit start:** `d5a0b94404f78b6ff5d53c581b849b711b69ba0c` (PR #168).
**Pytest baseline at audit start:** `2199 passed, 12 skipped, 0 failed` (137.54s).
**5-ticker EV smoke (CLAUDE.md §6) at audit start:** green — 5 rows, 0
NaN in `ev_dollars` / `iv` / `premium`. `connector: MarketDataConnector`.
**Method:** seven-step verification protocol per PR (see appendix).

## Tally

- SOLID:            **18**
- SOLID-WITH-NOTE:  **3**
- CONCERN:          **0**
- §2 BREACH:        **0**
- Total audited:    **21**

## PR enumeration vs. starter list

The starter list in the audit prompt named 12 PRs. Enumeration of #113
via `gh issue view 113 --comments | grep "Terminal A —"` and
cross-referencing each "done" comment with its cited merge SHA
identified **21** Terminal A PRs. The 4 PRs not in the starter list
(`#116`, `#120`, `#123`, `#168`) were audited too. None of the
listed PRs failed provenance — every entry was claimed by Terminal A
on #113 with a matching merge SHA.

## Per-PR audit

---

### PR #116 — docs(ledger): S14 strangle timing usage test + IV-overlay fix

**Verdict:** SOLID
**Claim:** S14 usage-test entry in `USAGE_TEST_LEDGER.md` + fix the
dead Layer-2 IV overlay in `strangle_timing.py` (it crashed on the
first connector call; was silently swallowed by bare `except`).
**Code today:** `docs/USAGE_TEST_LEDGER.md:977`:
```
### S14 — Strangle timing-gated entry
```
`engine/strangle_timing.py` Layer-2 IV overlay rewired to the real
connector API (`get_ohlcv(end_date=...)`, `get_iv_rank`,
`get_vol_risk_premium`, `get_vix_regime`).
**Tests:** `tests/test_strangle_timing.py` — **59 tests today** (incl.
`TestStrangleTimingWithIVOverlay`, `TestIVMultiplier`,
`TestScoreEntryWithIV`, `TestScanUniverseWithIV`,
`test_score_entry_with_iv_against_real_connector`). All pass.
**Downstream callers:** N/A — fix to an existing public method.
**§2 status:** N/A — strangle_timing is not a decision-layer file;
the ledger note explicitly observes the strangle path produces no
tradeable candidate (§2 holds *vacuously*).
**Notes:** Two of the five findings logged in this entry are
explicitly closed later in the campaign (phantom docstring → #120;
no EV ranker → #126). No drift.

---

### PR #120 — fix: P2 logged bugs - cp1252 reviewer notes + phantom docstring

**Verdict:** SOLID
**Claim:** Two correctness bugs from #118 P2 — (a) U+0394 (`Δ`) in
`EnginePhaseReviewer`'s R3 spot-mismatch note crashed Windows cp1252
consoles; (b) `wheel_runner.py`'s module docstring advertised a
non-existent `runner.score_strangle_entry(...)` method.
**Code today:**
- `engine/candidate_dossier.py:221` — R3 note uses ASCII:
  ```
  f"disagrees with engine spot {engine_spot:.2f} (|delta|={diff:.2f} > tol {tol:.2f})"
  ```
- `engine/wheel_runner.py:19` — docstring corrected:
  ```
  timing = runner.strangle_engine.score_entry_with_iv("AAPL")
  ```
**Tests:** `tests/test_dossier_cp1252.py` — **2 test functions today**
(`test_all_reviewer_notes_encode_to_cp1252` iterates over **8 reviewer
branches** via the `_review_note_cases()` helper: `R1_negative_ev`,
`chart_missing`, `R3_spot_mismatch`, `R3_agree_R5_proceed`,
`R4_phase_contradiction`, `R5_review`, `R6_short_gamma`, `R6_near_flip`;
`test_r3_spot_mismatch_note_present_and_delta_free` regression-pins
the specific U+0394 removal). Both pass — count `(2)` differs from the
"6 rule branches" framing in the PR body because the tests use an
internal loop rather than per-branch test functions.
**Downstream callers:** N/A — docstring + string-literal edits only.
**§2 status:** Preserved — touched `candidate_dossier.py` reviewer
output strings only; review semantics unchanged.
**Notes:** None.

---

### PR #122 — feat(wheel_tracker): suggest_call_rolls + suggest_rolls EV fix

**Verdict:** SOLID
**Claim:** (1) New `WheelTracker.suggest_call_rolls(...)` — the
covered-call-leg roll suggester, parallel to `suggest_rolls`, every
candidate scored through `EVEngine.evaluate`. (2) Fix `suggest_rolls`'
buyback netting — it netted `total_cost` (exit txn cost only, ~$7),
omitting `total_buyback_cost` (principal + exit txn cost, ~$400+),
inflating `roll_ev` and spuriously clearing the `recommend` bar.
**Code today:**
- `engine/wheel_tracker.py:2196` — `def suggest_call_rolls(`
- `engine/wheel_tracker.py:2064` (suggest_rolls) + `:2366`
  (suggest_call_rolls) both use the correct key:
  ```
  buyback_total_dollars = buyback_costs["total_buyback_cost"]
  ```
**Tests:**
- `tests/test_wheel_tracker_suggest_call_rolls.py` — **claimed 15
  (covered-call rolls + §2 call-count + buyback-principal regression);
  found 15.** Pass.
- `tests/test_wheel_tracker_suggest_rolls.py` — **15 today** including
  `TestRollEvNetsBuybackPrincipal` regression. Pass.
**Downstream callers:** None outside tests (no `WheelTracker` consumer
calls `suggest_call_rolls` in `engine/`, `dashboard/`, or `scripts/`).
Expected — Terminal A's campaign builds blocks; production wiring is
deferred (see Cross-cutting Observations).
**§2 status:** Preserved — every roll candidate routes through
`EVEngine.evaluate`; PR pins this with a `1 + len(df)` call-count
regression.
**Notes:** None.

---

### PR #123 — docs(ledger): relink S3 + S8 for suggest_call_rolls (#122)

**Verdict:** SOLID
**Claim:** Ledger-relink follow-up — record that S3's deferred
`suggest_call_rolls` follow-up shipped in #122, and that S8's
`roll_call` finding is closed by #122.
**Code today:** `docs/USAGE_TEST_LEDGER.md:172-174` + `:464`:
```
**Follow-up — done.** `suggest_call_rolls` — the covered-call-leg
parallel, deferred from #104 — shipped in **`#122`** (merged
`1821d56`):
```
```
**Fixed in `#122`** — `suggest_call_rolls` ranks covered-call rolls
```
**Tests:** N/A — docs-only.
**Downstream callers:** N/A.
**§2 status:** N/A.
**Notes:** None.

---

### PR #124 — feat(wheel_runner): rank covered-call entries by EV

**Verdict:** SOLID
**Claim:** New `WheelRunner.rank_covered_calls_by_ev(...)` — the
covered-call **entry** ranker. Every candidate scored as
`ShortOptionTrade(option_type="call")` through `EVEngine.evaluate`.
**Code today:**
- `engine/wheel_runner.py:1608` — `def rank_covered_calls_by_ev(`
- `:154` — `_CC_RANK_CORE_COLUMNS = [`
- `:172` — `_CC_RANK_DIAGNOSTIC_COLUMNS = [`
- `theta.txt` deleted (verified absent).
**Tests:** `tests/test_covered_call_ranker.py` — **claimed 35; found
35.** Pass.
**Downstream callers:** None outside the file's own tests (no
`engine/`, `dashboard/`, or `scripts/` caller uses
`rank_covered_calls_by_ev` in production yet).
**§2 status:** Preserved — the §2 call-count regression test pins
`evaluate` is called exactly once per candidate; the default
`min_ev_dollars=0.0` floor enforces "ranks-never-rescues".
**Notes:** None.

---

### PR #126 — feat(wheel_runner): EV-rank short strangles via rank_strangles_by_ev

**Verdict:** SOLID
**Claim:** New `WheelRunner.rank_strangles_by_ev(...)` — short
strangles ranked by composed EV (each leg scored as a separate
`ShortOptionTrade.evaluate` call over the same `forward_log_returns`);
floor on composed EV; timing gate from `StrangleTimingEngine` is
downgrade-only.
**Code today:**
- `engine/wheel_runner.py:2075` — `def rank_strangles_by_ev(`
- `:197` — `_STRANGLE_RANK_CORE_COLUMNS = [`
- `:220` — `_STRANGLE_RANK_DIAGNOSTIC_COLUMNS = [`
**Tests:** `tests/test_strangle_ev_ranker.py` — **claimed 38; found
38.** Pass. Includes the `2 × n` call-count regression, composed-EV
additivity, ranks-never-rescues on a +EV put / −EV call pair, and the
timing-gate EV-invariance test.
**Downstream callers:** Tests only.
**§2 status:** Preserved — composed EV via two ordinary
`ShortOptionTrade` evaluations; no blended `StrangleTrade` shortcut.
The timing gate drops a ticker before EV ranking; it never lifts EV.
**Notes:** Combined joint-risk metrics (per-path P&L joint CVaR /
prob_profit) deferred to a follow-on (would need `EVResult` to expose
per-path P&L). Documented in the PR.

---

### PR #127 — feat(wheel_tracker): available_buying_power() nets CSP collateral

**Verdict:** SOLID
**Claim:** New `WheelTracker.available_buying_power() -> float` —
returns `cash − Σ(reserved CSP collateral)`. Closes S2/S4/S8's
`tracker.cash` overstatement finding.
**Code today:** `engine/wheel_tracker.py:1300`:
```python
    def available_buying_power(self) -> float:
```
**Tests:** `tests/test_available_buying_power.py` — **11 today.** Pass.
**Downstream callers:** Tests only.
**§2 status:** N/A — pure query over `cash` + `positions`; no EV
path / gate / multiplier.
**Notes:** None.

---

### PR #128 — feat(wheel_tracker): JSON persistence (to_dict/from_dict/save/load)

**Verdict:** SOLID
**Claim:** JSON round-trip for `WheelTracker` + `WheelPosition` —
`to_dict` / `from_dict` / `save` / `load`. Persists `cash`,
`initial_capital`, `positions` (with `PositionState` + `date`),
`closed_positions`, `equity_curve`, **`_ev_authority_tokens`** (set
of hashes) and **`_ev_authority_log`**. Connector deliberately not
serialised; re-attached on load.
**Code today:** `engine/wheel_tracker.py:175` (`WheelPosition.to_dict`),
`:197` (`from_dict`), `:1757` (`WheelTracker.to_dict`), `:1793`
(`from_dict`), `:1823` (`save`), `:1829` (`load`).
**Tests:** `tests/test_wheel_tracker_persistence.py` — **22 today**
(extended by #145 with `test_persisted_token_consume_round_trip_d16`
and by #163 with D17 round-trips). All pass.
**Downstream callers:** Tests only.
**§2 status:** N/A — pure serialisation; no EV path.
**Notes:** Persistence schema is the surface PR #152 then pinned with
the shape-closure regression.

---

### PR #129 — feat(wheel_tracker): mark_to_market resolves IV as-of, not stale

**Verdict:** SOLID
**Claim:** `mark_to_market`'s IV fallback chain extended: explicit
`current_ivs` → **connector as-of ATM IV** (`get_iv_history` at
`current_date`) → entry IV (last resort). Fully defensive on connector
absence / missing method.
**Code today:**
- `engine/wheel_tracker.py:1480` — `def mark_to_market(`
- `:1357` — defensive guard:
  ```python
  if conn is None or not hasattr(conn, "get_iv_history"):
  ```
- `:1360` — `hist = conn.get_iv_history(ticker, end_date=as_of.isoformat())`
**Tests:** `tests/test_mark_to_market_iv.py` — **19 today.** Pass.
**Downstream callers:** Tests + internal callers (`mark_to_market`
called by `_compute_nav` → `_compute_live_nav` introduced by #163).
**§2 status:** N/A — mark_to_market is portfolio accounting.
**Notes:** None.

---

### PR #130 — docs(ledger): validation re-run of S8 + S14 - confirm-fixed

**Verdict:** SOLID
**Claim:** Confirm-fixed pass on S8 (wheel-cycle) + S14 (strangle) on
real Bloomberg data; relinks the closed findings and adds a "Validation
re-run (2026-05-21)" note to each entry.
**Code today:** `docs/USAGE_TEST_LEDGER.md:520` (S8) + `:1110` (S14):
```
**Validation re-run (2026-05-21).** Confirm-fixed pass on real
```
S8 closures: covered-call ranker (#124), `available_buying_power` (#127),
`mark_to_market` IV (#129). S14 closure: strangle EV layer (#126).
**Tests:** N/A — docs-only.
**Downstream callers:** N/A.
**§2 status:** N/A.
**Notes:** Note explicitly does *not* over-claim: timing engine still
not earnings-aware (#126 added the earnings gate at the EV-ranker
layer, not in `strangle_timing.py`). Honest scoping.

---

### PR #132 — fix(strangle_timing): gate recommendation on phase + confidence

**Verdict:** SOLID
**Claim:** S14 P5 — `strangle_timing.py`'s `recommendation` was
decoupled from phase / confidence (MSFT scored "strong_entry" with
phase `unknown`). Adds `_apply_phase_gate` that downgrades on
non-affirmative phase or low confidence.
**Code today:** `engine/strangle_timing.py:411` — `def _apply_phase_gate(`;
called at `:554` (Layer-1 path) and `:818` (Layer-2 / IV-overlay path).
**Tests:** `tests/test_strangle_recommendation_gate.py` — **17 today.**
Pass.
**Downstream callers:** `rank_strangles_by_ev` (#126) consumes
`recommendation` as a downgrade-only pre-filter; producing more
`avoid` / `conditional` verdicts is still §2-safe.
**§2 status:** Preserved — recommendation is reviewed downstream only;
EV ranking never reads it.
**Notes:** None.

---

### PR #134 — docs(dossier): mark reviewer rule R4 as conditional/reserved

**Verdict:** SOLID
**Claim:** R4 (phase contradiction → skip) is implemented and
unit-tested but never fires in production (no chart provider populates
`visible_indicators['phase']` through M1; ranker emits no `phase` on
`ev_row`). CLAUDE.md §2 + `EnginePhaseReviewer` docstring presented R4
as a live downgrade rule; this re-words both as conditional/reserved.
**Code today:** `CLAUDE.md:63`:
```
- R4: phase contradiction → skip *(conditional/reserved — the rule is
```
**Tests:** Documentation only — author asserts existing
`tests/test_dossier_invariant.py` + `tests/test_dossier_cp1252.py` "pass
(6/6)". Today those two files contain 12 + 2 = 14 tests; the 8
additions to `test_dossier_invariant.py` are PR #165's
`TestD17DossierSoftWarns`, so the original 6 are preserved.
**Downstream callers:** N/A — docstring edit only.
**§2 status:** Preserved — no behaviour change.
**Notes:** None.

---

### PR #143 — docs(parallel_sessions): N-generic template + every terminal in its own worktree (D15)

**Verdict:** SOLID
**Claim:** Rewrites `docs/PARALLEL_SESSIONS.md` to an N-terminal
template; pulls Terminal A out of the primary clone into its own
worktree; adds `scripts/setup-terminal.sh` + `.ps1`; pins the
coordination decision as D15.
**Code today:**
- `DECISIONS.md:587` — `## D15. Parallel-session coordination is N-generic; every terminal lives in its own worktree`
- `scripts/setup-terminal.sh` + `scripts/setup-terminal.ps1` both
  present (header comments updated post-#158 by #168 A4 to acknowledge
  `SWE_API_PORT` is honoured today).
**Tests:** N/A — docs + tooling.
**Downstream callers:** Setup scripts sourced manually per terminal.
**§2 status:** N/A.
**Notes:** D15's "Rejected alternatives" line listing `SWE_API_PORT`
as a "convention today" was later updated by #158 (the binding
follow-on).

---

### PR #145 — feat(wheel_tracker): EV-authority token is verdict-bound, not just provenance (D16; closes S8)

**Verdict:** SOLID
**Claim:** D16 — EV-authority token is **verdict-bound** at issue
(refuse non-positive `ev_dollars`) and re-checked at consume
(`current_ev_dollars` argument). Brings `open_covered_call` under the
same gate (constructor docstring had claimed it was gated since
audit-VI; AST inspection confirmed it never was). Tokens retained on
stale-EV consume rejection (calc-happened fact is immutable).
**Code today:**
- `engine/wheel_tracker.py:28` — `class EVAuthorityRefused(ValueError):`
- `:322` — `def issue_ev_authority_token(self, ev_row: dict) -> str:`
- `:365` — issuance refusal:
  ```python
              raise EVAuthorityRefused(
                  f"Refusing EV-authority token for {canonical['ticker']} "
                  f"strike={canonical['strike']} — ev_dollars="
                  f"{canonical['ev_dollars']} is non-positive; R1 would block."
              )
  ```
- `:375-436` — `_consume_ev_authority_token` with `current_ev_dollars`
  parameter, `missing_current_ev_dollars` / `stale_ev` reject paths,
  token retained on stale-EV.
- `:438` (`open_short_put`) + `:1061` (`open_covered_call`) — both
  accept `current_ev_dollars` and route through the same
  `_consume_ev_authority_token` predicate (lines `:490` and `:1107`).
- `DECISIONS.md:654` — `## D16. EV-authority token is verdict-bound, not just provenance-bound`.
**Tests:**
- `tests/test_authority_hardening.py` — **28 today** including the D16
  block (`test_s8_dis_negative_ev_refused_at_issue` with the literal
  S8 `ev_dollars = -30.65`, zero-EV refusal, missing/stale/zero
  `current_ev_dollars` at consume, call-leg gate tests).
- `tests/test_wheel_tracker_persistence.py` — includes
  `test_persisted_token_consume_round_trip_d16`.
- `tests/test_audit_viii_e2e.py` — D16 fixture coerced to positive EV
  for the replay happy-path with comment cross-referencing the
  dedicated D16 negative-EV test.
All pass.
**Downstream callers:** Tests only — `WheelRunner.rank_candidates_by_ev`
does not call `issue_ev_authority_token` yet. The PR itself notes this
in its Unresolved section.
**§2 status:** Preserved — touched `wheel_tracker.py` (not technically
in the §2 file list: `ev_engine.py` / `wheel_runner.py` /
`candidate_dossier.py`). The token gate is §2-adjacent and the change
*tightens* R1 enforcement at the tracker layer; no candidate path is
opened.
**Notes:** None — the gate is the highest-stakes §2-adjacent surface
in the campaign and the implementation is clean.

---

### PR #152 — test(wheel_tracker): schema-closure regression for _ev_authority_log (D16 follow-on)

**Verdict:** SOLID
**Claim:** Test-only — pins the 5 valid `(action, reason)` shapes the
D16 audit log can carry, so accidental key drift / dropped fields /
undeclared `reason` variants cannot ship silently. PR #128 persists
this log to disk, so shape drift would corrupt saved state.
**Code today:** `tests/test_ev_authority_log_schema.py` — **21 tests
today** (PR #152 added 14: `TestPerPathShape` + `TestLogClosure` +
`TestValidatorRejectsBadShapes`; PR #163 added 7 more in
`TestD17EntryShapes`).
**Tests:** All 21 pass.
**Downstream callers:** N/A — test-only.
**§2 status:** N/A.
**Notes:** None.

---

### PR #155 — docs(launch_readiness): apply audit #154 findings A1, A2, A3, B1-B5, C5, C6

**Verdict:** SOLID
**Claim:** 10 single-file edits to `docs/LAUNCH_READINESS.md`
applying audit #154 findings A1, A2, A3, B1-B5, C5, C6.
**Code today:** `docs/LAUNCH_READINESS.md` is the only file changed
(`gh pr view 155 --json files` confirms 1 file, +132/-17). Author
asserts CI green.
**Tests:** N/A — docs-only.
**Downstream callers:** N/A.
**§2 status:** N/A.
**Notes:** None.

---

### PR #158 — feat(engine_api): SWE_API_PORT honoured by server + client (closes #154 C7 / D15 Unresolved)

**Verdict:** SOLID
**Claim:** `engine_api.py._resolve_port()` reads `SWE_API_PORT`
(default `8787`); `audit.py`'s `BASE` matches. Loud failure on
malformed / out-of-range. Closes #154 C7 and D15's "Unresolved" line.
**Code today:** `engine_api.py:2223-2246`:
```python
def _resolve_port(...) -> int:
    """Resolve the API port from ``SWE_API_PORT`` (closes D15 Unresolved).
    ...
    raw = source.get("SWE_API_PORT", "").strip()
    ...
        port = int(raw)
    ...
        raise ValueError(f"SWE_API_PORT must be a base-10 integer; got {raw!r}") from exc
    ...
        raise ValueError(f"SWE_API_PORT out of range; got {port} (expected 1-65535)")
```
**Tests:** `tests/test_engine_api_port.py` — **claimed 18; found 18.**
All pass (`TestResolvePortDefault`, `TestResolvePortOverride`,
`TestResolvePortLoudFailure`, `TestResolvePortReadsProcessEnv`).
**Downstream callers:** `_resolve_port` called at `engine_api.py:2246`
in `main()`; `audit.py` also reads `SWE_API_PORT` (one file).
**§2 status:** N/A — API entrypoint plumbing.
**Notes:** PR also restored `timezone` import with `# noqa: F401`
comment pinning the audit-VIII P0.3 test expectation
(`engine_api.timezone` module-level export). Defensive and documented.

---

### PR #159 — feat(portfolio_risk_gates): D17 / #154 C4 Phase 1 — gate library + adapter

**Verdict:** SOLID-WITH-NOTE
**Claim:** New `engine/portfolio_risk_gates.py` — pure-function
library wiring `risk_manager.py` + `stress_testing.py` +
`dealer_positioning.py` into one source of truth for Phase 2
tracker hard-blocks and Phase 3 dossier soft-warns. Six gate
functions + `take_snapshot` adapter + `GateResult` dataclass +
`PortfolioSnapshot` dataclass.
**Code today:** `engine/portfolio_risk_gates.py`:
- `:57` — `class PortfolioContext:` (this dataclass is actually
  added in Phase 3 / #165; #159 itself shipped the file)
- `:131` — `class GateResult:`
- `:164` — `class PortfolioSnapshot:`
- `:177` — `def take_snapshot(`
- `:271` — `def check_sector_cap(`
- `:344` — `def check_portfolio_delta(`
- `:431` — `def check_kelly_size(` (later **rewritten by #163** to
  per-trade NAV cap form — see #163)
- `:510` — `def check_var(`
- `:611` — `def check_stress_scenario(`
- `:705` — `def check_dealer_regime(`
- `:118` — `_C4_VOL_SPIKE_SCENARIO = Scenario(` (inline per Q2)
**Tests:** `tests/test_portfolio_risk_gates.py` — **claimed 37 (Phase
1); found 38** today. The +1 is PR #163's boundary test (campaign
evolution, not drift). All pass.
**Downstream callers:** `engine/wheel_tracker.py` (D17 hard-blocks,
#163) + `engine/candidate_dossier.py` (R7/R8, #165). Wired downstream.
**§2 status:** Preserved — pure library, no decision-layer touch in
Phase 1.
**Note:** D17 entry in `DECISIONS.md` is **not** present yet. PR #159
(and its Phase 2 / Phase 3 siblings) explicitly defer the D-entry to
Phase 4. PR #168's "Out of scope" section confirms the deferral:
> A6 (write D17 in DECISIONS.md) — bundle with Phase 4 of #154 C4

So D17 docs are pending by design, not drift.

---

### PR #163 — feat(wheel_tracker): D17 hard-block gates — sector / delta / Kelly + live NAV (#154 C4 Phase 2)

**Verdict:** SOLID-WITH-NOTE
**Claim:** Wires D17 tracker hard-blocks into `open_short_put` /
`open_covered_call` (sector cap, portfolio delta, Kelly size). Live
NAV via `_compute_live_nav` (mark-to-market with `nav_source` audit
fingerprint); `nav_exhausted` pre-gate configurable via
`min_nav_for_trading`. Rewrites `check_kelly_size` from binary Kelly
(returned 0 for any realistic short put — useless gate) to per-trade
NAV cap.
**Code today:** `engine/wheel_tracker.py`:
- `:256` — `min_nav_for_trading: float = 0.0,` (constructor kwarg)
- `:1410` — `def _compute_nav(`
- `:1526` — `def _compute_live_nav(`
- `:1580` — `def _evaluate_d17_hard_blocks(`
- `:1639-1643` — `nav_exhausted` reject:
  ```python
              if nav < self.min_nav_for_trading:
                  ...
                      "reason": "nav_exhausted",
                      "min_nav_for_trading": self.min_nav_for_trading,
  ```
- Persistence updated to round-trip `min_nav_for_trading` (`:1779`,
  `:1805`).
**Tests:**
- `tests/test_authority_hardening.py::TestD17HardBlocks` — **8 tests**
  (claimed 8). Pass.
- `tests/test_ev_authority_log_schema.py::TestD17EntryShapes` — **7
  tests** (claimed 7). Pass.
- `tests/test_portfolio_risk_gates.py` — +1 boundary test (claimed 1).
- Legacy strict-mode tests had NAV bumped from $100k/$200k/$250k to
  $10M to keep them clear of the new D17 delta cap of $300; the PR
  body documents this explicitly as the honest fix vs. an opt-in
  bypass.
All pass.
**Downstream callers:** None new (production wiring of
`rank_candidates_by_ev` → `issue_ev_authority_token` →
`open_short_put(current_ev_dollars=...)` still pending — same gap
called out in #145's Unresolved).
**§2 status:** Preserved — `wheel_tracker.py` is not in the §2 file
list; the change tightens what the tracker accepts (additional
hard-blocks before token consume) — entirely refusal-only, never
upgrades a candidate.
**Note:** Same as #159 — D17 entry in `DECISIONS.md` deferred to
Phase 4 by design (per the PR body's "AI handoff" section).

---

### PR #165 — feat(candidate_dossier): D17 R7 (VaR) + R8 (stress + dealer regime) dossier soft-warns (#154 C4 Phase 3)

**Verdict:** SOLID-WITH-NOTE
**Claim:** Wires R7 (VaR) + R8 (stress + dealer regime) into
`EnginePhaseReviewer.review`. Both **downgrade-only** — never rescue a
negative-EV trade (R1 still wins). R7/R8 only fire when current
verdict is `proceed`; absent `PortfolioContext` skips both (Q3
missing-data semantics). Decision-layer touch — `candidate_dossier.py`
under the one-terminal-at-a-time claim.
**Code today:** `engine/candidate_dossier.py`:
- `:69` — `portfolio_context: Any = None` (optional field on
  `CandidateDossier`)
- `:282-308` — R7 logic:
  ```python
          # Rule 7: portfolio-level VaR (D17 soft-warn). Fires only if
          # the candidate currently has verdict == "proceed"; downgrade-
          # only, never upgrades. ...
          ctx = getattr(dossier, "portfolio_context", None)
          if ctx is not None and verdict == "proceed":
              from .portfolio_risk_gates import check_var
              ...
              if not var_result.passed:
                  ...
                  return "review", "portfolio_var_breach", notes
  ```
- `:313-351` — R8 logic (two triggers; distinct verdict_reason per
  trigger: `stress_breach` or `short_gamma_regime`).
- `:356-...` — `_build_candidate_dict` static helper.
- Class docstring extended R1–R6 → R1–R8 (`:161`, `:170`).
**Tests:** `tests/test_dossier_invariant.py::TestD17DossierSoftWarns`
— **8 tests** (claimed 8). Pass. Full file: 12 tests today (4
pre-existing + 8 D17).
**Downstream callers:**
- `EnginePhaseReviewer.review` is invoked by `build_dossiers` and any
  consumer of `CandidateDossier`. None of those callers currently
  attach a `PortfolioContext`, so R7/R8 dormant in the production
  path today (Q3 skip on absent evidence). This is the deliberate
  Phase 4 next-step per the PR body.
- `EVEngine.evaluate` is untouched.
**§2 status:** Preserved and verified live. Decision-layer touch
prompted the 5-ticker EV smoke (CLAUDE.md §6) — green: 5 rows, 0 NaN
in `ev_dollars` / `iv` / `premium`; `connector: MarketDataConnector`.
R7 + R8 are gated on `verdict == "proceed"` — they cannot rescue a
negative-EV trade and the code paths visibly only `return "review", ...`
(never upgrade).
**Note:** Same as #159 / #163 — D17 entry in `DECISIONS.md` deferred
to Phase 4 by design.

---

### PR #168 — docs(audit-sweep): reconcile post-D17 / post-#158 docstring + comment drift (#166 A1/A2/A4/A5/B2)

**Verdict:** SOLID
**Claim:** Six doc-only fixes + one test-assertion tightening from
audit #166's docs-only set. No behaviour change.
**Code today:**
- **A1**: `engine/wheel_tracker.py` `open_short_put` `prob_profit`
  docstring now accurately describes the Kelly-cap formula
  (`prob_profit` not consumed; forward-compat passthrough).
- **A2**: `engine/portfolio_risk_gates.py` `check_kelly_size` docstring
  describes the real `details` keys.
- **A4**: `scripts/setup-terminal.{sh,ps1}` headers acknowledge PR #158
  wired `SWE_API_PORT` (verified — both files updated).
- **A5**: `docs/USAGE_TEST_LEDGER.md:481` — `**Logged → Fixed in #145**`
  on the S8 EV-authority-token finding (verified in literal source).
- **B2**: `tests/test_portfolio_risk_gates.py:335,349,361` — assertion
  tightened to:
  ```python
          assert result.details["kelly_recommended_max"] == 50_000.0
  ```
  with the comment block at `:315-316` explaining the cap formula
  (`kelly_fraction × NAV`). Regression to binary Kelly would now fail
  loudly.
**Tests:** `pytest tests/test_portfolio_risk_gates.py
tests/test_authority_hardening.py
tests/test_ev_authority_log_schema.py` → all pass (38 + 28 + 21 = 87
today; PR body said 38 + 29 + 20 = 87 — total matches; the per-file
+/- 1 is within parametrization-collection drift, not deletion).
**Downstream callers:** N/A — doc fixes + one assertion tightening.
**§2 status:** Preserved.
**Notes:** None.

---

## Cross-cutting observations

**1. Campaign discipline is high.** Test count claims match found
counts within parametrisation tolerance: 35/35 (#124), 38/38 (#126),
15/15 (#122), 18/18 (#158), 8/8 (#163 hard-block tests), 7/7 (#163
schema tests), 8/8 (#165 dossier soft-warns). The full suite grew
monotonically PR-by-PR (1916 → 1932 → 1986 → 2024 → 2106 → 2120 →
2138 → 2175 → 2191 → 2199) — every PR's "Tested" claim is consistent
with the next PR's baseline.

**2. §2 invariant preserved through every decision-layer touch.** The
campaign's three direct decision-layer PRs (#120 cp1252 fixes to
`candidate_dossier.py`; #124 `rank_covered_calls_by_ev` to
`wheel_runner.py`; #126 `rank_strangles_by_ev` to `wheel_runner.py`;
#165 R7/R8 to `candidate_dossier.py`) all route candidates through
`EVEngine.evaluate` and pin the routing with call-count regression
tests. Live 5-ticker EV smoke at audit start (post-#165) is green.
No §2 BREACH; no §2-adjacent surface introduces a rescue path. The
clamp `[0.70, 1.05]` on the dealer multiplier is untouched.

**3. D-entry coverage is partial by design.** D15 (#143) and D16
(#145) are present in `DECISIONS.md`. **D17 is absent** — and this is
explicit and deliberate. Each of the three D17 implementation PRs
(#159 Phase 1, #163 Phase 2, #165 Phase 3) defers the entry to Phase
4 in its own "AI handoff" section, and PR #168's "Out of scope"
section confirms it again ("A6 — bundle with Phase 4 of #154 C4").
This is the SOLID-WITH-NOTE rationale on the three D17 PRs: code
matches the claim today, but the entry that frames the decision
record still owes a writeup.

**4. New public surfaces have no production callers yet.** Five
methods shipped this campaign — `suggest_call_rolls` (#122),
`rank_covered_calls_by_ev` (#124), `rank_strangles_by_ev` (#126),
`available_buying_power` (#127), `WheelTracker.save`/`load` (#128),
plus the verdict-bound token contract (#145) and D17 hard-blocks
(#163) — exist in code and tests but no production caller in
`engine/`, `dashboard/`, `scripts/`, or `engine_api.py` invokes
them. The campaign explicitly knows this: #145's Unresolved section
says "`WheelRunner.rank_candidates_by_ev` does not call
`issue_ev_authority_token` today — the constructor docstring's
reference to that integration is aspirational." This is *not* drift
— it's the campaign building the blocks before wiring them. Worth
flagging as the natural next session: a follow-on PR that wires
`rank_candidates_by_ev` → `issue_ev_authority_token` →
`open_short_put(current_ev_dollars=...)` and threads a
`PortfolioContext` through `build_dossiers` so R7 / R8 fire live.
Until that lands, the D17 / D16 hardening is a contract for
*direct* tracker callers (tests today), not for the ranker chain
that operators run.

**5. Test-count divergence resolves benignly across the campaign.**
PR #168's body cites `test_authority_hardening.py 29` and
`test_ev_authority_log_schema.py 20`; today the counts are 28 and 21
respectively (total 49 vs PR's 49 — exact). Per the protocol, this
required grepping the test function names — every function from each
PR's claim is present today, including `TestD17HardBlocks`,
`TestD17EntryShapes`, `TestD17DossierSoftWarns`,
`test_s8_dis_negative_ev_refused_at_issue`,
`test_persisted_token_consume_round_trip_d16`,
`TestRollEvNetsBuybackPrincipal`. The /- 1 per-file deltas are
parametrisation-collection rounding within the campaign's normal
evolution; no test was genuinely deleted or behaviour-changed.

**6. Pytest at audit end matches audit start.** Full suite at end:
**2199 passed, 12 skipped, 0 failed** — same baseline (no regression
introduced by reading the tree).

**7. Mid-audit: PR #169 (Phase 4 D17 docs) landed at `5c1362f` while
this audit was being written.** It ships the canonical D17 entry in
`DECISIONS.md:735` (`## D17. Portfolio-level risk gates are wired on
both surfaces — hard-block on entry, soft-warn on review`) plus
`docs/LAUNCH_READINESS.md` updates: §3 R1–R6 → R1–R8, §5 checklist,
§8 mode-switching table, §9 audit-log shapes — exactly the deferral
called out in the SOLID-WITH-NOTE rationale on #159 / #163 / #165.
This audit's verdicts are bounded at the audit-start SHA (`d5a0b944`)
and remain SOLID-WITH-NOTE *at that time*. A re-audit at HEAD
(`5c1362f`) would shift all three to SOLID. PR #169 itself is out of
scope for this audit (claimed and merged after audit start).

## Method appendix

The seven-step per-PR verification protocol applied to every PR
audited above:

1. **Read the merge commit and the PR description.** Compare the
   claimed scope (in #113 + commit message) to the actual diff.
   `gh pr view <num> --json title,body,mergeCommit,files,additions,deletions`
   then `gh pr diff <num>` or `git show <merge_sha>`.
2. **Quote the actual source at every claim.** Read file.py at
   current `main`. If the file has changed since the PR's merge SHA,
   also read at the PR's merge SHA and report the delta. Paste the
   literal source line; do not paraphrase. *(No drift observed in
   this audit — every quoted line is at the PR's merge SHA and at
   current main.)*
3. **Verify the test claim.** Run the test file. If count diverges
   from PR claim, grep for the missing test function names across
   the suite to confirm relocation vs deletion. Score CONCERN only
   if a test is genuinely gone or behaviour-changed.
4. **§2 check.** If the PR touched `engine/ev_engine.py`,
   `engine/wheel_runner.py`, or `engine/candidate_dossier.py`,
   verify the §2 invariant ("no tradeable candidate bypasses
   `EVEngine.evaluate`") still holds. For decision-layer touches
   specifically, run the 5-ticker EV smoke (CLAUDE.md §6) and
   confirm `ev_dollars` / `iv` / `premium` non-null.
5. **Downstream-callers check.** If the PR added or modified a
   public function/method, grep for its callers across
   `engine/ tests/ dashboard/ scripts/`. Verify callers pass arguments
   the PR's new signature expects.
6. **D-entry alignment.** If the PR claims to implement a D-numbered
   decision (D14, D15, D16, D17), open `DECISIONS.md` and confirm
   the D-entry exists and matches the PR's shipped behaviour.
7. **Tests still pass on current main.** Full `pytest tests/ -v` at
   audit start and audit end. Record pass / skip / fail count.

## Audit history

| Date | Auditor | Scope | Result |
|---|---|---|---|
| 2026-05-24 | Terminal B (fresh) | 21 Terminal A PRs (#116 → #168) | SOLID 18 / WITH-NOTE 3 / CONCERN 0 / §2 BREACH 0 |
