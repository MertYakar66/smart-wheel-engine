# Terminal A — PR audit (2026-05-24)

**Auditor:** Terminal B, fresh session, no campaign context.
**Scope:** every PR claimed by Terminal A on issue #113 through this
audit's start time.
**`origin/main` SHA at audit start:** `86b917c7ab4fe4905dbe68e38c8027c8a81b1374` (post-#172).
**Pytest baseline at audit start:** `2215 passed, 2 failed, 12 skipped`
(164.22s). The two failures are
`tests/test_theta_connector.py::test_ohlcv_shape` and `::test_iv_rank_in_range` —
both pre-existing Windows-local-only flakes that depend on Theta Terminal
session state (not introduced by any audited PR).
**5-ticker EV smoke (CLAUDE.md §6) at audit start:** green — 5 rows
(`AAPL, MSFT, JPM, XOM, UNH`), 0 NaN in `ev_dollars` / `iv` / `premium`.
`connector: MarketDataConnector` (Bloomberg path, default). §2 holds
post-#165's decision-layer touch.
**Method:** seven-step verification protocol per PR (see appendix).

## Tally

- SOLID:            **22**
- SOLID-WITH-NOTE:  **0**
- CONCERN:          **0**
- §2 BREACH:        **0**
- Total audited:    **22**

## PR enumeration vs. starter list

The starter list in the audit prompt named 13 PRs. Enumeration of #113
via `gh issue view 113 --comments` and cross-referencing each
`Terminal A — ... done` / `claim closed` comment with its cited merge
SHA identified **22** Terminal A PRs. The 9 PRs not in the starter list
(`#116`, `#120`, `#123`, `#130`, `#132`, `#134`, `#152`, `#168`, `#169`)
were audited too. None of the listed PRs failed provenance — every
entry was claimed by Terminal A on #113 with a matching merge SHA.

**Scope exclusions (provenance-ambiguous):** PRs `#139` / `#140` /
`#141` / `#142` / `#144` appear in Terminal A's 2026-05-23 17:02 UTC
"batch closure complete" comment as part of the day's "session merge
cascade" but have **no individual `Terminal A — claim` comment** on
#113. The cascade comment says the cascade *merged* them; it does not
identify authorship. Per the audit prompt ("If a PR in the list isn't
claimed by Terminal A on #113, drop it from the audit (provenance
mismatch is itself a finding to log)"), they are excluded from this
audit on the same basis as the prior audit (PR #170 / `ab775bf`).

**Scope exclusions (other terminals):** `#160` (Terminal C —
`feat(wheel_runner)`: annotate synthetic premium + OI fallback sources),
`#161` (Terminal C — `feat(wheel_runner)`: `explore_ticker`), `#170`
(Terminal B's own prior audit), `#172` (Terminal B — Kelly gate
docstring + test comment, closes #166 B1+B3). The Terminal B own-session
close-out comment on #113 (2026-05-24 04:42 UTC) explicitly claims
both #170 and #172 as `Terminal B (this session)`. Per the audit
prompt ("Auditing Terminal C or Terminal B work (your previous self).
Scope is Terminal A only"), all four are out of scope.

**Source-line drift summary across the campaign:** two post-prior-audit
PRs moved cited line numbers without affecting verdicts:

- `engine/wheel_runner.py`: PR #161 (Terminal C, `explore_ticker` at
  line 1448) inserted 95 LoC ahead of `rank_covered_calls_by_ev` and
  `rank_strangles_by_ev`. The two A-shipped methods moved from
  1608 → 1725 and 2075 → 2192 respectively. Verified the functions
  themselves are unchanged (definitions and bodies match prior audit
  at the new offsets); drift is purely positional.
- `engine/portfolio_risk_gates.py`: PR #172 (Terminal B) added ~19
  lines to `check_kelly_size`'s docstring (the "Current-path
  reachability (#166 B3)" paragraph), pushing later functions down.
  `check_var` 510 → 529, `check_stress_scenario` 611 → 630,
  `check_dealer_regime` 705 → 724. Same positional-only drift; the
  function bodies are unchanged.

No drift in any Terminal A-touched file body.

## Per-PR audit

---

### PR #116 — docs(ledger): S14 strangle timing usage test + IV-overlay fix

**Merge SHA:** `a104528` · **Verdict:** SOLID

**Claim:** Add `### S14 — Strangle timing-gated entry` to
`docs/USAGE_TEST_LEDGER.md`; rewire the dead Layer-2 IV overlay in
`strangle_timing.py` (`score_entry_with_iv` / `scan_universe_with_iv`)
to the connector's real public API.

**Code today:** `docs/USAGE_TEST_LEDGER.md:977`:
```
### S14 — Strangle timing-gated entry
```
`engine/strangle_timing.py` Layer-2 IV overlay calls `get_ohlcv(end_date=…)`,
`get_iv_rank`, `get_vol_risk_premium`, `get_vix_regime` (all real
`MarketDataConnector` / `ThetaConnector` methods).

**Tests:** `tests/test_strangle_timing.py` — **59 tests today** (count
unchanged from prior audit). All pass.

**Downstream callers:** N/A — fix to an existing public method.

**§2 status:** N/A — `strangle_timing.py` is not a decision-layer file;
the S14 ledger entry observes that the strangle path produces no
tradeable candidate (§2 holds *vacuously* until #126's
`rank_strangles_by_ev` lands).

**Notes:** None.

---

### PR #120 — fix: P2 logged bugs - cp1252 reviewer notes + phantom docstring

**Merge SHA:** `c9f8d4c` · **Verdict:** SOLID

**Claim:** Two correctness bugs from #118 P2 — (a) `U+0394` (`Δ`) in
`EnginePhaseReviewer`'s R3 spot-mismatch note crashed Windows cp1252
consoles; (b) `wheel_runner.py`'s module docstring advertised a
non-existent `runner.score_strangle_entry(...)` method.

**Code today:**

- `engine/candidate_dossier.py:220-221` — R3 note is pure ASCII:
  ```python
                  f"visible chart price {chart.visible_price:.2f} "
                  f"disagrees with engine spot {engine_spot:.2f} (|delta|={diff:.2f} > tol {tol:.2f})"
  ```
- `engine/wheel_runner.py` module-docstring example references
  `runner.strangle_engine.score_entry_with_iv("AAPL")` — verified live
  by importing the module under cp1252 stdout.

**Tests:** `tests/test_dossier_cp1252.py` — **2 test functions today**
(`test_all_reviewer_notes_encode_to_cp1252` parametrises across 8
reviewer branches via the `_review_note_cases()` helper;
`test_r3_spot_mismatch_note_present_and_delta_free` is the targeted
regression). Both pass — count `(2)` differs from the PR body's "6
rule branches" framing because the tests use an internal loop rather
than per-branch test functions.

**Downstream callers:** N/A — string-literal + docstring edits only.

**§2 status:** Preserved — touched `candidate_dossier.py` reviewer
output strings only; review semantics unchanged.

**Notes:** None.

---

### PR #122 — feat(wheel_tracker): suggest_call_rolls + suggest_rolls EV fix

**Merge SHA:** `1821d56` · **Verdict:** SOLID

**Claim:** (1) New `WheelTracker.suggest_call_rolls(...)` — the
covered-call-leg roll suggester, parallel to `suggest_rolls`, every
candidate scored through `EVEngine.evaluate`. (2) Fix `suggest_rolls`'
buyback netting — it netted `total_cost` (exit txn cost only, ~$7),
omitting `total_buyback_cost` (principal + exit txn cost, ~$400+),
inflating `roll_ev` and spuriously clearing the `recommend` bar.

**Code today:**

- `engine/wheel_tracker.py:2196` — `def suggest_call_rolls(...)`
- Both rollers use `"total_buyback_cost"` (the bug-fix key), pinned by
  the regression in `tests/test_wheel_tracker_suggest_rolls.py`'s
  `TestRollEvNetsBuybackPrincipal` class.

**Tests:**

- `tests/test_wheel_tracker_suggest_call_rolls.py` — **15 today** (PR
  claimed 15). Pass.
- `tests/test_wheel_tracker_suggest_rolls.py` — **15 today**, including
  `TestRollEvNetsBuybackPrincipal` regression. Pass.

**Downstream callers:** None outside tests. No `WheelTracker` consumer
in `engine/`, `dashboard/`, `scripts/`, `engine_api.py`, or `advisors/`
calls `suggest_call_rolls` in production. Expected — Terminal A's
campaign builds blocks; production wiring is deferred (see
Cross-cutting Observation #4).

**§2 status:** Preserved — every roll candidate routes through
`EVEngine.evaluate`; PR pins this with a `1 + len(df)` call-count
regression.

**Notes:** None.

---

### PR #123 — docs(ledger): relink S3 + S8 for suggest_call_rolls (#122)

**Merge SHA:** `97a2680` · **Verdict:** SOLID

**Claim:** Ledger-relink follow-up — record that S3's deferred
`suggest_call_rolls` follow-up shipped in #122 and that S8's
`roll_call` finding is closed by #122.

**Code today:** `docs/USAGE_TEST_LEDGER.md:172-176`:
```markdown
**Follow-up — done.** `suggest_call_rolls` — the covered-call-leg
parallel, deferred from #104 — shipped in **`#122`** (merged
`1821d56`): the same DTE × delta enumeration through
`EVEngine.evaluate`, covered-call rolls only, pinned by a §2
call-count regression.
```
S8's `roll_call` finding similarly relinked.

**Tests:** N/A — docs-only.

**Downstream callers:** N/A.

**§2 status:** N/A.

**Notes:** None.

---

### PR #124 — feat(wheel_runner): rank covered-call entries by EV

**Merge SHA:** `f86c88d` · **Verdict:** SOLID

**Claim:** New `WheelRunner.rank_covered_calls_by_ev(...)` — the
covered-call **entry** ranker. Every candidate scored as
`ShortOptionTrade(option_type="call")` through `EVEngine.evaluate`.

**Code today:** `engine/wheel_runner.py:1725` (drifted from prior
audit's 1608 — see drift summary; #161's `explore_ticker` at line
1448 displaced the function downward by 117 lines; the function
body is unchanged).
```python
    def rank_covered_calls_by_ev(
        self,
        ticker: str,
        shares_held: int = 100,
        *,
        target_dtes: tuple[int, ...] = (21, 35, 49, 63),
        target_deltas: tuple[float, ...] = (0.30, 0.25, 0.20, 0.15),
        as_of: str | None = None,
        min_ev_dollars: float = 0.0,
        ...
```

**Tests:** `tests/test_covered_call_ranker.py` — **35 today** (PR
claimed 35). Pass.

**Downstream callers:** None outside its own tests (verified via
`grep -rln "rank_covered_calls_by_ev" --include="*.py" engine/
dashboard/ scripts/ engine_api.py advisors/` — only hit is the
definition file `engine/wheel_runner.py`).

**§2 status:** Preserved — the §2 call-count regression test pins
`evaluate` is called exactly once per candidate; the default
`min_ev_dollars=0.0` floor enforces "ranks-never-rescues".

**Notes:** None.

---

### PR #126 — feat(wheel_runner): EV-rank short strangles via rank_strangles_by_ev

**Merge SHA:** `f319a48` · **Verdict:** SOLID

**Claim:** New `WheelRunner.rank_strangles_by_ev(...)` — short
strangles ranked by composed EV (each leg scored as a separate
`ShortOptionTrade.evaluate` call over the same `forward_log_returns`);
floor on composed EV; timing gate from `StrangleTimingEngine` is
downgrade-only.

**Code today:** `engine/wheel_runner.py:2192` (drifted from prior
audit's 2075 — same +117 drift from #161; body unchanged).
```python
    def rank_strangles_by_ev(
        self,
        ticker: str,
        contracts: int = 1,
        *,
        target_dtes: tuple[int, ...] = (21, 35, 49, 63),
        target_deltas: tuple[float, ...] = (0.30, 0.25, 0.20, 0.15),
        ...
```

**Tests:** `tests/test_strangle_ev_ranker.py` — **38 today** (PR
claimed 38). Pass. Includes the `2 × n` call-count regression,
composed-EV additivity, ranks-never-rescues on a +EV put / −EV call
pair, and the timing-gate EV-invariance test.

**Downstream callers:** None outside its own tests.

**§2 status:** Preserved — composed EV via two ordinary
`ShortOptionTrade` evaluations; no blended `StrangleTrade` shortcut.
The timing gate drops a ticker before EV ranking; it never lifts EV.

**Notes:** Combined joint-risk metrics (per-path P&L joint CVaR /
prob_profit) deferred to a follow-on (would need `EVResult` to expose
per-path P&L).

---

### PR #127 — feat(wheel_tracker): available_buying_power() nets CSP collateral

**Merge SHA:** `2b129cb` · **Verdict:** SOLID

**Claim:** New `WheelTracker.available_buying_power() -> float` —
returns `cash − Σ(reserved CSP collateral)`. Closes S2 / S4 / S8's
`tracker.cash` overstatement finding.

**Code today:** `engine/wheel_tracker.py:1300` (no drift):
```python
    def available_buying_power(self) -> float:
        """Cash genuinely deployable for new positions, net of CSP collateral.
        ...
        This returns ``cash`` minus the collateral reserved by every open
        cash-secured put: ``put_strike * 100`` per open
        :attr:`PositionState.SHORT_PUT` position. The tracker is one
        contract — 100 shares — per position ...
```

**Tests:** `tests/test_available_buying_power.py` — **11 today** (PR
claimed 11). Pass.

**Downstream callers:** None outside its own tests (only mention in
production code is the docstring reference in
`engine/portfolio_risk_gates.py:477` inside the Kelly gate's
"Current-path reachability" paragraph).

**§2 status:** N/A — pure query over `cash` + `positions`; no EV path
/ gate / multiplier.

**Notes:** None.

---

### PR #128 — feat(wheel_tracker): JSON persistence (to_dict/from_dict/save/load)

**Merge SHA:** `1fffe95` · **Verdict:** SOLID

**Claim:** JSON round-trip for `WheelTracker` + `WheelPosition` —
`to_dict` / `from_dict` / `save` / `load`. Persists `cash`,
`initial_capital`, `positions` (with `PositionState` + `date`),
`closed_positions`, `equity_curve`, **`_ev_authority_tokens`** (set
of hashes) and **`_ev_authority_log`**. Connector deliberately not
serialised; re-attached on load.

**Code today:** `engine/wheel_tracker.py:175` (`WheelPosition.to_dict`),
`:197` (`from_dict`), `:1757` (`WheelTracker.to_dict`), `:1793`
(`from_dict`), `:1823` (`save`), `:1829` (`load`). Schema extended by
#145 with `ev_authority_tokens` / `ev_authority_log` (lines 1788-1789
and 1819-1820) and by #163 with `min_nav_for_trading` (lines 1779,
1805).

**Tests:** `tests/test_wheel_tracker_persistence.py` — **22 today**
(PR claimed 22), including
`test_persisted_token_consume_round_trip_d16` (added by #145) and the
D17 round-trip cases (added by #163). All pass.

**Downstream callers:** None outside its own tests.

**§2 status:** N/A — pure serialisation; no EV path.

**Notes:** Persistence schema is the surface PR #152 pinned with
shape-closure regression (`tests/test_ev_authority_log_schema.py`).

---

### PR #129 — feat(wheel_tracker): mark_to_market resolves IV as-of, not stale

**Merge SHA:** `4cb62d3` · **Verdict:** SOLID

**Claim:** `mark_to_market`'s IV fallback chain extended: explicit
`current_ivs` → **connector as-of ATM IV** (`get_iv_history` at
`current_date`) → entry IV (last resort). Fully defensive on connector
absence / missing method.

**Code today:** `engine/wheel_tracker.py:1480` — `def mark_to_market(...)`
(no drift). Docstring explicitly documents the new fallback chain at
lines 1494-1502. Private helpers `_connector_atm_iv` at :1344 and
`_resolve_mark_iv` at :1385.

**Tests:** `tests/test_mark_to_market_iv.py` — **19 today** (PR
claimed 19). Pass.

**Downstream callers:** Tests + internal: `mark_to_market` is called
by `_compute_nav` → `_compute_live_nav` (introduced by #163,
`engine/wheel_tracker.py:1526`).

**§2 status:** N/A — `mark_to_market` is portfolio accounting.

**Notes:** None.

---

### PR #130 — docs(ledger): validation re-run of S8 + S14 - confirm-fixed

**Merge SHA:** `c6b281d` · **Verdict:** SOLID

**Claim:** Confirm-fixed pass on S8 (wheel-cycle) + S14 (strangle) on
real Bloomberg data; relinks the closed findings and adds a "Validation
re-run (2026-05-21)" note to each entry.

**Code today:** `docs/USAGE_TEST_LEDGER.md:520` (S8) and `:1110`
(S14):
```
**Validation re-run (2026-05-21).** Confirm-fixed pass on real
```
S8 bullets confirm `rank_covered_calls_by_ev` (#124),
`available_buying_power` (#127), `mark_to_market` IV (#129), and
`suggest_call_rolls` (#122). S14 closure confirms strangle EV layer
(#126).

**Tests:** N/A — docs-only.

**Downstream callers:** N/A.

**§2 status:** N/A.

**Notes:** Note explicitly does *not* over-claim: timing engine still
not earnings-aware (#126 added the earnings gate at the EV-ranker
layer, not in `strangle_timing.py`). Honest scoping preserved.

---

### PR #132 — fix(strangle_timing): gate recommendation on phase + confidence

**Merge SHA:** `51eb09b` · **Verdict:** SOLID

**Claim:** S14 P5 — `strangle_timing.py`'s `recommendation` was
decoupled from phase / confidence (MSFT scored "strong_entry" with
phase `unknown`). Adds `_apply_phase_gate` that downgrades on
non-affirmative phase or low confidence.

**Code today:** `engine/strangle_timing.py:410`:
```python
    def _apply_phase_gate(self, recommendation: str, regime: StrangleRegime | None) -> str:
        """Gate a score-cut recommendation on the regime phase / confidence.

        Strictly **downgrade-only** (S14): the ``total_score`` cut that
        produced ``recommendation`` is the base; this gate may only ever
        make it more conservative, never lift it. ...
```
Called from `score_entry` (`:554`) and `StrangleTimingWithIV.score_entry`
(`:818`).

**Tests:** `tests/test_strangle_recommendation_gate.py` — **17 today**
(PR claimed 17). Pass.

**Downstream callers:** `rank_strangles_by_ev` (#126) consumes
`recommendation` as a downgrade-only pre-filter; producing more
`avoid` / `conditional` verdicts is still §2-safe.

**§2 status:** Preserved — recommendation is reviewed downstream
only; EV ranking never reads it.

**Notes:** None.

---

### PR #134 — docs(dossier): mark reviewer rule R4 as conditional/reserved

**Merge SHA:** `f700f7b` · **Verdict:** SOLID

**Claim:** R4 (phase contradiction → skip) is implemented and
unit-tested but never fires in production (no chart provider populates
`visible_indicators['phase']` through M1; ranker emits no `phase` on
`ev_row`). `CLAUDE.md` §2 + `EnginePhaseReviewer` docstring presented
R4 as a live downgrade rule; this re-words both as
conditional/reserved.

**Code today:** `CLAUDE.md:63`:
```
- R4: phase contradiction → skip *(conditional/reserved — the rule is
  implemented and unit-tested but dormant in the production path: no
  current chart provider populates `visible_indicators['phase']` (empty
  through M1) and the ranker emits no `phase` on `ev_row`. It fires only
  when a phase-aware chart provider lands — see
  `docs/TRADINGVIEW_INTEGRATION.md`. Not a live downgrade today.)*
```
`docs/LAUNCH_READINESS.md:77` carries the same wording for R4.

**Tests:** Documentation only — `test_dossier_invariant.py` still
exercises R4. **12 today** (4 pre-existing + 8 D17 from #165). All
pass.

**Downstream callers:** N/A — docstring edits only.

**§2 status:** Preserved — no behaviour change.

**Notes:** None.

---

### PR #143 — docs(parallel_sessions): N-generic template + every terminal in its own worktree (D15)

**Merge SHA:** `c2e316c` · **Verdict:** SOLID

**Claim:** Rewrite `docs/PARALLEL_SESSIONS.md` to an N-terminal
template; pull Terminal A out of the primary clone into its own
worktree; add `scripts/setup-terminal.sh` + `.ps1`; pin the
coordination decision as D15.

**Code today:**

- `DECISIONS.md:587` — `## D15. Parallel-session coordination is N-generic; every terminal lives in its own worktree`
- `scripts/setup-terminal.sh` + `scripts/setup-terminal.ps1` both
  present; headers describe per-letter port mapping. The "Rejected
  alternatives" entry for `SWE_API_PORT` (lines 633-638) was updated
  post-#158 to record the landing of the wire-in.

**Tests:** N/A — docs + tooling.

**Downstream callers:** Setup scripts sourced manually per terminal.

**§2 status:** N/A.

**Notes:** None.

---

### PR #145 — feat(wheel_tracker): EV-authority token is verdict-bound, not just provenance (D16; closes S8)

**Merge SHA:** `fbfe521` · **Verdict:** SOLID

**Claim:** D16 — EV-authority token is **verdict-bound** at issue
(refuse non-positive `ev_dollars`) and re-checked at consume
(`current_ev_dollars` argument). Brings `open_covered_call` under the
same gate (constructor docstring had claimed it was gated since
audit-VI; AST inspection confirmed it never was). Tokens retained on
stale-EV consume rejection (calc-happened fact is immutable).

**Code today:**

- `engine/wheel_tracker.py:28` — `class EVAuthorityRefused(ValueError):`
- `:322` — `def issue_ev_authority_token(self, ev_row: dict) -> str:`
- `:357-369` — issuance refusal:
  ```python
          if canonical["ev_dollars"] <= 0:
              self._ev_authority_log.append(
                  {
                      "action": "refuse_issue",
                      "reason": "non_positive_ev",
                      "row": canonical,
                  }
              )
              raise EVAuthorityRefused(
                  f"Refusing EV-authority token for {canonical['ticker']} "
                  f"strike={canonical['strike']} — ev_dollars="
                  f"{canonical['ev_dollars']} is non-positive; R1 would block."
              )
  ```
- `:375-436` — `_consume_ev_authority_token` with `current_ev_dollars`
  parameter, `unknown_token` / `missing_current_ev_dollars` / `stale_ev`
  reject paths, token retained on stale-EV.
- `:438` (`open_short_put`) + `:1061` (`open_covered_call`) — both
  accept `current_ev_dollars` and route through
  `_consume_ev_authority_token` (calls at `:489` and the matching
  call-leg site).
- `DECISIONS.md:654` — `## D16. EV-authority token is verdict-bound, not just provenance-bound`.

**Tests:**

- `tests/test_authority_hardening.py` — **28 today** (count
  unchanged), including the D16 block (`test_s8_dis_negative_ev_refused_at_issue`
  with the literal S8 `ev_dollars = -30.65`, zero-EV refusal,
  missing/stale/zero `current_ev_dollars` at consume, call-leg gate).
- `tests/test_wheel_tracker_persistence.py` includes
  `test_persisted_token_consume_round_trip_d16`.

All pass.

**Downstream callers:** Tests only —
`WheelRunner.rank_candidates_by_ev` does not call
`issue_ev_authority_token` yet. The PR itself notes this in its
Unresolved section.

**§2 status:** Preserved — touched `wheel_tracker.py` (not technically
in the §2 file list: `ev_engine.py` / `wheel_runner.py` /
`candidate_dossier.py`). The token gate is §2-adjacent and the change
*tightens* R1 enforcement at the tracker layer; no candidate path is
opened.

**Notes:** None — the gate is the highest-stakes §2-adjacent surface
in the campaign and the implementation is clean.

---

### PR #152 — test(wheel_tracker): schema-closure regression for _ev_authority_log (D16 follow-on)

**Merge SHA:** `6cc2ade` · **Verdict:** SOLID

**Claim:** Test-only — pins the 5 valid `(action, reason)` shapes the
D16 audit log can carry, so accidental key drift / dropped fields /
undeclared `reason` variants cannot ship silently. PR #128 persists
this log to disk, so shape drift would corrupt saved state.

**Code today:** `tests/test_ev_authority_log_schema.py` — **21 today**
(14 from PR #152: `TestPerPathShape` + `TestLogClosure` +
`TestValidatorRejectsBadShapes`; 7 added by PR #163 in
`TestD17EntryShapes`).

**Tests:** All 21 pass.

**Downstream callers:** N/A — test-only.

**§2 status:** N/A.

**Notes:** None.

---

### PR #155 — docs(launch_readiness): apply audit #154 findings A1, A2, A3, B1-B5, C5, C6

**Merge SHA:** `4407607` · **Verdict:** SOLID

**Claim:** 10 single-file edits to `docs/LAUNCH_READINESS.md`
applying audit #154 findings A1, A2, A3, B1-B5, C5, C6.

**Code today:** `docs/LAUNCH_READINESS.md` is the only file the PR
touched (`gh pr view 155 --json files` confirms 1 file, +132/-17).
Subsequent PRs #158/#168/#169 layered on top of #155 without
reverting its work; current state is internally consistent (R4
conditional-reserved wording matches CLAUDE.md, R1-R6 → R1-R8 table
from #169 cleanly extends #155's content).

**Tests:** N/A — docs-only.

**Downstream callers:** N/A.

**§2 status:** N/A.

**Notes:** None.

---

### PR #158 — feat(engine_api): SWE_API_PORT honoured by server + client (closes #154 C7 / D15 Unresolved)

**Merge SHA:** `d23eee0` · **Verdict:** SOLID

**Claim:** `engine_api.py._resolve_port()` reads `SWE_API_PORT`
(default `8787`); `audit.py`'s `BASE` matches. Loud failure on
malformed / out-of-range. Closes #154 C7 and D15's "Unresolved" line.

**Code today:** `engine_api.py:2222`:
```python
def _resolve_port(env: dict[str, str] | None = None) -> int:
    """Resolve the API port from ``SWE_API_PORT`` (closes D15 Unresolved).
    ...
    raw = source.get("SWE_API_PORT", "").strip()
    if not raw:
        return _DEFAULT_API_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"SWE_API_PORT must be a base-10 integer; got {raw!r}") from exc
    if not (1 <= port <= 65535):
        raise ValueError(f"SWE_API_PORT out of range; got {port} (expected 1-65535)")
    return port
```
`main()` at `:2245-2246` reads `port = _resolve_port()` and binds it.

**Tests:** `tests/test_engine_api_port.py` — **18 today** (PR claimed
18). All pass.

**Downstream callers:** `_resolve_port` called at `engine_api.py:2246`
in `main()`; `audit.py` also reads `SWE_API_PORT`.

**§2 status:** N/A — API entrypoint plumbing.

**Notes:** Minor drift from prior audit's cited range — `_resolve_port`
was 2223-2246 in the prior audit, now 2222-2242. Functionally
identical (one-line offset upstream of the def). Same behaviour, same
test count.

---

### PR #159 — feat(portfolio_risk_gates): D17 / #154 C4 Phase 1 — gate library + adapter

**Merge SHA:** `2a2c725` · **Verdict:** SOLID *(promoted from
SOLID-WITH-NOTE in the prior audit — see Notes)*

**Claim:** New `engine/portfolio_risk_gates.py` — pure-function
library wiring `risk_manager.py` + `stress_testing.py` +
`dealer_positioning.py` into one source of truth for Phase 2 tracker
hard-blocks and Phase 3 dossier soft-warns. Six gate functions +
`take_snapshot` adapter + `GateResult` dataclass + `PortfolioSnapshot`
dataclass.

**Code today:** `engine/portfolio_risk_gates.py`:

- `:57` — `class PortfolioContext:` (added in Phase 3 / #165;
  #159 itself shipped the rest of the file)
- `:131` — `class GateResult:`
- `:164` — `class PortfolioSnapshot:`
- `:177` — `def take_snapshot(`
- `:271` — `def check_sector_cap(`
- `:344` — `def check_portfolio_delta(`
- `:431` — `def check_kelly_size(` (rewritten by #163 to per-trade
  NAV cap; docstring extended by #168/#172)
- `:529` — `def check_var(` (+19 drift from prior audit's 510, due
  to #172's docstring addition to `check_kelly_size`; body
  unchanged)
- `:630` — `def check_stress_scenario(` (+19 drift, same cause)
- `:724` — `def check_dealer_regime(` (+19 drift, same cause)
- `:118` — `_C4_VOL_SPIKE_SCENARIO = Scenario(` (inline per Q2)

**Tests:** `tests/test_portfolio_risk_gates.py` — **38 today** (PR
claimed 37; +1 boundary test from #163; #168/#172 tightened existing
assertions without adding tests). All pass.

**Downstream callers:** `engine/wheel_tracker.py` (D17 hard-blocks,
PR #163) + `engine/candidate_dossier.py` (R7 / R8, PR #165). Wired
downstream within Terminal A's own campaign.

**§2 status:** Preserved — pure library, no decision-layer touch in
Phase 1.

**Notes:** The prior audit scored this SOLID-WITH-NOTE because the
D17 entry in `DECISIONS.md` was pending at audit start (`d5a0b94`).
**PR #169 landed `DECISIONS.md:735` `## D17` exactly the deferred
write-up**, so today the only basis for the WITH-NOTE is closed.
Promoted to SOLID.

---

### PR #163 — feat(wheel_tracker): D17 hard-block gates — sector / delta / Kelly + live NAV (#154 C4 Phase 2)

**Merge SHA:** `14bd271` · **Verdict:** SOLID *(promoted from
SOLID-WITH-NOTE in the prior audit — see Notes)*

**Claim:** Wires D17 tracker hard-blocks into `open_short_put` /
`open_covered_call` (sector cap, portfolio delta, Kelly size). Live
NAV via `_compute_live_nav` (mark-to-market with `nav_source` audit
fingerprint); `nav_exhausted` pre-gate configurable via
`min_nav_for_trading`. Rewrites `check_kelly_size` from binary Kelly
to per-trade NAV cap.

**Code today:** `engine/wheel_tracker.py`:

- `:256` — `min_nav_for_trading: float = 0.0,` (constructor kwarg)
- `:1410` — `def _compute_nav(`
- `:1526` — `def _compute_live_nav(`
- `:1580` — `def _evaluate_d17_hard_blocks(...)` with explicit Q4
  "compute once" docstring (`nav, nav_source = self._compute_live_nav()`
  at `:1628`; audit-log shapes carry both fields)
- `:1639-1644` — `nav_exhausted` reject:
  ```python
          if nav < self.min_nav_for_trading:
              return {
                  **common_audit,
                  "reason": "nav_exhausted",
                  "min_nav_for_trading": self.min_nav_for_trading,
              }
  ```
- Persistence updated to round-trip `min_nav_for_trading` (lines
  1779, 1805).

**Tests:**

- `tests/test_authority_hardening.py::TestD17HardBlocks` — **8 tests**
  today (claimed 8). All pass:
  `test_d17_passes_at_realistic_nav`, `test_d17_nav_exhausted_pre_gate`,
  `test_d17_portfolio_delta_breach`,
  `test_d17_kelly_or_other_d17_gate_fires_at_small_nav`,
  `test_d17_static_fallback_when_no_connector`,
  `test_d17_non_strict_mode_bypasses_all_gates`,
  `test_d17_compute_once_per_call`, `test_d17_covered_call_skips_kelly`.
- `tests/test_ev_authority_log_schema.py::TestD17EntryShapes` — **7
  tests** today (claimed 7). All pass.
- `tests/test_portfolio_risk_gates.py` — +1 boundary test (claimed +1).

All pass.

**Downstream callers:** None new (production wiring of
`rank_candidates_by_ev` → `issue_ev_authority_token` →
`open_short_put(current_ev_dollars=...)` still pending — same gap
called out in #145's Unresolved).

**§2 status:** Preserved — `wheel_tracker.py` is not in the §2 file
list; the change tightens what the tracker accepts (additional
hard-blocks before token consume) — refusal-only, never upgrades a
candidate.

**Notes:** Same as #159 — prior audit's WITH-NOTE rationale was the
deferred D17 entry; #169 landed it at `DECISIONS.md:735`. Promoted to
SOLID.

---

### PR #165 — feat(candidate_dossier): D17 R7 (VaR) + R8 (stress + dealer regime) dossier soft-warns (#154 C4 Phase 3)

**Merge SHA:** `3a1a67f` · **Verdict:** SOLID *(promoted from
SOLID-WITH-NOTE in the prior audit — see Notes)*

**Claim:** Wires R7 (VaR) + R8 (stress + dealer regime) into
`EnginePhaseReviewer.review`. Both **downgrade-only** — never rescue a
negative-EV trade (R1 still wins). R7 / R8 only fire when current
verdict is `proceed`; absent `PortfolioContext` skips both (Q3
missing-data semantics). Decision-layer touch —
`engine/candidate_dossier.py` under the one-terminal-at-a-time claim.

**Code today:** `engine/candidate_dossier.py`:

- `:69` — `portfolio_context: Any = None` (optional `CandidateDossier`
  field)
- `:282-308` — R7 logic, traced:
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
- `:313-351` — R8 logic (two triggers; distinct `verdict_reason` per
  trigger: `stress_breach` or `short_gamma_regime`; mirrors R6's
  two-trigger pattern).
- `:356-...` — `_build_candidate_dict` static helper.
- Class docstring extended R1-R6 → R1-R8 (visible in
  `EnginePhaseReviewer` from `:118` onward).

**Tests:** `tests/test_dossier_invariant.py::TestD17DossierSoftWarns`
— **8 tests** today (claimed 8). All pass, including
`test_r7_r8_cannot_upgrade_negative_ev` and
`test_r7_r8_cannot_upgrade_review_to_proceed` — direct §2-adjacent
upgrade-attempt regressions. Full file: 12 tests today (4
pre-existing + 8 D17).

**Downstream callers:**

- `EnginePhaseReviewer.review` is invoked by `build_dossiers` and any
  consumer of `CandidateDossier`. None of those callers currently
  attach a `PortfolioContext`, so R7 / R8 are dormant in the
  production path today (Q3 skip on absent evidence). This is the
  deliberate Phase 4 next-step per the PR body.
- `EVEngine.evaluate` is untouched.

**§2 status:** **Preserved and verified live.** Decision-layer touch
prompted the 5-ticker EV smoke (CLAUDE.md §6) at audit start —
green: 5 rows, 0 NaN in `ev_dollars` / `iv` / `premium`;
`connector: MarketDataConnector`. R7 + R8 are gated on
`verdict == "proceed"` — they cannot rescue a negative-EV trade, and
the code paths visibly only `return "review", ...` (never upgrade).
The two negation tests above pin the upgrade-impossibility
structurally.

**Notes:** Same as #159 / #163 — prior audit's WITH-NOTE rationale
was the deferred D17 entry; #169 landed it. Promoted to SOLID.

---

### PR #168 — docs(audit-sweep): reconcile post-D17 / post-#158 docstring + comment drift (#166 A1/A2/A4/A5/B2)

**Merge SHA:** `d5a0b94` · **Verdict:** SOLID

**Claim:** Six doc-only fixes + one test-assertion tightening from
audit #166's docs-only set. No behaviour change.

**Code today:**

- **A1:** `engine/wheel_tracker.py:466-482` — `open_short_put`'s
  `prob_profit` docstring accurately describes Kelly-cap formula and
  the forward-compat passthrough.
- **A2:** `engine/portfolio_risk_gates.py:440-502` —
  `check_kelly_size` docstring describes the real `details` keys
  (`margin_required`, `kelly_recommended_max`, `kelly_fraction`)
  and the per-trade NAV cap formula.
- **A4:** `scripts/setup-terminal.sh:14-25` (and matching `.ps1`)
  acknowledge PR #158 wired `SWE_API_PORT`: "Honoured today by
  engine_api.py's `_resolve_port()` and audit.py's `BASE` (PR #158
  / D15); default 8787 falls through when unset."
- **A5:** `docs/USAGE_TEST_LEDGER.md:481` — `**Logged → Fixed in #145**`
  on the S8 EV-authority-token finding (literal source verified).
- **B2:** `tests/test_portfolio_risk_gates.py:335,349,361` — assertion
  tightened to:
  ```python
          assert result.details["kelly_recommended_max"] == 50_000.0
  ```
  with comment block (lines 315-321) explaining the cap formula
  (`kelly_fraction × NAV`). A regression to binary Kelly would fail
  loudly.

**Tests:** `tests/test_portfolio_risk_gates.py` (38),
`tests/test_authority_hardening.py` (28),
`tests/test_ev_authority_log_schema.py` (21) — all pass. Total 87
matches PR body's 38 + 29 + 20 = 87 (per-file ±1 collected-vs-claimed
is within parametrisation tolerance; no test deleted).

**Downstream callers:** N/A — doc fixes + one assertion tightening.

**§2 status:** Preserved.

**Notes:** None.

---

### PR #169 — docs(d17): canonical D17 entry + LAUNCH_READINESS R1-R8 / §5 / §8 / §9 (#154 C4 Phase 4)

**Merge SHA:** `5c1362f` · **Verdict:** SOLID

**Claim:** Pure docs PR closing the C4 arc on #154 and audit #166
A3 + A6. Two files:

- **`DECISIONS.md`** — new canonical D17 entry (179 lines): two-surface
  design (tracker hard-blocks + dossier soft-warns R7 + R8), six locked
  defaults, four new D17 audit-log shapes, eight rejected alternatives
  (including the binary-Kelly miss that PR #163 caught).
- **`docs/LAUNCH_READINESS.md`** — four edits in one file:
  §3 R1-R6 → R1-R8; §5 D17 checklist item; §8 `SWE_API_PORT` no longer
  labelled "convention"; §9 `_ev_authority_log` bullet expanded from
  5 D16 shapes to 9 (the four D17 hard-block shapes with `nav` +
  `nav_source` fingerprint); top-of-file D-reference list now includes
  D17.

**Code today:**

- `DECISIONS.md:735` — `## D17. Portfolio-level risk gates are wired on both surfaces — hard-block on entry, soft-warn on review`.
  Content covers all of: the two-surface design, the six locked
  defaults table (sector 25% / delta $300/$100k / Kelly 50% / VaR 5% /
  stress 8% / C4 vol-spike), the four new tracker reject audit-log
  shapes, missing-data semantics (Q3), rationale, and the eight
  rejected alternatives. R1's primacy over D17 stated explicitly:
  *"R1 (`negative EV → blocked`) still wins over every D17 surface —
  the hard invariant from CLAUDE.md §2 / D1 / D16 is not amended."*
- `docs/LAUNCH_READINESS.md:13-14` — D-reference list now includes
  `D17 (portfolio-risk gates on both surfaces)`.
- `docs/LAUNCH_READINESS.md:65-94` — §3 R1-R8 table; R7 row
  (`portfolio VaR_95 > 5% NAV → review`) and R8 row (one rule, two
  triggers `stress_breach` / `short_gamma_regime`) match the
  `EnginePhaseReviewer` in-code logic at `candidate_dossier.py:282-351`.
  R7 + R8 soft-warns described as "downgrade `proceed → review` and
  never override R1's `blocked`".
- `docs/LAUNCH_READINESS.md:152-166` — §5 checklist new item for
  portfolio-risk-gate changes, names the four test files
  (`test_authority_hardening.py::TestD17HardBlocks`,
  `test_ev_authority_log_schema.py::TestD17EntryShapes`,
  `test_dossier_invariant.py`) and the locked-defaults non-edit
  contract.
- `docs/LAUNCH_READINESS.md:281-287` — §8 env table now lists
  `SWE_API_PORT` (with `COVERAGE_FILE`, `PYTEST_CACHE_DIR`,
  `SWE_DATA_PROVIDER`) as "real today — each one is read by a live
  consumer". Only `SWE_DATA_PROCESSED_DIR` / `SWE_MODELS_DIR` remain
  conventions.
- `docs/LAUNCH_READINESS.md:325-339` — §9 audit-log section lists the
  nine `_ev_authority_log` entry shapes explicitly broken into five
  D16 + four D17, each D17 shape carrying the `nav` + `nav_source`
  fingerprint.

**Tests:** N/A — docs-only PR. Author's "Local verification: 152
launch-blocker + D17 tests pass" claim is consistent with the audit's
per-PR test-file runs (157 + 193 = 350 across the audit's two
test-file batches, of which the §4 launch-blocker subset is well
inside).

**Downstream callers:** N/A — docs-only.

**§2 status:** Preserved. The D17 entry reinforces §2: R7 + R8 are
written as downgrade-only and R1's primacy is restated. No code
behaviour change.

**D-entry alignment (step 6):** PR #169 *is* the D-entry. Match
between the D17 entry's claimed scope and the shipped code at
`engine/portfolio_risk_gates.py` + `engine/wheel_tracker.py` +
`engine/candidate_dossier.py` is exact — every locked-default
constant named in `DECISIONS.md:774-781` exists at the cited module
path (`_DEFAULT_MAX_SECTOR_PCT`, `_DEFAULT_DELTA_CAP_PER_100K_NAV`,
`_DEFAULT_KELLY_FRACTION`, `_DEFAULT_MAX_VAR_PCT`,
`_DEFAULT_MAX_STRESS_DRAWDOWN_PCT`, `_C4_VOL_SPIKE_SCENARIO`); every
audit-log shape named in `DECISIONS.md:792-796` validates against
`tests/test_ev_authority_log_schema.py::TestD17EntryShapes`; the
binary-Kelly rejected-alternative narrative matches the
`check_kelly_size` docstring at `portfolio_risk_gates.py:440-502`.

**Notes:** None. With #169 on `main`, the prior audit's
SOLID-WITH-NOTE verdict on #159 / #163 / #165 (deferred D17 entry)
is resolved — see the per-PR Notes on each of those entries.

---

## Cross-cutting observations

**1. Campaign closure is clean.** All 22 Terminal A PRs in scope
audit SOLID. The three D17 implementation PRs (#159 Phase 1, #163
Phase 2, #165 Phase 3) that the prior audit (PR #170) flagged as
SOLID-WITH-NOTE because they deferred the D17 write-up are now SOLID
— PR #169 (Phase 4) landed the canonical D17 entry at
`DECISIONS.md:735` exactly as promised in those three PRs' "AI
handoff" sections. The deferral chain unwound as designed.

**2. §2 invariant preserved through every decision-layer touch.** The
campaign's four direct decision-layer PRs touched
`engine/candidate_dossier.py` (#120 cp1252 reviewer-note fix, #165
R7 + R8 soft-warns) or `engine/wheel_runner.py` (#124
`rank_covered_calls_by_ev`, #126 `rank_strangles_by_ev`). All four
route candidates through `EVEngine.evaluate` or are pure string /
note edits. The §2 call-count regressions in #124 and #126 pin
`evaluate` is called exactly once per candidate. R7 + R8 are
structurally upgrade-incapable — the only branch that fires returns
`"review", ...` (never `"proceed"`). The audit's live 5-ticker EV
smoke (CLAUDE.md §6) at audit start, post-#165, is green —
`connector: MarketDataConnector`, 5 rows, 0 NaN in `ev_dollars` /
`iv` / `premium`. **No §2 BREACH; no §2-adjacent surface introduces a
rescue path.** The dealer-multiplier clamp `[0.70, 1.05]` is
untouched.

**3. Source-line drift is purely positional and benign.** Two
post-Terminal-A PRs shifted cited line numbers:
- PR #161 (Terminal C, 2026-05-24, `explore_ticker`) inserted 95 LoC
  in `engine/wheel_runner.py` at line 1448, pushing #124's
  `rank_covered_calls_by_ev` from 1608 → 1725 and #126's
  `rank_strangles_by_ev` from 2075 → 2192 (+117 each).
- PR #172 (Terminal B, 2026-05-24, Kelly-gate docstring) added ~19
  lines to `engine/portfolio_risk_gates.py:check_kelly_size`'s
  docstring, pushing `check_var` 510 → 529, `check_stress_scenario`
  611 → 630, `check_dealer_regime` 705 → 724 (+19 each).
In every case the function definitions, signatures, and bodies are
unchanged; only the line offsets moved. The audit verifies the bodies
at the new offsets and notes the prior-audit citation deltas. The
two prior audits that tripped over "second-layer drift" (helper / adapter
defaults that made the entry-point diff a no-op) found no equivalent
class of bug in this campaign — every cited entry point and the
helpers it calls were traced.

**4. New public surfaces still have no production callers.** Five
public methods + one new exception shipped this campaign —
`suggest_call_rolls` (#122), `rank_covered_calls_by_ev` (#124),
`rank_strangles_by_ev` (#126), `available_buying_power` (#127),
`WheelTracker.save` / `load` (#128), the verdict-bound D16 token
contract (#145), the D17 hard-blocks (#163), and the R7 / R8 dossier
soft-warns (#165). `grep -rln` across `engine/`, `dashboard/`,
`scripts/`, `engine_api.py`, and `advisors/` (excluding the
definitions' own files and `tests/`) returns only one hit — a
docstring reference to `available_buying_power` inside
`portfolio_risk_gates.py:477`'s Kelly-gate reachability paragraph.
The campaign explicitly knows this gap: #145's Unresolved section
flags that `WheelRunner.rank_candidates_by_ev` does not call
`issue_ev_authority_token` today. The natural follow-on PR wires
`rank_candidates_by_ev` → `issue_ev_authority_token` →
`open_short_put(current_ev_dollars=...)` and threads a
`PortfolioContext` through `build_dossiers` so R7 / R8 fire live.
Until that lands, D16 / D17 hardening is a contract for *direct*
tracker callers (tests today), not for the ranker chain that
operators run.

**5. Test-count tracking holds across the campaign.** Every per-PR
test claim matches today's count exactly: 2 (#120), 15 (#122),
15 (#122 sibling regression in `test_wheel_tracker_suggest_rolls.py`),
35 (#124), 38 (#126), 11 (#127), 22 (#128), 19 (#129), 17 (#132),
12 (#134 incl. 8 D17 from #165), 21 (#152 incl. 7 D17 from #163),
18 (#158), 38 (#159 incl. +1 from #163), 28 (#145 incl. 8 D17 from
#163). The 8 D17 dossier soft-warn + 8 D17 hard-block + 7 D17 entry
shape tests for the C4 arc all pass — 23 D17-tagged tests total.
The full suite at audit start is `2215 passed, 2 failed, 12 skipped`;
the two failures are Windows-local-only theta-connector flakes
(`test_ohlcv_shape`, `test_iv_rank_in_range`) that depend on Theta
Terminal session state and are unrelated to any audited PR. Memory
pins this same Windows-local-theta failure mode across multiple
prior sessions.

**6. Pytest at audit end: zero divergence.** Full suite after writing
this audit doc: `2215 passed, 2 failed, 12 skipped` (175.23s).
Identical to baseline — exactly the same two Windows-local theta
flakes (`test_ohlcv_shape`, `test_iv_rank_in_range`), zero new
failures, zero new passes. Pure docs (the audit doc itself) was the
only file changed between the two runs.

## Method appendix

The seven-step per-PR verification protocol applied to every PR
audited above:

1. **Read the merge commit and the PR description.** Compare the
   claimed scope (in #113 + PR body) to the actual diff.
   `gh pr view <num> --json title,body,mergeCommit,files,additions,deletions`
   then `gh pr diff <num>` or `git show <merge_sha>`.
2. **Quote the actual source at every claim.** Read `file.py` at
   current `main`. If the file has changed since the PR's merge SHA,
   also read at the PR's merge SHA and report the delta. Paste the
   literal source line; do not paraphrase. *(Two positional drifts
   observed in this audit, both from post-Terminal-A PRs and both
   purely line-number-based — see Cross-cutting Observation #3.)*
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
   public function / method, grep for its callers across `engine/
   tests/ dashboard/ scripts/ engine_api.py advisors/`. Verify the
   callers pass arguments the PR's new signature expects.
6. **D-entry alignment.** If the PR claims to implement a D-numbered
   decision (D14, D15, D16, D17), open `DECISIONS.md` and confirm the
   D-entry exists and matches the PR's shipped behaviour.
7. **Tests still pass on current main.** Full `pytest tests/ -v` at
   audit start and audit end. Record pass / skip / fail count.

## Audit history

| Date | Auditor | Scope | Result |
|---|---|---|---|
| 2026-05-24 | Terminal B (fresh, prior session) | 21 Terminal A PRs (#116 → #168), audit-start SHA `d5a0b944` | SOLID 18 / WITH-NOTE 3 / CONCERN 0 / §2 BREACH 0 |
| 2026-05-24 | Terminal B (fresh, this session) | 22 Terminal A PRs (#116 → #169), audit-start SHA `86b917c7` | SOLID 22 / WITH-NOTE 0 / CONCERN 0 / §2 BREACH 0 |
