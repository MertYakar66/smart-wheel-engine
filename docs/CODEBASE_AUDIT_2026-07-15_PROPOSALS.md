# Codebase audit — pre-registered fix proposals (2026-07-15)

**Status: SPEC ONLY. NOTHING IS IMPLEMENTED.** Each proposal below is a
ready-to-approve change spec produced by the 2026-07-15 weakness audit
(companion register: `docs/CODEBASE_AUDIT_2026-07-15.md`; evidence thread:
issue #493). The operator approves, rejects, or amends each one; only then does
implementation begin, one at a time, on a branch, with tests.

**Governance:** the decision-layer trio (`engine/ev_engine.py`,
`engine/wheel_runner.py`, `engine/candidate_dossier.py`) is never edited without
explicit consent — flagged per proposal below. Nothing here has been applied.

## Priority index

| P | Sev | Consent | Trio | File | Proposal |
|---|---|---|---|---|---|
| 1 | high | YES | — | `engine/risk_manager.py:363` | Pre-registered fix: remove the stray /100 in gamma_dollars so portfoli |
| 2 | high | no | — | `engine_api.py:646` | Harden CORS localhost allow-list: exact hostname match instead of unbo |
| 3 | medium | YES | YES | `engine/candidate_dossier.py:366` | Pre-registration: activate the dead R6 dealer-positioning downgrade in |
| 4 | medium | YES | YES | `engine/wheel_runner.py:3082` | Thread as_of into get_fundamentals in the covered-call and strangle ra |
| 5 | medium | YES | YES | `engine/wheel_runner.py:899` | Make WheelRunner provider selection observable: log resolved connector |
| 6 | medium | no | — | `data/bloomberg_loader.py:1100` | Align data/bloomberg_loader.get_current_risk_free_rate with the connec |
| 7 | medium | no | — | `engine/stress_testing.py:467` | Standardize the Monte-Carlo t(5) spot shock to unit variance in stress |
| 8 | medium | no | — | `engine/stress_testing.py:791` | Fix annual-vs-daily theta unit bug in stress_testing.greeks_scenario_m |
| 9 | medium | no | — | `engine/wheel_tracker.py:2624` | Remove phantom entry commission+slippage from roll suggesters' hold_ev |
| 10 | medium | YES | YES | `engine/candidate_dossier.py:526` | Emit `contracts` on the put ranker row so dossier soft-warns R7–R10 si |

---

## Pre-registered change proposal: gamma_dollars 100x convexity understatement

### Status
SPEC ONLY — nothing implemented. Approve Option A (recommended) or Option B, or reject. Needs human consent: **yes** (behavioral shift in the R7 decision-layer reviewer + a risk-limit recalibration), even though **no trio file** (`ev_engine.py`/`wheel_runner.py`/`candidate_dossier.py`) is edited.

### Problem (grounded)
`RiskManager.calculate_portfolio_greeks` uses two mismatched dollar-Greek conventions:

- `engine/risk_manager.py:352` — `multiplier = direction * pos["contracts"] * 100` (the `100` = shares/contract).
- `engine/risk_manager.py:362` — `delta_dollars += pos_greeks["delta"] * multiplier * spot` → **includes** the ×100.
- `engine/risk_manager.py:363` — `gamma_dollars += pos_greeks["gamma"] * multiplier * spot * spot / 100` → the `/100` **cancels** the ×100 contract multiplier, so `gamma_dollars = gamma · direction · contracts · S²` (multiplier dropped). The line-49 comment `# Gamma * underlying price^2 * 100 / 100` literally records the cancellation.

Every consumer then applies gamma against a **raw fractional return**, exactly like delta, via `0.5 · gamma_dollars · r²`:

| Site | Code | Input scale |
|---|---|---|
| `_parametric_var:542` | `0.5 * abs(gamma_dollars) * horizon_vol**2` | fractional vol |
| `_historical_var:651` | `0.5 * gamma_dollars * hist_returns**2` | fractional returns |
| `calculate_covariance_var:784,854` | gamma_d built w/ `/100`, `0.5*abs(gamma_d)*vol_h**2` | fractional vol |
| `run_stress_tests:1259/1303/1349/1376` | `0.5 * gamma_dollars * spot_move**2` | spot_move ∈ {-0.10,…} fractional |
| `check_risk_limits:1479` | `abs(gamma_dollars) > max_portfolio_gamma_dollars` | limit |

Because delta_dollars keeps the ×100 and gamma_dollars does not, **the convexity term is 100x too small for every short-gamma book** — precisely the structural tail of a short-put wheel. The identical decimal↔points error was already fixed for **vega** at the consumer side (`* 100.0`, comment "understated 100x", lines 558/667/869); gamma never got the same treatment. `calculate_monte_carlo_var` (full reval) never reads gamma_dollars, so the analytic and MC paths disagree ~100x on convexity today.

### Proposed change — Option A (recommended): unify on "per unit fractional return"
1. `risk_manager.py:363` — drop `/ 100`.
2. `risk_manager.py:784` — drop the identical `/ 100` in `calculate_covariance_var`'s `gamma_d`.
3. `risk_manager.py:73` — rescale `max_portfolio_gamma_dollars` `50000 → 5_000_000` (×100) to preserve the limit gate's pass/fail semantics after gamma_dollars grows 100x. **Flagged as a risk-policy value for you to confirm.**
4. `risk_manager.py:49` — fix comment to `# Gamma * multiplier * underlying price^2 (dollar-gamma per unit return)`.
5. `docs/GREEKS_UNIT_CONTRACT.md` §Gamma (22-25) — set `gamma_dollars = gamma * spot^2 * contracts * multiplier`, `gamma_pnl = 0.5 * gamma_dollars * (fractional_return)^2` (self-consistent, matches delta's doc).
6. `tests/test_risk_manager.py` — add the single-position gamma-P&L test.

No P&L consumer site changes under A — they self-correct.

### Alternative — Option B (lower blast, mirrors vega precedent)
Keep `/100` in gamma_dollars (a "per-1% move" display scale) and multiply the **7** consumer gamma expressions by `100` like the vega `* 100.0` fix. Leaves the limit default and dashboard numbers untouched; costs 7 edits (miss-one risk) and keeps delta≠gamma conventions split. Recommend A for one unified convention.

### Is the safe fix smaller than the finding implies?
Yes, mostly. The **correctness core is two character-level deletions** (`/100` at lines 363 and 784). The limit rescale (line 73), comment (49), doc, and test are the project-discipline hygiene around it. The finding's "multiply every consumer by 100" is Option B's path; Option A avoids touching consumers entirely.

### Blast radius
- Analytic VaR/CVaR (parametric/historical/covariance) and `run_stress_tests` gamma terms corrected ~100x up for short-gamma books.
- `check_risk_limits` gamma gate — needs the limit rescale or it trips spuriously.
- **R7 `check_var` soft-warn** (`portfolio_risk_gates.py:894`) fires proceed→review MORE often on short-gamma books. This is the decision-layer behavior change driving `needs_consent`. **CLAUDE.md §2 invariant is preserved**: reviewers stay downgrade-only; a bigger/more-correct VaR never rescues a negative-EV trade.
- MC VaR unaffected (finally agrees with analytic on convexity). Dashboard $-gamma renders 100x larger (correct).
- `stress_testing.py` (R8 path) is a **separate module** — not touched here; may share the bug → its own audit.

### Acceptance criteria (falsifiable)
1. Single short ATM put: `gamma_dollars == gamma·(-1)·1·100·S²` within 1e-6 (pre-fix fails ×100).
2. `0.5·gamma_dollars·(0.05)²` within 5% of the BSM second-order residual `reprice_pnl − delta_dollars·r` (pre-fix ~100x low).
3. Covariance/historical gamma contribution now same order as MC convexity on a short-gamma book.
4. With limit=5_000_000, a book near the old 50_000 boundary keeps the same `check_risk_limits` verdict.
5. GREEKS_UNIT_CONTRACT.md §Gamma reproduces criterion 2 with no residual 100x; test_greeks_unit_invariants stays green.
6. Full `pytest tests/ -v` green (no test pins the old gamma_dollars magnitude — verified by grep).
7. test_dossier_invariant green; R7 only ever downgrades.

### Test plan
New: `test_gamma_dollars_carries_contract_multiplier`, `test_gamma_convexity_pnl_matches_reprice`, `test_gamma_limit_semantics_after_rescale`. Regression: full suite, with focus on risk_manager VaR ordering/MC determinism/stress shape, portfolio_risk_gates R7, dossier invariant, dashboard display. Control: 5-ticker EV smoke unchanged. Out-of-scope follow-up: separately audit `stress_testing.py` for the same convention.
---

# Pre-registered change proposal — Harden CORS localhost allow-list (engine_api.py)

**Status:** spec only, nothing implemented. Interface layer; does **not** touch the decision-layer trio (`ev_engine.py` / `wheel_runner.py` / `candidate_dossier.py`). Normal branch + PR; no §2/§3 hard-invariant surface involved.

## Problem (grounded)
`_resolve_cors_origin` (engine_api.py:632-652) chooses which request `Origin` to reflect in `Access-Control-Allow-Origin`. The loopback gate at **engine_api.py:646** is:

```python
if any(origin.startswith(p) for p in _LOCALHOST_ORIGIN_PREFIXES):
    return origin
```

with boundary-less prefixes at engine_api.py:622-629 (`"http://localhost"`, `"http://127.0.0.1"`, `"http://[::1]"`, + https). `startswith` has no host boundary, so any origin whose host *starts with* one of those strings is reflected. Reproduced:

| Origin | `startswith` gate | Result today |
|---|---|---|
| `http://localhost.evil.com` | matches `http://localhost` | **echoed into ACAO (bug)** |
| `http://127.0.0.1.evil.com` | matches `http://127.0.0.1` | **echoed (bug)** |
| `http://127.0.0.1@evil.com` | matches `http://127.0.0.1` | **echoed (bug)** |

This defeats the R3 default-deny intent documented in the function's own docstring (engine_api.py:638-641). The header is emitted on every JSON response (`_send_json`, engine_api.py:673-676) and the preflight (`do_OPTIONS`, engine_api.py:709-712).

**Impact is bounded but real.** No `Access-Control-Allow-Credentials` is ever set (none in engine_api.py) and the port is loopback-only by default (`_resolve_host` -> `127.0.0.1`, engine_api.py:3455-3470). So the exposure is a cross-origin **read** of this unauthenticated local API by a page the victim opens at `http://localhost.evil.com` (or a DNS-rebinding variant) — precisely the class R3 was added to close. Severity **high** as reported; the caveats are why it is a default-deny bypass rather than an authenticated-data breach.

## Proposed change
Parse the origin and exact-match host + scheme. `urlparse` is already imported (engine_api.py:67) — **no new import**.

```python
# engine_api.py:622-629  (replace the prefix tuple)
_LOCALHOST_ORIGIN_SCHEMES = frozenset({"http", "https"})
_LOCALHOST_ORIGIN_HOSTS   = frozenset({"localhost", "127.0.0.1", "::1"})

# engine_api.py:646  (replace the any(startswith) line)
parts = urlparse(origin)
if parts.scheme in _LOCALHOST_ORIGIN_SCHEMES and parts.hostname in _LOCALHOST_ORIGIN_HOSTS:
    return origin
```

`urlparse().hostname` strips IPv6 brackets (`http://[::1]:3000` -> `::1`) and userinfo (`http://127.0.0.1@evil.com` -> `evil.com`), so all three bypasses resolve to a host outside the set and are denied; every legitimate loopback origin (any port) still matches. The empty-guard (643-644) and the `SWE_API_CORS_ORIGIN` exact-string branch (648-651, already correct) are untouched.

**Tighter than the finding's sketch on purpose:** I keep scheme restricted to `{http, https}` (preserving the original prefix list's http/https-only intent and additionally denying `file://localhost`), rather than "any scheme". The only *loosening* vs. today is that mixed-case (`HTTP://LOCALHOST`) now matches — which is correct, since Origin scheme/host are case-insensitive.

## Files touched
- `engine_api.py` — constant at lines 622-629; one line in `_resolve_cors_origin` at 646.
- `tests/test_engine_api_hardening.py` — add negative-case coverage adjacent to the existing CORS tests (lines 156-181).

## Blast radius
`_resolve_cors_origin` <- `_cors_origin_header` (658-664) <- `_send_json` (673, every response) and `do_OPTIONS` (709, preflight). Only other references are the four unit tests. Output governs only whether `Access-Control-Allow-Origin` + `Vary: Origin` are emitted — never bodies, status codes, the ranker, or the trio. Non-browser callers send no `Origin` and short-circuit at 643-644 (unchanged). Net: strictly fewer origins reflected — three spoofs denied, zero legitimate origins lost.

## Falsifiable acceptance criteria
1. `_resolve_cors_origin("http://localhost.evil.com", env={})` -> `None` (today: returns the origin).
2. `_resolve_cors_origin("http://127.0.0.1.evil.com", env={})` -> `None` (today: returns the origin).
3. `_resolve_cors_origin("http://127.0.0.1@evil.com", env={})` -> `None`.
4. `_resolve_cors_origin("file://localhost", env={})` -> `None`.
5. Regression: `http://localhost:3000`, `https://localhost`, `http://127.0.0.1:8787`, `http://[::1]:3000` each return themselves unchanged.
6. Unchanged: `https://evil.example.com` -> `None`; `None`/`""` -> `None`; `SWE_API_CORS_ORIGIN` exact-match test still passes.
7. `pytest tests/ -v` green; no decision-layer test diffs.

## Test plan
- Add `test_cors_denies_localhost_prefix_spoofs` (parametrized: `localhost.evil.com`, `127.0.0.1.evil.com`, `[::1].evil.com`, userinfo, `localhostx`, `file://localhost`, `null`) — all assert `is None`.
- Keep the four existing CORS tests as regression guards; optionally add `HTTP://LOCALHOST:3000` as accepted to lock the case-insensitive behavior.
- Pre-fix sanity: run criteria 1-3 against current code and confirm they FAIL (proves the test bites).
- `pytest tests/test_engine_api_hardening.py -v`, then full `pytest tests/ -v` (CLAUDE.md §4.3).

## Consent
Does not touch the trio and hits no §3 NEVER item -> **no trio-consent gate**. Ship via branch + PR per project rules once this spec is approved.
---

## Pre-registration — Activate the dead R6 dealer-positioning downgrade

**Status:** SPEC ONLY. Nothing implemented. Requires explicit human consent (touches the decision-layer trio member `engine/candidate_dossier.py`).

### 1. Problem (grounded)
R6 is unreachable in production. Its guard reads `ms = getattr(dossier, "market_structure", None)` / `if ms is not None and verdict == "proceed":` (`engine/candidate_dossier.py:366-367`), but:
- `CandidateDossier.market_structure` defaults to `None` (`candidate_dossier.py:83`).
- `build_dossiers`, the only production constructor, sets `ticker`, `ev_row`, `chart_context`, `portfolio_context`, `vix_level` — never `market_structure` (`candidate_dossier.py:670-676`).

So `ms` is always `None` on every operator path and lines 368-389 never execute. The ranker computes a per-candidate `MarketStructure` (`wheel_runner.py:2258`), passes it to `EVEngine.evaluate` for the multiplier (`wheel_runner.py:2301`), then **discards the object** — only scalars `dealer_regime` (`:2433`) and `nearest_put_wall_strike` (`:2443`) reach `ev_row`, gated on `include_diagnostic_fields` which `build_candidate_dossiers` sets true (`:4266`) and forwards to `build_dossiers` (`:4292`).

`dossier.market_structure` is **read** only at `candidate_dossier.py:366` and **set** only in tests/verification scripts. Corroborated by `archive/2026-06/ADVERSARIAL_WEAKNESS_REVIEW_2026-06-15.md:320-322`. Unlike R4 (docstring labels it "reserved/dormant"), R6 is presented as live in CLAUDE.md §2 and `candidate_dossier.py:201-205`, overstating the engine's live protections.

### 2. Proposed change (minimal — smaller than the finding's sketch)
Inside the R6 block only: prefer `dossier.market_structure` when present (unchanged), else read `regime` from `ev_row["dealer_regime"]` and (short_gamma branch) `ev_row["nearest_put_wall_strike"]`, applying identical downgrade logic and verdict_reasons. No change to `wheel_runner.py`/`ev_engine.py` — the scalars R6 needs already reach `ev_row`. This avoids the finding's proposal of carrying a non-serializable `MarketStructure` object out of `rank_candidates_by_ev` (which would change the one central ranker's contract or pollute `ev_row.to_dict()` used by `engine_api`).

Downgrade-only and invariant-safe: R6 runs only when verdict is already `proceed` (R1 blocks negative/non-finite EV first), moves only `proceed → review`, never touches `ev_raw` or the `[0.70,1.05]` dealer clamp. No MarketStructure and no diagnostics → no-op, matching today and the R7-R11 "no soft-warn on absent evidence" convention.

**Alternative (docs-only, no consent, lower value):** relabel R6 "reserved/dormant" like R4 in the docstring + CLAUDE.md §2 so the contract stops presenting a dead rule as live, leaving behavior unchanged. Preferred path is activation, since R6 was clearly wired to be live and both other trio members already emit its inputs.

### 3. Files touched
- `engine/candidate_dossier.py` — R6 block (~359-389) + docstring point 6 (~201-205) + field comment (~80-83). **TRIO.**
- `tests/test_r6_dealer_wiring.py` (new) or extend `tests/test_decision_layer_wiring.py` — mirrors `tests/test_r11_elevated_vol.py:109`.
- `CLAUDE.md` §2 R6 bullet — document the ev_row-fallback wiring (mirror the R11 bullet).
- (optional) `docs/LAUNCH_READINESS.md` R6 row.

### 4. Blast radius
Reaches `/api/tv/dossier` and the dashboard trade ticket via `build_candidate_dossiers → build_dossiers → EnginePhaseReviewer.review`. Effect: some `proceed` → `review` when dealer positioning yields a chain-backed `short_gamma_amplifying` (strike ≥ put wall) or `near_flip` regime. On the Bloomberg-CSV sandbox (chains often unavailable) `dealer_regime` is None → still a no-op. Never rescues/upgrades (§2 preserved). Reviewers given a real `market_structure` (existing tests) unchanged. No serialization impact.

### 5. Acceptance criteria (falsifiable)
AC1 short_gamma fires (`dealer_short_gamma_above_put_wall`); PRE-FIX = `proceed`. AC2 near_flip fires (`dealer_near_flip`); PRE-FIX = `proceed`. AC3/AC4 no false positives (long_gamma / strike-below-wall stay `proceed`). AC5 negative-EV row stays `blocked/negative_ev` (R6 never runs). AC6 explicit-object precedence keeps existing R6 tests green. AC7 `pytest tests/ -v` fully green.

### 6. Test plan
RED first (assert AC1/AC2 fail on the unmodified tree, proving R6 dead) → apply edit → GREEN (AC1-AC6) → optional end-to-end via `build_candidate_dossiers` with a fake short-gamma chain → mandatory full `pytest tests/ -v` (trio cross-cutting) incl. `test_dossier_downgrade_property.py` and launch-blocker subset → branch + PR (never main), PR body noting the pre-registered, consented trio change.

### 7. Consent gate
Touches `engine/candidate_dossier.py` (trio) and changes live verdict distribution (activates a dormant downgrade). Do not implement until the human approves this pre-registration.
---

## Pre-registered change proposal — PIT dividend carry-q in the covered-call and strangle rankers

**Status:** spec only — no code written. Touches the decision-layer trio (`engine/wheel_runner.py`), so **explicit human consent is required before implementation** (CLAUDE.md §3).

### Problem (grounded)
`rank_covered_calls_by_ev` (engine/wheel_runner.py def @2824) and `rank_strangles_by_ev` (def @3460) accept an `as_of` and honor it for spot, IV, risk-free rate, and earnings — but resolve fundamentals with a **dateless snapshot** call:
- `engine/wheel_runner.py:3082` → `conn.get_fundamentals(ticker)`
- `engine/wheel_runner.py:3715` → `conn.get_fundamentals(ticker)`

The `dividend_yield` from that dict becomes `div_q`, which drives the BSM strike solve, the synthetic premium, `ShortOptionTrade.dividend_yield`, and `EVEngine.evaluate` (covered calls: 3242 / 3271 / 3313 / 3322; strangles: 3879,3908 / 3937,3969 / 4010,4026 / 4033). On any historical run the carry-q is the **2026 snapshot yield applied to a past valuation date** — lookahead.

The puts ranker already does this correctly at `engine/wheel_runner.py:1661-1667`, and the production connector's own docstring calls out the gap verbatim at `engine/data_connector.py:1540-1541` ("the covered-call and strangle rankers still call the snapshot form (no `as_of`)"). The connector already resolves a PIT dividend when asked (engine/data_connector.py:1567-1570); the rankers just never ask.

### Proposed change (minimal, mirrors precedent)
Replace each bare snapshot call with the puts ranker's exact pattern:

```python
if as_of is None:
    fundamentals = conn.get_fundamentals(ticker) or {}
else:
    try:
        fundamentals = conn.get_fundamentals(ticker, as_of=as_of) or {}
    except TypeError:
        fundamentals = conn.get_fundamentals(ticker) or {}
```

The `TypeError` fallback is required — ThetaConnector (engine/theta_connector.py:620), data/consolidated_loader.py:532, and ~20 test stubs define `get_fundamentals(self, ticker)` with no `as_of`.

This is exactly the finding's fix sketch; **the safe fix is not smaller**. No signature change, no downstream edit. `as_of=None` (live) is a provable no-op — identical call. Only `dividend_yield` is made PIT by the connector, so the fundamentals IV-fallback fields are untouched, matching the puts ranker.

### Files touched
| File | Trio? | Edit |
|---|---|---|
| `engine/wheel_runner.py` | **yes** | lines 3082 and 3715 → as_of-conditional block (copy of 1661-1667) |
| `tests/test_ranker_iv_pit.py` | no | extend `TestFundamentalsAsOfThreading` with CC + strangle wire/no-op/fallback/EV-sensitivity tests |
| `engine/data_connector.py` | no (docstring) | refresh the stale wiring statement at 1540-1541 |

Optional: one-line DECISIONS.md entry paralleling the #428/#429 puts note.

### Blast radius
- Production callers: `scripts/s47_trader_session_2026_03_20.py:692`, `scripts/audit_full_wheel.py:215` (audit/analysis scripts). No engine_api/dashboard caller. `rank_strangles_by_ev` has no non-test caller.
- **Locked claims S27/S32/S34/S35 cannot move** — their reproducer (`scripts/run_paper_book.py:531`) calls only `rank_candidates_by_ev`. No #429-style re-baseline needed.
- Live path unchanged (as_of=None no-op). Invariant-safe: `div_q` is an input *to* `EVEngine.evaluate`; nothing bypasses evaluate or rescues a negative-EV candidate.

### Falsifiable acceptance criteria
1. **AC1/AC2 (wire):** spy connector records `as_of="2026-03-15"` on `get_fundamentals` for both rankers; all non-None recorded values equal the ranker's `as_of`. (Fails today.)
2. **AC3 (live no-op):** `as_of=None` runs call `get_fundamentals` only with `as_of=None`.
3. **AC4 (legacy fallback):** a `TypeError`-raising `get_fundamentals(self, ticker)` stub still ranks non-empty at `as_of="2026-03-15"`.
4. **AC5 (substantive PIT effect):** a connector whose `dividend_yield` varies by `as_of` yields a different `strike`/`premium`/EV column across two dates — proving the PIT dividend reaches BSM/evaluate (identical under current code; differs under the fix).
5. **AC6 (locked claims):** `tests/test_backtest_regression.py` fingerprint tests pass with no re-baseline.
6. **AC7 (suite):** `pytest tests/ -v` green; ev-smoke returns five non-null rows.

### Test plan
Add CC/strangle analogues to `TestFundamentalsAsOfThreading` (mirror 484-512) incl. an EV-sensitivity test with a negative control (confirm it fails pre-fix). Run the targeted PIT + CC + strangle files, the fast backtest-regression fingerprint subset (expect no re-baseline), the full `pytest tests/ -v` (trio mandate), and an ev-smoke live sanity check with the selected provider logged.
---

## Pre-registered change proposal — Observable provider selection in `WheelRunner.connector`

### Finding (confirmed, grounded)
`engine/wheel_runner.py:893-908`, the `connector` property, selects the data connector from `SWE_DATA_PROVIDER` with **no log line** and a **catch-all `else`**:

```python
provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg").lower()   # :899
if provider == "theta":
    self._connector = ThetaConnector(str(self.data_dir))              # :903
else:
    self._connector = MarketDataConnector(str(self.data_dir))         # :907  <-- catch-all
```

Any token that is not exactly `"theta"` — `"thetadata"`, `"theta_terminal"`, `"bbg"`, a typo — silently yields the Bloomberg `MarketDataConnector` with nothing in the process log. This violates **CLAUDE.md §4.1**: *"Always log which provider was actually selected — silent provider selection is a recurring bug source."* The file already has `logger = logging.getLogger(__name__)` (`:39`) and uses `logger.warning(...)` at `:411/:431/:550/:566/:600/:1008/:2054`, so no new import is needed. The sibling site `engine_api.py:215-229` logs its selection but shares the silent-fallback shape — the two sites disagree, and `wheel_runner.py` (the canonical selection site per `MODULE_INDEX.md:46`) is the one with zero visibility.

### Change (observability-only; conservative)
Inside the property:
1. `provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg").strip().lower()`; empty/whitespace → default `"bloomberg"` (no warning).
2. Recognized set `{"theta", "bloomberg"}`. `"theta"`→`ThetaConnector`; `"bloomberg"`/empty→`MarketDataConnector`; **any other non-empty token → `MarketDataConnector` (fallback preserved) plus a `logger.warning` echoing the token, naming the recognized set, and stating the fallback.**
3. On every resolution, one `logger.info` naming the resolved token + the concrete class instantiated. Fires at most once per instance (property caches `self._connector`).

**Smaller than the finding's "treat as an error":** default stays warn+fallback so the existing `test_unknown_provider_falls_back_to_bloomberg` and the ~dozens of `WheelRunner`-constructing scripts don't break. **Optional** hard `ValueError` behind an opt-in `SWE_STRICT_PROVIDER` flag (default OFF) — human may accept or drop.

### Why it is §2/§3-safe
The property returns the identical connector object for every input as today; it contains no `ev_raw`/`ev_dollars`/`EVResult`/dealer/verdict reference, does not touch `rank_candidates_by_ev`, the `[0.70,1.05]` clamp, or the downgrade-only reviewer contract. The only new output is log records.

### Files touched
- `engine/wheel_runner.py` — the `connector` property (~893-908). **TRIO file → explicit human consent required.**
- `tests/test_wheel_runner_coverage.py` — extend `TestWheelRunnerConnector` (~69-96) with `caplog` assertions.
- *Follow-up, NOT bundled:* `engine_api.py::get_connector` (215-229) for cross-site consistency.

### Falsifiable acceptance criteria
- **AC1** unset/`bloomberg`/`theta` → one INFO record with the correct concrete class. Fails if absent/wrong.
- **AC2** `"thetadata"`/`"theta_terminal"` → one WARNING containing token + recognized set + "fallback". Fails if silent.
- **AC3** returned class identical to today for every input; existing fallback test passes unmodified; no EV/dealer reference in the property.
- **AC4** unset/`""`/whitespace → no WARNING.
- **AC5** second `.connector` access emits no additional record (cached).
- **AC6 (optional)** `SWE_STRICT_PROVIDER` truthy + bad token → `ValueError`; unset → AC2.
- Suite: `pytest tests/ -v`, launch-blocker subset, and `tests/test_preflight_environment.py` stay green.

### Test plan
Add `caplog` tests for AC1/AC2/AC4/AC5 in `TestWheelRunnerConnector`; keep `test_unknown_provider_falls_back_to_bloomberg` for AC3; add a strict-mode test if AC6 is accepted. Run full `pytest tests/ -v` (trio edit), the `launch-blockers` skill, `tests/test_preflight_environment.py`, and the 5-ticker `ev-smoke` regression. Manual: `SWE_DATA_PROVIDER=thetadata python -c "import logging; logging.basicConfig(level=logging.INFO); from engine.wheel_runner import WheelRunner; WheelRunner().connector"` → one WARNING naming the fallback.

### Consent
Touches the decision-layer trio (`wheel_runner.py`). **Nothing is implemented in this proposal** — it is a spec for human approval. On approval: branch + PR per CLAUDE.md §3 (never commit to `main`).
---

# Pre-registered change proposal — Fix the unfixed rate-fallback twin (#378/W37 sibling)

**Status:** spec only, nothing implemented. Data-layer only; does NOT touch the decision-layer trio, so no trio-consent gate — but still branch + PR per CLAUDE.md §3.

## Problem (grounded)
`data/bloomberg_loader.py:1084-1113` `get_current_risk_free_rate(rates_df, tenor)` returns a hard-coded `0.05` at three missing-data branches:
- line **1100** — `rates_df is None or rates_df.empty`
- line **1106** — no `rate_*` columns present
- line **1111** — chosen tenor is all-NaN after `dropna`

There is no `fallback` parameter and no log line, even though a module `logger` already exists (`data/bloomberg_loader.py:38`).

This is the still-unfixed twin of the accessor fixed under **W37 / #378**. The sibling `engine/data_integration.py:300-356` already defaults `fallback=float("nan")` and returns `fallback` at every missing branch; its docstring (lines 327-332) is the explicit record:
> "#378 / W37: the missing-data fallback was a **silent 0.05**, diverging from the connector's NaN-on-missing contract — a future EV-path caller would get a guessed 5% instead of a fail-closed NaN."

The canonical contract is NaN-on-missing: `MarketDataConnector.get_risk_free_rate` (asserted by `tests/test_data_connector.py:606-616`). Two same-named accessors now disagree on missing-data semantics, and the loader injects a fabricated 5% with no audit trail.

## Why this is medium, not high
Blast-radius trace shows this accessor does **not** feed the EV/decision path:
- Called only by `data/pipeline.py:331` (`DataPipeline.load_rates`, no fallback) and re-exported at `data/__init__.py:62`.
- `DataPipeline._risk_free_rate` → `DataPipeline.get_risk_free_rate()` (`pipeline.py:542-550`), which has its **own** `return 0.05` at line 550.
- The only non-test consumer of `DataPipeline.get_risk_free_rate()` is the docstring example at `pipeline.py:26`. No engine file imports `DataPipeline`; the ranker/EV path resolves its rate via the *fixed* `engine.data_integration` twin (`wheel_runner.py:1058-1066`, `engine_api.py:2053-2067`).

So this is a latent contract-divergence + observability hazard, not live EV corruption. Fixing it (a) forecloses the exact "future EV-path caller gets a guessed 5%" risk W37 pre-empted, (b) restores parity with the connector, (c) adds an audit log.

## Proposed change
Mirror the W37 fix on the loader twin, plus one behavior-preserving call-site update:
1. `data/bloomberg_loader.py::get_current_risk_free_rate` — add `fallback: float = float("nan")`; return `fallback` at 1100/1106/1111; `logger.warning(...)` on each branch naming the reason.
2. `data/pipeline.py:331` — call `get_current_risk_free_rate(self._rates, fallback=0.05)` so `DataPipeline`'s documented "Returns 0.05 if no rates data loaded" stays byte-for-byte identical.
3. Tests — update `tests/test_bloomberg_loader.py::test_default_rate_no_data` (465-468) to the NaN-default + explicit-fallback contract; add tenorless / all-NaN / caplog cases; keep the DataPipeline parity test (584-590) green.

**Files touched:** `data/bloomberg_loader.py`, `data/pipeline.py`, `tests/test_bloomberg_loader.py`.
**Not touched:** the trio (`engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/candidate_dossier.py`); `data/__init__.py` (name unchanged, signature backward-compatible); `data/consolidated_loader.py`.

## Scope honesty
- **Smaller safe variant available:** logging-only (keep 0.05, just log) is zero-behavior-change but leaves the contract divergence — the entire "unfixed twin" point — unresolved. The parity fix is recommended; the reviewer may down-scope to logging-only.
- **Adjacent, out of scope:** `data/consolidated_loader.py:565-576` still returns a silent `0.05` (569) **and** carries the separate `/100 if >1` E2 heuristic (576). It is on the primary DataPipeline consolidated path (`pipeline.py:247`). Flagged as a distinct finding so the twin-set is known to be not fully closed by this proposal.

## Falsifiable acceptance criteria
1. `pd.isna(get_current_risk_free_rate(None))` is True (was 0.05).
2. `get_current_risk_free_rate(None, fallback=0.05) == 0.05`.
3. Tenorless frame → NaN default, `fallback` when passed.
4. All-NaN tenor → NaN default, `fallback` when passed.
5. Each fallback branch emits exactly one WARNING record (caplog).
6. Valid-data path unchanged: `test_get_current_rate` (458-463) passes with no edit.
7. `DataPipeline.get_risk_free_rate()` still returns 0.05 on missing rates (parity test 584-590 + new regression case pass).
8. `pytest tests/ -v` fully green.

## Test plan
Targeted `pytest tests/test_bloomberg_loader.py tests/test_data_integration.py tests/test_data_integrity_bloomberg.py tests/test_data_connector.py -v`, then full `pytest tests/ -v` (shared re-exported accessor). Branch + PR; no direct commit to main.
---

# Pre-registered change proposal — Standardize Monte-Carlo t(5) spot shock to unit variance

**Status:** Spec only. Nothing implemented. Awaiting human go/no-go.
**Severity:** medium · **Touches decision trio (ev_engine/wheel_runner/candidate_dossier):** no · **CLAUDE.md §2 invariants touched:** none

## 1. Problem (grounded)

`engine/stress_testing.py::monte_carlo_stress`, lines 466-470:

```python
466  df = 5  # degrees of freedom
467  z = float(stats.t.rvs(df, random_state=rng.integers(2**31)))
468  avg_iv = np.mean([p["iv"] for p in positions]) if positions else 0.20
469  daily_vol = avg_iv / np.sqrt(252)
470  spot_change = z * daily_vol * np.sqrt(horizon_days)
```

Line 470 uses `daily_vol * sqrt(horizon_days)` as the horizon standard deviation of the spot move — valid only if `z` is a unit-variance shock. A raw `stats.t.rvs(5)` has variance `df/(df-2) = 5/3 ≈ 1.667`, so std ≈ **1.2910**. The simulated spot-change **body is ~29.1% wider** than the implied vol specifies. This is over-dispersion layered on top of the intended fat tails — not the fat tails themselves. Every downstream risk number (`var_95`, `var_99`, `cvar_95`, `prob_10pct_loss`, lines 494-500) is therefore systematically too extreme.

The correct pattern already exists in this codebase on the HAR-RV path — `engine/forward_distribution.py:305-306`:

```python
z = rng.standard_t(df_t, size=(n_scenarios, horizon_days))
# Rescale to unit variance for the t distribution
z = z * np.sqrt((df_t - 2) / df_t)
```

The MC path just omits that one line.

## 2. Proposed change (minimal — exactly the finding's sketch)

Insert one line after `engine/stress_testing.py:467`:

```python
z = float(stats.t.rvs(df, random_state=rng.integers(2**31)))
z *= np.sqrt((df - 2) / df)  # rescale t(5) to unit variance (mirrors forward_distribution.py:306)
```

For df=5 this multiplies each draw by sqrt(3/5) ≈ 0.7746, making the body's variance exactly 1 while preserving the t(5) shape (and therefore the intended fat tails — kurtosis is scale-invariant). The downstream IV shock at line 473 (`iv_shock = -spot_change * 2`) rides along and is corrected by the same factor in its spot-coupled term; the additive `rng.normal(0, vol_of_vol*avg_iv)` term at line 474 is independent and unchanged.

**Explicitly out of scope** (kept as-is): the per-iteration `rng.integers(2**31)` reseed, the `-spot_change * 2` IV coupling constant, and the `df = 5` choice. No smaller safe fix exists; nothing larger is justified. `(df - 2) / df` is float division in Python 3 — no cast needed.

## 3. Files touched

- `engine/stress_testing.py` — 1-line insert inside `monte_carlo_stress`.
- `tests/test_tail_copula_stress_invariants.py` — 1 new regression test class beside the existing W46/W47 MC invariants.

No trio files. No API, reviewer, or config changes.

## 4. Blast radius

`monte_carlo_stress` callers (full repo grep): `tests/test_stress_testing.py:169`, `tests/test_tail_copula_stress_invariants.py` (W46/W47), `scripts/feature_smoke_test.py:986`, `dashboard/quant_dashboard.py:847` (read-only display menu). **Not** called by `EVEngine.evaluate`, `wheel_runner.rank_candidates_by_ev`, or any `EnginePhaseReviewer` rule.

R8's stress soft-warn uses the **deterministic** `_C4_VOL_SPIKE_SCENARIO` (-10% spot / +30% IV) via `run_scenario` (`engine/portfolio_risk_gates.py:159-165`, `check_stress_scenario` :944-1003) — **not** `monte_carlo_stress`. So no verdict, no `ev_raw`/`ev_dollars`, and no dealer multiplier is affected. CLAUDE.md §2 invariants remain intact.

**Governance note for the approver:** a prior task's `<rule_outs>` deliberately forbade editing `stress_testing.py` (DECISIONS.md:994-1000; workaround comment at `engine/portfolio_risk_gates.py:154-158`). That was scoped to that task, not a standing freeze. Approving this proposal means explicitly re-opening `stress_testing.py` for this fix.

## 5. Falsifiable acceptance criteria

1. **Dispersion matches vol:** sample std of the reconstructed spot-change body = `avg_iv/sqrt(252)*sqrt(horizon)` within ~2% at N=200k (pre-fix ratio ~1.291 → post-fix ~1.00).
2. **Unit-variance shock:** std of `t.rvs(5)*sqrt(3/5)` over N≥200k ∈ [0.97, 1.03].
3. **Fat tails preserved:** excess kurtosis of the standardized shock stays ~6 (> 3), i.e. body narrows without going Gaussian.
4. **Invariants hold:** W46 (seed reproducibility) and W47 (`cvar_95 <= var_95 <= 0`) and `test_monte_carlo_stress` still pass — none pin magnitudes.
5. **Scope proof:** `git diff --name-only` shows only the two files; full `pytest tests/ -v` + launch-blocker subset green.

## 6. Test plan

1. New W-series regression test: standardized draw std ∈ [0.97,1.03] and excess kurtosis > 3; plus an assertion that the *un-standardized* reference draw has std > 1.25 (proves the test would have been red pre-fix).
2. Dispersion test: reconstruct line-470 expression with fixed seed; assert realized horizon vol == implied within 2%.
3. `pytest tests/test_stress_testing.py tests/test_tail_copula_stress_invariants.py -v`.
4. Full `pytest tests/ -v` + launch-blocker subset (decision layer is cross-cutting per CLAUDE.md §4.3).
5. Manual: run `monte_carlo_stress` on the W46 fixture with seed=42 before/after — confirm |var_95| shrinks while `cvar_95 <= var_95 <= 0` holds.
6. Branch + PR (CLAUDE.md §3); never commit to main.

## 7. Why this is safe

One line, matching an already-shipped pattern in `forward_distribution.py`. Purely narrows an over-dispersed diagnostic that no decision-path code consumes. No invariant, reviewer rule, or trio file is touched.
---

## Pre-registered change proposal — Daily-theta fix in `greeks_scenario_matrix`

**Status:** spec only — nothing implemented. Approve or reject before any code is written.
**Touches decision-layer trio (ev_engine / wheel_runner / candidate_dossier):** No.
**Requires trio consent per CLAUDE.md:** No (change is confined to `engine/stress_testing.py`, a diagnostic surface).

### Problem (grounded)
`engine/stress_testing.py::greeks_scenario_matrix` emits **annual** theta in two output columns, while the canonical contract and every other Greeks surface use **daily** theta.

- Canonical rule — `docs/GREEKS_UNIT_CONTRACT.md:27-30`: theta is "Dollar change per calendar day"; "Pricer returns annual theta; convert: `daily_theta = annual_theta / 365`."
- Pricer emits per-year theta — `engine/option_pricer.py:492-505` (analytic `black_scholes_all_greeks`); docstring confirmation at `engine/option_pricer.py:245-246`.
- Offending site 1 — `engine/stress_testing.py:791` → `greeks_surface["theta"]` (emitted line 802): `total_theta += greeks["theta"] * multiplier` (no `/365`).
- Offending site 2 — `engine/stress_testing.py:848` → `time_decay["remaining_theta"]` (emitted line 855): `total_theta += new_greeks["theta"] * multiplier` (no `/365`).

Result: both columns are ~365x too large and inconsistent with the rest of the engine.

**Three siblings already convert** (this is a genuine miss, not a deliberate exception):
`engine/stress_testing.py:596`, `engine/stress_testing.py:1012`, and `engine/risk_manager.py:357` all divide the pricer's theta by 365, two of them with explicit "convert annual to daily" comments citing the contract.

### Proposed change (minimal — the finding's option (a))
Two one-line edits, each mirroring the compliant siblings:

```python
# engine/stress_testing.py:791  (greeks_surface loop)
- total_theta += greeks["theta"] * multiplier
+ # pricer returns annual theta; convert to daily per GREEKS_UNIT_CONTRACT.md
+ total_theta += (greeks["theta"] / 365) * multiplier

# engine/stress_testing.py:848  (time_decay loop)
- total_theta += new_greeks["theta"] * multiplier
+ # pricer returns annual theta; convert to daily per GREEKS_UNIT_CONTRACT.md
+ total_theta += (new_greeks["theta"] / 365) * multiplier
```

No column renames. `theta` and `remaining_theta` are retained and now carry the same per-day unit as everywhere else.

**Why not option (b) (rename to `theta_annual` + document the deviation):** the contract mandates per-day theta engine-wide and is the single source of truth; renaming would legitimize a lone violation instead of fixing it. Rejected.

**Why this is the complete fix:** `pnl_surface` uses full repricing (price differences, line 752) and has no theta-unit exposure; delta/gamma/vega have no per-year convention. These are the only two lines in the function that aggregate raw pricer theta without conversion.

### Files touched
- `engine/stress_testing.py` — two 1-line edits (791, 848) + a clarifying comment at each.
- `tests/test_stress_testing.py` — add value-level regression tests (below). No existing assertions change.
- No documentation change (the fix conforms code to the existing contract).

### Blast radius
Only caller of `greeks_scenario_matrix` in the repo is `tests/test_stress_testing.py:351`, which asserts column *presence*, not magnitude — it stays green. No production surface (EV engine, wheel_runner, candidate_dossier, engine_api, dashboard API, advisors) reads these columns. The dashboard's similarly named `greeks_surface` (`dashboard/quant_dashboard.py:242`) is an unrelated method. Off the EV ranking path; no invariant (R1-R11, dealer clamp, ev_raw/ev_dollars) is affected.

### Falsifiable acceptance criteria
1. `greeks_surface["theta"]` at `spot_change == 0` == pricer annual theta `/365 x multiplier` (rel-tol 1e-9).
2. `time_decay["remaining_theta"]` at `days_elapsed == 0` == same daily-theta aggregate (rel-tol 1e-9).
3. New theta == old theta `/365` (~365x magnitude drop) — pins direction and size.
4. `greeks_surface["theta"]` (unshocked) == `RiskManager.calculate_portfolio_greeks().theta` for the same positions (rel-tol 1e-6).
5. Existing `test_greeks_scenario_matrix` still passes.
6. `pnl_surface`, `delta`, `delta_dollars`, `gamma`, `vega` outputs are unchanged for a fixed position set.

### Test plan
- `test_greeks_surface_theta_is_daily`, `test_time_decay_remaining_theta_is_daily`, `test_greeks_surface_theta_matches_risk_manager` (expected values recomputed independently from `black_scholes_all_greeks`, so tests don't re-encode the bug).
- Run `pytest tests/test_stress_testing.py tests/test_risk_manager.py tests/test_extreme_numerics.py -v`; optional full `pytest tests/ -q`.

### Consent / process
Branch + PR (never commit to `main`, CLAUDE.md §3). Non-trio, non-invariant change — proceeds on approval of this spec; no separate decision-layer sign-off required.
---

## Pre-registered change proposal — Finding #9

### Remove phantom entry commission+slippage from roll suggesters' `hold_ev`

**Status:** pre-registration (spec only — no code written). Awaiting human approve/reject.
**Trio touched:** No (`engine/wheel_tracker.py` only; not `ev_engine.py` / `wheel_runner.py` / `candidate_dossier.py`).
**§2 invariant impact:** None — every EV still flows through `EVEngine.evaluate`; only the post-processing arithmetic on the hold anchor is corrected. Reviewers/dealer clamp untouched.

---

### 1. Problem (grounded)

`suggest_rolls` and `suggest_call_rolls` rank candidate rolls against holding the current position. The hold anchor `hold_ev` re-prices the existing option as a synthetic "re-sell" through `EVEngine.evaluate`, then subtracts the notional re-sell premium to "recover the pure forward P&L" (docstring `engine/wheel_tracker.py:2454-2458`):

```
# engine/wheel_tracker.py:2624 (put)   /  :2998 (call)
hold_ev = hold_result.ev_dollars - buyback_value_per_share * multiplier
```

But `EVEngine.evaluate` charges the synthetic re-sell an **entry** commission + slippage and folds them into the premium before building the P&L:

```
# engine/ev_engine.py:350-366  compute entry_commission, entry_slippage
# engine/ev_engine.py:375      net_premium_in = gross_premium - entry_commission - entry_slippage
# engine/ev_engine.py:401      pnls built from net_premium_in
# engine/ev_engine.py:604      ev_dollars = ev_raw * regime_mult   (regime_mult = 1.0 for the hold trade)
```

With `regime_mult = 1` this gives

```
ev_dollars = gross_premium - entry_commission - entry_slippage - E[payout] - E[assign]
hold_ev    = ev_dollars - gross_premium
           = -entry_commission - entry_slippage - E[payout] - E[assign]
```

The intended "pure forward P&L" of holding is `-E[payout] - E[assign]` — **holding re-opens nothing, so it incurs no entry commission and no entry slippage.** The `-entry_commission - entry_slippage` term is phantom.

`roll_ev` (`engine/wheel_tracker.py:2736` / `:3106`) has **no** matching phantom: its entry cost (inside `new_result.ev_dollars`) is real, and `buyback_total_dollars` is a real exit cost. So `hold_ev` is over-charged relative to `roll_ev`, and since `recommend = roll_ev > hold_ev` (`:2750` / `:3120`), the suggester is **biased toward recommending a roll**.

**Magnitude.** For the synthetic hold trade, spread = ask − bid = `0.10 × buyback` (`:2615-2616`), `open_interest = 1000` (no illiquidity bump), `DEFAULT_SLIPPAGE_PCT = 0.15`, commission `$0.65` (`engine/transaction_costs.py:25,27`):

```
phantom ≈ 1.5 × buyback_value_per_share + 0.65   dollars / contract
```

≈ $1.4 on a $0.50 buyback, ≈ $3.6 on a $2 buyback, ≈ $12.7 on a deep-ITM $8 buyback. **The bias grows with how challenged the position is** — exactly the `include_defensive=True` regime where the roll/hold decision matters most.

---

### 2. Proposed change

Add the engine-charged entry transaction cost back into `hold_ev` in both methods. `EVResult` already exposes `total_transaction_cost` (`engine/ev_engine.py:161,621`) = `2 × (entry_commission + entry_slippage)` because the engine sets `exit_commission = entry_commission` and `exit_slippage = entry_slippage` (`engine/ev_engine.py:368-372`). So the entry leg is `total_transaction_cost / 2`, scaled by the regime multiplier the engine actually applied (`EVResult.regime_multiplier`, `:620`).

```python
# engine/wheel_tracker.py:2624  (identical edit at :2998 with that scope's hold_result)
# ev_dollars charged this synthetic re-sell an ENTRY commission+slippage
# (ev_engine.py:350-366 -> net_premium_in :375 -> scaled by regime_mult :604).
# Holding re-opens nothing, so that entry cost is phantom; add it back so hold
# is charged zero transaction cost. total_transaction_cost is the round-trip sum
# with entry==exit by construction (ev_engine.py:368-372), hence /2 for entry.
entry_txn_cost = (hold_result.total_transaction_cost / 2.0) * hold_result.regime_multiplier
hold_ev = hold_result.ev_dollars - buyback_value_per_share * multiplier + entry_txn_cost
```

Plus a docstring refresh at `:2444-2458` and `:2848` to show the `+ entry_txn_cost` add-back and state that holding is charged zero transaction cost.

**Why this shape (conservative).** The minimal correct change is the `+ entry_txn_cost` add-back. On the designed path `regime_multiplier == 1.0` (the hold trade passes `regime_multiplier=1.0` at `:2618` and no `market_structure`), so `entry_txn_cost` reduces exactly to `total_transaction_cost / 2`; the `* regime_multiplier` factor is a no-op included only as free insurance so the neutralization stays exact if the heavy-tail penalty (`engine/ev_engine.py:567-568`) ever makes `regime_mult != 1` for a hold trade. `roll_ev`, the candidate grid, the drops accumulator, the `.attrs` summaries, sort order, and every `EVEngine.evaluate` call are byte-for-byte unchanged.

**Explicitly out of scope (flagged, not fixed):** the `- buyback_value_per_share * multiplier` premium subtraction is exact only when `regime_mult == 1`. If the heavy-tail penalty fires on the hold trade (needs ≥200 forward scenarios + a heavy GPD fit), a residual `(regime_mult − 1) × gross_premium` term remains. Removing that would require the engine to expose an unscaled forward P&L — a trio change — and is a separate, lower-priority issue.

---

### 3. Files touched

- `engine/wheel_tracker.py` — two ~3-line edits (`:2624` put, `:2998` call) + two docstring formula blocks (`~:2444-2458`, `~:2848`).
- Additive tests in `tests/test_wheel_tracker_suggest_rolls.py` and `tests/test_wheel_tracker_suggest_call_rolls.py`.

No trio files. No API/dashboard/production-decision-path files.

---

### 4. Blast radius

`suggest_rolls` / `suggest_call_rolls` are management-layer suggesters **off** the EV-authority ranking path — `engine/wheel_runner.py` mentions them only in docstrings (`:747,:756-757,:2852,:2861`); the rankers never call them. Real callers: the roll test suite, `scripts/s47_trader_session_2026_03_20.py:574`, and the `docs/verification_artifacts/.../i7_roll_economics.py` snapshot. Observable effect: `hold_ev` rises by the phantom, which can flip some `recommend` True→False (never False→True). The `recommend ⇒ roll_ev > hold_ev` invariant is preserved by construction. Existing tests assert only structural invariants (no hard-coded `hold_ev` dollar values), so none should break; the i7/s47 recorded numbers legitimately shift and want a re-run note (non-blocking).

---

### 5. Falsifiable acceptance criteria

- **AC1/AC2** — In both methods, `hold_ev == ev_dollars − buyback×100 + (total_transaction_cost/2)×regime_multiplier` within 1e-6 (re-evaluated independently on the same synthetic hold trade).
- **AC3** — `hold_ev` strictly rises vs pre-fix; delta == `(total_transaction_cost/2)×regime_multiplier` (≈ `1.5×buyback + 0.65` on defaults) and > 0. *Falsifiable:* a zero/negative delta means the fix is wrong.
- **AC4** — `roll_ev` and `new_ev_dollars` byte-identical old-vs-new for the same inputs.
- **AC5** — post-fix `recommend=True` rows are a subset of pre-fix (no False→True flip).
- **AC6** — every `recommend=True` row still satisfies `roll_ev > hold_ev` (existing Spec-4 tests pass).
- **AC7** — drops/`.attrs`/sort/evaluate-call-count unchanged; `pytest tests/ -v` green.

---

### 6. Test plan

1. `test_hold_ev_charges_zero_transaction_cost` (put) — deterministic `forward_log_returns`, re-evaluate the synthetic hold trade, assert AC1 + AC3.
2. Mirror for `suggest_call_rolls` (AC2).
3. Before/after guard: assert `new_hold_ev − old_hold_ev == (total_transaction_cost/2)×regime_multiplier > 0` and recommend-subset (AC3, AC5) over an identical grid with shared `forward_log_returns`.
4. Assert `roll_ev`/`new_ev_dollars` columns identical old-vs-new (AC4).
5. Re-run `tests/test_wheel_tracker_suggest_rolls.py`, `..._call_rolls.py`, `test_suggest_rolls_defensive.py`, `test_suggest_rolls_drops.py` unchanged (AC6, AC7).
6. Full suite `pytest tests/ -v` (CLAUDE.md §4.3).
7. Non-blocking: re-run `i7_roll_economics.py`, record shifted numbers as an updated artifact.
---

## Pre-registered change proposal — Finding #10

**Emit `contracts` on the put ranker row so dossier soft-warns R7–R10 size multi-contract candidates correctly**

Status: SPEC ONLY — nothing implemented. Touches the decision-layer trio (`wheel_runner.py`, `candidate_dossier.py`) → **requires explicit human consent before any code is written.**

### 1. Problem (grounded)

`rank_candidates_by_ev` honors its public `contracts` parameter (`engine/wheel_runner.py:1277`) in every money field of the emitted row — `collateral = strike*100*contracts` (`:2332`, emitted `:2362`) and `ev_dollars`/`cvar_5` scale via `EVEngine.evaluate`'s `multiplier = 100 * max(trade.contracts,1)` (`engine/ev_engine.py:344`). **But the put ranker row dict (`:2344-2409`) never writes a `contracts` key** — unlike the CC row (`:3366`) and strangle row (`:4083`).

Downstream, the D17 soft-warns therefore size the candidate as a single contract:

| Rule | Code | Reads |
|---|---|---|
| R7 (VaR) | `candidate_dossier.py:400` → `_build_candidate_dict` | `"contracts": 1` hardcoded at **:611** |
| R8 (stress/dealer) | `:433` → `_build_candidate_dict` | same hardcode **:611** |
| R9 (sector cap) | `:476`, notional `:480` | `ev_row.get("contracts", 1)` → key absent → **1** |
| R10 (single-name) | `:522`, notional `:526` | `ev_row.get("contracts", 1)` → key absent → **1** |

So within one row, `collateral`/`ev_dollars` say N contracts but the four soft-warns say 1. Since these reviewers are **downgrade-only**, the error is one-directional: an oversized N-contract candidate that should be downgraded proceed→review is silently left at proceed. The hard §2 invariant is **not** breached — the tracker recomputes true notional at `open_short_put` time as a HARD refusal — so this is a soft-warn **preview-fidelity / audit-honesty** bug (medium), not a rescue path.

Dormant at the default `contracts=1`; active for any caller passing `contracts>1` (public parameter, forwarded by `explore_ev_grid`). A parallel consumer, `engine_api._enrich_alert` (`engine_api.py:3181`, notional `:3202`), has the identical latent bug and self-heals with the same fix.

### 2. Why the existing tests didn't catch it

`tests/test_dossier_r9_r10_audit.py` hand-builds `ev_row` **with** an explicit `"contracts"` key in every fixture (e.g. `_proceeding_dossier(..., contracts=...)`), so it validates the reviewer in isolation and passes — but never exercises the real `rank_candidates_by_ev → build_dossiers → reviewer` seam where the column is missing. That seam is the gap.

### 3. Proposed change (minimal — smaller than the finding's two options)

The finding offered "emit `contracts` OR use `collateral` in R9/R10". **Emitting `contracts` is strictly better and sufficient**, because it also fixes R7/R8 (which need a contract *count*, not a notional, so `collateral` can't help them) and the `engine_api` path for free.

1. `engine/wheel_runner.py` (~line 2349, in the put ranker row dict): add `"contracts": contracts,`. The arg is already in scope; `df = pd.DataFrame(rows)` at `:2536` has no `columns=` reindex and the empty path is a bare `pd.DataFrame()` at `:1496`, so **no pinned schema to update**.
2. `engine/candidate_dossier.py:611`: replace `"contracts": 1,` with a parsed read (mirroring the helper's strike/dte/iv parses and R9/R10's `int(ev_row.get("contracts", 1))` convention — **no `or 1`**, so explicit `contracts=0` stays 0 per S42 Finding #3).

**Explicitly NOT touched** (conservative scope): R9/R10 reviewer bodies (they self-heal once the column exists — keep the `strike*100*contracts` recompute), `engine_api._enrich_alert` (self-heals), and no new column-schema constant.

### 4. Blast radius

Change #1 is a purely additive column on the universal ranker frame: safe for `build_dossiers`/reviewers (key-with-default reads), `engine_api` (JSON-additive), `select_book` (checks only `{collateral, ev_dollars}`), `explore_ev_grid`, dashboard, and scripts (no positional column assumptions; no pinned put schema). Only genuine risk: a test asserting the put ranker's exact column set — grep of `tests/` shows exact-set assertions only for CC/roll/connector frames, none for `rank_candidates_by_ev`. Change #2's only callers are R7/R8; identical at `contracts=1`.

### 5. Acceptance criteria & test plan

See the structured `acceptance_criteria` (AC1–AC7) and `test_plan`. Headline falsifiable expectation: the soft-warn now trips at the same N-contract size that `collateral`/`ev_dollars` already report, and the default `contracts=1` path is byte-identical (backtest S27/S32/S34/S35 zero drift).

### 6. Rollback

Two-line revert; the additive column is independently removable. No data migration, no schema constant, no persisted-state change.

### 7. Consent gate

Both edits are in the trio (`wheel_runner.py`, `candidate_dossier.py`). Per CLAUDE.md §3, do not implement without explicit human approval of this spec.

---

## Carried forward — the validation-phase re-baseline queue (findings §5)

These six measured facts from the just-closed validation phase (V1–V6) sit
alongside the audit proposals above as inputs to the same re-baseline
discussion. They are **measured findings, not patches**; none is applied. See
`docs/VALIDATION_PHASE_FINDINGS_2026-07-13.md` §5 for full evidence.

- **F-V1-1 — onset blindness.** Entry-VIX conditioning is structurally blind to
  crisis onset; calm-entry crisis rows breach at ~3.4× severity. Lever:
  damage-bounders (R10) + forward-looking tail inputs, not faster trailing fits.
- **F-V1-2 — winner's curse.** Top-bin tail optimism (6.32% breach vs the 5%
  bound), selection concentrates tail optimism where the money goes.
- **F-V1-4 — breach severity.** ~30% of breaches ≥ 3× modeled ES; cvar_5
  understates conditional-on-breach loss in exactly the strata that trade.
- **F-V3-1 — R11 mis-aim.** The VIX=25 cut sits below the measured lift>1
  crossover (~27.5+); constrains any R11 recalibration (not a removal).
- **F-V5-1 — the blind spots compose.** R10+R9+collateral admits up to
  36.5%-NAV single-cycle damage at calm-VIX onset; only the top-bin filter
  bounds worst-case below ruin.
- **F-V6-1 — EV-sign refusal is not a crisis defense.** On 2007–2009 the book
  stayed EV-positive through the Oct–Dec 2008 grind (unconditional after the
  V6-r1 re-read); survival is carried by assignment-and-hold, not refusal.

**Cross-link:** audit proposal #3 (activate the dead R6 dealer downgrade) and
the F-V1-1/F-V5-1 damage-bounder theme point the same direction — the
short-gamma / dealer-positioning configurations are where both the audit and
the validation phase locate the residual risk.
