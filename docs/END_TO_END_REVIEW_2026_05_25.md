# End-to-end product review — 2026-05-25

**Reviewer:** Terminal A, fresh session, no campaign context.
**Scope:** logical + structural review of the whole product (data,
quant, decision, interface), with §2 as the load-bearing invariant.
**`origin/main` SHA at review start:** `e83eaca` (post-#194,
post-S28/S29/S30 ledger, post-#185/#186/#193 test PRs,
post-#173/#194/#195 audit/review docs).
**Pytest baseline:** 28/28 on the three recent test PRs
(`tests/test_evengine_event_lockout.py` + `tests/test_dealer_multiplier_evengine_integration.py`
+ `tests/test_consume_ranker_row_anchor.py`, 21.74s) and 62/62 on the
launch-blocker invariant subset (`test_audit_invariants.py` +
`test_authority_hardening.py` + `test_dossier_invariant.py`, 0.80s).
**5-ticker EV smoke (CLAUDE.md §4):** green — `MarketDataConnector`,
5 rows, 0 NaN in `ev_dollars` / `iv` / `premium`. `dealer_multiplier`
uniformly `1.0` (Bloomberg connector lacks chain access — expected;
matches S29 finding F3).

---

## Headline — no §2 breach; four open follow-ups; one new structural note

**`EVEngine.evaluate` is the only path to a tradeable verdict at HEAD.**
Every candidate-producing surface — `WheelRunner.rank_candidates_by_ev`
(puts), `rank_covered_calls_by_ev`, `rank_strangles_by_ev`,
`WheelTracker.suggest_rolls` / `suggest_calls`,
`WheelRunner.build_candidate_dossiers`, the `/api/candidates` and
`/api/tv/{ranked,dossier,webhook}` HTTP endpoints, and the
module-level `engine.ev_engine.score_candidates` helper — calls
`EVEngine.evaluate` directly. Non-ranker `engine_api.py` endpoints
carry the documented `authority: "heuristic_diagnostic"` +
`tradeable_endpoint: "/api/candidates"` contract. The dealer multiplier
clamp `[0.70, 1.05]` holds by construction (math, not runtime clamp).
`EnginePhaseReviewer` R1–R8 are structurally downgrade-only (R6/R7/R8
guard on `verdict == "proceed"` before downgrading; R1 returns
`"blocked"`; R7/R8 only return `"review"`). The token-based
launch gate (D16 in `engine/wheel_tracker.py:322-436`) refuses
non-positive EV at issuance AND consumption. No new §2-bypass surface
introduced by the 9 commits since `d24f041`.

**Four already-flagged follow-ups remain open after this review,**
each meaningful in its own way:

1. **+inf bypass of R1/R5 (in-process, latent).** Confirmed live by
   this review's driver — same boolean truth table as RELIABILITY_ARC_REVIEW
   C1. The network surface structurally closes the gap today
   (`engine/tv_signals.py` `TVAlert.parse` known-field whitelist
   excludes `ev_dollars`; `_enrich_alert` server-computes it; S20 G3
   confirmed no user-controllable injection path). The one-line
   `math.isfinite` defense at `engine/candidate_dossier.py:202` is
   not yet applied. Cost: tiny; benefit: removes the latent residual.
2. **Wire `rank_candidates_by_ev` → `issue_ev_authority_token` →
   `open_short_put(current_ev_dollars=...)` automatically.**
   Production callers must currently invoke
   `WheelTracker.consume_ranker_row` (PR #174) by hand. The audit
   doc TERMINAL_A_AUDIT.md cross-cutting #4 names this as "the
   natural follow-on PR"; ENGINE_BACKTEST `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`
   F5 names it as the production-readiness fix for fragile
   BP-saturation protection. D16 / D17 hardening is a contract for
   *direct* tracker callers (tests today), not for the ranker chain
   that operators run.
3. **F4 tail-risk gap on the forward-distribution + POT-GPD pipeline.**
   COST April 2022 (mean realized **−$7,500** with `prob_profit=0.833`
   throughout a 22% drop) and UNH November 2024 (−$19,000 with
   `prob_profit=0.98` through a 19% drop) — both reproduced in S22
   (pre-IV-PIT-fix) and S27 (post-fix). PR #196 (Terminal C, open)
   pins the symptom as a regression test; the underlying fix is open.
   ENGINE_BACKTEST F4 calls this **the highest-leverage finding for
   the engine team**.
4. **Two divergent verdict paths between `/api/tv/dossier` and
   `/api/tv/webhook`.** `/api/tv/dossier` routes through
   `EnginePhaseReviewer.review` (R1–R8). `/api/tv/webhook` →
   `_enrich_alert` runs an *inline* verdict block at
   `engine_api.py:2082-2099` with the same shape as R1/R5 plus extras
   (`days_to_earnings < 5`, `prob_profit >= 0.65`, `agrees`) but
   never invokes `EnginePhaseReviewer`. Same EV row could produce
   different verdicts across the two endpoints. RELIABILITY_ARC_REVIEW
   observation #3 flagged this for Terminal A decision-layer
   follow-up; not unified on `main`.

**One new structural note that warrants explicit logging** beyond what
prior audits caught: `src/backtest/wheel_backtest.py` is a parallel
**heuristic** backtester (`_score_entry` uses RV / RSI / trend / MA
composite — NOT `EVEngine.evaluate`; `_estimate_put_price` fabricates
premiums via BSM, not real chain data) that opens tracker positions
without an EV-authority token. It lives in deprecated `src/` per
PROJECT_STATE §4 but has no warning header naming itself
non-§2-compliant. Not a §2 breach (no production caller; sandboxed
in its own backtester), but a user could mistake it for "the engine's
backtester" and get heuristic-driven results. See finding **C5** below.

---

## Methodology

Four-pass invariant-led audit:

| Pass | Anchor | Question |
|---|---|---|
| 1 | §2 surface enumeration | Find every candidate-producing path; verify each calls `EVEngine.evaluate`. |
| 2 | Recent-commit skepticism | For each of 9 commits since `d24f041`, verify claims are pinned by code/tests at HEAD; mine `Unresolved` / `AI handoff` for forgotten follow-ups. |
| 3 | Layering & dormant-surface census | Verify the four-layer model by import graph; walk PROJECT_STATE §3 (WIP) and §4 (deprecated) claims against current code. |
| 4 | Contract-vs-implementation per route | Read each of the four authoritative routes; verify signature, side-effects, downgrade-only invariant, dealer clamp at actual lines. Confirm R4 dormancy. |

**Borrowed format:** structural blend of `RELIABILITY_ARC_REVIEW.md`
(C-labeled claims with file:line citations) and `TERMINAL_A_AUDIT.md`
(per-pass tally with cross-cutting observations at the tail). New
findings are F-numbered for cross-reference.

**Out of scope, intentionally:**

- The four prior audits (#173 Terminal A audit, #194 reliability arc
  review, #195 audit-of-audit, #197 predictive-validity arc review)
  cover what they cover; I did not re-audit their primary subjects
  unless something in Pass 2 raised skepticism.
- I did not run the full pytest suite as a primary signal (the
  audits did; baseline `2215 passed / 2 Windows-local-flake / 12
  skipped`). I spot-ran 90 tests across the three recent test PRs +
  launch-blocker invariant subset.
- News pipeline (`financial_news/`, `news_pipeline/`) read at the
  caller / boundary level only — off the EV decision path per
  PROJECT_STATE §4.
- Next.js dashboard (`dashboard/src/`) — per the orientation prompt
  "(not in use this cycle)".
- Theta data-pull internals — out of scope per Theta-Terminal-lock
  protocol.

---

## Tally

- **§2 BREACH:** 0
- **CONCERN (real risk surfaces that should change):** 5 (C1–C5)
- **WITH-NOTE (documentation drift / minor observability gaps):** 10 (N1–N10)
- **SOLID (positives explicitly tallied):** 12 (4 in Pass 1, 3 in Pass 2, 5 in Pass 3) — plus all 4 authoritative routes pass the contract-vs-implementation check in Pass 4.

---

# Pass 1 — §2 surface enumeration

## SOLID — all 11 EVEngine.evaluate call sites in documented routes

Grep across `engine/`, `engine_api.py`, `advisors/`, `financial_news/`,
`news_pipeline/`, `scripts/`, `dashboard/` for `\.evaluate\(`:

| File:line | Caller |
|---|---|
| `engine/ev_engine.py:76` | docstring example |
| `engine/ev_engine.py:640` | `score_candidates` module helper |
| `engine/wheel_runner.py:1415` | `rank_candidates_by_ev` (puts) |
| `engine/wheel_runner.py:2241` | `rank_covered_calls_by_ev` |
| `engine/wheel_runner.py:2787` | `rank_strangles_by_ev` (put leg) |
| `engine/wheel_runner.py:2793` | `rank_strangles_by_ev` (call leg) |
| `engine/wheel_tracker.py:2290` | `suggest_rolls` hold_ev |
| `engine/wheel_tracker.py:2401` | `suggest_rolls` roll_ev |
| `engine/wheel_tracker.py:2658` | `suggest_call_rolls` hold_ev |
| `engine/wheel_tracker.py:2765` | `suggest_call_rolls` roll_ev |
| `engine_api.py:960` | `committee.evaluate(...)` — *advisor* committee, not `EVEngine` |

Of the eleven, ten are `EVEngine.evaluate` and all sit inside the
documented authoritative routes. The eleventh
(`engine_api.py:960`) is the **advisor** committee — a separate
class, and the `/api/committee` handler explicitly carries
`authority: "heuristic_diagnostic"` + `tradeable_endpoint:
"/api/candidates"` (`engine_api.py:1008-1009`).

**Verdict:** SOLID. The grep is complete; no orphan callers.

## SOLID — engine_api.py non-ranker endpoints all carry authority contracts

| Endpoint | Handler:line | Authority field |
|---|---|---|
| `/api/candidates` | `_handle_candidates:433` | `"ev_ranked"` |
| `/api/screen` | `_handle_screen:729` | `"heuristic_research_only"` + warning |
| `/api/analyze` | `_handle_analyze:597` | `"heuristic_diagnostic"` |
| `/api/committee` | `_handle_committee:773` | `"heuristic_diagnostic"` |
| `/api/strangle` | `_handle_strangle:1311` | `"heuristic_diagnostic"` |
| `/api/strikes` | `_handle_strikes:1124` | `"heuristic_diagnostic"` |
| `/api/tv/ranked` | `_handle_tv_ranked:1906` | (calls `rank_candidates_by_ev`) |
| `/api/tv/dossier` | `_handle_tv_dossier:1811` | (calls `build_candidate_dossiers`) |
| `/api/tv/webhook` | `_handle_tv_webhook:1575` | `_enrich_alert` ladder; ev-authoritative |

Each non-ranker endpoint either routes through `WheelRunner` or
includes the explicit `tradeable_endpoint` redirect to
`/api/candidates`. Discipline is consistent; no rogue handler.

## SOLID — dealer multiplier clamp `[0.70, 1.05]` by construction

`engine/dealer_positioning.py:720-746` shows the mapping:

```python
def dealer_regime_multiplier(ms: MarketStructure | None) -> float:
    if ms is None:
        return 1.0
    conf = max(0.0, min(1.0, float(ms.confidence)))   # conf clamped to [0, 1]
    if ms.regime == "long_gamma_dampening":
        return 1.0 + 0.05 * conf                       # ∈ [1.00, 1.05]
    if ms.regime == "short_gamma_amplifying":
        return 1.0 - 0.30 * conf                       # ∈ [0.70, 1.00]
    if ms.regime == "near_flip":
        return 0.85                                     # flat
    return 1.0
```

The range `[0.70, 1.05]` is **mathematically guaranteed** by the
construction, not by a post-hoc clamp. Pinned at the integration
boundary by PR #193's `TestDealerMultiplierBoundsAtEVResultLevel`
(`tests/test_dealer_multiplier_evengine_integration.py:115-179` —
verified passing in this review).

The compounding into `ev_dollars` is at `engine/ev_engine.py:488`:
`regime_mult *= dealer_mult`. `ev_raw` (`engine/ev_engine.py:366`)
is `float(np.mean(pnls))` — untouched. `ev_dollars = ev_raw *
regime_mult` (line 502) is the only product. **`ev_raw` is never
multiplied by a reviewer-injected value.** ✓

## SOLID — EnginePhaseReviewer R1–R8 are structurally downgrade-only

`engine/candidate_dossier.py:197-353`. Rule-by-rule:

| Rule | Line | Verdict it can emit | Guard |
|---|---|---|---|
| R1 (negative-EV block) | `:202-204` | `"blocked"` | none — fires unconditionally |
| R2 (chart missing) | `:207-210` | `"review"` | chart `is_ok()` false |
| R3 (spot mismatch) | `:215-223` | `"skip"` | engine_spot > 0 & visible_price set & diff > tol |
| R4 (phase contradiction) | `:228-238` | `"skip"` | both phases present (dormant — see Pass 4) |
| R5 (EV threshold) | `:241-248` | `"proceed"` ∨ `"review"` | ev >= min_proceed_ev → proceed else review |
| R6 (dealer downgrade) | `:258-280` | `"review"` | `verdict == "proceed"` (downgrade-only) |
| R7 (portfolio VaR) | `:288-311` | `"review"` | `verdict == "proceed"` (downgrade-only) |
| R8 (stress + dealer regime) | `:321-351` | `"review"` | `verdict == "proceed"` (downgrade-only) |

R6/R7/R8 cannot move a `"blocked"` or `"skip"` upward — the guard
`verdict == "proceed"` is the structural check. They cannot move
`"review"` upward either — they only return `"review"` themselves.

## WITH-NOTE N1 — CLAUDE.md §2 documents R1–R6; reality is R1–R8

CLAUDE.md §2 (`CLAUDE.md:58-72`) enumerates rules R1 through R6.
R7 (portfolio VaR, D17) and R8 (stress + dealer regime, D17) shipped
in PR #165 and are documented in `engine/candidate_dossier.py:161-179`
+ `docs/LAUNCH_READINESS.md` + `DECISIONS.md` D17. CLAUDE.md was not
updated. Not a §2 violation — R7/R8 are downgrade-only by structure —
but the canonical reference says less than the code does.

## WITH-NOTE N2 — engine_api.py header lists 32 endpoints; reality is 34

`engine_api.py:7-42` lists exactly 32 endpoints. Live route table
(`do_GET`/`do_POST` at `:253-413`) is 34: the two missing from the
header are `/api/news` (GET; `:370`) and `/api/news/ingest`
(POST; `:410`). PROJECT_STATE.md §1's claim "32 endpoints listed in
the file header" is literally correct re: the header but understates
the actual surface by 2.

## CONCERN C1 — +inf bypasses R1 and R5 (latent, in-process)

Already documented as the C7b mechanism in RELIABILITY_ARC_REVIEW.md
C1, but re-confirmed live by this review's driver:

```
ev=                         -25.0  ->  verdict='blocked'   reason='negative_ev'
ev=                          -inf  ->  verdict='blocked'   reason='negative_ev'
ev=                           0.0  ->  verdict='review'    reason='ev_below_proceed_threshold'
ev=                           5.0  ->  verdict='review'    reason='ev_below_proceed_threshold'
ev=                          10.0  ->  verdict='proceed'   reason='ev_above_threshold'
ev=                          25.0  ->  verdict='proceed'   reason='ev_above_threshold'
ev=                           inf  ->  verdict='proceed'   reason='ev_above_threshold'   <-- bypass
ev=                           nan  ->  verdict='review'    reason='ev_below_proceed_threshold'  <-- silent
```

`+inf` slides through both R1 (`if ev < 0` is False for +inf) and R5
(`if ev >= 10.0` is True for +inf). `NaN` silently degrades to
`"review"` because `NaN >= 10.0` is False; the audit log records this
as an EV-below-threshold review, which is misleading (the truth is
"engine produced an unparseable value").

**Why this is CONCERN (not BREACH):** No production code path injects
`+inf` into a dossier `ev_row` today. `EVEngine.evaluate` returns
`float(np.mean(pnls))` where `pnls` is bounded by finite terminal
prices × finite multipliers; producing `+inf` would require a numerical
overflow at the BSM or simulation layer (not observed today). The
network surface confirms no exfiltration (S20 G3): `TVAlert.parse`'s
known-field set at `engine/tv_signals.py:543-552` excludes
`ev_dollars`; `_enrich_alert` server-computes EV; `_sanitize_nans`
scrubs the reply path. The structural risk is purely in-process:
**if a future code path exposes a hand-built `ev_row` → dossier
path** (e.g., a new admin endpoint that accepts a pre-computed
`ev_row`, or an MCP tool that synthesises one), the residual reopens
immediately.

**Fix cost:** one-line `if not math.isfinite(ev) or ev < 0:` at
`engine/candidate_dossier.py:202`. RELIABILITY_ARC_REVIEW's
suggestion. Not landed.

## CONCERN C2 — two divergent verdict paths between dossier and webhook

Both routes are §2-compliant in isolation, but they implement
**different rule sets** for the same downstream concept ("verdict on
a candidate"):

| Layer | Code site | Negative-EV verdict | Proceed-threshold rule | Extras |
|---|---|---|---|---|
| Dossier path (`EnginePhaseReviewer`) | `engine/candidate_dossier.py:202-353` | `"blocked"` | `ev >= 10` (R5) | R3 spot mismatch, R6 dealer, R7 VaR, R8 stress |
| Webhook path (`_enrich_alert`) | `engine_api.py:2082-2099` | `"skip"` | `ev >= 10 AND prob_profit >= 0.65 AND agrees` | Hard event gate `days_to_earnings < 5` |

Same EV row, two endpoints, two possible verdicts.

Example divergence: a candidate with `ev_dollars=$12`, `prob_profit=0.55`,
chart agrees → dossier returns `"proceed"`; webhook returns `"review"`.

Already flagged by RELIABILITY_ARC_REVIEW observation #3 as a
Terminal A decision-layer follow-up; not yet landed. The unification
target would be: have `_enrich_alert` build a `CandidateDossier` and
call `EnginePhaseReviewer.review` rather than carrying its own
inline ladder. The extras (`prob_profit >= 0.65`, `agrees`) could
become R9 / R10 on the reviewer, preserving today's webhook
behaviour but unifying the rule surface.

## CONCERN C3 — R7 + R8 dormant on the live `/api/tv/dossier` endpoint

R7 (portfolio VaR) and R8 (stress + dealer regime) are
downgrade-only soft-warns implemented in `engine/candidate_dossier.py:282-351`.
They both require a non-None `portfolio_context` to fire (guarded at
`:288` and `:321`).

`engine_api.py` never passes `portfolio_context`. Grep:

```
$ grep -rn "portfolio_context" engine_api.py
(no matches)
```

So `/api/tv/dossier` (`engine_api.py:1811-1904`) never fires R7 or
R8. Their contract is intact; their reach to production is zero. The
operator running the live dashboard receives dossiers with R7 / R8
skipped silently.

This is consistent with the design (no portfolio data attached →
soft-warns can't fire), but it means D17 / R7 / R8 hardening is
*structural* coverage today, not *runtime* coverage. TERMINAL_A_AUDIT
cross-cutting #4 flagged the corresponding gap (D16 is a contract
for direct tracker callers, not for the ranker chain). The fix is
the same as C4 below: thread `WheelTracker.portfolio_context_snapshot`
through `build_candidate_dossiers`.

## CONCERN C4 — production wire from ranker to tracker is manual

`engine/wheel_runner.py:631` ranks candidates. Each row is fully
EV-evaluated. But the chain to a `WheelTracker` position requires
the operator to call `consume_ranker_row` (PR #174) themselves —
`rank_candidates_by_ev` does not auto-issue tokens or auto-open
positions, and no production code wires the chain.

Grep:

```
$ grep -n "issue_ev_authority_token\|consume_ranker_row" engine/wheel_runner.py
(no matches)
```

The implications:

- **D16 (token-bound EV authority) and D17 (portfolio-risk hard-blocks)
  are dormant on the production path** until the operator runs the
  wire by hand. `engine_api.py` does not run them either — it serves
  ranked rows as JSON; the actual "fire" happens in the user's
  broker outside the system.
- ENGINE_BACKTEST F5 ("BP-saturation protection is fragile") proved
  this matters: S22's bigger EVs accidentally over-tied-up BP and
  prevented some bad trades; S27's smaller EVs let those trades
  fire, and the executed-trade mean realized went negative
  (−$72/trade despite a higher per-row hit rate).
- The fix is named in both audits (TERMINAL_A_AUDIT cross-cutting
  #4, ENGINE_BACKTEST F5): wire `rank_candidates_by_ev` →
  `issue_ev_authority_token` → `consume_ranker_row` →
  `open_short_put(current_ev_dollars=...)` and attach a
  `PortfolioContext` to `build_candidate_dossiers` so R7/R8 fire
  live.

The board (#113) shows no Terminal A claim for this. It is the
single largest "ship it" item identified by the campaign.

## CONCERN C5 — heuristic backtester in deprecated `src/` has no warning header

`src/backtest/wheel_backtest.py` (363+ lines) is a parallel
backtester that:

1. **Scores candidates via `_score_entry` (`:363-414`)** — a
   heuristic combining `rv_rank_252`, `trend_20d`, `rsi_14`,
   `above_sma_200`, `drawdown_52w`. **Not `EVEngine.evaluate`.**
2. **Fabricates premiums via `_estimate_put_price` (`:285`)** — BSM
   evaluation against a hardcoded `realized_vol_20` as proxy IV.
   **Not real chain data.**
3. **Opens tracker positions via `tracker.open_short_put` (`:292-299`)
   with no `ev_authority_token`** and the default
   `require_ev_authority=False`. Even in `WheelTracker`'s strict
   mode this would refuse — but the backtester runs in non-strict.

This is **not a §2 BREACH** — no production caller routes through
`WheelBacktest`; the recent ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md
(S27) correctly used `WheelRunner.rank_candidates_by_ev`, not this
heuristic path. But:

- `src/` is documented as "phantom architecture" in PROJECT_STATE §4
  with the directive "do not add new modules under `src/`", yet
  `wheel_backtest.py` is far from a stub.
- The file's docstring (`:1-50`) describes it as "Event-driven wheel
  strategy backtester" with no warning that it bypasses the EV path.
- A user picking it up via `from src.backtest import WheelBacktest`
  (the natural import path; the module has full
  `__init__.py` exports) and running it on the same 2022-2024 window
  would get heuristic-driven results plausibly mistakable for "what
  the engine would have done."

**Recommended treatment:** either (a) add a top-of-file warning
banner naming the file non-§2-compliant and pointing at
`WheelRunner.rank_candidates_by_ev` + `consume_ranker_row` as the
sanctioned backtest harness, or (b) delete it (PROJECT_STATE §4's
"plan: either fully remove or fully repopulate" pointer). The
existence of `backtests/simulator.py` (similarly heuristic, but
self-described as "PLACEHOLDER until Bloomberg data arrives" — at
least the framing is honest) suggests (a) is the lighter touch.

---

# Pass 2 — recent-commit skepticism

Nine commits since `d24f041`. For each: PR claims verified at HEAD;
`Unresolved` / `AI handoff` sections mined for forgotten items.

## SOLID — three test PRs (#185, #186, #193) all do what they claim

Verified by reading the test files end-to-end and running them on
HEAD: **28/28 passed in 21.74s.**

- **PR #185** (`9bb90a2`, `test(ev_engine)` event-lockout) —
  `tests/test_evengine_event_lockout.py` pins the short-circuit at
  `engine/ev_engine.py:262-285` via 5 blocking + 3 non-blocking
  cases + 1 schema regression + 1 §2-adjacent test
  (`TestBlockedPathBypassesDealerMultiplier`) that confirms no
  `MarketStructure` can lift a blocked candidate. All 10 tests use
  real `EVEngine` + real `EventGate` + real `ScheduledEvent`. No
  mocks on the hot path.
- **PR #193** (`cc54619`, `test(dealer_positioning)`) —
  `tests/test_dealer_multiplier_evengine_integration.py` pins exact
  boundary values (1.05, 0.70, 0.85, 1.00) at `EVResult.dealer_multiplier`,
  the asymmetric clamp, the `regime_multiplier *= dealer_mult`
  compounding at `engine/ev_engine.py:488`, and the proportionality
  invariant for `ev_dollars` ratio = dealer multiplier ratio.
  9 tests; real fixtures; `n=199` chosen specifically to keep
  `heavy_tail=False` (per the file docstring) so the
  `heavy_tail_penalty` doesn't confound assertions.
- **PR #186** (`ef38cee`, `test(wheel_tracker)` anchor) —
  `tests/test_consume_ranker_row_anchor.py` pulls **real ranker
  rows** from `WheelRunner.rank_candidates_by_ev` against the real
  Bloomberg CSVs and feeds them through `consume_ranker_row`. The
  hand-built `_ev_row` in `test_decision_layer_wiring.py` is no
  longer the only proof of contract: schema drift between ranker
  output and `consume_ranker_row`'s key reads now fails immediately.
  9 tests; happy path (AAPL +EV), refusal path (META −EV), round-trips
  through `mark_to_market` + `portfolio_context_snapshot`, empty-frame
  no-op, duplicate-consume rejection, strict-mode full chain.

## SOLID — three ledger PRs (#190, #191, #192) line citations all check out

- **S30 HMM** (`1a735e7`, PR #190) — `engine/regime_hmm.py:76` is
  the `GaussianHMM` class ✓; `engine/regime_hmm.py:265-280` is
  `position_multiplier` with weights `{crisis: 0.2, bear: 0.5,
  normal: 1.0, bull_quiet: 1.25}` ✓; `engine/wheel_runner.py:1300`
  is `combined_regime_mult = float(hmm_regime_mult * skew_mult *
  news_mult * credit_mult)` ✓. S30's matrix observations
  (AAPL crisis 0.200 on 2025-04-04, recovery to 0.817 on 2025-04-15)
  are not re-reproducible without re-running the driver (it's not
  committed; lives under `%TEMP%\s29\`) but the structural claims
  about the HMM are verified by the live code.
- **S28 dividends** (`3bcf117`, PR #191) — `engine/ev_engine.py:355-361`
  is the dividend penalty gate ✓; `engine/wheel_tracker.py:88-112`
  is `_solve_call_strike` with `[spot*1.01, spot*2.0]` Brent
  bracket and the documented OTM-only docstring quote ✓; the
  diagnostic-column populate-regardless-of-gate F2 observation
  (MSFT 25Δ CC with `days_to_ex_div=62`, `dte=35`, but
  `expected_dividend=0.91` in the row) is reproducible by code
  reading.
- **S29 skew** (`e2d268a`, PR #192) — `engine/wheel_runner.py:147`
  is `_resolve_pit_atm_iv` ✓; `engine/wheel_runner.py:1252-1285` is
  the `skew_mult` block gated on `use_skew_dynamics and chain_df is
  not None and len(chain_df) > 0` ✓; the Bloomberg connector lacks
  `get_options` / `get_option_chain` (verified by grep of
  `engine/data_connector.py`) ✓. The shocking F1 data finding
  (1,353,901 rows with `put_iv == call_iv` exactly = **100%** of
  populated rows) is the kind of finding that would normally take
  an external auditor weeks to surface. It explains why
  `_resolve_pit_atm_iv`'s `(put_iv + call_iv) / 2` averaging is
  provably equivalent to `X / X = X` on Bloomberg data.

## SOLID — three audit/review docs (#173, #194, #195) are honest about scope

- **TERMINAL_A_AUDIT.md** (`2ccb936`, PR #173) — 22-PR audit,
  SOLID 22 / WITH-NOTE 0 / CONCERN 0 / §2 BREACH 0. Cross-cutting
  observation #4 (D16 / D17 not on the ranker chain) is the
  single open follow-up; consistent with this review's C4.
- **RELIABILITY_ARC_REVIEW.md** (`e83eaca`, PR #194) — re-verifies
  S18/S19/S20 against `d24f041`. C1 (`+inf` bypass) reproduces
  exactly today (my driver above). C2 (`request_queue_size = 5`
  cap) reproduces. Observation #2 (`request_queue_size = 128` fix
  not landed) and #3 (two divergent verdict paths) are both this
  review's C2 + N12.
- **AUDIT_OF_AUDIT_REVIEW.md** (`#195`, open PR by Terminal B) —
  reviewed by reading the diff via `gh pr view 195`; not on `main`
  yet, so not in this review's scope to verify.

## WITH-NOTE N3 — S30 lists its driver path as `%TEMP%\s29\` (typo)

`docs/USAGE_TEST_LEDGER.md:5212-5213`:

```
Driver under `%TEMP%\s29\`, not committed;
```

The driver for S30 is described as living under `%TEMP%\s29\`
(should be `\s30\`). Cosmetic. The S28 entry has the same shape
("Drivers under `%TEMP%\s27\`" at `:4599-4600`, should be `\s28\`)
and the S29 entry too ("Driver under `%TEMP%\s28\`" at `:4903-4904`,
should be `\s29\`). All three ledger entries used the previous
campaign's directory name. Probably reflects a re-use of the same
`%TEMP%` directory across runs.

## WITH-NOTE N4 — S30's "documented multiplier range `[0.0, 1.25]`" is technically the post-validation envelope

S30 §F3 says "HMM multipliers respect the documented `[0.20, 1.25]`
envelope". The HMM's per-state weights are `{crisis: 0.2, bear: 0.5,
normal: 1.0, bull_quiet: 1.25}`, so a *pure-state* posterior yields
exactly those values. Mixed-state posteriors yield convex
combinations in `[0.2, 1.25]`. So the HMM-only range is `[0.20, 1.25]`
as S30 states.

But the **combined** `regime_mult` fed to `ShortOptionTrade.regime_multiplier`
at `engine/wheel_runner.py:1300` is `hmm * skew * news * credit`,
and `EVEngine.evaluate` clamps the input to `[0.0, 1.25]` at
`engine/ev_engine.py:462-473`. The HMM alone respects `[0.2, 1.25]`;
the combined product is validated to `[0.0, 1.25]`. Not a
contradiction — S30 is testing the HMM contract specifically — but a
reader of S30 might conflate the two. Minor framing note.

## CONCERN re-surfaced — F4 tail-risk gap from S27 / ENGINE_BACKTEST (covered above as headline #3)

PR #196 (Terminal C, open) pins this as a regression test. The
underlying engine fix is the highest-leverage open item across the
campaign.

---

# Pass 3 — layering & dormant-surface census

## SOLID — engine/ does not import from advisors/, dashboard/, financial_news/, or news_pipeline/

Grep for cross-layer imports in `engine/`:

```
$ grep -E "^(from|import) (advisors|dashboard|financial_news|news_pipeline)" engine/*.py
(no matches)
```

The decision layer (`engine/`) is structurally insulated from the
interface layer (`dashboard/`, `advisors/`) and the
research/news platforms (`financial_news/`, `news_pipeline/`). ✓

## WITH-NOTE N5 — engine/ has two `from src.features.technical import` lines

Grep found:

```
engine/strangle_timing.py:31:from src.features.technical import TechnicalFeatures
engine/tv_signals.py:48:from src.features.technical import TechnicalFeatures
```

Plus a third in `engine_api.py:1161` (`from src.features.technical
import TechnicalFeatures` inside `_handle_chart`). So `src/features/technical.py`
is actively imported by **three production sites**, two of them in
`engine/`. PROJECT_STATE §4's "phantom architecture" framing
under-states how alive `src/features/` actually is.

## WITH-NOTE N6 — PROJECT_STATE §4's pyproject claim is stale

PROJECT_STATE.md §4 (`:308-314`) says:

> The pyproject entrypoint `wheel = "src.cli:app"` targets a
> `src/cli.py` that does **not exist**; coverage and
> `[tool.hatch] packages = ["src"]` treat `src/` as load-bearing.

The `wheel = "src.cli:app"` entry **does not exist in pyproject.toml**
on `main` today. `pyproject.toml` has no `[project.scripts]` section
at all (verified by reading the file end-to-end). ROADMAP B5 closed
this; PROJECT_STATE §4 was not updated. The second half of the
PROJECT_STATE claim is still true: `[tool.hatch.build.targets.wheel] packages`
includes `"src"` (`pyproject.toml:72`), and
`[tool.coverage.run] source` includes `"src"` (`:109`). So `src/`
is still load-bearing for build + coverage; it's just not also a
broken entrypoint anymore.

## SOLID — `validation/` was removed per D14 ✓

```
$ test -d validation/ && ls validation/ || echo "removed"
removed
```

## SOLID — TradingView MCP Stage 3 wiring matches PROJECT_STATE §3 ✓

`engine/tradingview_bridge.py:611-656` shows `build_default_provider`
gating MCP behind `SWE_USE_MCP_CHART` env var:

```python
if enable_mcp is None:
    enable_mcp = os.environ.get("SWE_USE_MCP_CHART", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
providers: list[ChartContextProvider] = []
if enable_mcp:
    from engine.mcp_client import MCPCLIClient
    providers.append(MCPChartProvider(client=MCPCLIClient()))
providers.append(FilesystemChartProvider(base_dir=screenshots_dir))
```

PROJECT_STATE §3 ("WIP: TradingView MCP integration") is accurate.

## WITH-NOTE N7 — `_classify` TODO(live-verify) markers still present

`engine/mcp_client.py:324-352` is the `_classify` function that maps
CLI error strings to canonical `MCPClientError` codes. The
`TODO(live-verify)` marker at `:327` is still present:

```python
def _classify(step: str, blob: str) -> MCPClientError:
    """...
    TODO(live-verify): the substring checks are written against
    *plausible* error wording — the upstream README does not document
    the CLI's error strings. Confirm against a live server and tighten.
    """
```

Matches the existing memory `[[review-findings-backlog]]` (residual
F1-F11 closed; only the `_classify` live-verify remains).

## SOLID — SVI tooling has zero non-test callers, exactly as PROJECT_STATE §3 says

```
$ grep -rn "VolatilitySurfaceBuilder\|SVICalibrator\|create_empirical_surface"
```

Production callers: only `engine/__init__.py` re-exports + the
class definitions themselves in `engine/volatility_surface.py`.
Non-production callers: `tests/test_audit_improvements.py`,
`scripts/feature_smoke_test.py`. **Zero live consumers.** PROJECT_STATE
§3 "Open decision: pick a missing-data contract" remains open;
the SVI tooling is a code museum until that decision lands.

## WITH-NOTE N8 — engine/ has five other dormant abstractions PROJECT_STATE doesn't flag

Searched for `from engine.X import` patterns across `engine/`:

| Module | Production callers in engine/ | Notes |
|---|---|---|
| `engine/policy_config.py` | 0 | Docstring claims modules "should accept a TradingPolicyConfig" — none do. Magic numbers remain scattered. |
| `engine/signal_context.py` | 0 (re-exported in `__init__.py:99`) | Docstring shows usage; no caller. |
| `engine/observability.py` | 0 in engine/ (imported by `data/__init__.py:109`) | Not on the decision path. |
| `engine/portfolio_intelligence.py` | 0 | Docstring shows usage; no caller. |
| `engine/skew_dynamics.NelsonSiegelTermStructure` | 0 | Per S29 F4. |
| `engine/skew_dynamics.ivs_dislocation_score` | 0 | Per S29 F4. |

PROJECT_STATE §3 names SVI dormancy explicitly. The five above are
similarly dormant but not flagged. Not §2 concerns; code-museum
hygiene.

## WITH-NOTE N9 — `_handle_news_ingest` docstring claims event-gate integration that doesn't exist

`engine_api.py:2158-2189` is `_handle_news_ingest`. The docstring
(`:2166-2167`) says:

> The endpoint also extracts ticker mentions and checks them against
> the event gate.

The handler body does no such thing — it deserializes the payload,
appends to `_NEWS_BUFFER`, trims, and returns. No event-gate
integration. Docstring drift.

## SOLID — news-stack duplication is as PROJECT_STATE §4 describes

Three modules, only one on the EV path:

| Path | Imported by engine/ ranker? |
|---|---|
| `financial_news/` (RSS/scraping platform) | No (only by `data/__init__.py`) |
| `news_pipeline/` (browser-agent pipeline) | No (only by `scripts/orchestrate.py`) |
| `engine/news_sentiment.py` | Yes — `engine/wheel_runner.py:756` |

✓ matches PROJECT_STATE §4. The duplication is intentional (the
two non-EV stacks are research-tier per `MODULE_INDEX.md`).

## WITH-NOTE N10 — `models/` directory is empty, but referenced

```
$ ls models/
(empty)
$ grep "models/" ml/wheel_model.py
416:    output_path: str = 'models/wheel_entry_model.joblib',
469:    parser.add_argument("-o", "--output", default="models/wheel_entry_model.joblib")
```

PROJECT_STATE §4's claim that `models/` is "retained, not deleted as
a placeholder" because `ml/wheel_model.py` references it — ✓
verified. The directory itself is empty (no trained model artifacts
committed). `ml/wheel_model.py:save` does `mkdir(parents=True)` per
PROJECT_STATE so the dir is not strictly required to pre-exist.

---

# Pass 4 — contract vs implementation, four routes

## Route 1 — `EVEngine.evaluate` (`engine/ev_engine.py:237-541`)

**Signature contract** (`:237-261`): accepts `ShortOptionTrade` +
optional `forward_log_returns` / `price_scenarios` / `trade_start` /
`trade_end` / `market_structure`. Returns `EVResult`. Documents that
`market_structure`, when supplied, multiplies the final regime
scaling by `dealer_regime_multiplier` clamped to `[0.70, 1.05]`,
and that "Dealer positioning NEVER upgrades a negative-EV trade:
the raw ev_raw path is untouched, and the dealer multiplier only
scales the final `ev_dollars` output."

**Implementation match:**

- Event lockout short-circuit at `:262-285` returns BEFORE any math
  if `event_gate is not None and trade_start is not None and trade_end is not None`
  and `is_blocked` returns True. Returns `EVResult(ev_dollars=0.0,
  …, event_lockout_reason=reason, metadata={"blocked": True})`. ✓
  pinned by PR #185.
- `regime_multiplier` input validated to `[0.0, 1.25]` at `:462-473`
  with anomaly metadata for audit replay. NaN → 1.0; negative → 0.0;
  over-cap → 1.25.
- Heavy-tail penalty (default 0.5) multiplied at `:474-475` when
  `heavy_tail` is True.
- Dealer multiplier resolved via lazy import at `:485-488` and
  compounded `regime_mult *= dealer_mult`.
- `ev_dollars = ev_raw * regime_mult` at `:502`. ✓ ev_raw never
  multiplied directly by a reviewer-injected value.

**Contract vs implementation: ✓ aligned.**

## Route 2 — `WheelRunner.rank_candidates_by_ev` (`engine/wheel_runner.py:631-1672`)

**Signature contract** (`:631-712`): 17 kwargs including
`min_history_days=504`, `enforce_history_gate=True`,
`enforce_chain_quality_gate=True`, `use_event_gate=True`,
`use_dealer_positioning=True`, `use_skew_dynamics=True`,
`use_news_sentiment=True`, `use_credit_regime=True`. Returns
DataFrame sorted by `ev_per_day` descending, plus
`.attrs["drops"]` diagnostic list. Documents that "every candidate's
EV comes from a direct `EVEngine.evaluate` call" (`:1935-1937`).

**Implementation match:**

- `ev_eng = EVEngine(event_gate=event_gate)` at `:737`; same engine
  reused for all candidates.
- `evaluate` call at `:1415` is the single per-candidate scoring
  site for puts.
- `min_ev_dollars` threshold gate at `:1432-1442` emits a `drops`
  entry with `gate="ev_threshold"`. Pure observability.
- Capital-efficiency columns `collateral` (`strike * 100 *
  contracts`) and `roc` (`ev_dollars / collateral`) at `:1451-1452`,
  computed strictly from `res.ev_dollars` — "they re-present the
  EV authority's output, never rescue it" (`:1449-1450`).

**Contract vs implementation: ✓ aligned.** The function is large
(1041 lines) but the structure is clear: gate-by-gate, with `drops`
emitted at every drop site, and the EV evaluation as the single
authoritative scoring step.

## Route 3 — `EnginePhaseReviewer` (`engine/candidate_dossier.py:118-353`)

**Signature contract** (`:118-187`): rules R1–R8 documented, each
explicit about downgrade-only intent. R4 explicitly marked
"Conditional — reserved" (dormant).

**Implementation match:**

- R1 at `:202-204`: `if ev < 0` → `"blocked"`, "negative_ev". ✓
- R2 at `:207-210`: chart missing/errored → `"review"`. ✓
- R3 at `:215-223`: spot mismatch → `"skip"`. ✓
- R4 at `:228-238`: phase contradiction → `"skip"` if chart phase in
  `{"compression", "expansion"}` AND disagrees with engine phase.
  **Dormant confirmation:** no production chart provider populates
  `visible_indicators["phase"]` (grep'd `engine/tradingview_bridge.py`:
  `:453` says "`visible_indicators` stays empty in M1"); no ranker
  output emits `phase` on `ev_row` (grep `'"phase":'` in
  `engine/wheel_runner.py`: no matches). R4 fires only in test
  fixtures (`tests/test_dossier_invariant.py:46` sets it explicitly).
- R5 at `:241-248`: `ev >= self.min_proceed_ev` → `"proceed"`. ✓
- R6 at `:258-280`: guarded on `verdict == "proceed"` AND
  `ms is not None`. ✓ downgrade-only.
- R7 at `:288-311`: guarded on `verdict == "proceed"` AND
  `ctx is not None`. ✓ downgrade-only; skips silently on missing data.
- R8 at `:321-351`: guarded on `verdict == "proceed"` AND
  `ctx is not None`. Two triggers (stress drawdown >8% NAV OR
  candidate's underlying in short-gamma regime). ✓ downgrade-only.

**Contract vs implementation: ✓ aligned, with the documented R4
dormancy genuinely realised in code.**

## Route 4 — `engine_api.py` (the HTTP surface, 34 endpoints)

**Signature contract** (PROJECT_STATE §1): "HTTP API on `:8787`;
32 endpoints listed in the file header." Actual count is 34 (see N2).

**Implementation match:** see Pass 1 for the per-endpoint authority
discipline. The two endpoints missing from the header (`/api/news`,
`/api/news/ingest`) serve a separate in-memory news ring buffer
(`_NEWS_BUFFER`) and produce no candidate verdicts.

**Contract vs implementation: ✓ aligned (with the documentation
drift noted in N2).**

---

## Cross-cutting observations

**1. The campaign is in a "deferred follow-up" pattern.** Five distinct
items have been correctly identified by prior audits / S-tests /
backtests but not yet landed: C1 (+inf), C2 (two-path divergence),
C3 (R7/R8 dormant at endpoint), C4 (rank-to-tracker auto-wire), and
the F4 tail-risk fix. Each one has known scope, known fix, and a
named follow-up doc. The risk isn't that they're unknown; the risk
is that they continue to accumulate while the test/realism layer
grows around them. Three of the four (C1, C3, C4) are partially
"covered" by tests in the sense that the contract is pinned — but
the *production reach* of those contracts is partial.

**2. The realism-check Sn campaign (S22→S30) added enormous epistemic
value.** S29's discovery that 100% of 1.35M rows in the Bloomberg IV
file have `put_iv == call_iv EXACTLY` is the kind of structural
finding that would normally cost weeks to surface in an external
audit. S28's discovery that the dividend file forward-truncates 75%
of S&P 500 tickers is similarly load-bearing. S30's confirmation
that the HMM correctly recognises a broad-market crisis across 5/5
names is a positive that pre-S30 the engine couldn't claim. **The
realism-check pattern (engine output vs. reference behaviour) should
become a permanent feature of every new knowledge-surface PR.**

**3. The decision layer's structural integrity is exceptional for a
4-layer Python project of this size.** 11 EVEngine.evaluate call sites,
all in documented routes. 9 reviewer rules, all downgrade-only.
A token-based EV authority gate. Asymmetric dealer clamp with
mathematical guarantees rather than runtime clamps. 62/62
launch-blocker tests passing. The campaign that built this discipline
(audit-I through audit-VIII + D16/D17) shows in the code.

**4. The dormant-surface count is higher than PROJECT_STATE §3/§4
acknowledges.** Six engine/ modules with zero production callers (N8)
+ the SVI tooling + the two skew-dynamics surfaces (S29 F4) + the
heuristic backtester (C5) + the unused `models/` dir (N10) all share
a shape: "infrastructure that was built ahead of the consumer." The
project would benefit from a brief sweep through MODULE_INDEX.md
adding a "live consumers: N" column.

**5. CLAUDE.md is the one piece of high-traffic documentation that's
drifting.** §2 lists R1–R6 (reality: R1–R8). §4's `src/` description
is over-stated (N6). The drift is small but CLAUDE.md is auto-loaded
on every session — the cost-of-error is high relative to the line
count.

**6. The seven cross-doc follow-ups can be consolidated into three
PRs.** Looking across all the open AI-handoff sections:

   - **PR α (ranker chain auto-wire)** — addresses C4, C3, ENGINE_BACKTEST
     F5, TERMINAL_A_AUDIT cross-cutting #4 in a single change. Wire
     `rank_candidates_by_ev` → `issue_ev_authority_token`, thread
     `PortfolioContext` through `build_candidate_dossiers`. Decision-
     layer touch — requires §2 second-review per PARALLEL_SESSIONS
     rule.
   - **PR β (in-process EV sanitisation)** — addresses C1 and C2.
     Add `math.isfinite` to R1, unify `_enrich_alert` ladder by
     calling `EnginePhaseReviewer.review`. Decision-layer touch.
   - **PR γ (observability sweep)** — N1 (CLAUDE.md R7/R8), N2
     (endpoint count), N6 (PROJECT_STATE §4 update), N9 (docstring
     drift), S28/S29/S30 column-honesty fixes. Doc + diagnostic-only
     edits. No §2 surface.

---

## Method appendix

For each finding I:

1. **Read the code at HEAD** at the cited line(s) and quoted the
   literal source where the rule lived.
2. **Ran a live driver** to confirm the boolean truth where the
   claim was behavioural (the +inf / NaN / -inf table in C1).
3. **Ran the relevant test subset** to confirm the contract pinning
   is real (28 + 62 tests, all green).
4. **Grep'd for callers** before declaring a surface dormant — the
   distinction between "untested" and "no live caller" matters.
5. **Cross-referenced PROJECT_STATE / CLAUDE.md / DECISIONS.md** to
   distinguish "new finding" from "known drift" — the four open
   follow-ups in the headline are all already-known per the prior
   audits; this review adds C5 + the dormant-surface count (N8 + N10).

Drivers and one-off scripts ran under `%TEMP%` per the
[[sys-path-worktree-shadow]] convention; not committed.

**Audit history**

| Date | Auditor | Scope | Result |
|---|---|---|---|
| 2026-05-25 | Terminal A (this session) | End-to-end, post-#194 | §2 BREACH 0 / CONCERN 5 / WITH-NOTE 10 / SOLID 12 |
