# Restart plan — the Operator's rulings of 2026-09-16 turned into work

**Status:** plan of record for the restart. Companion to
`docs/RESTART_BRIEF_2026-09-11.md` (the analysis) and `DECISIONS.md` D29 (the
rulings). Every track below is written as a sharpened Execution Prompt per
`docs/PROMPTING_STANDARD.md`, so a Strategist (Claude Code or Codex) can hand it
to an Executor with at most the Operator answers filled in.

## 0. The rulings, in one table

| # | Ruling (Operator, 2026-09-16) | Where it lands |
|---|---|---|
| R1 | Build the exit evaluator: a position tracker that evaluates open positions and learns from past trades and mistakes (adopts DECISIONS D25) | Track C |
| R2 | The agent acts as macro and micro strategist on incoming data (volatility trends, whether it is a good time to sell or buy options); news feed joins later | Track D |
| R3 | Tradeable expiries are 7, 14, 21, at most 28 days, not 35; trade close to events to capture elevated premium | Track B |
| R4 | One protocol only; previous working schemes retired | Track E (done in this pass for the tree; history rewrite declined, see §6) |
| R5 | Solidify the working scheme and prompting guidelines | `OPERATING_MODEL.md` v3, `docs/PROMPTING_STANDARD.md` (done in this pass) |
| R6 | Delete all news-feed implementations for now; redesign later | done in this pass (Python, API, tests; dashboard UI in the same PR series) |
| R7 | Strategist–Operator–Executor chain kept; two equal Strategists (Claude Code, Codex); main Executor is Claude Code in VS Code on the Operator's machine; more executors allowed | `OPERATING_MODEL.md` v3 §2 |
| R8 | Away window is overnight, 8 to 10 hours | `OPERATING_MODEL.md` v3 §3.1 |
| R9 | All July loops and terminals are terminated | recorded; no zombie risk |
| R10 | The Operator has never used the product for real trading and wants it to work | the ordering of the tracks below |
| R11 | Bloomberg Terminal access is gone for good; data comes from online sources and subscriptions | Track A |

Answers recorded as "I don't know": the 2026-07-15 consent, the 2026-06-28 cascade
authorisation, the reason for the no-model-name ruling, and whether the held
findings were an acceptance or a stopgap. Consequences: merge authority moves to
branch protection (nothing depends on remembering who clicked); model
attribution is allowed again (D29); F1 and F3 are re-triaged as fixes to make,
F4 rides the re-baseline.

## 1. Order of work

The engine has never been used live. The shortest path to "it works" is: fresh
data (A), then the short-dated menu (B), then the exit evaluator (C), with the
commentary layer (D) alongside once A is in. E is housekeeping that makes the
rest safe to run unattended. Do A first; everything downstream re-baselines
against whatever A produces, and B changes every locked result, so B and A
share one re-baseline.

## 2. Concerns raised before the work starts

**Trading close to events inverts today's safety gate.** The engine drops any
candidate whose holding window touches an earnings date (±5 days). That gate
filtered 48.7% of name-weeks in the $500k campaign and is one of the two guards
that produced the crisis refusal record. Selling premium into earnings is a
legitimate strategy, but the forward distribution that prices it today is an
unconditional sample of 5 years of N-day returns, in which an earnings week
appears about once in thirteen. Pricing an earnings-containing week with that
distribution understates the tail exactly where the premium is richest, and the
engine's known top-bin over-confidence gets worse, not better. So Track B is not
a parameter change. It needs an event-conditioned forward distribution (sample
only windows that contained the same kind of event) and a validation pass before
the gate policy flips. The plan keeps the gate on by default until that pass
passes.

**7-day contracts change what data the engine needs.** Weekly expiries exist for
liquid names only; the current universe gate (504 bars of history) says nothing
about weekly-option liquidity. Track A must carry a liquidity field the ranker
can gate on, or 7-DTE rows will be priced for names that have no weekly chain.

**Every locked result is a 35-DTE result.** The four regression snapshots, the
calibration studies, the $500k campaign and the parameter-freeze holdout are all
35 DTE. After Track B nothing in the evidence table transfers; the engine must
be re-validated at the new horizons before the real-money verdict is revisited.

**Deleting history is the one instruction not executed.** See §6.

## 3. Track A — data without Bloomberg

```
<run-mode> change (design first as read-only; implementation in a second run)
<request-as-sharpened>
  Goal: give the engine a live, refreshable data path that needs no Bloomberg Terminal.
  Success: `python scripts/refresh_data.py` (new) advances every connector input the
    engine reads to yesterday's close from online sources, on the Operator's machine,
    in under an hour, and the EV smoke ranks with today's date without staleness warnings.
  Tried before: yfinance pullers exist for treasury (in use), OHLCV/options/fundamentals/
    earnings (unconsumed `_yf` parallels); Theta connector and larder exist (options,
    EOD stocks, IV snapshots; no IV history at the Operator's tier); FRED, CBOE, EDGAR
    adapters exist. The committed CSVs are a frozen 2018-01-02 → 2026-07-02 history.
  Must not break: the connector's CSV panel schemas (the engine reads them); point-in-time
    discipline (every panel keeps a knowledge date); the split-adjusted-vs-raw rule (D20/D11).
  Scope of run: a provider census + design, then the pullers. No engine change.
  Operator answers: WHICH SUBSCRIPTIONS EXIST OR ARE ACCEPTABLE — Theta Data (tier?),
    Polygon / Massive, Tiingo, Alpaca, ORATS, IVolatility, none (free only)? This decides
    the IV-history source, the one input with no free replacement.
<objective> Produce docs/DATA_SOURCING_2026-09.md: for each of the 13 connector inputs
  (10 CSVs + 3 broad-pull panels) the replacement source, its cost, its PIT property, its
  refresh cadence and a sample pull; then implement the pullers that write the SAME CSV
  schemas so MarketDataConnector needs no change. IV history: propose computing daily
  ATM IV from Theta EOD option quotes (the option_history larder + produce_option_premiums)
  or a vendor series; the Operator picks.
<context> docs/DATA_POLICY.md §2 matrix; engine/data_connector.py `_FILES`;
  docs/DATA_SUFFICIENCY_REVIEW_2026-07-21.md §4 (the last refresh procedure);
  scripts/pull_*_yf.py; engine/external_data/*; tests/test_preflight_environment.py
  (EXPECTED_FRONTIER); the frontier-pinned data tests.
<scope> docs/DATA_SOURCING_2026-09.md; scripts/refresh_data.py; scripts/pull_*.py edits;
  tests/test_data_refresh.py (schema + PIT tests on synthetic fixtures).
<out-of-scope> engine/*, the trio, snapshots, any change to what the engine computes.
<owns> the files above. <reads> engine/data_connector.py, engine/theta_connector.py.
<invariants> Data integrity (provider logged; no silent substitution; PIT absolute). Honesty.
<gates> before adding any paid dependency; before the first commit that changes a tracked CSV
  (data commits are audited; keep the frozen Bloomberg history under a dated subfolder).
<verification> python -m pytest tests/test_data_refresh.py tests/test_preflight_environment.py -q;
  the EV smoke with as_of=None shows no frontier warning; check_manifest_coverage OK.
<push> claude/data-sourcing
<summary> Run Summary per OPERATING_MODEL.md §4.4 as a PR comment.
<constraints> Never delete the committed Bloomberg CSVs; never mix split-adjusted and raw prices.
```

## 4. Track B — 7/14/21/28-day menu and the event-aware policy

```
<run-mode> change, in three runs (menu → event-conditioned distribution → policy flip)
<request-as-sharpened>
  Goal: the engine offers 7, 14, 21 and 28-day puts and can price trades near events.
  Success (run 1): rank_candidates_by_ev takes a dte_menu and returns one row per (ticker,
    dte) with per-row liquidity and event flags; default menu (7,14,21,28); event gate ON.
  Success (run 2): an event-conditioned forward distribution exists and is validated: on
    2018–2026 data, prob_profit calibration for earnings-containing windows within 5 points
    of realized at every menu DTE, measured by the existing calibration harness.
  Success (run 3): a configurable event policy (block | price) with `price` allowed only
    when the run-2 gate passed; the reviewer adds a rule that downgrades event trades
    whose premium does not exceed the event-conditioned tail cost.
  Tried before: dte_target is already a parameter (35); select_book iterates dtes;
    forward_distribution takes horizon_days; the event gate is whole-window ±5 days.
  Must not break: EVEngine.evaluate authority; downgrade-only reviewers; D21 horizon
    semantics (fix it in the same pass, not before).
  Scope of run: as above; each run is its own PR with a lane claim.
  Operator answers: menu (7,14,21,28) confirmed; events priced, not blocked, once validated;
    which delta target for short DTE (keep 0.25?) — ANSWER NEEDED.
<objective> see Success. <context> engine/wheel_runner.py rank_candidates_by_ev;
  engine/forward_distribution.py (tiers); engine/event_gate.py; docs/PROB_PROFIT_CALIBRATION_2026-05-28.md;
  docs/REBASELINE_D19_D21_RECAL_SCOPE.md; the calibration accumulator in engine/paper_book.py.
<scope> run 1: wheel_runner (menu), tests; run 2: forward_distribution (new tier), a study
  doc with the calibration table, tests; run 3: event_gate policy, candidate_dossier
  (new rule R12), tests, re-baseline of the four snapshots.
<out-of-scope> data pulls (Track A); exit logic (Track C); any change to ev_dollars semantics.
<owns> per run, listed in its prompt. <invariants> Decision integrity; Modelling; Validation honesty.
<gates> every trio edit; the snapshot re-baseline; the policy flip (Operator signs the run-2 result).
<verification> launch blockers + full fast lane per run; run 2 adds the calibration table
  as pasted output; run 3 re-runs the regression harness and pins new snapshots.
<push> claude/dte-menu, claude/event-conditioned-dist, claude/event-policy
<summary> per OPERATING_MODEL.md §4.4 as PR comments.
<constraints> The gate stays ON until run 2 passes; no synthetic-premium-only validation
  for event trades (use the Theta larder's real mids where they exist).
```

## 5. Track C — exit evaluator and the post-mortem loop (D25 adopted)

```
<run-mode> change, in two runs (evaluator → post-mortem)
<request-as-sharpened>
  Goal: the engine evaluates open positions (hold / roll / close / accept assignment) and
    learns from closed trades.
  Success (run 1): `WheelTracker.evaluate_open_positions(as_of)` scores every open leg
    through EVEngine.evaluate (roll legs re-scored, never a shortcut), returns a ranked
    action list with reasons, and the D17 caps + D16 token apply to rolls (closes F1).
  Success (run 2): a post-mortem record per closed trade (forecast vs realized, regime at
    entry, event proximity, sizing) accumulates in the SIM/IBKR namespaces, and a monthly
    report proposes rule changes as PROPOSALS (never applies them).
  Tried before: suggest_rolls / suggest_call_rolls compute hold_ev vs roll_ev; the IBKR
    EV-calibration loop and the paper-book calibration accumulator exist; D25 is the
    reserved decision; F1 is the roll bypass.
  Must not break: brokerage read-only; downgrade-only; no order routing.
  Operator answers: hold-to-expiry remains the default; the evaluator is ADVISORY (no
    auto-action) — CONFIRM.
<objective> see Success. <context> engine/wheel_tracker.py (roll_put/roll_call, suggest_rolls),
  docs/IBKR_LIVE_BOOK_INTEGRATION.md §3, tests/test_held_finding_roll_ev_bypass.py,
  scripts/ibkr_ev_calibration.py, engine/paper_book.py calibration accumulator.
<scope> run 1: engine/wheel_tracker.py, engine/exit_evaluator.py (new), engine_api /api/positions/evaluate,
  tests; run 2: engine/post_mortem.py (new), scripts/post_mortem_report.py, tests.
<out-of-scope> the ranker; the dashboard (a follow-up); any automatic order.
<invariants> Decision integrity; Brokerage safety; Honesty.
<gates> before touching wheel_tracker's roll path; before exposing the API endpoint.
<verification> the F1 xfail flips to pass (remove the marker in the same PR); launch blockers;
  full fast lane; a paper-book replay showing evaluator actions on a seeded book.
<push> claude/exit-evaluator, claude/post-mortem
<summary> per OPERATING_MODEL.md §4.4 as PR comments.
<constraints> Rolls re-scored through EVEngine.evaluate only; no new path from a negative-EV
  leg to a tradeable one.
```

## 6. Track D — strategist commentary (macro and micro)

```
<run-mode> change
<request-as-sharpened>
  Goal: a daily written brief in which the agent reads the engine's own state and the
    market inputs and comments as a strategist: volatility rising or falling, term
    structure, regime, whether selling or buying premium is favoured now, which names
    and expiries the menu favours, what to avoid.
  Success: `python scripts/market_brief.py` writes docs-free output to the SIM namespace
    and the dashboard terminal shows it; every number in the brief is traceable to a
    connector field or ranker column; the brief never changes a verdict (§7).
  Tried before: engine/trade_memo.py (Ollama memos), the regime banner, VIX regime,
    credit regime, dealer positioning (Theta only), the news panel (removed).
  Must not break: reviewers downgrade only; the brief is operator-transparency, not an input.
  Operator answers: model for prose (local Ollama vs API) — ANSWER NEEDED; cadence (daily
    pre-market) — CONFIRM.
<objective> see Success. <scope> engine/market_brief.py (new), scripts/market_brief.py,
  engine_api /api/brief, dashboard terminal panel (follow-up), tests.
<out-of-scope> news ingestion (later redesign); any EV multiplier.
<invariants> Decision integrity (no upgrade path); Honesty (every figure sourced).
<gates> before adding an LLM dependency; before the API endpoint.
<verification> unit tests on the brief's data assembly; a rendered sample pasted in the summary.
<push> claude/market-brief
<constraints> No number in the brief may be generated by the language model; the model
  writes prose around engine-supplied figures.
```

## 7. Track E — protocol v3 adoption (housekeeping)

Done in this pass: OPERATING_MODEL v3, PROMPTING_STANDARD, hook and template
updates, archived old-protocol docs deleted, coordination issues closed. Left
for the first Operator-run steps: turn on branch protection for `main`
(required checks, no direct push, base must be main); merge this branch's PR,
then PR #523; run the docs currency pass (60 stale worklog statuses, audit
register and worklist marked shipped, `docs/PRODUCTION_READINESS.md`
refreshed); open the campaign issue for Track A.

## 8. What was not executed: rewriting history

The Operator asked to delete the previous protocols "even from the history".
The old protocol documents are deleted from the tree in this pass (`archive/`
copies included) and the coordination issues are closed. Rewriting git history
was **not** done, for three reasons the Operator should weigh before confirming
it as a separate, explicit action: it invalidates every clone, worktree and open
PR (#523, #507) and needs a force-push to `main`; the protocols also live in
GitHub issues, which a git rewrite does not touch; and the record of how the
rules were arrived at is the project's memory of *why* they exist. If the
Operator confirms, the procedure is a filter-repo rewrite of the six deleted
governance files and the two archive files, followed by a coordinated re-clone.
