# Roadmap

What is intentionally next — work that is scoped but not done. This
file is the *forward* companion to `PROJECT_STATE.md` (which records
*current* state) and `CHANGELOG.md` (which records *past* state).

If you finish a roadmap item, **don't delete the entry** — move it
into `CHANGELOG.md` with the commit SHA, then strike the entry here
with a final note pointing to the changelog row.

Each item carries a **status**:
- `next` — committed; the next agent who picks this up should ship it
- `blocked` — explicit dependency unmet, named in the entry
- `parked` — intentionally not now; the entry says when to revisit
- `open question` — needs a human decision before scoping

---

## Open work — refreshed 2026-09-16

The live queue after the restart rulings (`DECISIONS.md` D29). Each row points
at its owning doc; the Execution Prompts are in `docs/RESTART_PLAN_2026-09-16.md`.
Order (revised 2026-09-17): F and E now, A when a subscription is chosen, then B
(A and B share one re-baseline), then C; D once A is live.

| Track | Item | Status | Owning doc |
|---|---|---|---|
| **F** | **Repository structure and efficiency pass** — the Operator rules row by row on the candidate table (advisors, ml, studies, src remnants, dormant engine modules, TradingView MCP workspace, docs mass) and on the wheel_runner ranker/ladder unification | `done` 2026-09-17 on PR #524 for every row except the `wheel_runner` unification (excluded by ruling; open as its own campaign). Follow-up candidates in plan §7a. | `docs/RESTART_PLAN_2026-09-16.md` §7a |
| **A** | **Data without Bloomberg** — provider census, pullers writing the same connector schemas, `scripts/refresh_data.py`; IV-history source is the Operator's subscription call | `parked` until a subscription is chosen (2026-09-17) | `docs/RESTART_PLAN_2026-09-16.md` §3 |
| **B** | **7/14/21/28-day menu + event-aware policy** — menu plumbing → event-conditioned forward distribution + calibration gate → configurable event policy (block \| price) + reviewer rule; re-baseline; delta target deferred (keep 0.25 until ruled) | `next` after A | plan §4 |
| **C** | **Exit evaluator + post-mortem loop** (D25 adopted; advisory, confirmed 2026-09-17; closes F1) | `next` after B | plan §5 |
| **D** | **Strategist commentary layer** (macro + micro brief, engine-sourced figures, prose by an API model) | `next` once A is live | plan §6 |
| **E** | **Protocol v3 adoption** — merge this restart PR (#524, which carries #523 and the Track F cuts; `main` stays unprotected by ruling), docs currency pass (60 stale worklog statuses; audit register and worklist marked shipped; `docs/PRODUCTION_READINESS.md` refresh), campaign issue for Track A | `next` (Operator steps first) | plan §7 |
| — | **News layer redesign** | `parked` until A–C land | D29 |
| — | **F4 IV-fallback guard**, D19 exit-cost netting, D21 horizon units, recalibration | folded into Track B's re-baseline | `docs/REBASELINE_D19_D21_RECAL_SCOPE.md` |

Superseded 2026-09-16: the "Re-baseline session" and "Bloomberg data acquisition"
rows of the 2026-06-09 queue (no Terminal exists; Track A replaces the pull
plan, Track B carries the re-baseline); the "R11 onset-aware trigger" research
card was refuted 2026-06-29 and stays parked; the prob_profit top-bin item is now
inside Track B's calibration gate.

---

## Track A — Decision-layer correctness (closed)

All three items shipped; per-PR detail in `CHANGELOG.md` 2026-05.

- ~~**A1. TradingView MCP chart provider (`MCPChartProvider`)**~~ —
  `done`. Stages 1–3 (PR #95; opt-in via `SWE_USE_MCP_CHART`;
  `DECISIONS.md` D12/D13). Residual: the `TODO(live-verify)` markers
  in `engine/mcp_client.py` need a live TradingView Desktop +
  tradingview-mcp server to confirm. *(Removed 2026-09-17 with the MCP
  workspace — `DECISIONS.md` D30.)*
- ~~**A2. iv_surface missing-data contract**~~ — `done` (2026-05-30):
  chose **fail loudly** (`SurfaceDataUnavailable` + `require_surface`;
  `DECISIONS.md` D9; pinned by `tests/test_iv_surface_failloud.py`).
- ~~**A3. `engine/__init__.py` re-exports the modern decision layer**~~ —
  `done`. Focused 7-symbol re-export, shipped after a pre-edit grep
  proved no existing import site could break.

## Track B — Documentation drift repair (closed)

All six one-shot doc repairs landed; detail in `CHANGELOG.md` 2026-05.

- ~~B1 `README.md` wholesale rewrite~~ · ~~B2 `docs/CONTRIBUTING.md`
  phantom-deps fix~~ · ~~B3 `docs/ARCHITECTURE.md` archived (D14)~~ ·
  ~~B4 `dashboard/README.md` re-positioned~~ · ~~B5 `pyproject.toml`
  phantom entrypoint removed + package list fixed + phantom deps
  (`prefect`, `ib_insync`) dropped~~ · ~~B6 `tradingview/README.md`
  dead link fixed~~ — all `done`. *(B2's `docs/CONTRIBUTING.md` was later
  consolidated into `OPERATING_MODEL.md` §9.8, 2026-07-28.)*

## Track C — Hygiene + governance follow-ups

- ~~**C1. Track-vs-gitignore for the bloomberg yfinance CSVs**~~ —
  `done` (2026-05-30): keep tracking as **data commits** (the
  point-in-time audit trail wins; recorded in `docs/DATA_POLICY.md` §5).
- ~~**C2. Stage the `tradingview/` analyst-workspace files**~~ —
  `done` (PR #78, `4e9c3f3`, 2026-05-15).

### ~~C3. Drop `engine/.gitkeep` and the empty `models/` placeholder~~
**Status:** `done` (2026-06-09, D27 scripts/ pass) — with a scope
correction. The four `.gitkeep`s in **populated** directories
(`scripts/` 67 files, `engine/` 52, `backtests/` 19, `data_raw/` 11)
were removed. The placeholders in **empty-by-design** directories
were deliberately retained as load-bearing: `models/` (the
`ml/wheel_model.py` referenced output path), `notebooks/`,
`data_processed/` (gitignored tree), and
`tradingview/{models,pine,research}` (C2 workspace dirs) — removing
those would remove the directories from git.

## Track E — Coverage push (closed)

Landed **82% on the CI scope** with the `--cov-fail-under` gate moved
70 → 80. `DECISIONS.md` D10 carries the full rationale — including why
E5b (the last ~10pp, in research-tier news plumbing) was cancelled as
coverage theater.

| # | Scope | Outcome |
|---|---|---|
| E1 | 7 EV-adjacent modules → 88-100% | PR #65 — found the NaT crash in `event_gate` |
| E2 | external_data adapters → 97-98% | PR #66 — `requests-mock` template |
| E3 | `engine/theta_connector` 11% → 78% | PR #67 |
| E4 | `event_calendar` → 88%, `risk_manager` → 83% | PR #68 |
| E5a | `news_pipeline/recovery/*` → 63-94% | PR #69 |
| E5b | browser_agents + scrapers + orchestrator | **cancelled** — see `DECISIONS.md` D10 |
| E6 | `--cov-fail-under` 70 → 80 | shipped with E-track close |

## Track F — Lint debt (closed)

~~F1~~ — `done` (PR #79 `9e15dbf`, 2026-05-15): 75 → 0 ruff errors
across the CI scope, one rule per commit. Surfaced two real bugs
(F821 missing `TYPE_CHECKING` imports in `engine/ev_engine.py`; B023
closure traps in `engine/wheel_runner.py` / `engine/earnings_drift.py`).
Detail in `CHANGELOG.md` 2026-05.

---

## Track D — Things explicitly out of scope (do not propose)

Reproduced from `CLAUDE.md`'s NEVER list so a fresh agent doesn't
have to find it. Adding any of these requires explicit user consent
and probably a redesign of the EV path:

- Tick-level order flow / microstructure (Theta v3 doesn't expose it)
- Auto-execution / broker wiring / OMS / order routing
- Non-US equities or non-S&P 500 names
- Non-wheel strategies beyond short puts + covered calls + strangles
  (timing-gated)
- Anything that overrides `EVEngine.evaluate` (see `DECISIONS.md` D1)

---

## How to maintain this file

- New work goes under the matching track (A / B / C). Pick a number
  one higher than the existing largest in that track, and add a row
  to the **Open work** table at the top so the live queue stays a
  one-stop router.
- When you start an item, set status to `in flight` and put your
  branch name on the entry.
- When you finish, move the entry to `CHANGELOG.md` with the SHA;
  leave a strike-through here pointing to the changelog row.
- Do not delete `parked` items unless the parking reason no longer
  applies — the parking reason itself is the value.
