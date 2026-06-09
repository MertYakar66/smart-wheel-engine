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

## Open work — refreshed 2026-06-09

The live queue. Each row points at its owning doc; this table is a
router, not the spec.

| Item | Status | Owning doc |
|---|---|---|
| **Re-baseline session** — the D19 (exit-cost netting) + D21 (calendar→trading-day horizon) deferred fixes + the open data queue + S-snapshot re-pin, executed as one coordinated session | `next` | `docs/NEXT_DATA_SESSION_RUNBOOK.md` (PR #381 — the single authoritative runbook) |
| **Bloomberg data acquisition** — pull-broadly plan + the no-code pull checklist | `next` (needs operator Terminal access) | `docs/DATA_ACQUISITION_ROADMAP.md`, `docs/BLOOMBERG_PULL_LIST.md` |
| **prob_profit top-bin over-confidence** — wire the POT-GPD tail machinery (`engine/tail_risk.py`) into the `prob_profit` computation path | `open question` (research) | `PROJECT_STATE.md` §3 "prob_profit calibration", `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` |
| **R11 onset-aware trigger** — persistence-based VIX trigger (fire after N consecutive days >25: catch the 2022 grind, skip the 2020 spike) | `parked` (research card) | `DECISIONS.md` D23 post-ship validation; the r11-onset-aware card in `docs/worklog/` |
| C3 placeholder cleanup | `parked` (low value) | Track C below |

---

## Track A — Decision-layer correctness (closed)

All three items shipped; per-PR detail in `CHANGELOG.md` 2026-05.

- ~~**A1. TradingView MCP chart provider (`MCPChartProvider`)**~~ —
  `done`. Stages 1–3 (PR #95; opt-in via `SWE_USE_MCP_CHART`;
  `DECISIONS.md` D12/D13). Residual: the `TODO(live-verify)` markers
  in `engine/mcp_client.py` need a live TradingView Desktop +
  tradingview-mcp server to confirm.
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
  dead link fixed~~ — all `done`.

## Track C — Hygiene + governance follow-ups

- ~~**C1. Track-vs-gitignore for the bloomberg yfinance CSVs**~~ —
  `done` (2026-05-30): keep tracking as **data commits** (the
  point-in-time audit trail wins; recorded in `docs/DATA_POLICY.md` §5).
- ~~**C2. Stage the `tradingview/` analyst-workspace files**~~ —
  `done` (PR #78, `4e9c3f3`, 2026-05-15).

### C3. Drop `engine/.gitkeep` and the empty `models/` placeholder
**Status:** `parked` (low value)
**Issue:** `engine/` is fully populated, so its `.gitkeep` is harmless
noise. `models/` is an empty referenced-path placeholder
(`ml/wheel_model.py`'s default output directory — see
`PROJECT_STATE.md` §4). The third placeholder originally named here,
`validation/`, was already removed in D14.
**Why parked:** removing empty placeholders is rarely worth a PR;
bundle into a future cleanup.

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
