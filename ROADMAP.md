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

## Track A — Decision-layer correctness

### A1. TradingView MCP chart provider — `MCPChartProvider`
**Status:** `done`
**Owner contract:** `docs/TRADINGVIEW_MCP_INTEGRATION.md`
**Seam:** `engine/tradingview_bridge.py` — `MCPChartProvider` +
`engine/mcp_client.py` `MCPCLIClient`.
**Done when:** ✅ the import-guarded contract test in
`tests/test_dossier_invariant.py::test_mcp_provider_*` is active and
passes; the chained provider falls through to filesystem on MCP
failure (no quiet substitution).
**Shipped:** Stage 1 (skeleton) + Stage 2 (`MCPCLIClient`) via PR #95;
Stage 3 wires `MCPChartProvider` into `build_default_provider` behind
the `SWE_USE_MCP_CHART` env var — opt-in, co-located transport
(`DECISIONS.md` D13). Remaining: `TODO(live-verify)` markers in
`engine/mcp_client.py` need a live TradingView Desktop + tradingview-mcp
server to confirm JSON field names.

### A2. iv_surface integration — pick the missing-data contract
**Status:** `done` (2026-05-30) — decided **(a) fail loudly**. The SVI
tooling is wired in behind `SurfaceDataUnavailable` + `require_surface`
(`engine/volatility_surface.py`); first production caller is
`scripts/diagnose_iv_surface.py` (fail-loud, non-zero exit on uncovered
tickers); contract recorded in `DECISIONS.md` D9; pinned by
`tests/test_iv_surface_failloud.py`. See `CHANGELOG.md` 2026-05.

### A3. `engine/__init__.py` re-exports the modern decision-layer
**Status:** `done` — the parking premise turned out to be wrong. A
pre-edit grep proved every existing import site uses the full
submodule path (`from engine.ev_engine import EVEngine`, etc.), so
adding entries to `__all__` could only *enable* new imports, never
break existing ones. Shipped a focused 7-symbol re-export (EVEngine,
EVResult, ShortOptionTrade, WheelRunner, EnginePhaseReviewer,
CandidateDossier, MarketStructure) plus a docstring line pointing at
CLAUDE.md §1. See `CHANGELOG.md` 2026-05.

---

## Track B — Documentation drift to repair

These were named in `PROJECT_STATE.md` §5 and are still pending. Each
is a one-shot doc edit, not code.

### B1. `README.md` — describes the wrong product
**Status:** `done` — wholesale rewrite. README now leads with the
four-layer model + EV invariant, points at `python engine_api.py` /
the Next.js dashboard / `morning_run.py` as the real entry points,
removes the CLI-dashboard and broker-env-var references, and links
the full Tier-1 + Tier-2 + on-demand doc set.

### B2. `docs/CONTRIBUTING.md` — installs phantom deps
**Status:** `done` — rewrote setup to use `pip install -r requirements.txt`
plus the explicit dev-tooling install. The pyproject install
(`pip install -e ".[dev]"`) is explicitly called out as known-stale
with a pointer to B5. Took the "docs/CONTRIBUTING.md tells the
contributor which deps are optional and why" branch of the original
Done-when.

### B3. `docs/ARCHITECTURE.md` — describes the wrong tree
**Status:** `done` — archived in D14 to `archive/2026-05/ARCHITECTURE.md`.
`MODULE_INDEX.md` is the live per-module map.

### B4. `dashboard/README.md` — wrong product name
**Status:** `done` — wholesale rewrite. Re-positioned as the Next.js
cockpit for the Smart Wheel Engine (primary surface: engine
cockpit; secondary: the financial-news component that piggybacks
on the same app). Build/dev commands retained; added an explicit
"How the engine cockpit talks to the engine" section pinning the
Next.js → `:8787` proxy contract.

### B5. `pyproject.toml` — phantom entrypoint and wrong package list
**Status:** `done` — closed in this branch. The `[project.scripts]
wheel = "src.cli:app"` block was removed (the target file does not
exist; no consumer relied on the script). `[tool.hatch.build.targets.wheel]
packages` was expanded from `["src"]` to `["engine", "advisors",
"data", "financial_news", "news_pipeline", "src"]` so a built wheel
now contains the live install surface. `src` is retained per
DECISIONS.md D2 (still imported by tests + four production modules).
The two confirmed-phantom dependencies (`prefect`, `ib_insync`) were
also removed from `[project.dependencies]` — neither is imported by
any tracked Python file, and `ib_insync` would have violated the
CLAUDE.md NEVER-rule "no broker integration." `streamlit` is kept;
it is a real consumer (`local_agent/ui/streamlit_app.py`). See
`CHANGELOG.md` 2026-05.

### B6. `tradingview/README.md` — references a non-existent doc
**Status:** `done` — fixed during the foundation pass
(see `PROJECT_STATE.md` §3.6). Link now points at
`docs/TRADINGVIEW_INTEGRATION.md`.

---

## Track C — Hygiene + governance follow-ups

### C1. Decide whether the bloomberg yfinance CSVs should be tracked
**Status:** `done` (2026-05-30) — decided **keep tracking as data
commits** (the *Track and treat refreshes as data commits* option).
Rationale: the point-in-time "what data did we run on?" audit trail
(`docs/DATA_POLICY.md` §4 PIT discipline) outweighs the
commit-per-refresh history noise, and it needs zero migration.
`sp500_earnings_yf.csv`, `sp500_fundamentals_yf.csv`,
`treasury_yields.csv` stay tracked; a refresh is a data commit.
Recorded in `docs/DATA_POLICY.md` §5. See `CHANGELOG.md` 2026-05.

### C2. Stage the new `tradingview/` analyst workspace files
**Status:** `done` — shipped via PR #78
(`claude/audit-fixes-no-coverage`), merged to main as `4e9c3f3` on
2026-05-15. Tracked the six listed files: `tradingview/CLAUDE.md`,
`tradingview/OVERVIEW.md`, `tradingview/launch-tradingview-cdp.sh`,
and the three `.gitkeep` placeholders for `models/`, `pine/`, and
`research/`. The `tradingview-mcp-jackson/` nested repo and the
`*.docx` deliverables remain gitignored as planned. See
`CHANGELOG.md` 2026-05.

### C3. Drop `engine/.gitkeep` and the empty `models/` and
`validation/` placeholders
**Status:** `parked` (low value)
**Issue:** `engine/` is fully populated; the .gitkeep is harmless
noise. `models/` and `validation/` are empty placeholders from an
earlier scaffold.
**Why parked:** removing empty placeholders is rarely worth a PR;
bundle into a future cleanup.

---

## Track E — Coverage push (CLOSED — see CHANGELOG 2026-05)

The original goal was "90%+ on the EV-adjacent surface." The
landed outcome is **82% on the CI scope**
(`src + engine + advisors + financial_news`), with the
`--cov-fail-under` gate raised from 70 → 80. The remaining ~10pp
to 90% lives in research-tier code (browser agents, scrapers,
orchestrator) and would be coverage theater — see E5b below.

| # | Scope | Status | Outcome |
|---|---|---|---|
| E1 | 7 EV-adjacent modules → 88-100% | ✅ shipped | PR #65 — `f395aa6`. Found NaT crash in `event_gate`. |
| E2 | external_data adapters (4 files) → 97-98% | ✅ shipped | PR #66 — `354f440`. `requests-mock` template. |
| E3 | `engine/theta_connector` 11% → 78% | ✅ shipped | PR #67 — `10ddc9d`. Mocked v3 endpoints. |
| E4 | `event_calendar` 31% → 88%, `risk_manager` 63% → 83% | ✅ shipped | PR #68 — `526ce67`. |
| E5a | `news_pipeline/recovery/*` to 63-94% | ✅ shipped | PR #69 — `3754779`. Async paths deferred. |
| **E5b** | browser_agents + scrapers + orchestrator + async health | **❌ cancelled** | See rationale below. |
| E6 | Raise `--cov-fail-under` 70 → 80 | ✅ shipped | This PR. |

### Why E5b was cancelled (2026-05-06)

The remaining ~10pp gap to 90% sits in
`news_pipeline/{browser_agents,scrapers,orchestrator}.py` and the
async paths of `recovery/health.py`. Three reasons not to chase it:

1. **Research-tier, not on the EV path.** `MODULE_INDEX.md`
   "Other top-level dirs" classifies these as research /
   experimental. Per `DECISIONS.md` D1, news content reaches the
   EV engine only through `engine/news_sentiment.py` (already at
   94% via E1), which reads from disk. The browser plumbing that
   *produces* those files is consumed asynchronously, off the
   decision path. **Post-D18 (2026-05-26):** even
   `engine/news_sentiment.py`'s EV-influence channel is severed
   (`sentiment_multiplier` returns constant 1.0), strengthening
   the case — no news subsystem feeds the EV authority any more.
2. **Infrastructure investment far exceeds value.** Coverage would
   require a Playwright + aiohttp + SessionManager mock harness —
   ~hundreds of lines of fixture infrastructure for code that's
   exercised by `morning_run.py`'s actual browser sessions every
   day.
3. **Goodhart's law.** Pushing the percentage by writing tests for
   research-tier code reduces the percentage's signal value about
   the *EV decision contract* (which is what should be locked
   down). 80% on the value-bearing code is more honest than 90%
   averaged with 90% on plumbing tests that don't pin invariants.

If the research-tier modules ever migrate onto the EV path, the
calculus changes and this should be revisited.

---

## Track F — Lint debt cleanup (closed)

### F1. Close the residual ruff errors
**Status:** `done` — closed by PR #79 (`9e15dbf`), merged
2026-05-15. The mechanical pass (PR #64 `1fb2c33`) left 44
judgement-required errors as of `3754779`; additional churn between
that commit and PR #79 grew the count to 75, all cleared in one
rule-per-commit pass. CI scope (`src/ engine/ data/ advisors/
financial_news/ tests/ scripts/ utils/ news_pipeline/ dashboard/`)
verified clean post-merge. Surfaced two real bugs during the cleanup:
F821 in `engine/ev_engine.py` (missing `if TYPE_CHECKING:` imports —
`typing.get_type_hints` would have raised `NameError`); B023 closure
traps in `engine/wheel_runner.py` and `engine/earnings_drift.py`
(rebound via default-arg capture — latent if either nested function
is ever stored or deferred). See `CHANGELOG.md` 2026-05.

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
  one higher than the existing largest in that track.
- When you start an item, set status to `in flight` and put your
  branch name on the entry.
- When you finish, move the entry to `CHANGELOG.md` with the SHA;
  leave a strike-through here pointing to the changelog row.
- Do not delete `parked` items unless the parking reason no longer
  applies — the parking reason itself is the value.
