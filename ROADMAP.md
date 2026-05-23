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
**Status:** `open question`
**Context:** `engine/volatility_surface.py` (SVI calibrator + builder)
has zero non-test callers. Theta `iv_surface/` snapshot covers
28/503 tickers; `iv_surface_history/` is now 381/503 after the
2026-05-04 pull (122 strict-mode rejections).
**The question:** for the uncovered tickers, do we (a) fail loudly,
or (b) use a clearly-named `flat_iv_fallback` (never silent)?
**Done when:** the contract is recorded in `DECISIONS.md` D9 and the
SVI tooling has at least one production caller, OR the SVI tooling
is removed from `engine/__init__.py` re-exports and marked
**deprecated** in `MODULE_INDEX.md`.

### A3. `engine/__init__.py` re-exports the modern decision-layer
**Status:** `parked` (revisit when an EV-layer refactor is in flight)
**Issue:** the package currently re-exports the legacy quant layer
(`option_pricer`, `monte_carlo`, `regime_detector`, …) but not
`EVEngine`, `WheelRunner`, `EnginePhaseReviewer`, `MarketStructure`.
A fresh agent has to discover the full submodule paths.
**Why parked:** touching `engine/__init__.py` ripples through every
import site. Bundle with a real refactor instead of doing in
isolation.

---

## Track B — Documentation drift to repair

These were named in `PROJECT_STATE.md` §5 and are still pending. Each
is a one-shot doc edit, not code.

### B1. `README.md` — describes the wrong product
**Status:** `next`
**Issue:** the root README references the Python CLI dashboard
(`python -m dashboard.quant_dashboard`), broker env vars
(`BROKER_API_KEY`, `BROKER_SECRET`) which are out of scope per
`CLAUDE.md`'s NEVER list, and a 6-folder project structure (actual is 20+).
**Done when:** README reflects the real entry points
(`python engine_api.py`, the Next.js dashboard, `morning_run.py`),
removes broker references, and points at `AGENTS.md` /
`PROJECT_STATE.md` / `MODULE_INDEX.md` for orientation.

### B2. `docs/CONTRIBUTING.md` — installs phantom deps
**Status:** `next`
**Issue:** says `pip install -e ".[dev]"` will install from a
pyproject still listing `streamlit`, `prefect`, `ib_insync` as hard
deps. None are part of the EV decision path.
**Done when:** either pyproject is cleaned (see B5) or
docs/CONTRIBUTING.md tells the contributor which deps are optional and
why.

### B3. `docs/ARCHITECTURE.md` — describes the wrong tree
**Status:** `next`
**Issue:** documents `src/data`, `src/features`, `src/execution` as
the live architecture. Actual quant layer is `engine/`.
**Done when:** the doc is rewritten against `engine/` and references
`MODULE_INDEX.md` for the per-module map, OR the doc is replaced
with a one-line redirect to `MODULE_INDEX.md`.

### B4. `dashboard/README.md` — wrong product name
**Status:** `next`
**Issue:** still says "FinanceNews — AI Financial News Platform".
The directory was reused; the README was not.
**Done when:** README describes the Next.js dashboard for the engine,
its build/dev commands, and how it connects to `engine_api.py`.

### B5. `pyproject.toml` — phantom entrypoint and wrong package list
**Status:** `next`
**Issue:** `[project.scripts] wheel = "src.cli:app"` references a
file that does not exist. `[tool.hatch.build.targets.wheel]
packages = ["src"]` would build a wheel that excludes `engine/`,
`engine_api.py`, `advisors/`, the dashboard, etc.
**Done when:** the script entry is either removed or pointed at a
real entry; the packages list reflects the real install surface.

### B6. `tradingview/README.md` — references a non-existent doc
**Status:** `next`
**Issue:** line 11 says "the underlying architecture is documented in
`TRADINGVIEW_INTEGRATION_REPORT.md` at the repo root" — that file
does not exist.
**Done when:** the link is updated to point at
`docs/TRADINGVIEW_INTEGRATION.md` (parent doc) and
`docs/TRADINGVIEW_MCP_INTEGRATION.md` (MCP-specific design contract).

---

## Track C — Hygiene + governance follow-ups

### C1. Decide whether the bloomberg yfinance CSVs should be tracked
**Status:** `open question`
**Issue:** `data/bloomberg/sp500_earnings_yf.csv`,
`sp500_fundamentals_yf.csv`, `treasury_yields.csv` are tracked but
re-generated on every yfinance pull. They show local modifications
right now (refreshed earnings + treasury yields).
**Options:**
- *Track and treat refreshes as data commits* — current behaviour;
  forces a commit-and-push every refresh
- *Gitignore and add to a daily refresh script* — cleaner history;
  loses the "what data did we run on?" historical record
- *Track but freeze* — refresh in a separate `data/refresh/` overlay
  that is gitignored

### C2. Stage the new `tradingview/` analyst workspace files
**Status:** `next`
**Files to commit on next branch:** `tradingview/CLAUDE.md`,
`tradingview/OVERVIEW.md`, `tradingview/launch-tradingview-cdp.sh`,
`tradingview/models/.gitkeep`, `tradingview/pine/.gitkeep`,
`tradingview/research/.gitkeep`. The `tradingview-mcp-jackson/`
nested repo and the `*.docx` deliverables stay gitignored.

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
   decision path.
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

## Track F — Lint debt cleanup (next)

### F1. Close the residual 44 ruff errors
**Status:** `next`
**Context:** PR #64 (`1fb2c33`) closed 187/229 ruff errors via
`ruff check --fix` + `ruff format`. The residual 44 need
judgement-required fixes:
- `UP038` (~14): `isinstance(x, (A, B))` → `isinstance(x, A | B)`.
  `--unsafe-fixes` autofixable but each one deserves an eyeball
  for `__instancecheck__` sites.
- `B904` (~8): `raise X` inside `except` → `raise X from <orig>`.
  Per-site judgement on which cause to chain.
- `B023` (~6): closure trap on loop variables. Real bugs hiding
  in the lint output.
- `F841` (~5): unused locals. Delete or `_`-prefix.
- `B019` (~5): mutable default args / cached values.
- `F821` (~3): undefined names. **Investigate before
  suppressing** — if real, runtime should already be failing.
- Misc (~3): `E741` (ambiguous names), `C408` (unnecessary
  collection call), `P031`, `E402`, `C410`, `B007`.
**Approach:** one rule per commit so each is independently
reviewable. Optionally suppress in `pyproject.toml [tool.ruff]
ignore` with a per-rule justification comment.
**Done when:** `ruff check src/ engine/ advisors/ financial_news/
tests/ scripts/ utils/ news_pipeline/ dashboard/` is clean
(or every remaining error has a documented suppression).

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
