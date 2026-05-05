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
**Status:** `next`
**Owner contract:** `docs/TRADINGVIEW_MCP_INTEGRATION.md`
**Seam:** `engine/tradingview_bridge.py` — add a new
`MCPChartProvider(ChartContextProvider)` that joins the existing
`ChainedChartProvider` ordering.
**Done when:** the import-guarded contract test in
`tests/test_dossier_invariant.py::test_mcp_provider_*` activates and
passes; the chained provider falls through to filesystem on MCP
failure (no quiet substitution).
**Why now:** the design contract is locked, the seam is clean,
and the MCP server (`tradingview/tradingview-mcp-jackson/`) is the
mechanism Mert already uses for the analyst workspace — sharing it
with the engine is a cheap consolidation.

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
`CLAUDE.md` §4, and a 6-folder project structure (actual is 20+).
**Done when:** README reflects the real entry points
(`python engine_api.py`, the Next.js dashboard, `morning_run.py`),
removes broker references, and points at `AGENTS.md` /
`PROJECT_STATE.md` / `MODULE_INDEX.md` for orientation.

### B2. `CONTRIBUTING.md` — installs phantom deps
**Status:** `next`
**Issue:** says `pip install -e ".[dev]"` will install from a
pyproject still listing `streamlit`, `prefect`, `ib_insync` as hard
deps. None are part of the EV decision path.
**Done when:** either pyproject is cleaned (see B5) or
CONTRIBUTING.md tells the contributor which deps are optional and
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
`TRADINGVIEW_INTEGRATION.md` (parent doc) and
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

## Track E — Coverage to 90%+ (multi-PR)

Goal: raise CI `--cov-fail-under` from 70 to 90+ on the
`engine/ + advisors/ + financial_news/ + news_pipeline/` scope.
Baseline measured 2026-05-05: **63%** (12,884 statements, 4,324
missing). Target: **90%+**. See `DECISIONS.md` D10 for the
"invariants first, then 90% as a forcing function" framing.

Each phase is **one reviewable PR**, not one big-bang push.

### E1. Phase 1 — easy wins (this branch)
**Status:** `in flight` (`claude/coverage-phase-1`)
**Modules:** `engine/observability` (0% → 90%+),
`engine/earnings_drift` (0% → 90%+), `engine/contracts` (40% → 90%+),
`engine/policy_config` (33% → 90%+), `engine/event_gate` (57% → 90%+),
`engine/news_sentiment` (43% → 90%+), `engine/tail_risk` (71% → 90%+).
**Side effect:** uncovered + fixed `NaT`-handling crash in
`event_gate.from_bloomberg_calendar` (see PR description).
**Expected gain:** 63% → ~70%.

### E2. Phase 2 — external data adapters with `requests-mock`
**Status:** `next`
**Modules:** `engine/external_data/cboe_adapter` (24%),
`engine/external_data/edgar_adapter` (24%),
`engine/external_data/fred_adapter` (52%),
`engine/external_data/yfinance_adapter` (24%).
**Approach:** install `requests-mock` (or `responses`); stub each
remote endpoint with realistic + adversarial fixtures; assert the
adapter's contract (no silent failures, retries, schema parsing).
**Expected gain:** ~70% → ~75%.

### E3. Phase 3 — `theta_connector` with mocked v3 endpoints
**Status:** `next`
**Module:** `engine/theta_connector` (currently 11% — 470 stmts,
404 missing).
**Approach:** mock the Theta v3 HTTP API surface. Will require
fixture data for OHLCV, IV history, chains, surfaces. Reuse the
existing `data_processed/theta/**` parquets as fixture seeds where
possible.
**Expected gain:** ~75% → ~80%.

### E4. Phase 4 — deep-coverage on big in-scope modules
**Status:** `next`
**Modules:** `engine/risk_manager` (63%, 702 stmts),
`engine/event_calendar` (31%, 368 stmts),
`engine/strangle_timing` (76%, 334 stmts).
**Approach:** target the missing branches surfaced by
`--cov-report=term-missing`. Heavy on edge cases and adversarial
inputs.
**Expected gain:** ~80% → ~85%.

### E5. Phase 5 — `news_pipeline/orchestrator` + scrapers + browser agents
**Status:** `next` (largest scope; may split into E5a / E5b)
**Modules:** `news_pipeline/orchestrator` (0%, 358 stmts),
`news_pipeline/scrapers/{aggregator,browser_scraper,rss_scraper}.py`
(0%-51%), `news_pipeline/browser_agents/*` (0% on most).
**Approach:** build a `pytest`-friendly Playwright + browser-session
mocking harness. Largest infrastructure investment of the five
phases. Likely worth its own design doc before starting.
**Expected gain:** ~85% → 90%+.

### E6. Raise `--cov-fail-under` to 90 in `pyproject.toml`
**Status:** `blocked` on E5
**Action:** flip the gate once Phase 5 lands. Can be a one-line PR.

---

## Track D — Things explicitly out of scope (do not propose)

Reproduced from `CLAUDE.md` §4 so a fresh agent doesn't have to find
it. Adding any of these requires explicit user consent and probably
a redesign of the EV path:

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
