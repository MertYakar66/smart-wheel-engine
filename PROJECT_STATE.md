# Project State

**Last updated:** 2026-04-29 (against `origin/main` at `82b01fe`).

This file records *temporal* state — what is authoritative now, what is
in progress, what is deprecated. It is the half-life partner of
`CLAUDE.md` (which is the *structural* contract). Update this file
when you finish a meaningful unit of work or discover that something
described here is no longer accurate.

---

## 1. Authoritative — do not bypass

| Module | Public entry | Locked by |
|---|---|---|
| `engine/ev_engine.py` | `EVEngine.evaluate` (line 234) | `tests/test_audit_invariants.py`, `tests/test_audit_viii_*` |
| `engine/wheel_runner.py` | `WheelRunner.rank_candidates_by_ev` (line 445) | `tests/test_authority_hardening.py`, `tests/test_audit_viii_real_data_smoke.py` |
| `engine/candidate_dossier.py` | `EnginePhaseReviewer` (line 113), rules R1–R6 | `tests/test_dossier_invariant.py` |
| `engine_api.py` | HTTP API on `:8787`; 29 endpoints listed in the file header | `tests/test_tv_api.py`, `tests/test_tv_dossier.py` |

These four routes are the only sanctioned paths from raw inputs to a
tradeable verdict. Reviewers (chart provider, news sentiment, advisor
committee, dealer positioning) can downgrade outputs — never upgrade.

## 2. Recent decision-layer audits

Each row links to the commit that shipped the change. Use
`git log --grep "^audit-<N>"` for the full diff and rationale.

| Audit | What shipped | Tests added |
|---|---|---|
| `audit` (`8ca561c`) | PIT bug fixes, TV webhook hardening, institutional EV engine | — |
| `audit-ii` (`3be3f2a`) | EV engine wired into runner; forward distributions; empirical surface; early-assignment-div; survivorship audit; calibration gate; sqrt impact | — |
| `audit-iii` (`81a42b1`) | POT-GPD CVaR; 4-state Gaussian HMM; Nelson-Siegel skew dynamics; Student-t copula; event gate | — |
| `audit-iv` (`2440891`) | TradingView visual-layer bridge + candidate dossier (Mode B) | `test_tv_dossier.py` |
| `audit-v` (`4afe7ea`, `48fe29b`) | Market-level dealer positioning (GEX, walls, flip, regime); P0/P1 unify decision authority; survivorship + chain quality + stress residual gates | `test_dealer_positioning.py` |
| `audit-vi` (`7e1bda7`) | Closed authority leaks across `tv` webhook, analyze, strangle, strikes, wheel_tracker EV gate | `test_authority_hardening.py` |
| `audit-vii` (`506b348`) | Unified orchestrator; HMM regime wiring; Grok/X agent; news API; ML guard | — |
| `audit-viii` (`e4c30e1`) | EV-path unit bugs (IV / risk-free rate percent↔decimal); roll/close P&L double-count; committee authority leak | `test_audit_viii_unit_invariants.py`, `test_audit_viii_e2e.py`, `test_audit_viii_real_data_smoke.py` (20 new tests) |

After audit-VIII the suite reports 1087 passed / 0 failed and 287
deprecation warnings (down from 1067+1 / 578).

## 3. Work in progress

### TradingView MCP integration

- **Design contract:** `docs/TRADINGVIEW_MCP_INTEGRATION.md`
  (commit `c064652`). Pins the seam, the canonical `ChainedChartProvider`
  ordering, the four hard invariants (including no-quiet-substitution
  on MCP failure), and the M1 scope (3 MCP tools).
- **Implementation seam:** `engine/tradingview_bridge.py` — new class
  `MCPChartProvider(ChartContextProvider)` to be added.
- **Contract test:** `tests/test_dossier_invariant.py::test_mcp_provider_*`
  is import-guarded; dormant until `engine.tradingview_bridge.MCPChartProvider`
  exists, then auto-activates.
- **Status:** nothing has shipped on the MCP class itself.

### iv_surface integration decision

- Theta `iv_surface/` coverage is 28/503 tickers (5.6% — mega-caps +
  sector ETFs only). `iv_surface_history/` is absent on the current
  Drive snapshot.
- The SVI tools in `engine/volatility_surface.py`
  (`VolatilitySurfaceBuilder`, `create_empirical_surface`,
  `SVICalibrator`) are exported but **have zero non-test callers as of
  2026-04-25**. Audit pass: no live silent-fallback paths.
- `get_iv_surface()` returns an empty DataFrame on missing data, not a
  flat-IV stub.
- **Open decision:** before wiring SVI surfaces into a feature or the
  decision path, pick a missing-data contract — fail loudly on the
  ~475 uncovered tickers, or use a clearly-named fallback
  (`flat_iv_fallback`, never silent).

### `.claude/` SessionStart hook

- `0e451f6 feat: .claude/settings.json SessionStart hook + dossier invariant test`
- `40d1ec4 fix: SessionStart hook — gate pip install behind venv-or-opt-in`
- The hook prints provider warning, dataset presence, theta manifest
  recency, missing-deps reminder, and connector class. Active and
  tested as of today.

## 4. Deprecated / phantom — do not extend

- `src/` — phantom architecture from an earlier scaffold. Real
  decision-layer code is in `engine/`, not `src/`. The pyproject
  entrypoint `wheel = "src.cli:app"` targets a `src/cli.py` that does
  **not exist**; coverage and `[tool.hatch] packages = ["src"]`
  treat `src/` as load-bearing. Empty subpackages: `src/execution/`,
  `src/models/`, `src/risk/`. Partially populated: `src/data/` (only
  `schemas.py`, `validators.py`), `src/features/` (mirrors what
  `data/features/` consumes). Plan: either fully remove or fully
  repopulate. Until then, do not add new modules under `src/`.
- `models/`, `validation/` — empty placeholder dirs (`.gitkeep` only).
- `dashboard/quant_dashboard.py` — legacy Python CLI dashboard. The
  primary dashboard is the Next.js app under `dashboard/src/`.
  README.md still describes the legacy CLI as the main entry point.
- News-stack duplication — `financial_news/` (34 files,
  RSS/scraping/clustering platform), `news_pipeline/` (29 files,
  browser-agent pipeline driving `morning_run.py`),
  `engine/news_sentiment.py` (downgrade-only reviewer on the EV path),
  `scripts/pull_news_sentiment.py` (one-shot puller). Only
  `engine/news_sentiment.py` feeds the EV path. Verify before adding
  a new news source.

## 5. Documentation drift to repair

These are stale relative to `CLAUDE.md` and the live code, and have
not been fixed in this review pass:

- `README.md` — describes a CLI dashboard
  (`python -m dashboard.quant_dashboard`) and broker env vars
  (`BROKER_API_KEY`, `BROKER_SECRET`) that are out of scope per
  `CLAUDE.md` §4. Project structure listing is 6 dirs; actual is 20+.
- `CONTRIBUTING.md` — `pip install -e ".[dev]"` will install from a
  pyproject still listing `streamlit`, `prefect`, `ib_insync` as hard
  deps; none are part of the EV decision path.
- `docs/ARCHITECTURE.md` — describes `src/data`, `src/features`,
  `src/execution` as the live architecture. Actual quant layer is
  `engine/`. CLAUDE.md §1 supersedes.
- `dashboard/README.md` — still says "FinanceNews — AI Financial News
  Platform". Directory was reused; README was not.
- `pyproject.toml` —
  `[project.scripts] wheel = "src.cli:app"` points at a missing file;
  `[tool.hatch.build.targets.wheel] packages = ["src"]` excludes
  `engine/`, `engine_api.py`, `advisors/`, `dashboard/`, etc., so the
  built wheel would not contain the live code.
- `engine/__init__.py` — re-exports the legacy quant layer
  (option_pricer, monte_carlo, regime_detector, signals, etc.) but
  does **not** re-export `EVEngine`, `WheelRunner`,
  `EnginePhaseReviewer`, `MarketStructure`. Modern decision-layer
  symbols can only be imported via their full submodule paths. This
  is silent — not broken — but misleading to a fresh agent.

## 6. Branch + workflow policy

- Default branch: `main`. Don't edit `main` directly.
- Feature work happens on `claude/<short-slug>` branches. The branch
  that introduced these docs (`claude/handoff-docs`) was a
  documentation-only foundation review.
- CI runs on push to `main` / `develop` and on PRs (`.github/workflows/ci.yml`).
- The `.pre-commit-config.yaml` is committed; install with
  `pre-commit install`.
- `.claude/settings.json` SessionStart hook validates dataset presence,
  Theta manifest recency, and connector class on every fresh session.
