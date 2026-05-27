# File Manifest

Exhaustive per-file index of the Smart Wheel Engine repository, grouped by
directory. One line per file: what it is and why it exists. This is the
companion to `MODULE_INDEX.md` — that file is the curated decision-layer
view (authority / reviewer / multiplier roles); this file is the complete
map.

Conventions:

- Data files (`*.csv`, `*.parquet`, `*.xlsx`) and per-ticker shards are
  described at the directory level, not enumerated.
- Gitignored trees are out of scope: `dashboard/node_modules/`,
  `dashboard/.next/`, `data_processed/theta/**`, non-AAPL feature shards,
  `tradingview/tradingview-mcp-jackson/`, the `Theta/` install.
- Lines describe **purpose only**. Module status, staleness and
  doc-truth observations are out of scope for this file — see
  `PROJECT_STATE.md` and `ROADMAP.md`.

See `DECISIONS.md` D14 for the tiered layout this manifest reflects.

---

## Repository root

| File | Purpose |
|---|---|
| `AGENTS.md` | Tier-1 canonical agent entry doc — the read order any AI agent follows on entering the repo, plus the hard EV invariant. |
| `CLAUDE.md` | Tier-1 entry contract — the four-layer mental model, the hard EV invariant, the NEVER list, the fresh-session bring-up, and the on-demand pointer block. |
| `README.md` | Tier-1 human entry point; routes agents to `AGENTS.md` and the doc set. |
| `PROJECT_STATE.md` | Tier-2 — temporal state: what is authoritative, in progress, or deprecated right now. |
| `MODULE_INDEX.md` | Tier-2 — per-module purpose and decision-layer role classification. |
| `TESTING.md` | Tier-2 — test taxonomy, the launch-blocker subset, the "what to run when you touch X" map. |
| `DECISIONS.md` | Tier-2 — the locked architectural decision log with rationale and rejected alternatives. |
| `COMMIT_GUIDE.md` | Tier-2 — the commit-message and PR format standard. |
| `FILE_MANIFEST.md` | Tier-2 — this file: the exhaustive per-file index. |
| `CHANGELOG.md` | Human-readable history of meaningful changes, grouped by month. |
| `ROADMAP.md` | Scoped-but-not-done work by track; forward companion to `PROJECT_STATE.md`. |
| `LICENSE` | MIT license. |
| `engine_api.py` | Interface-layer entry point — the stdlib HTTP API server on `:8787` serving the Next.js dashboard. |
| `morning_run.py` | Entry point for the browser-driven, zero-API-cost multi-LLM morning news pipeline. |
| `audit.py` | Standalone smoke-test client that hits a running `engine_api.py` and runs domain-grouped backend checks. |
| `conftest.py` | pytest configuration — hypothesis profiles, shared fixtures, custom markers. |
| `pyproject.toml` | Packaging and tooling configuration (ruff, mypy, pytest, coverage). |
| `requirements.txt` | Runtime dependency list. |
| `.pre-commit-config.yaml` | Pre-commit hook config — whitespace hygiene, ruff, ruff-format, bandit, mypy, detect-secrets. |
| `.gitignore` | Ignore rules — secrets, Python artefacts, caches, gitignored data trees, the Theta install, with allow-list exceptions for tracked `.claude/` content. |
| `.gitattributes` | Pins LF line endings and marks binary types; prevents CRLF churn from the Drive mount. |
| `.python-version` | Pins the Python version for version managers. |
| `.env.example` | Template for the gitignored `.env` secrets file; lists the optional API keys. |
| `fetch_data.bat` | Windows double-click launcher — Theta health check then Theta backfill. |
| `pull.bat` | Windows double-click launcher — stash edits, pull `origin/main`, restore stash. |
| `pull_branch.bat` | Windows double-click launcher — stash edits and check out a feature branch (branch name edited in-file). |

## `.claude/`

| File | Purpose |
|---|---|
| `.claude/settings.json` | Claude Code harness config — registers the SessionStart hook. |
| `.claude/hooks/session_start.sh` | SessionStart hook — provider warning, dataset presence, Theta manifest recency, dependency batching, connector smoke. |
| `.claude/commands/launch-blockers.md` | Tier-3 slash command — runs the launch-blocker test subset. |
| `.claude/commands/ev-smoke.md` | Tier-3 slash command — runs the 5-ticker EV-ranker smoke check. |
| `.claude/commands/backtest-regression.md` | Tier-3 slash command — runs the four ledger-backtest reproducers (S27/S32/S34/S35) against the current engine. Long-running (~4–5 h); excluded from per-PR CI. |

## `.github/`

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | CI pipeline — environment validation, lint/type check, test+coverage matrix (excludes `backtest_regression` marker), quant validation, security scan, integration. |
| `.github/workflows/backtest-regression.yml` | Manual-dispatch workflow that runs the backtest-regression suite (S27/S32/S34/S35 reproducers, ~4–5 h). Cron disabled until CSV hydration in CI is solved; today's primary entry point is the `.claude/commands/backtest-regression.md` skill on the dev laptop. |

## `archive/`

Point-in-time and superseded artifacts, retained for history, not maintained. See `archive/README.md`.

| File | Purpose |
|---|---|
| `archive/README.md` | Index of archived files — original path and archival reason for each. |
| `archive/2026-05/OptionsEngine.txt` | Archived narrative end-to-end usage walkthrough (point-in-time, unmaintained). |
| `archive/2026-05/ARCHITECTURE.md` | Archived architecture doc describing a planned `src/`-based layout that does not match the repo. |
| `archive/2026-05/DATA_COLLECTION_REPORT.md` | Archived dated data-collection phase report. |

## `advisors/` — investment committee (advisory-only)

| File | Purpose |
|---|---|
| `advisors/__init__.py` | Committee package re-export hub; the `TalebAdvisor` import is guarded. |
| `advisors/base.py` | `BaseAdvisor` ABC — response-schema enforcement and shared trade-assessment helpers. |
| `advisors/committee.py` | `CommitteeEngine` — runs all advisors, aggregates votes; also portfolio-review and post-mortem modes; `format_committee_report`. |
| `advisors/integration.py` | `EngineIntegration` — converts engine dicts into `AdvisorInput`; `quick_evaluate` helper. |
| `advisors/schema.py` | Advisor dataclasses and enums (`AdvisorInput`, `CandidateTrade`, `CommitteeOutput`, portfolio-review/post-mortem schemas). |
| `advisors/scorecard.py` | `AdvisorScorecard` — tracks advisor prediction accuracy, calibration and P&L. |
| `advisors/buffett.py` | `BuffettAdvisor` — business-quality / margin-of-safety heuristic critic. |
| `advisors/munger.py` | `MungerAdvisor` — inversion and cognitive-bias-detection critic. |
| `advisors/simons.py` | `SimonsAdvisor` — statistical-significance / Kelly-sizing / regime-fit quant critic. |
| `advisors/taleb.py` | `TalebAdvisor` — tail-risk / fragility-score critic. |

## `backtests/` — research backtesting

| File | Purpose |
|---|---|
| `backtests/__init__.py` | Re-exports the simulator and walk-forward classes. |
| `backtests/simulator.py` | `WheelBacktester` — a simplified placeholder backtester (constant-IV approximation). |
| `backtests/walk_forward.py` | Walk-forward validation framework (anchored / rolling / purged k-fold, parameter-stability analysis, out-of-sample tracking). |
| `backtests/.gitkeep` | Directory placeholder. |

## `backtests/regression/` — backtest regression harness

The four reproducers that pin S27/S32/S34/S35 against the current engine. Snapshots, pytest harness, and the on-fail re-baseline workflow land in PR2; see `docs/LAUNCH_READINESS.md` §6 (PR2) and `.claude/commands/backtest-regression.md` (PR2).

| File | Purpose |
|---|---|
| `backtests/regression/__init__.py` | Package marker for the regression sub-package. |
| `backtests/regression/universes.py` | `UNIVERSE_24` (S17-minus-WMT, the S22/S27/S32/S35 set) and `UNIVERSE_100` (first 100 from `MarketDataConnector.get_universe()`). Static tuples; no import-time disk access. |
| `backtests/regression/_common.py` | Shared driver: `run_backtest()` loops trading days through `WheelRunner.rank_candidates_by_ev` + `WheelTracker`, forward-replays realized P&L on every ranked row, returns a metrics dict. Plus friction overlay (`friction_adjusted_premium`, `friction_open_cost`, `friction_assignment_cost`), `assert_data_window_available`, `ohlcv_sha256`, JSON snapshot I/O. |
| `backtests/regression/s27_ivpit_24t_100k.py` | S27 reproducer — $100k / 24 tickers / 2022-2024 / frictionless. Mirrors `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`. |
| `backtests/regression/s32_friction_24t_1m.py` | S32 reproducer — $1M / 24 tickers / 2022-2024 / three friction levels. Mirrors `docs/ENGINE_BACKTEST_S32_FRICTION.md`. |
| `backtests/regression/s34_universe_100t_1m.py` | S34 reproducer — $1M / 100 tickers / 2022-2024 / three friction levels / `top_n=15`. Mirrors the S34 section of `docs/SOUNDNESS_REVIEW_2026-05-26.md`. |
| `backtests/regression/s35_oos_24t_100k.py` | S35 reproducer — $100k / 24 tickers / 2018-2020 OOS / three friction levels. Mirrors `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md`. PR4 re-baselines against the post-PIT-fix engine. |
| `backtests/regression/snapshots/s27_ivpit_24t_100k.json` | Locked S27 snapshot — aggregate / per-year / per-quartile metrics from current post-PIT-fix engine. Generated via `--update-snapshot` against `data_csv_sha256` recorded in the fingerprint. Re-baseline workflow in `TESTING.md`. |
| `backtests/regression/snapshots/s32_friction_24t_1m.json` | Locked S32 snapshot — $1M / 24 tickers / 2022-2024 / three friction levels (none / bid_ask / full). Generated via the shared-rank `run_backtest_multi_friction` driver. |
| `backtests/regression/snapshots/s34_universe_100t_1m.json` | Locked S34 snapshot — $1M / 100 tickers / 2022-2024 / three friction levels. Universe-expansion backtest from `docs/SOUNDNESS_REVIEW_2026-05-26.md`. ρ ≈ 0.329 matches doc within 0.002. |
| `backtests/regression/snapshots/s35_oos_24t_100k.json` | Locked S35 snapshot — $100k / 24 tickers / 2018-2020 OOS / three friction levels. PR4 re-baseline. ρ ≈ 0.500 matches doc within 0.003 (driver-invariant signal); execution count doubled (19 → 40). |

## `config/`

| File | Purpose |
|---|---|
| `config/__init__.py` | Re-exports `Config`, `ConfigManager`, the sub-config dataclasses and preset factories. |
| `config/settings.py` | Centralized config — strategy / risk / execution / data / regime / edge / backtest dataclasses with YAML/JSON/env loading. |

## `dashboard/` — Next.js dashboard + legacy Python CLI

| File | Purpose |
|---|---|
| `dashboard/package.json` | npm manifest — Next.js, React, Drizzle/SQLite, Vercel AI SDK, recharts, rss-parser. |
| `dashboard/package-lock.json` | npm lockfile (committed dependency tree). |
| `dashboard/next.config.ts` | Next.js config; marks `better-sqlite3` server-external. |
| `dashboard/tsconfig.json` | TypeScript config; `@/*` path alias. |
| `dashboard/eslint.config.mjs` | Flat ESLint config extending `eslint-config-next`. |
| `dashboard/postcss.config.mjs` | PostCSS config wiring the Tailwind plugin. |
| `dashboard/components.json` | shadcn/ui generator config. |
| `dashboard/drizzle.config.ts` | Drizzle-kit config — SQLite dialect, schema and db paths. |
| `dashboard/.env.example` | Sample dashboard env vars (Finnhub, FRED, Ollama, Valyu, Daytona). |
| `dashboard/.gitignore` | Dashboard-scoped ignore rules. |
| `dashboard/README.md` | Dashboard project README. |
| `dashboard/__init__.py` | Python package init — re-exports the legacy `QuantDashboard` and helpers. |
| `dashboard/quant_dashboard.py` | Legacy standalone Python CLI dashboard — option pricing, Greeks, VaR, stress, sizing (imports only `engine/`). |
| `dashboard/web_vitals.py` | `WebVitalsTracker` — records Core Web Vitals and checks per-page budgets. |
| `dashboard/public/*.svg` | Default Next.js scaffold icon assets. |
| `dashboard/src/app/favicon.ico` | Browser favicon. |
| `dashboard/src/app/layout.tsx` | Root Next.js layout — global metadata and CSS. |
| `dashboard/src/app/page.tsx` | Root index — redirects to `/top`. |
| `dashboard/src/app/globals.css` | Global Tailwind styles and terminal color palette. |
| `dashboard/src/app/(main)/layout.tsx` | Layout for the standard web app — nav plus centered container. |
| `dashboard/src/app/(main)/top/page.tsx` | "TOP" command-center page — breaking strip, top stories, category sections. |
| `dashboard/src/app/(main)/feed/page.tsx` | News feed page — story cards with sector filter and RSS refresh. |
| `dashboard/src/app/(main)/calendar/page.tsx` | Macro calendar page. |
| `dashboard/src/app/(main)/research/page.tsx` | AI research chat page streaming from `/api/chat`. |
| `dashboard/src/app/(main)/story/[id]/page.tsx` | Story detail page — sources, timeline, exposure mechanisms. |
| `dashboard/src/app/(main)/ticker/[symbol]/page.tsx` | Per-ticker page — quote, price chart, related news. |
| `dashboard/src/app/(main)/watchlist/page.tsx` | Watchlist page — add/remove tickers, prices, alerts. |
| `dashboard/src/app/(terminal)/layout.tsx` | Layout for the terminal route. |
| `dashboard/src/app/(terminal)/terminal/page.tsx` | Bloomberg-style terminal dashboard — 6-panel grid, command line, engine data. |
| `dashboard/src/app/api/engine/route.ts` | Server-side proxy bridge to the Python engine API on `:8787`. |
| `dashboard/src/app/api/stories/route.ts` | Stories list API — query by sector/ticker, exposure-ranked. |
| `dashboard/src/app/api/stories/[id]/route.ts` | Single-story detail API. |
| `dashboard/src/app/api/stories/[id]/impact/route.ts` | Story impact API — quotes plus event-study templates. |
| `dashboard/src/app/api/chat/route.ts` | Chat API — streams from Ollama via the AI SDK. |
| `dashboard/src/app/api/market/route.ts` | Market quote API — Finnhub with cached fallback. |
| `dashboard/src/app/api/ingest/route.ts` | POST trigger for the RSS ingestion pipeline. |
| `dashboard/src/app/api/watchlist/route.ts` | Watchlist CRUD API — GET enriches with prices. |
| `dashboard/src/app/api/alerts/route.ts` | Alerts API — list/dismiss. |
| `dashboard/src/app/api/events/route.ts` | Calendar-events CRUD API. |
| `dashboard/src/app/api/categories/route.ts` | News-categories CRUD API. |
| `dashboard/src/app/api/briefings/route.ts` | Briefings API — generate/get morning/evening digests. |
| `dashboard/src/app/api/execute/route.ts` | Code-execution API — Daytona or template executor. |
| `dashboard/src/app/api/exposure/route.ts` | User-exposure CRUD API. |
| `dashboard/src/app/api/schedule/route.ts` | Ingestion-schedule API — status/history/trigger. |
| `dashboard/src/app/api/stream/route.ts` | Server-Sent-Events endpoint pushing new headlines. |
| `dashboard/src/components/nav.tsx` | Top navigation bar for the web app. |
| `dashboard/src/components/ui/*.tsx` | shadcn/ui base primitives (badge, button, card, input, scroll-area, skeleton). |
| `dashboard/src/components/terminal/*.tsx` | Terminal-app panels and controls (panel, status-bar, market/options/chart/agent/news/watchlist/macro/chat panels, command-line). |
| `dashboard/src/db/index.ts` | SQLite/Drizzle connection — lazy init, table creation, idempotent migrations. |
| `dashboard/src/db/schema.ts` | Drizzle ORM schema — the news/story tables. |
| `dashboard/src/hooks/useEngineData.ts` | React hooks against `/api/engine` — engine data, ticker analysis, committee review. |
| `dashboard/src/lib/utils.ts` | `cn()` Tailwind class-merge helper. |
| `dashboard/src/types/index.ts` | Shared TypeScript types for the dashboard. |
| `dashboard/src/services/briefing-generator.ts` | Generates/persists morning/evening/breaking briefings. |
| `dashboard/src/services/code-execution.ts` | Code-execution abstraction — Daytona plus local template executor. |
| `dashboard/src/services/data-provider.ts` | `FinanceDataProvider` singleton aggregating RSS/Finnhub/EDGAR/FRED. |
| `dashboard/src/services/edgar.ts` | SEC EDGAR client — ticker-to-CIK and recent filings. |
| `dashboard/src/services/entity-extraction.ts` | Entity extraction — Ollama NLP with regex fallback. |
| `dashboard/src/services/exposure-ranking.ts` | Exposure-first story ranking against holdings/watchlist/factors. |
| `dashboard/src/services/impact-analysis.ts` | Impact analysis — factor/horizon/sentiment tagging. |
| `dashboard/src/services/macro-data.ts` | FRED API client for macro time series. |
| `dashboard/src/services/market-data.ts` | Finnhub quote client and market-snapshot cache. |
| `dashboard/src/services/news-categories.ts` | News-category taxonomy and keyword/ticker matching. |
| `dashboard/src/services/rss-feeds.ts` | Static config of financial RSS feed sources. |
| `dashboard/src/services/rss-ingestion.ts` | RSS feed parser/ingester with ticker extraction. |
| `dashboard/src/services/scheduled-ingestion.ts` | Orchestrates the multi-step ingestion pipeline. |
| `dashboard/src/services/story-clustering.ts` | Story-graph clustering — Jaccard dedup, contradiction detection. |
| `dashboard/src/services/retrieval/index.ts` | `RetrievalOrchestrator` — multi-provider search aggregation. |
| `dashboard/src/services/retrieval/rss-provider.ts` | RSS-backed retrieval provider. |
| `dashboard/src/services/retrieval/valyu-provider.ts` | Valyu-backed retrieval provider. |

## `data/` — data layer (Bloomberg-CSV provider + feature pipeline)

| File | Purpose |
|---|---|
| `data/__init__.py` | Package re-export hub for the data layer. |
| `data/bloomberg.py` | `BloombergConnector` — live Bloomberg Terminal connector via `blpapi` or Excel COM. |
| `data/bloomberg_import.py` | One-shot Bloomberg OHLCV CSV loader plus per-ticker feature computation. |
| `data/bloomberg_loader.py` | Per-ticker Bloomberg CSV parsers with column normalization. |
| `data/consolidated_loader.py` | `ConsolidatedBloombergLoader` — loads the consolidated `sp500_*.csv` panels. |
| `data/feature_pipeline.py` | `FeaturePipeline` — wires the `src/features/` modules into a layered compute DAG. |
| `data/feature_provenance.py` | Feature lineage / lag-audit registry (`ProvenanceRegistry`, `FeatureProvenance`). |
| `data/feature_store.py` | `FeatureStore` — Parquet-backed feature persistence with atomic writes, locking, TTL cache. |
| `data/observability.py` | Structured logging, metrics collection, tracing and alerting for the data pipeline. |
| `data/orchestrator.py` | `PipelineOrchestrator` — DAG executor with retry and checkpoint/resume. |
| `data/pipeline.py` | `DataPipeline` — master data interface; auto-detects CSV format; survivorship-bias audit. |
| `data/quality.py` | `DataQualityFramework` — schema / completeness / consistency / anomaly / freshness validation; the chain-quality gate on the EV path. |
| `data/bloomberg/EXTRACTION_GUIDE.md` | Runbook of Bloomberg Excel formulas to regenerate the CSV panels. |
| `data/bloomberg/*.csv` | Consolidated wide-format Bloomberg data panels — OHLCV, IV/vol, earnings, dividends, fundamentals, credit, analyst, institutional, macro, VIX, short interest, index membership, sector ETFs, treasury yields, VIX term structure. |
| `data/bloomberg/sp500_short_interest.csv.xlsx` | A short-interest export carrying a double `.csv.xlsx` extension (an `.xlsx` file; not loadable by the CSV connector as-named). |
| `data/features/<group>/ticker=AAPL/{data.parquet,metadata.json,stats.json}` | Committed AAPL-only feature-store sample shards across the 8 feature groups; other tickers regenerate via `scripts/backfill_features.py`. |
| `data/features/_lineage/`, `data/features/_registry/` | Feature-store lineage table and registry index. |

## `data_processed/`

Mostly gitignored regenerable Theta/yfinance pulls. Tracked content:

| File | Purpose |
|---|---|
| `data_processed/trade_universe/2025-11-22_trade_universe.csv` | A labeled per-contract ML training dataset snapshot. |
| `data_processed/.gitkeep` | Directory placeholder. |

## `data_raw/`

| File | Purpose |
|---|---|
| `data_raw/sp500_constituents_current.csv` | The canonical S&P 500 constituent universe list. |
| `data_raw/ohlcv/*.csv` | Sample per-ticker yfinance OHLCV fixtures (5 tickers). |
| `data_raw/yfinance/options/*.csv` | Sample dated yfinance option-chain fixtures (5 tickers). |
| `data_raw/.gitkeep` | Directory placeholder. |

## `docs/` — documentation set

| File | Purpose |
|---|---|
| `docs/DATA_POLICY.md` | Data tiers, provider matrix, what never enters git, point-in-time discipline, refresh procedures. |
| `docs/DATA_SPECIFICATION.md` | Data architecture and schemas. |
| `docs/LAPTOP_SETUP.md` | Machine bring-up — cloning, env, Theta Terminal, regenerating local data. |
| `docs/LAUNCH_READINESS.md` | The launch-blocker gate checklist consolidating the EV invariant, the four authoritative routes, and the dossier rules. |
| `docs/LAUNCH_READINESS_ANALYSIS_2026-05-26.md` | Comprehensive launch-readiness analysis pulling every backtest (S22/S27/S32/S34/S35) + every review PR + every shipped fix into one yes/no real-life-trading verdict. Headline finding: signal is real and scale-AND-window-invariant (ρ = 0.22-0.50 across capital scales and time windows) but **dollar-alpha is window-dependent** (engine +27pp vs SPY in 2022-2024, −41pp in 2018-2020 — same parameters). Verdict: NO autonomous; conditional supervised at ≤ $100k with window-sensitivity caveat. |
| `docs/PRODUCTION_READINESS.md` | The real-money deployment gate. Consolidates findings from the S22 / S27 / S32 backtests + the four review docs (#194 / #195 / #197 + S32) into one answer to "should we deploy this engine against a real brokerage account?" Names three blockers (F4 tail-risk widening, D17 live-wire to `engine_api.py`, strategy capacity at >$100k), four caveats, and a deployment decision matrix. Complementary to `LAUNCH_READINESS.md` (code-quality merge gates). |
| `docs/THETA_INSTRUCTIONS.md` | Quick reference for refreshing every Theta-sourced dataset. |
| `docs/THETA_USAGE.md` | Theta Terminal v3 per-endpoint reference, tier behaviour, wire-format codes. |
| `docs/THETA_PULL_SESSION_NOTES.md` | Operational checklist and gotchas for a laptop Theta pull. |
| `docs/TRADINGVIEW_INTEGRATION.md` | Parent guide for the two TradingView roles — engine bridge and analyst workspace. |
| `docs/TRADINGVIEW_MCP_INTEGRATION.md` | Design contract for the MCP-driven chart provider. |
| `docs/GREEKS_UNIT_CONTRACT.md` | Canonical Greeks unit conventions. |
| `docs/GOVERNANCE.md` | Model governance framework. |
| `docs/MODEL_CARDS.md` | Per-model documentation cards. |
| `docs/USAGE_TEST_LEDGER.md` | Record of end-to-end usage tests — purpose, setup, findings, follow-up PRs. |
| `docs/PARALLEL_SESSIONS.md` | How the repo is worked by N parallel Claude Code terminals — roles, lanes, coordination board. |
| `docs/SESSION_HANDOFF.md` | A point-in-time snapshot of in-flight work for a session handoff. |
| `docs/TERMINAL_A_AUDIT.md` | Independent engineering audit of Terminal A's coordinated PR run on board #113 — per-PR seven-step protocol, cross-cutting observations, audit-history table. Append-only on re-audit. |
| `docs/AUDIT_OF_AUDIT_REVIEW.md` | Meta-verification of `docs/TERMINAL_A_AUDIT.md` (PR #173) — seven meta-checks M1–M7 on the audit's headline tally, D17 promotion logic, second-layer drift discipline, the two named line-drifts, test counts, exclusion-list defensibility, and the no-production-callers observation. |
| `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md` | S27 follow-up to PR #178's `ENGINE_BACKTEST_2022_2024.md`: re-runs the same 2022-2024 backtest against the post-fix engine (`claude/fix-ranker-iv-pit-aware` `d26a8d6`). Side-by-side ρ / quartile / per-year / tail-episode comparison. Verdict: signal preserved at ρ=0.22 (halved); 2022 bear actually stronger; F4 tail-risk gap confirmed real. |
| `docs/ENGINE_BACKTEST_S32_FRICTION.md` | S32 — $1M friction-modeled simulation closing S22 Caveat 3. Same window / universe / engine as S27 but 10× capital with three-layer friction overlay (bid/ask + commission + assignment slippage). Three parallel WheelTracker instances per friction level. Headline: friction drag is 0.27% NAV (much smaller than S22's "2-5% per leg" worst case); but capital deployment averages 10.8% at $1M, so engine returns +1.85% vs SPY +24% — the +27pp-over-SPY narrative inverts at scale. |
| `docs/ENGINE_BACKTEST_S34_UNIVERSE.md` | S34 — Universe expansion to 100 SP500 tickers at $1M (closes PRODUCTION_READINESS Blocker B3). Tests S32 F3's hypothesis: expand universe 24→100, leave everything else identical. Result: engine +35.61% NAV (full friction) vs SPY ~+24% = **+11.6pp OVER SPY** (vs S32's −22pp UNDER SPY). 34pp swing on universe size alone. ρ = 0.33 (higher than S32's 0.19); 22.1% capital deployment (2× S32); zero BP rejections. Universe expansion materially closes the capacity gap; multi-contract / strategy-stack remain candidates for further deployment but are not required. |
| `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` | S35 — 2018-2020 out-of-window cross-validation against S22 / S27. Same 24-ticker universe and $100k capital; only the time window changes. Headline: signal generalizes (ρ = 0.50 in 2020 — double S27's 0.22) but dollar-alpha does NOT (engine +3.57% vs SPY ~+45% = −41pp underperformance). The "+27pp over SPY" property turns out to be both $100k-specific (per S32) AND 2022-2024-window-specific. Plus the discovery of a 504-day OHLCV history gate. |
| `docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md` | S38 — 5-year multi-window backtest at $1M / 100 tickers / 2020-01-02 → 2024-12-31. Closes the gap S34 ($1M/100t/2022-2024 = +11.6pp) and S35 ($100k/24t/2018-2020 = −41pp) opened. Headline: engine returns +33.18% (full friction) vs SPY ~+85% = **−52pp UNDERPERFORMANCE** over the 5-year window. Confirms dollar-alpha is window-specific. Realized executed P&L is **negative** (−$28,647); all NAV growth is equity-beta on assigned positions (108.6%). Spearman ρ = 0.358 (signal generalizes; never negative per year). COVID refusal rate 97.8% (defensive behavior preserved at scale). |
| `docs/RELIABILITY_ARC_REVIEW.md` | Independent verification of the reliability arc (S18 load/scale, S19 failure-mode chaos, S20 engine_api concurrency) — C7b +inf mechanism, request_queue_size ceiling, v5 race-vector mechanisms, IV-PIT cross-arc coherency. |
| `docs/END_TO_END_REVIEW_2026_05_25.md` | Four-pass end-to-end product review against `origin/main` @ `e83eaca`: §2 surface enumeration, recent-commit skepticism, layering/dormant-surface census, contract-vs-implementation per route. Tally: §2 BREACH 0 / CONCERN 5 / WITH-NOTE 10 / SOLID 12. Headline: no breach; four already-known follow-ups remain open (+inf R1/R5 bypass, ranker→tracker auto-wire, F4 tail-risk gap, dossier/webhook verdict-path divergence); one new structural note (heuristic backtester in deprecated `src/`). |
| `docs/PREDICTIVE_VALIDITY_REVIEW.md` | Independent meta-verification of the S22 + S27 backtests (PRs #178 + #184) — nine claim-checks P1–P9 (PIT discipline, Spearman recomputation, realized P&L spot-check, IV-PIT bug live repro, COST April 2022 tail-risk gap, BP-saturation accident, §2 on test methodology, qualitative 10+10 trade sanity, caveat honesty). Answers the user's question *does the engine produce realistic output?* — verdict: yes, with two structural caveats (F4 tail-risk, F5 BP-fragility). |
| `docs/F4_TAIL_RISK_DIAGNOSTIC.md` | Mechanical characterisation of why `prob_profit = 0.8333` stayed constant across COST's 31.5% drop in April-May 2022 (the F4 finding from S22 / S27 / S32). Verified live: at the engine's default `lookback_years=5.0` with `non_overlapping=True` at 35-day horizon, only ~30 samples populate the empirical forward distribution; advancing `as_of` by 14 days does not add new sample points. Proposes Fix A + B1 (shorter lookback + regime-conditioned distribution widening) with a definition-of-done checklist for closing `PRODUCTION_READINESS.md` §3 Blocker B1. |
| `docs/SOUNDNESS_REVIEW_2026-05-26.md` | Second-pass critical re-verification of the session's S22/S27/S32/S34/S35 conclusions. Verifies §2 invariant, P&L formula consistency across 532 executed rows, Spearman ρ statistical significance, refusal behavior. Surfaces two new findings the earlier docs missed: (1) ~92% of S34's NAV gain is equity beta on assigned stocks not put-selection alpha, (2) BKNG concentration drives 110% of S34's executed realized P&L (ρ is robust to removing it). Engine is mechanically sound; framing needs the equity-beta-vs-ranking-alpha clarification. |
| `docs/SESSION_REPORT_2026-05-26.md` | Machine-readable session ledger for the deployment-readiness verification campaign (2026-05-26). Captures user intent, what was tested, how it was tested, results across §2 verification + B1/B2/B3 blocker remediation + S38 multi-window stress + engine-API hardening, and the prioritized next-agent backlog. Companion to PRODUCTION_READINESS / LAUNCH_READINESS_ANALYSIS / SOUNDNESS_REVIEW. 10 PRs shipped; B2 closed; B3 structurally closed; B1 negative-result documented. |
| `docs/ENGINE_REALISM_VERIFICATION_2026-05-26.md` | Live realism + reliability battery against `origin/main` @ 9f0afaf. 6 observable tests: §2 launch-blocker subset (93/93), 5-ticker smoke, IV PIT match vs Bloomberg file (all 5 within 0.015% rel diff), EV magnitude regime-multiplier dominance (Spearman corr(iv, ev_dollars)=0 → regime feature working as designed), F4 reproducibility (prob_profit drifted from documented 0.8333 to 0.9032 post-PIT-fix; gap still real and worse), refusal behaviour at 3 anchor dates (COVID earnings 100% refusal, mid-COVID 0% with mixed-sign EV, normal 60% refusal). Engine is reliable and realistic on verified surfaces. F4 doc needs prob_profit refresh. |
| `docs/verification_artifacts/README.md` | Conventions + index for the verification-artifact directory. Captures the doc → driver → raw-output relationship so future agents can re-run any verification and diff against the historically-captured output. |
| `docs/verification_artifacts/realism_verify_driver.py` | Driver for ENGINE_REALISM_VERIFICATION_2026-05-26: 5-ticker smoke + IV PIT match vs Bloomberg file + EV magnitude + F4 reproducibility + 3-anchor refusal check. Read-only client of the production ranker. Re-runnable from any worktree (edit `WORKTREE` constant). |
| `docs/verification_artifacts/realism_2026-05-26_raw_output.txt` | Captured stdout from `realism_verify_driver.py` against `origin/main` @ 9f0afaf. Companion observable for `docs/ENGINE_REALISM_VERIFICATION_2026-05-26.md`. |
| `docs/verification_artifacts/f4_baseline_driver.py` | Multi-case F4 pre-fix baseline driver (COST 2022-04-04, UNH 2024-11-11, AAPL 2026-02-13 control). Captures `prob_profit` / `cvar_5` / `heavy_tail` + recomputed realised 35-day forward returns. Companion to Terminal A's incoming `claude/fix-f4-regime-conditioned-widening` branch — diff the post-fix output against the captured baseline to validate the structural fix. |
| `docs/verification_artifacts/f4_baseline_2026-05-26_raw_output.txt` | Pre-fix baseline output. Headline: COST 2022-04-04 reproduces `prob_profit=0.8333` exactly (F4 doc value is NOT stale; corrects the earlier realism-doc "drift" claim which was a date mismatch). UNH 2024-11-11 the cleanest F4 reproducer: `prob_profit=0.8571` vs realised −20.27%. `heavy_tail=False` on both F4 cases — POT-GPD not firing on realised-tail dates. |
| `docs/ENGINE_SUBSYSTEM_AUDIT.md` | Structural read-through audit of 46 `engine/` files and 10 `advisors/` files, complementary to S31 / S33 / S36 / S37's verification axes. Surveys danger patterns (silent excepts, ignored kwargs, fabricated data, EV bypasses) and confirms the post-soundness-campaign state is structurally clean. No new bugs surfaced; 3 small observability nudges queued (log `wheel_runner.py` bare excepts, log GEX skip rate in `dealer_positioning`, replace `strangle_timing` legacy bare excepts). The dossier R1-R6 contract noted as the most carefully-engineered single file. |
| `docs/BACKTEST_REGRESSION_CAMPAIGN.md` | Campaign report for the backtest regression harness (4-PR series: scaffolding → snapshots → CI split → S35 re-baseline). Architecture, methodology, snapshot-vs-doc comparison tables for S27/S32/S34/S35, on-fail re-baseline workflow, compute profile, and known limitations. Reference doc for other agents picking up or auditing this work. |
| `docs/bloomberg_refresh_runbook.md` | Point-in-time runbook for refreshing the Bloomberg connector CSVs. |
| `docs/data_inventory_2026-05-17.md` | Point-in-time data-inventory analysis report. |
| `docs/optionsengine_audit_2026-05-17.md` | Point-in-time accuracy audit of the archived `OptionsEngine.txt` walkthrough. |
| `docs/CONTRIBUTING.md` | Contributor workflow guide. |
| `docs/SECURITY.md` | Security policy and best practices. |
| `docs/Claude_Prompting_Master_Guide.md` | General Claude prompt-engineering reference (not project-specific). |

## `engine/` — quant + decision layer

| File | Purpose |
|---|---|
| `engine/__init__.py` | Package init re-exporting the legacy quant-layer symbols (pricing, risk, regime, signals, Monte Carlo, portfolio). |
| `engine/ev_engine.py` | `EVEngine.evaluate` — the authoritative probabilistic expected-value computation for short-option trades. |
| `engine/wheel_runner.py` | `WheelRunner` — the orchestrator and authoritative ranker (`rank_candidates_by_ev`, covered-call/strangle rankers, `select_book`, dossier builder); provider selection. |
| `engine/candidate_dossier.py` | The EV-plus-chart `CandidateDossier` artifact and `EnginePhaseReviewer` (the downgrade-only R1–R6 rules). |
| `engine/chart_context.py` | `ChartContext` dataclass and the `ChartContextProvider` protocol. |
| `engine/tradingview_bridge.py` | Pluggable TradingView chart-capture providers (filesystem, Playwright, MCP, chained) and the default-provider factory. |
| `engine/mcp_client.py` | `MCPCLIClient` — the `tv`-CLI subprocess client backing the MCP chart provider. |
| `engine/tv_signals.py` | Deterministic TradingView Pine-parity signal computation and `TVAlert` webhook parsing. |
| `engine/signal_context.py` | Builds the context dicts the signal framework consumes from the Bloomberg data loaders. |
| `engine/signals.py` | Signal-generation framework — IV-rank / trend / profit-target / stop-loss / DTE / event signals and the aggregator. |
| `engine/news_sentiment.py` | `NewsSentimentReader` — reads news-pipeline sentiment from disk and maps it to a clamped EV multiplier. |
| `engine/event_calendar.py` | Earnings / dividend / FOMC / CPI / NFP / GDP / expiry calendar and a JSON-backed ingestion manager. |
| `engine/event_gate.py` | `EventGate` — the hard pre-EV lockout for candidates whose holding window touches a scheduled event. |
| `engine/forward_distribution.py` | PIT-safe forward-return distribution builder (empirical → block bootstrap → HAR-RV cascade). |
| `engine/transaction_costs.py` | Transaction-cost model — commissions, slippage, assignment fees, sqrt market impact, Reg-T margin. |
| `engine/tail_risk.py` | Peaks-over-Threshold Generalised-Pareto extreme-value tail estimation. |
| `engine/portfolio_copula.py` | Gaussian and Student-t copula joint-portfolio simulation for tail-aware CVaR. |
| `engine/regime_detector.py` | Rule-based market regime classifier (volatility / trend / term structure). |
| `engine/regime_hmm.py` | Pure-numpy 4-state Gaussian HMM regime detector and position-size multiplier. |
| `engine/dealer_positioning.py` | `DealerPositioningAnalyzer` — GEX/DEX/walls/gamma-flip/regime; the clamped dealer EV multiplier. |
| `engine/skew_dynamics.py` | Nelson-Siegel IV term-structure fitting and skew-slope/momentum signals. |
| `engine/realized_vol.py` | OHLC realised-volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) and the vol-risk-premium bundle. |
| `engine/earnings_drift.py` | `EarningsDriftAnalyzer` — post-earnings drift and move distributions per ticker and sector. |
| `engine/strangle_timing.py` | Volatility-lifecycle timing engine for short-strangle entry scoring and phase classification. |
| `engine/option_pricer.py` | Black-Scholes-Merton pricing, full first/second/third-order Greeks, an IV solver, BAW American approximation. |
| `engine/binomial_tree.py` | Cox-Ross-Rubinstein binomial-lattice pricer for American options with discrete dividends. |
| `engine/monte_carlo.py` | Block bootstrap, Merton jump-diffusion, and Longstaff-Schwartz American pricing engines. |
| `engine/volatility_surface.py` | SVI volatility-surface parameterization, calibration and construction tools. |
| `engine/model_validation.py` | `CrossModelValidator` — cross-model governance comparing pricing models with acceptance gates. |
| `engine/shared_valuation.py` | Shared trade-simulation logic used by both the backtester and the label generator. |
| `engine/risk_manager.py` | Position sizing (Kelly), portfolio Greeks, VaR family, sector exposure, hierarchical risk parity. |
| `engine/stress_testing.py` | Stress-testing — historical/hypothetical scenarios, Greeks sensitivity ladders, scenario reports. |
| `engine/wheel_tracker.py` | `WheelTracker` — wheel position lifecycle (short put → assignment → covered call → exit) with EV-scored roll suggestions. |
| `engine/portfolio_risk_gates.py` | D17 / #154 C4 — pure-function library of portfolio-risk gates (sector cap, portfolio delta, Kelly size, VaR, stress scenario, dealer regime) shared by the tracker hard-blocks (Phase 2) and the dossier soft-warns R7+R8 (Phase 3). Wires the existing `risk_manager.py` + `stress_testing.py` + `dealer_positioning.py` machinery that S15 found unimported by the decision-layer trio. |
| `engine/portfolio_tracker.py` | `PortfolioTracker` — portfolio bookkeeping: tax lots, time-weighted returns, allocation, dividends. |
| `engine/portfolio_intelligence.py` | Congressional and institutional (13F) trading trackers cross-referenced against a watchlist. |
| `engine/performance_metrics.py` | Backtest performance reports — return, Sharpe/Sortino, drawdown, profit factor. |
| `engine/payoff_engine.py` | Payoff-diagram grids, IV expected-move bands, BSM-delta strike recommendations. |
| `engine/data_connector.py` | `MarketDataConnector` — the Bloomberg-CSV data provider (default). |
| `engine/theta_connector.py` | `ThetaConnector` — the live ThetaData v3 connector with strict per-endpoint failure semantics. |
| `engine/data_integration.py` | Loads Bloomberg earnings/dividend/treasury CSVs into calendar objects; resolves the risk-free rate. |
| `engine/contracts.py` | Protocol/contract definitions and validators for the pricer/risk/stress interfaces. |
| `engine/policy_config.py` | `TradingPolicyConfig` — centralized runtime policy knobs with JSON load/save. |
| `engine/observability.py` | Audit-trail tooling — trace context, decision journal, JSON audit logger. |
| `engine/dependency_check.py` | Environment-parity gate — checks installed packages with a require-dependencies decorator. |
| `engine/trade_memo.py` | `MemoGenerator` — institutional trade memos combining engine analysis, the committee, and a local Ollama model. |
| `engine/external_data/__init__.py` | Subpackage init re-exporting the four free-data adapters. |
| `engine/external_data/fred_adapter.py` | `FREDAdapter` — FRED economic series and a derived credit-stress regime. |
| `engine/external_data/cboe_adapter.py` | `CBOEAdapter` — VIX-family / SKEW / MOVE index closes from free endpoints. |
| `engine/external_data/edgar_adapter.py` | `EDGARAdapter` — SEC EDGAR Form 4 / 13F / short-interest data. |
| `engine/external_data/yfinance_adapter.py` | `YFinanceAdapter` — cross-asset (DXY, oil, gold, sector ETF) data. |
| `engine/.gitkeep` | Directory placeholder. |

## `financial_news/` — standalone news platform (off the EV path)

| File | Purpose |
|---|---|
| `financial_news/__init__.py` | Package root for the macro/SP500 event-intelligence platform. |
| `financial_news/models.py` | Legacy "Bloomberg-style" data models — a parallel older model layer. |
| `financial_news/schema.py` | Canonical v2 dataclass schema and enums plus default sources/categories/rules. |
| `financial_news/pipeline.py` | Legacy async orchestrator (`NewsPipeline`, `PipelineScheduler`) with its own CLI. |
| `financial_news/scheduler.py` | Event-aware scheduler running AM/PM batches off the macro calendar. |
| `financial_news/verification_engine.py` | SQLite-backed candidate-verification workflow. |
| `financial_news/calendar/__init__.py` | Re-exports the macro calendar. |
| `financial_news/calendar/macro_calendar.py` | Hardcoded release schedules and the `MacroCalendar` lookup class. |
| `financial_news/connectors/__init__.py` | Re-exports the official-source connectors. |
| `financial_news/connectors/base.py` | `BaseConnector` — async HTTP base with rate limiting and retry. |
| `financial_news/connectors/discovery.py` | Tier-3 RSS headline discovery plus a corroboration engine. |
| `financial_news/connectors/eia.py` | `EIAConnector` — EIA petroleum-status and energy-news fetcher. |
| `financial_news/connectors/fed.py` | `FedConnector` — Federal Reserve press-release / monetary-policy RSS fetcher. |
| `financial_news/connectors/sec_edgar.py` | `SECEdgarConnector` — SEC EDGAR filings via the JSON/Atom API. |
| `financial_news/processing/__init__.py` | Re-exports the new and legacy processing components. |
| `financial_news/processing/brief_generator.py` | `BriefGenerator` — tiered AM/PM brief and story-summary generation. |
| `financial_news/processing/classifier.py` | `ArticleClassifier` — deterministic rule-based category classification. |
| `financial_news/processing/clusterer.py` | `StoryClustering` — clusters articles into stories. |
| `financial_news/processing/entity_extractor.py` | Legacy regex/rule entity, ticker and topic extraction. |
| `financial_news/processing/impact_scorer.py` | `ImpactScorer` — a market-impact score from source diversity and severity. |
| `financial_news/processing/ranker.py` | `StoryRanker` — weighted macro/SP500 relevance ranking. |
| `financial_news/processing/story_clusterer.py` | Legacy `StoryClusterer` clustering implementation. |
| `financial_news/sources/__init__.py` | Re-exports the legacy source fetchers. |
| `financial_news/sources/base.py` | Legacy `BaseSourceFetcher` ABC. |
| `financial_news/sources/gdelt.py` | `GDELTFetcher` — GDELT DOC API news discovery. |
| `financial_news/sources/rss_feeds.py` | `RSSFetcher` — official central-bank/government RSS fetcher. |
| `financial_news/sources/sec_edgar.py` | Legacy SEC EDGAR fetcher built on `httpx`. |
| `financial_news/storage/__init__.py` | Re-exports the canonical and legacy stores. |
| `financial_news/storage/database.py` | `NewsDatabase` — the canonical SQLite store. |
| `financial_news/storage/news_store.py` | Legacy `NewsStore` SQLite store. |
| `financial_news/ui/__init__.py` | Re-exports the dashboard UI. |
| `financial_news/ui/dashboard.py` | `NewsDashboard` — a Streamlit news UI. |
| `financial_news/utils/__init__.py` | Empty utils-package marker. |
| `financial_news/data/.gitignore` | Ignores the local SQLite database files in the news-platform data directory. |

## `local_agent/` — experimental autonomous browser agent

| File | Purpose |
|---|---|
| `local_agent/__init__.py` | Package root for the experimental browser agent. |
| `local_agent/main.py` | `AgentOrchestrator` — plan → DOM-act → execute → verify loop; CLI entry point. |
| `local_agent/mcp_server.py` | FastMCP server exposing browser-execution tools to Claude Desktop over stdio. |
| `local_agent/agents/__init__.py` | Re-exports the agent classes. |
| `local_agent/agents/base_agent.py` | `BaseAgent` — multi-provider LLM client (Claude API / Ollama). |
| `local_agent/agents/dom_actor.py` | `DOMActorAgent` — reads a DOM snapshot and executes a Playwright action. |
| `local_agent/agents/planner.py` | `PlannerAgent` — decomposes a goal into a JSON step plan. |
| `local_agent/agents/verifier.py` | `VerifierAgent` — verifies action success via URL/DOM state changes. |
| `local_agent/browser/__init__.py` | Re-exports the tab manager. |
| `local_agent/browser/tab_manager.py` | `TabManager` — multi-tab Playwright management with SSRF-validated navigation. |
| `local_agent/memory/__init__.py` | Re-exports the memory components. |
| `local_agent/memory/chroma_manager.py` | `ChromaManager` — ChromaDB vector store for pages and task plans. |
| `local_agent/memory/logger.py` | `StructuredLogger` — SQLite + JSON structured task logging. |
| `local_agent/ui/__init__.py` | Re-exports the Streamlit render helpers. |
| `local_agent/ui/components.py` | Reusable Streamlit UI components. |
| `local_agent/ui/streamlit_app.py` | Streamlit dashboard for the browser agent. |
| `local_agent/utils/__init__.py` | Re-exports config and retry/error helpers. |
| `local_agent/utils/config.py` | `AgentConfig` — env-driven pydantic config. |
| `local_agent/utils/efficiency.py` | Caching and performance utilities (LRU/action caches, batching, monitors). |
| `local_agent/utils/error_handling.py` | Agent exception hierarchy and retry/recovery helpers. |
| `local_agent/utils/security.py` | Security hardening — SSRF validation, sanitization, rate limiting, emergency stop. |
| `local_agent/tests/*` | Pytest tests for the agent, memory, Ollama, Playwright, security, and end-to-end flows. |

## `ml/` — research ML models (off the EV path)

| File | Purpose |
|---|---|
| `ml/__init__.py` | Re-exports the earnings and wheel models. |
| `ml/earnings_model.py` | `EarningsPredictor` — IV-crush / move-vs-implied prediction. |
| `ml/model_governance.py` | Model lifecycle governance — model cards, drift detection, champion/challenger, registry. |
| `ml/wheel_model.py` | `WheelEntryModel` — a research GBM entry classifier (default output dir: `models/`). |

## `models/`

| File | Purpose |
|---|---|
| `models/.gitkeep` | Placeholder keeping the otherwise-empty `models/` directory tracked; `models/` is the default model-output path named by `ml/wheel_model.py`. |

## `news_pipeline/` — browser-agent news pipeline (drives `morning_run.py`)

| File | Purpose |
|---|---|
| `news_pipeline/__init__.py` | Package root; lazy-imports browser agents to avoid a hard Playwright dependency. |
| `news_pipeline/orchestrator.py` | `NewsPipelineOrchestrator` — the multi-stage scrape → preprocess → verify → format → editorial → publish pipeline. |
| `news_pipeline/publisher.py` | `NewsPublisher` — publishes finalized stories to API / SQLite / file. |
| `news_pipeline/slo.py` | SLO definitions and the per-stage latency/availability tracker. |
| `news_pipeline/models/__init__.py` | Re-exports the pipeline data models. |
| `news_pipeline/models/schema.py` | Pipeline dataclasses with to/from-dict serialization. |
| `news_pipeline/browser_agents/__init__.py` | Re-exports browser-agent types and session classes. |
| `news_pipeline/browser_agents/base.py` | `BrowserModelSession` ABC and the session-pool manager. |
| `news_pipeline/browser_agents/chatgpt_agent.py` | Browser automation for ChatGPT. |
| `news_pipeline/browser_agents/claude_agent.py` | Browser automation for Claude (verification and editorial). |
| `news_pipeline/browser_agents/gemini_agent.py` | Browser automation for Gemini (verification with search). |
| `news_pipeline/browser_agents/grok_agent.py` | Browser automation for Grok (X/Twitter market sentiment). |
| `news_pipeline/browser_agents/robustness.py` | CSS-selector success-rate tracking and DOM-drift detection. |
| `news_pipeline/browser_agents/types.py` | Playwright-free enums and dataclasses. |
| `news_pipeline/local_llm/__init__.py` | Re-exports the local preprocessor. |
| `news_pipeline/local_llm/preprocessor.py` | `LocalPreprocessor` — Ollama-based news filtering and categorization. |
| `news_pipeline/recovery/__init__.py` | Re-exports the recovery components. |
| `news_pipeline/recovery/checkpoints.py` | Atomic JSON stage checkpoints with resume support. |
| `news_pipeline/recovery/fallbacks.py` | Provider fallback chains and degraded-mode configuration. |
| `news_pipeline/recovery/health.py` | Provider health monitoring and availability tracking. |
| `news_pipeline/scrapers/__init__.py` | Re-exports the scrapers. |
| `news_pipeline/scrapers/aggregator.py` | `NewsAggregator` — runs all scrapers in parallel and deduplicates. |
| `news_pipeline/scrapers/base.py` | `NewsScraper` ABC and the common item model. |
| `news_pipeline/scrapers/browser_scraper.py` | Playwright scraper for RSS-less sites. |
| `news_pipeline/scrapers/rss_scraper.py` | RSS/Atom feed scraper. |
| `news_pipeline/security/__init__.py` | Re-exports the security components. |
| `news_pipeline/security/classifier.py` | `SensitivityClassifier` — content sensitivity tiering. |
| `news_pipeline/security/routing_policy.py` | `RoutingPolicy` — local-only / sanitize / external routing decisions. |
| `news_pipeline/security/sanitizer.py` | `Sanitizer` — redacts PII and credentials before external transmission. |

## `notebooks/`

| File | Purpose |
|---|---|
| `notebooks/.gitkeep` | Placeholder for exploratory Jupyter notebooks. |

## `scripts/` — data pullers, diagnostics, Bloomberg-export assets

| File | Purpose |
|---|---|
| `scripts/pull_all.py` | Orchestrates every puller in dependency order, skipping steps whose upstream is unavailable. |
| `scripts/pull_ohlcv.py` | xbbg/Bloomberg pull of daily OHLCV for all constituents. |
| `scripts/pull_liquidity.py` | xbbg/Bloomberg pull of daily liquidity metrics. |
| `scripts/pull_options_greeks.py` | xbbg/Bloomberg pull of IV term structure / skew. |
| `scripts/pull_short_interest.py` | xbbg/Bloomberg pull of short-interest data. |
| `scripts/pull_historical_fundamentals.py` | xbbg/Bloomberg pull of quarterly fundamentals. |
| `scripts/pull_fundamentals_yf.py` | yfinance pull of a per-ticker fundamentals snapshot. |
| `scripts/pull_earnings_yf.py` | yfinance pull of past and upcoming earnings dates. |
| `scripts/pull_treasury_yields_yf.py` | yfinance pull of Treasury yield indices. |
| `scripts/pull_vol_indices.py` | Theta-then-yfinance pull of the volatility-index family. |
| `scripts/pull_news_sentiment.py` | Pulls and scores per-ticker news from Polygon/Finnhub/Benzinga. |
| `scripts/pull_theta_indices_history.py` | Theta pull of VIX-family index OHLC history. |
| `scripts/pull_theta_iv_surface_history.py` | Theta pull of historical IV surfaces with strict partial-coverage rejection. |
| `scripts/pull_theta_options_flow.py` | Theta pull of daily per-ticker options-flow aggregates. |
| `scripts/pull_theta_corp_actions.py` | Theta pull of stock splits and dividends. |
| `scripts/pull_theta_option_tape.py` | Theta pull of intraday option trade+quote tape. |
| `scripts/pull_theta_vix_futures.py` | Theta pull of the VIX futures curve. |
| `scripts/theta_backfill.py` | Tier-aware Theta bulk-backfill CLI with subcommands. |
| `scripts/theta_health_check.py` | Theta Terminal health probe across every v3 endpoint the engine uses. |
| `scripts/probe_theta_capabilities.py` | Probes the Theta tier and writes the capability map. |
| `scripts/backfill_features.py` | Recomputes the feature store for every universe ticker in parallel. |
| `scripts/diagnose_candidates.py` | Read-only EV-ranker funnel report for zero-trade debugging. |
| `scripts/feature_smoke_test.py` | End-to-end smoke-test harness exercising the data layer, EV engine and API. |
| `scripts/quant_benchmark_gate.py` | Hard acceptance gate validating the quant engine against academic reference values. |
| `scripts/orchestrate.py` | Unified daily orchestrator (morning / intraday / evening / full). |
| `scripts/run_pipeline.py` | CLI front-end to the data `PipelineOrchestrator`. |
| `scripts/validate_environment.py` | Environment validation for CI — Python version, dependencies, env vars, directory structure. |
| `scripts/check_env.py` | A small environment smoke check printing core-package versions. |
| `scripts/check_manifest_coverage.py` | CI guard — fails the build when a tracked file is absent from FILE_MANIFEST.md (or vice versa); wired into the `FILE_MANIFEST Coverage` job in `.github/workflows/ci.yml`. |
| `scripts/setup-terminal.sh` | Parallel-session env loader for bash / Git Bash / WSL — source with a terminal letter (`source scripts/setup-terminal.sh a`) to export per-terminal `SWE_API_PORT`, `COVERAGE_FILE`, `PYTEST_CACHE_DIR`, `SWE_DATA_PROCESSED_DIR`, `SWE_MODELS_DIR`, `SWE_DATA_PROVIDER`. See `docs/PARALLEL_SESSIONS.md` "Env vars per terminal". |
| `scripts/setup-terminal.ps1` | PowerShell companion to `setup-terminal.sh` — dot-source (`. .\scripts\setup-terminal.ps1 a`) for native Windows shells. Sets the same six env vars. |
| `scripts/process_bloomberg_exports.py` | Cleans and validates Bloomberg-exported CSVs into the per-ticker layout. |
| `scripts/download_sp500_constituents.py` | Scrapes the current S&P 500 constituent list from Wikipedia. |
| `scripts/download_ohlcv.py` | yfinance per-ticker OHLCV downloader. |
| `scripts/download_yf_ohlcv.py` | yfinance OHLCV downloader with multi-index header cleanup. |
| `scripts/download_yf_options.py` | yfinance option-chain downloader. |
| `scripts/test_bloomberg.py` | Bloomberg connectivity tester (a CLI tool, not a pytest file). |
| `scripts/bloomberg_excel_extractor.bas` | Excel VBA module automating Bloomberg data extraction. |
| `scripts/bloomberg_excel_extractor_v2.bas` | Revised Bloomberg Excel VBA extractor. |
| `scripts/bloomberg_export.vba` | Simplified Bloomberg VBA export macro. |
| `scripts/export_sheets_to_csv.vba` | Excel VBA macro exporting ticker worksheets to CSV. |
| `scripts/bloomberg_bql_pulls.md` | Copy/paste BQL query reference for pulling Bloomberg datasets. |
| `scripts/*_formulas.txt` | Generated per-ticker Bloomberg `=BDH(...)` formula lists for Excel paste (ohlcv / iv / earnings / dividends, plus the combined `bloomberg_formulas.txt`). |
| `scripts/.gitkeep` | Directory placeholder. |

## `src/` — feature-engineering / schema / backtest modules

See `DECISIONS.md` D2 for `src/`'s status.

| File | Purpose |
|---|---|
| `src/__init__.py` | Package marker. |
| `src/features/__init__.py` | Re-exports the nine feature classes. |
| `src/features/technical.py` | `TechnicalFeatures` — SMA/EMA/RSI/MACD/Bollinger/ATR/Hurst indicators. |
| `src/features/volatility.py` | `VolatilityFeatures` — realised-vol estimators and IV rank/percentile. |
| `src/features/options.py` | `OptionsFeatures` — flow ratios, P(profit), expected move, premium yield. |
| `src/features/dynamics.py` | `OptionsDynamics` — change-based features (ΔOI, ΔIV, Δskew). |
| `src/features/events.py` | `EventVolatility` — earnings/macro IV ramp and crush, gap distribution. |
| `src/features/regime.py` | `RegimeDetector` — trend/vol/liquidity regime classification. |
| `src/features/vol_edge.py` | `VolatilityEdge` — IV-RV spread/ratio/zscore and composite edge score. |
| `src/features/labels.py` | `LabelGenerator` — ML training labels (CSP outcome, forward returns, touch). |
| `src/features/assignment.py` | `AssignmentFeatures` — probability-of-touch and roll-vs-assignment scoring. |
| `src/data/__init__.py` | Re-exports the data schemas and validator. |
| `src/data/schemas.py` | Pydantic schemas for OHLCV, options flow, fundamentals, vol, etc. |
| `src/data/validators.py` | `DataValidator` — pandas-based data validation. |
| `src/backtest/__init__.py` | Re-exports the wheel backtester. |
| `src/backtest/wheel_backtest.py` | Event-driven wheel backtester (research/simulation only). |
| `src/execution/__init__.py` | Empty package stub. |
| `src/models/__init__.py` | Empty package stub. |
| `src/risk/__init__.py` | Empty package stub. |

## `tests/` — test suite

| File | Purpose |
|---|---|
| `tests/__init__.py` | Test-package marker. |
| `tests/quant_benchmarks.py` | Non-test helper — the quantitative tolerance registry used as release gates. |
| `tests/fixtures/theta_v3_*.csv` | Captured live Theta v3 SPY responses used as connector test fixtures. |
| `tests/test_audit_invariants.py` | Launch-blocker invariants — Greeks unit contract, PIT safety, TV webhook HMAC, EV-engine invariants. |
| `tests/test_audit_viii_e2e.py` | Launch-blocker invariant — the end-to-end TV-webhook → EV-ranker → tracker authority chain. |
| `tests/test_audit_viii_unit_invariants.py` | Launch-blocker invariant — IV/rate percent-vs-decimal normalisation and the rolled-leg P&L accumulator. |
| `tests/test_audit_viii_real_data_smoke.py` | Real-Bloomberg smoke test of the EV ranker (module-level skip without the CSVs). |
| `tests/test_authority_hardening.py` | Launch-blocker invariant — TV / strangle / tracker route through the EV authority. |
| `tests/test_backtest_regression.py` | Backtest-regression harness — runs S27/S32/S34/S35 reproducers against the current engine and compares to committed snapshots in `backtests/regression/snapshots/`. Gated behind the `backtest_regression` marker (long-running). |
| `tests/test_ev_authority_log_schema.py` | Schema-closure regression for `WheelTracker._ev_authority_log` — pins the five D16 entry shapes (`issue`, `refuse_issue`, `consume`, `reject` × {unknown_token, missing_current_ev_dollars, stale_ev}) and detects drift (unknown action, missing required key, accidental extra key). |
| `tests/test_engine_api_port.py` | Unit tests for `engine_api._resolve_port()` — pins the `SWE_API_PORT` contract (default 8787, env override, loud failure on malformed / out-of-range). Closes D15 Unresolved per #154 C7. |
| `tests/test_tv_nonce_register_lock.py` | Regression tests pinning the explicit `_TV_SEEN_NONCES_LOCK` around `_tv_seen_register` (S20 C3). Asserts lock existence + 64-worker contention behavior (exactly 1 worker accepts a duplicate digest, all 64 distinct digests accepted). |
| `tests/test_portfolio_risk_gates.py` | Unit tests for `engine/portfolio_risk_gates.py` (D17 / #154 C4 Phase 1) — pins the adapter (`take_snapshot`) per `WheelPosition` state plus the five gate functions' pass/fail/skip semantics against the locked D17 defaults. |
| `tests/test_dossier_invariant.py` | Launch-blocker invariant — the downgrade-only `EnginePhaseReviewer` contract. |
| `tests/test_decision_layer_wiring.py` | Launch-blocker invariant — the production wire `rank_candidates_by_ev` → `WheelTracker.consume_ranker_row` → `open_short_put(current_ev_dollars=…)` and `PortfolioContext` through `build_dossiers` so D16 / D17 hardening fires live for the ranker chain operators run (closes the prior-audit cross-cutting #4 gap). |
| `tests/test_consume_ranker_row_anchor.py` | Real-input anchor for the production chain — pulls rows from a live `WheelRunner.rank_candidates_by_ev` against the Bloomberg CSVs and feeds them unmodified through `WheelTracker.consume_ranker_row`. Schema-drift inoculation between the ranker tail and `consume_ranker_row`'s key reads. Complements `test_decision_layer_wiring.py`'s hand-built `_ev_row` coverage. |
| `tests/test_diagnostic_column_honesty.py` | Pin two diagnostic-column honesty contracts in the ranker output: `rank_covered_calls_by_ev`'s `expected_dividend` reads 0.0 when the EVEngine dividend gate would not fire (S28 Fix #1); `rank_candidates_by_ev`'s `skew_source` provenance reads `"unavailable"` when the skew block did not execute (S29 Fix #1). |
| `tests/test_ev_non_finite_defense.py` | Pin the non-finite EV defense across both verdict paths: dossier-side R1a (`EnginePhaseReviewer` returns `("blocked", "ev_non_finite")` on `+inf` / `-inf` / `NaN` before R1's negative-EV check), webhook-side mirror in `_enrich_alert`, and the shared `MIN_PROCEED_EV_DOLLARS` constant wiring. |
| `tests/test_ranker_tracker_wire.py` | Pin the end-to-end production wire: `WheelRunner.build_candidate_dossiers` threads `portfolio_context` through to `build_dossiers` so D17 R7/R8 fire live; `WheelRunner.consume_into_tracker` runs the rank → token → `open_short_put` chain in one call, capturing per-row outcomes and catching D16/D17 refusals into a structured list (loop-safe). |
| `tests/test_launch_blockers.py` | Launch-blocker invariant — `/api/candidates` EV authority, research-only flags, the history/chain/stress gates. |
| `tests/test_audit_improvements.py` | Quant-correctness for the 2026-04 audit deliverables (forward distributions, empirical surface, survivorship). |
| `tests/test_quant_upgrades.py` | Quant-correctness for the audit-III modules (tail risk, HMM, skew, copula, event gate). |
| `tests/test_ev_engine_upgrades.py` | `EVEngine` upgrades — deterministic fallback, regime-multiplier clamp, Omega ratio. |
| `tests/test_evengine_event_lockout.py` | `EVEngine.evaluate` event-lockout short-circuit — pins the blocked-branch return shape, boundary cases on `EventGate._event_touches_window`'s symmetric arithmetic, and §2-adjacent: dealer multiplier NOT applied when blocked. |
| `tests/test_covered_call_ranker.py` | Launch-blocker invariant — the covered-call EV ranker schema and authority. |
| `tests/test_strangle_ev_ranker.py` | Launch-blocker invariant — the strangle EV ranker composition and authority. |
| `tests/test_ranker_transparency.py` | Launch-blocker invariant — ranker drop-log, regime label and `ev_raw` transparency. |
| `tests/test_ranker_iv_pit.py` | Pin S23 F3 fix — the three rankers prefer `conn.get_iv_history(end_date=as_of)` over the snapshot `fundamentals['implied_vol_atm']`, with defensive fallback for connectors without `get_iv_history`. |
| `tests/test_wheel_runner_select_book.py` | Launch-blocker invariant — `select_book` as a pure post-processor. |
| `tests/test_wheel_runner_coverage.py` | `WheelRunner` coverage — analysis summary, connector selection, wheel score, screening. |
| `tests/test_explore_ticker.py` | `WheelRunner.explore_ticker` — single-ticker (delta × DTE) grid surfacing for short-put EV exploration via the production EV path. |
| `tests/test_option_pricer.py` | Quant-correctness — Black-Scholes pricing/Greeks, put-call parity, vectorized pricing. |
| `tests/test_binomial_tree.py` | Quant-correctness — CRR binomial pricing and cross-model validation. |
| `tests/test_monte_carlo.py` | Quant-correctness — block bootstrap, jump diffusion, Longstaff-Schwartz. |
| `tests/test_advanced_quant.py` | Quant-correctness — third-order Greeks, BAW American pricing, multi-asset VaR. |
| `tests/test_quant_fixtures.py` | Quant-correctness — textbook-value regression for pricing and volatility estimators. |
| `tests/test_greeks_unit_invariants.py` | Launch-blocker invariant — the Greeks unit contract. |
| `tests/test_realized_vol.py` | Quant-correctness — the realised-volatility estimators. |
| `tests/test_tail_risk.py` | Quant-correctness — POT-GPD tail estimation. |
| `tests/test_f4_tail_risk_gap.py` | F4 regression-watch from PR #178 / #184 — pins that today's forward-distribution + POT-GPD pipeline does NOT widen tail metrics for the COST 2022-04 / UNH 2024-11 19-24% realized drops. Synthetic two-regime tests isolate the 504-day-lookback-dilution mechanism (H1) and demonstrate the fix direction. Two `xfail(strict=False)` tests track the `heavy_tail` flag until the gap is fixed. |
| `tests/test_portfolio_copula_coverage.py` | Quant-correctness — copula edge paths (PSD repair, Cholesky fallback, CVaR ladder). |
| `tests/test_risk_manager.py` | `RiskManager` — sizing, portfolio Greeks, VaR, sector exposure, HRP. |
| `tests/test_stress_testing.py` | `StressTester` — scenarios, sensitivity, Greeks stress ladder. |
| `tests/test_payoff_engine.py` | `PayoffEngine` — payoff diagrams, expected-move bands, strike recommendations. |
| `tests/test_regime_detector.py` | The rule-based regime classifier. |
| `tests/test_dealer_positioning.py` | Dealer positioning — analyzer math, the clamped multiplier, reviewer rule R6. |
| `tests/test_dealer_multiplier_evengine_integration.py` | Dealer-multiplier integration boundaries — pins that `[0.70, 1.05]` survives `EVEngine.evaluate` to `EVResult.dealer_multiplier`, the asymmetric-by-design clamp at the EVResult level, the `regime_mult *= dealer_mult` compounding, and the §2 "scales ev_dollars only" claim as proportionality. Companion to PR #185's blocked-path test. |
| `tests/test_extreme_numerics.py` | Numerical stability under near-expiry / near-zero-vol / deep-moneyness extremes. |
| `tests/test_edge_cases.py` | Engine edge cases across pricing, costs and the tracker state machine. |
| `tests/test_properties.py` | Hypothesis property-based invariants for pricing, RSI, IV-rank, Kelly. |
| `tests/test_point_in_time.py` | Anti-lookahead — rolling features and labels use only past data. |
| `tests/test_pit_leaks.py` | Launch-blocker invariant — news and credit-regime overlays honour a historical `as_of`. |
| `tests/test_contracts.py` | `engine.contracts` interface-validation helpers. |
| `tests/test_policy_config.py` | `TradingPolicyConfig` load/save/validate. |
| `tests/test_observability.py` | `engine.observability` trace context, decision journal, audit logger. |
| `tests/test_event_calendar.py` | `engine.event_calendar` queries, builder, ingestion manager. |
| `tests/test_event_gate.py` | `EventGate` lockout, buffer windows, candidate filtering. |
| `tests/test_event_gate_back_buffer.py` | Pin S23 F1 fix — `MarketDataConnector.get_recent_earnings` complements `get_next_earnings`; the three rankers register past earnings on the gate so the symmetric back-buffer fires. |
| `tests/test_earnings_drift.py` | `EarningsDriftAnalyzer` post-earnings drift statistics. |
| `tests/test_signals.py` | The signal-generation framework and aggregator. |
| `tests/test_strangle_timing.py` | The strangle-timing engine — regime classification, entry scoring, IV overlay. |
| `tests/test_strangle_recommendation_gate.py` | The strangle phase/confidence downgrade-only recommendation gate. |
| `tests/test_data_connector.py` | `MarketDataConnector` query methods against synthetic CSVs. |
| `tests/test_theta_connector.py` | Live-Terminal smoke test of `ThetaConnector` (auto-skips). |
| `tests/test_theta_connector_v3.py` | HTTP-mocked `ThetaConnector` v3 endpoints and the per-endpoint failure contract. |
| `tests/test_theta_connector_coverage.py` | HTTP-mocked `ThetaConnector` edge-branch coverage. |
| `tests/test_data_integration.py` | `engine.data_integration` Bloomberg calendar/rate loaders. |
| `tests/test_bloomberg_loader.py` | Bloomberg CSV ingestion and the data pipeline. |
| `tests/test_data_pipeline.py` | The data-engineering pipeline — feature store, quality, observability. |
| `tests/test_data_quality.py` | `data.quality` — the options-consistency gate regression. |
| `tests/test_data_validation.py` | `utils.data_validation` — IV normalisation, option/OHLCV validation. |
| `tests/test_features.py` | `src.features` feature-calculation correctness. |
| `tests/test_external_data_cboe.py` | HTTP-mocked `CBOEAdapter`. |
| `tests/test_external_data_edgar.py` | HTTP-mocked `EDGARAdapter`. |
| `tests/test_external_data_fred.py` | HTTP-mocked `FREDAdapter`. |
| `tests/test_external_data_yfinance.py` | HTTP-mocked `YFinanceAdapter`. |
| `tests/test_wheel_lifecycle.py` | `WheelTracker` partial assignment and roll mechanics. |
| `tests/test_wheel_cycle.py` | A `print()`-driven `WheelTracker` cycle demo script (not a pytest file). |
| `tests/test_wheel_backtest.py` | `src.backtest.wheel_backtest` run, scoring and metrics. |
| `tests/test_wheel_tracker_persistence.py` | `WheelTracker` JSON persistence round-trips. |
| `tests/test_wheel_tracker_suggest_rolls.py` | Launch-blocker invariant — `suggest_rolls` properties and roll-EV regression. |
| `tests/test_wheel_tracker_suggest_call_rolls.py` | Launch-blocker invariant — `suggest_call_rolls` properties and roll-EV regression. |
| `tests/test_suggest_rolls_drops.py` | Pin S22 F1 fix — `suggest_rolls` and `suggest_call_rolls` emit `.attrs["drops"]` mirroring the ranker drop-log pattern; per-filter-site drop entries with conformant schema. |
| `tests/test_mark_to_market_iv.py` | `WheelTracker.mark_to_market` IV-staleness fix regression. |
| `tests/test_available_buying_power.py` | `WheelTracker.available_buying_power` CSP-collateral netting. |
| `tests/test_portfolio_tracker.py` | `PortfolioTracker` transactions, holdings, returns, snapshots. |
| `tests/test_transaction_costs.py` | A `print()`-driven transaction-costs demo script (not a pytest file). |
| `tests/test_transaction_costs_coverage.py` | `engine.transaction_costs` coverage backfill — slippage tiers, round-trip cost. |
| `tests/test_tv_api.py` | The TradingView bridge HTTP endpoints in `engine_api.py`. |
| `tests/test_tv_signals.py` | `engine.tv_signals` — signal computation, IV overlay, Pine-constant parity. |
| `tests/test_tv_dossier.py` | Launch-blocker invariant — the TV visual-context dossier layer and providers. |
| `tests/test_tv_dossier_d17_wire.py` | D17 portfolio-context live wire on `/api/tv/dossier` — verifies opt-in `portfolio_context` query params parse into a `PortfolioContext` consumed by `EVEngine.evaluate` (closes B2). |
| `tests/test_mcp_client.py` | Subprocess-mocked `MCPCLIClient` — the five-call capture sequence and failure modes. |
| `tests/test_dossier_cp1252.py` | Regression — reviewer notes are cp1252-encodable. |
| `tests/test_advisors.py` | The advisor committee — schemas, advisors, aggregation, engine integration. |
| `tests/test_new_modules.py` | Coverage backfill — Taleb advisor, committee modes, runner import smoke. |
| `tests/test_financial_news.py` | The `financial_news` platform — schema, macro calendar, verification engine. |
| `tests/test_news_processing.py` | `financial_news` article classification. |
| `tests/test_news_pipeline.py` | The `news_pipeline` package — models, security, recovery, publisher. |
| `tests/test_news_sentiment.py` | `NewsSentimentReader` — sentiment reading and the EV multiplier. |
| `tests/test_adversarial_news.py` | Adversarial news-scenario smoke test. |
| `tests/test_recovery_checkpoints.py` | `news_pipeline.recovery.checkpoints` save/load/resume. |
| `tests/test_recovery_fallbacks.py` | `news_pipeline.recovery.fallbacks` degraded-mode and fallback chains. |
| `tests/test_recovery_health.py` | `news_pipeline.recovery.health` provider health tracking. |
| `tests/test_dashboard.py` | The legacy `QuantDashboard` CLI surface. |
| `tests/test_infrastructure.py` | Repo-level infrastructure components — env validation, benchmarks, health, SLO. |
| `tests/test_iv_surface_history_puller.py` | Regression for the IV-surface-history puller. |
| `tests/test_theta_indices_puller.py` | Regression for the Theta indices-history puller. |

## `tradingview/` — Pine indicator + analyst-workspace assets

| File | Purpose |
|---|---|
| `tradingview/README.md` | Hands-on setup checklist for the engine bridge (install the Pine indicator, wire the webhook). |
| `tradingview/CLAUDE.md` | Session-orientation contract for Claude acting as the TradingView analyst. |
| `tradingview/OVERVIEW.md` | Operating overview of the analyst function. |
| `tradingview/smart_wheel_signals.pine` | The Pine v5 indicator mirroring `engine/tv_signals.py`. |
| `tradingview/alert_payload_schema.json` | JSON Schema for the webhook payload. |
| `tradingview/launch-tradingview-cdp.sh` | Launches TradingView Desktop with CDP for the analyst workspace. |
| `tradingview/{models,pine,research}/.gitkeep` | Placeholders for analyst-workspace output directories (contents gitignored). |

## `utils/` — shared utilities

| File | Purpose |
|---|---|
| `utils/__init__.py` | Re-exports the validation, dates, logging and metadata helpers. |
| `utils/data_validation.py` | Option/OHLCV validation, IV normalization, liquidity filtering. |
| `utils/dates.py` | Trading-day calendar, date normalization, DTE conversions. |
| `utils/health.py` | `HealthChecker` — Kubernetes-style liveness/readiness checks. |
| `utils/logging_config.py` | Logging setup helpers. |
| `utils/metadata.py` | Git-fingerprint and metadata-sidecar helpers for reproducibility. |
| `utils/security.py` | Audit logging, input validation, secrets management, rate limiting. |
