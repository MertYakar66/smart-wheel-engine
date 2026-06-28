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
| `conftest.py` | pytest configuration — hypothesis profiles, shared fixtures, custom markers. |
| `pyproject.toml` | Packaging and tooling configuration (ruff, mypy, pytest, coverage). |
| `requirements.txt` | Runtime dependency list. |
| `.pre-commit-config.yaml` | Pre-commit hook config — whitespace hygiene, ruff, ruff-format, bandit, mypy, detect-secrets. |
| `.gitignore` | Ignore rules — secrets, Python artefacts, caches, gitignored data trees, the Theta install, with allow-list exceptions for tracked `.claude/` content. |
| `.gitattributes` | Pins LF line endings and marks binary types; prevents CRLF churn from the Drive mount. |
| `.python-version` | Pins the Python version for version managers. |
| `.env.example` | Template for the gitignored `.env` secrets file; lists the optional API keys. |

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
| `.github/pull_request_template.md` | Default PR body — the COMMIT_GUIDE §3 sections (Summary / Changes / Why / Tests / §2 surface / Tried-rejected / Unresolved / AI-handoff) plus the `lane-claim` block required by `scripts/check_lane_claim.py` for decision-layer PRs. |

## `archive/`

Point-in-time and superseded artifacts, retained for history, not maintained. See `archive/README.md`.

| File | Purpose |
|---|---|
| `archive/README.md` | Index of archived files — original path and archival reason for each. |
| `archive/2026-05/OptionsEngine.txt` | Archived narrative end-to-end usage walkthrough (point-in-time, unmaintained). |
| `archive/2026-05/ARCHITECTURE.md` | Archived architecture doc describing a planned `src/`-based layout that does not match the repo. |
| `archive/2026-05/DATA_COLLECTION_REPORT.md` | Archived dated data-collection phase report. |
| `archive/2026-05/bloomberg_excel_extractor.bas` | Archived V1 of the Bloomberg Excel VBA extractor — superseded by `scripts/bloomberg_excel_extractor_v2.bas` ("fixed version with longer wait times"). |
| `archive/2026-05/download_ohlcv.py` | Archived early yfinance OHLCV downloader — superseded by `scripts/download_yf_ohlcv.py` (adds multi-index header cleanup). |
| `archive/2026-06/SESSION_HANDOFF.md` | Archived point-in-time session handoff (2026-05-18); carried a SUPERSEDED banner — live state is `PROJECT_STATE.md` / `DECISIONS.md`. |
| `archive/2026-06/Claude_Prompting_Master_Guide.md` | Archived generic Claude prompt-engineering reference (not project-specific; zero inbound refs). |
| `archive/2026-06/DATA_SPECIFICATION.md` | Archived aspirational Parquet data-layer design that never matched on-disk reality — superseded by `docs/DATA_POLICY.md` + `docs/DATA_INVENTORY.md`. |
| `archive/2026-06/pull.bat` | Archived Windows pull launcher — zero refs; not part of the live (agent-session) workflow. |
| `archive/2026-06/pull_branch.bat` | Archived Windows branch-checkout launcher — same rationale; default branch name was long dead. |
| `archive/2026-06/fetch_data.bat` | Archived Windows Theta-backfill launcher — the documented workflow invokes the Python scripts directly. |
| `archive/2026-05/END_TO_END_REVIEW_2026_05_25.md` | Archived four-pass end-to-end product review against `origin/main` @ `e83eaca`. Point-in-time snapshot; superseded by `docs/VERIFICATION_INDEX_2026-05-28.md` as the canonical verification index. |
| `archive/2026-05/LAUNCH_READINESS_ANALYSIS_2026-05-26.md` | Archived 2026-05-26 launch-readiness analysis. Point-in-time review of S22/S27/S32/S34/S35 pre-#260 engine; superseded by `docs/PRODUCTION_READINESS.md` (live deployment gate) and `docs/VERIFICATION_INDEX_2026-05-28.md`. |
| `archive/2026-05/SOUNDNESS_REVIEW_2026-05-26.md` | Archived second-pass critical re-verification of S22/S27/S32/S34/S35 conclusions. Surfaced the equity-beta-dominance and BKNG-concentration findings now folded into `docs/PRODUCTION_READINESS.md`. Point-in-time. |
| `archive/2026-05/PREDICTIVE_VALIDITY_REVIEW.md` | Archived PR #197 meta-verification of S22 + S27 backtests (P1–P9). Headline ρ ≈ 0.22 now carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md` Tested surfaces table. Point-in-time. |
| `archive/2026-05/RELIABILITY_ARC_REVIEW.md` | Archived PR #194 independent verification of the reliability arc (S18 / S19 / S20). Headline "PASS-with-caveat" carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md`. Point-in-time. |
| `archive/2026-05/AUDIT_OF_AUDIT_REVIEW.md` | Archived PR #195 meta-verification of `archive/2026-05/TERMINAL_A_AUDIT.md`. Headline "22/22 SOLID, 0 §2 breaches missed" carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md`. Point-in-time. |
| `archive/2026-05/ENGINE_SUBSYSTEM_AUDIT.md` | Archived structural read-through audit of 46 `engine/` + 10 `advisors/` files. Point-in-time; "no new bugs" finding carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md`. |
| `archive/2026-05/TERMINAL_A_AUDIT.md` | Archived independent engineering audit of Terminal A's 22-PR coordinated run on board #113. Point-in-time; tally carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md` §1. |
| `archive/2026-05/SESSION_REPORT_2026-05-26.md` | Archived machine-readable session ledger for the 2026-05-26 deployment-readiness campaign. Point-in-time; superseded by `docs/VERIFICATION_INDEX_2026-05-28.md` as the campaign-level reference. |
| `archive/2026-05/ENGINE_REALISM_VERIFICATION_2026-05-26.md` | Archived 2026-05-26 realism + reliability battery against `origin/main` @ 9f0afaf. Pre-#260 / pre-#262 engine snapshot; superseded on the live surface by `docs/REALISM_VERIFICATION_2026-05-28.md` (post-F4 + R9 + R10). |
| `archive/2026-05/optionsengine_audit_2026-05-17.md` | Archived 2026-05-17 accuracy audit of the (also-archived) `OptionsEngine.txt` walkthrough. Point-in-time. |
| `archive/2026-05/data_inventory_2026-05-17.md` | Archived 2026-05-17 point-in-time data-inventory analysis report. |

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
| `backtests/survivorship.py` | R3 survivorship-aware harness — PIT universe via `consolidated_loader.get_universe_as_of` (membership presence; `min_weight` is the dead all-zeros sentinel), a `deep_history=True` connector, delisting-aware `terminal_spot` (last close on/before expiry, else 0 — never NaN-drop), and `run_survivorship_backtest` routing 100% through `rank_candidates_by_ev`. Not on the decision-layer path. |

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
| `backtests/regression/s43_rolling_multiwindow.py` | S43 harness — 4-window rolling multi-window backtest at $1M / 100t / three friction levels (W1=2018-2022, W2=2019-2023, W3=2020-2024 = S38 re-run, W4=2021-2025). Post-#260 engine. Mirrors `docs/ENGINE_BACKTEST_S43_ROLLING_MULTIWINDOW.md`. |
| `backtests/regression/s43_analyze.py` | S43 analyzer — Spearman ρ, per-year breakdown, refusal rates, concentration / top-tickers, R10 post-hoc audit from rank_log + tracker_state. Companion to `s43_rolling_multiwindow.py`. |
| `backtests/regression/s43_reconstruct.py` | S43 reconstruction helper for W1 (which launched before tracker dump extension) — replays rank_log to approximate concentration / deployment time-series. Documented caveat in the S43 doc methodology section. |
| `backtests/regression/s43_scan.py` | S43 §2 invariant scan — counts non-finite EVs and ev≤0 tradeable verdicts across all 4 windows × 3 friction levels. 184,602 rows scanned; zero violations. |
| `backtests/regression/snapshots/s27_ivpit_24t_100k.json` | Locked S27 snapshot — aggregate / per-year / per-quartile metrics from current post-PIT-fix engine. Generated via `--update-snapshot` against `data_csv_sha256` recorded in the fingerprint. Re-baseline workflow in `TESTING.md`. |
| `backtests/regression/snapshots/s32_friction_24t_1m.json` | Locked S32 snapshot — $1M / 24 tickers / 2022-2024 / three friction levels (none / bid_ask / full). Generated via the shared-rank `run_backtest_multi_friction` driver. |
| `backtests/regression/snapshots/s34_universe_100t_1m.json` | Locked S34 snapshot — $1M / 100 tickers / 2022-2024 / three friction levels. Universe-expansion backtest from `docs/SOUNDNESS_REVIEW_2026-05-26.md`. ρ ≈ 0.329 matches doc within 0.002. |
| `backtests/regression/snapshots/s35_oos_24t_100k.json` | Locked S35 snapshot — $100k / 24 tickers / 2018-2020 OOS / three friction levels. PR4 re-baseline. ρ ≈ 0.500 matches doc within 0.003 (driver-invariant signal); execution count doubled (19 → 40). |

## `config/`

| File | Purpose |
|---|---|
| `config/__init__.py` | Re-exports `Config`, `ConfigManager`, the sub-config dataclasses and preset factories. |
| `config/settings.py` | Centralized config dataclasses with YAML/JSON/env (`WHEEL_*`) loading. **Dormant** — zero importers; the live runtime config is `engine/policy_config.py`. (`ml/wheel_model.py` reads the `WHEEL_*` env vars directly, not through this module.) |

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
| `dashboard/src/app/not-found.tsx` | Global 404 page — branded, links back to the Cockpit / TOP. |
| `dashboard/src/app/global-error.tsx` | Root-level error boundary (catches root-layout errors; renders its own html/body) with a retry. |
| `dashboard/src/app/(main)/layout.tsx` | Layout for the standard web app — nav plus centered container. |
| `dashboard/src/app/(main)/loading.tsx` | Suspense fallback for the news-app routes — skeleton list under the nav. |
| `dashboard/src/app/(main)/error.tsx` | Error boundary for the news-app routes — retry, nav stays usable. |
| `dashboard/src/app/(main)/top/page.tsx` | "TOP" command-center page — breaking strip, top stories, category sections. |
| `dashboard/src/app/(main)/feed/page.tsx` | News feed page — story cards with sector filter and RSS refresh. |
| `dashboard/src/app/(main)/calendar/page.tsx` | Macro calendar page. |
| `dashboard/src/app/(main)/research/page.tsx` | AI research chat page streaming from `/api/chat`. |
| `dashboard/src/app/(main)/story/[id]/page.tsx` | Story detail page — sources, timeline, exposure mechanisms. |
| `dashboard/src/app/(main)/ticker/[symbol]/page.tsx` | Per-ticker page — quote, price chart, related news. |
| `dashboard/src/app/(main)/watchlist/page.tsx` | Watchlist page — add/remove tickers, prices, alerts. |
| `dashboard/src/app/(terminal)/layout.tsx` | Layout for the terminal route. |
| `dashboard/src/app/(terminal)/loading.tsx` | Suspense fallback for the terminal/cockpit routes — monospace skeleton. |
| `dashboard/src/app/(terminal)/error.tsx` | Error boundary for the terminal/cockpit routes — monospace, retry, engine-down hint. |
| `dashboard/src/app/(terminal)/terminal/page.tsx` | Bloomberg-style terminal dashboard — 6-panel grid, command line, engine data. |
| `dashboard/src/app/api/engine/route.ts` | Server-side proxy bridge to the Python engine API on `:8787`. Forwards the full PIT parameter set for `candidates` (as_of/dte/delta/min_ev/universe_limit) and adds a `dossier` action proxying `/api/tv/dossier` (top_n/timeframe/screenshots_dir + optional nav/holdings/puts_held/regime_map for the D17 portfolio gates). |
| `dashboard/src/app/(terminal)/cockpit/page.tsx` | Decision-cockpit page — read top-to-bottom and act. Regime banner → selection funnel → candidate cockpit table → one-click dossier drawer. Client component; fetches `/api/engine?action=candidates` + `?action=vix`; PIT controls (as_of/dte/delta/scan/top-N). All numbers from the engine; no decision logic here. |
| `dashboard/src/types/cockpit.ts` | Wire-shape types for the cockpit (`EngineCandidate`, `CandidatesResponse`, `Dossier`, `DossierResponse`, `VixRegime`) matching `engine_api.py::_handle_candidates` / `_handle_tv_dossier` exactly. |
| `dashboard/src/lib/cockpit-trust.ts` | Trust-calibration helpers — the one place encoding what to trust/distrust: R11 thresholds (VIX 25 / top-bin 0.90, mirroring `engine.candidate_dossier`), `confidenceTrust`, `vixRegimeLabel`, verdict colours, formatters. Mid-range prob_profit trusted; top bin in elevated vol flagged (crisis-realized ~0.57). |
| `dashboard/src/components/cockpit/distribution-bar.tsx` | Per-row P&L distribution bar — cvar5 whisker · p25/p50/p75 box · breakeven line. Foregrounds the modeled tail (the short-put body pins at max premium). The headline visual; ev_dollars is deliberately not drawn. |
| `dashboard/src/components/cockpit/calibrated-prob.tsx` | Calibration-aware prob_profit indicator — 0-1 dot, green mid-range, amber/red top bin; ghost marker at the crisis-realized ~0.57 in elevated vol. Never draws a confident green high-confidence reading. |
| `dashboard/src/components/cockpit/regime-banner.tsx` | Top "weather" banner — VIX level + regime label, R11 active/dormant, as_of, universe-scan summary, defensive-posture note. |
| `dashboard/src/components/cockpit/cockpit-table.tsx` | Candidate cockpit table — each row a decision unit: verdict badge, distribution bar, calibrated confidence, strike/premium/DTE/IV/collateral/ROC/CVaR5, de-emphasized EV·rank. Row click opens the dossier drawer. |
| `dashboard/src/components/cockpit/dossier-drawer.tsx` | Expandable dossier panel — verdict card + plain-language reviewer-chain trace (R1–R11) derived from the EV row + market VIX using the engine's R11 thresholds; chart/book-dependent rules marked needs-chart/needs-book + EV diagnostics. |
| `dashboard/src/components/cockpit/funnel.tsx` | Selection funnel — universe → scanned → ranked → shown. Flags that per-gate drop reasons (`frame.attrs["drops_summary"]`) are not serialized by `/api/candidates` (API follow-up), rather than inventing numbers. |
| `dashboard/src/components/cockpit/verdict-card.tsx` | Glanceable verdict card (quant-card FORMAT) — verdict + calibration-flagged confidence + distribution sparkline + top reasons. Used in the dossier drawer header. |
| `dashboard/src/components/cockpit/concentration-meters.tsx` | Single-name concentration meter (R10 · 10% cap) — bars per actionable candidate's collateral as % of an adjustable book NAV, with the cap line. Flags R9 sector bars (need the engine sector map) + live R9/R10 verdicts as a `/api/tv/dossier` follow-up rather than guessing sectors. |
| `dashboard/src/components/cockpit/frontier-chip.tsx` | Shared data-frontier staleness chip (at-frontier / Nd-behind / beyond-frontier / vs-today fallback) — single owner of the 4-branch copy + severity logic used by the cockpit header and regime banner. |
| `dashboard/src/app/(terminal)/portfolio/page.tsx` | Portfolio performance viewer (design D26) — read-only, observational. Client component; fetches the read-only `/api/portfolio/*` endpoints via `usePortfolioData` with `mock.ts` as the typed fallback; shows a live/mock data-source indicator. No EV authority, no order routing. |
| `dashboard/src/app/api/portfolio/[sub]/route.ts` | Proxy bridge to the engine's read-only performance-viewer endpoints — forwards `GET /api/portfolio/{summary,positions,returns,income,risk,history}` to `:8787` (no-store), whitelisting the sub-path. Mirrors `api/engine/route.ts`. |
| `dashboard/src/components/portfolio/use-portfolio-data.ts` | Client data layer for the viewer — fetches the six `/api/portfolio/*` slices in parallel via the proxy, returns the SAME shapes the components consume with per-slice `mock.ts` fallback + a `live` flag. |
| `dashboard/src/components/portfolio/mock.ts` | Typed shapes + UI constants for the viewer (ACCOUNT/RETURNS/EQUITY/HOLDINGS/SECTORS/CURRENCY/SINGLE_NAME/SECTOR_EXPOSURE, WheelState/Period); doubles as the typed fallback when the engine is unreachable. |
| `dashboard/src/components/portfolio/parts.tsx` | Shared presentational primitives — `PfCard`, `WheelBadge`, `PeriodToggle`, signed-USD/pct formatters, `pnlColor`. |
| `dashboard/src/components/shell/wheelhouse-header.tsx` | Shared "Wheelhouse" page chrome — sticky branding header (accent dot + page label + key-stat / status slots) and `CrossPageNav` cross-page tabs (Cockpit/Portfolio/Terminal/News). Presentational only; lifted from the /portfolio design (D26) so Cockpit + Terminal share one visual language. No EV authority, no data fetching. |
| `dashboard/src/components/portfolio/kpi-cards.tsx` | Six KPI cards (net-liq, period total-return, unrealized / realized-YTD P&L, 30-day premium, win-rate); prop-driven with mock defaults. |
| `dashboard/src/components/portfolio/equity-curve.tsx` | Portfolio-vs-SPY equity area chart + premium-income bar chart (Recharts); period-windowed; prop-driven. |
| `dashboard/src/components/portfolio/allocation.tsx` | Sector-allocation donut + currency split; stable per-sector palette so live + mock render identically; prop-driven. |
| `dashboard/src/components/portfolio/holdings-table.tsx` | Sortable per-symbol holdings table with wheel-state badges, %-NAV bars, and the single-name breach flag; prop-driven. |
| `dashboard/src/components/portfolio/risk-radar.tsx` | Concentration meters (single-name R10 / sector R9 caps) + margin-health gauge fed by the real excess-liquidity cushion; prop-driven. |
| `dashboard/src/components/portfolio/ask-bar.tsx` | Conversational ask-bar affordance (suggestion chips) — visual only; the engine-backed query layer is a later phase (design D26 §6.2 Phase B). |
| `dashboard/src/components/portfolio/income-panel.tsx` | Real Income section from `/api/portfolio/income` (previously served-but-never-fetched): realized/premium/win-rate chips, monthly realized P&L bars, ranked per-ticker league table; honest empty-ledger state. |
| `dashboard/src/components/portfolio/margin-panel.tsx` | Margin & leverage panel — loan balance, maintenance margin, excess-liquidity cushion, leverage from served summary fields; prop-driven, null-honest. |
| `dashboard/src/app/api/stories/route.ts` | Stories list API — query by sector/ticker, exposure-ranked. |
| `dashboard/src/app/api/stories/[id]/route.ts` | Single-story detail API. |
| `dashboard/src/app/api/chat/route.ts` | Chat API — streams from Ollama via the AI SDK. |
| `dashboard/src/app/api/market/route.ts` | Market quote API — Finnhub with cached fallback. |
| `dashboard/src/app/api/ingest/route.ts` | POST trigger for the RSS ingestion pipeline. |
| `dashboard/src/app/api/watchlist/route.ts` | Watchlist CRUD API — GET enriches with prices. |
| `dashboard/src/app/api/alerts/route.ts` | Alerts API — list/dismiss. |
| `dashboard/src/app/api/events/route.ts` | Calendar-events CRUD API. |
| `dashboard/src/app/api/categories/route.ts` | News-categories CRUD API. |
| `dashboard/src/app/api/briefings/route.ts` | Briefings API — generate/get morning/evening digests. |
| `dashboard/src/app/api/exposure/route.ts` | User-exposure CRUD API. |
| `dashboard/src/app/api/schedule/route.ts` | Ingestion-schedule API — status/history/trigger. |
| `dashboard/src/app/api/stream/route.ts` | Server-Sent-Events endpoint pushing new headlines. |
| `dashboard/src/components/nav.tsx` | Top navigation bar for the web app. |
| `dashboard/src/components/ui/*.tsx` | shadcn/ui base primitives (badge, button, card, input, scroll-area, skeleton). |
| `dashboard/src/components/terminal/*.tsx` | Terminal-app panels and controls (panel, status-bar, market/options/news/watchlist/macro/chat panels, live-book + dealer-positioning + ticker-analysis panels, TradingView link row, command-line, error boundary). All engine/book reads labeled; no fabricated data. |
| `dashboard/src/db/index.ts` | SQLite/Drizzle connection — lazy init, table creation, idempotent migrations. |
| `dashboard/src/db/schema.ts` | Drizzle ORM schema — the news/story tables. |
| `dashboard/src/hooks/useEngineData.ts` | React hooks against `/api/engine` — engine data, ticker analysis, committee review. |
| `dashboard/src/lib/utils.ts` | `cn()` Tailwind class-merge helper. |
| `dashboard/src/types/index.ts` | Shared TypeScript types for the dashboard. |
| `dashboard/src/instrumentation.ts` | Next.js instrumentation hook — boots the news ingestion cron (nodejs runtime only, double-start guarded) and seeds default categories. |
| `dashboard/src/services/briefing-generator.ts` | Generates/persists morning/evening/breaking briefings. |
| `dashboard/src/services/edgar.ts` | SEC EDGAR client — ticker-to-CIK and recent filings. |
| `dashboard/src/services/entity-extraction.ts` | Entity extraction — Ollama NLP with regex fallback. |
| `dashboard/src/services/exposure-ranking.ts` | Exposure-first story ranking against holdings/watchlist/factors. |
| `dashboard/src/services/impact-analysis.ts` | Impact analysis — factor/horizon/sentiment tagging. |
| `dashboard/src/services/macro-data.ts` | FRED API client for macro time series. |
| `dashboard/src/services/market-data.ts` | Finnhub quote client and market-snapshot cache. |
| `dashboard/src/services/news-categories.ts` | News-category taxonomy and keyword/ticker matching. |
| `dashboard/src/services/rss-feeds.ts` | Static config of financial RSS feed sources. |
| `dashboard/src/services/news-alerts.ts` | Ingest-time news-trigger alert evaluator (watchlist symbol match) — writes alert rows; no price-alert claims. |
| `dashboard/src/services/news-cron.ts` | node-cron schedule for the ingestion pipeline — started once from `instrumentation.ts`, module-guarded, `NEWS_CRON=0` opt-out. |
| `dashboard/src/services/universe-cache.ts` | Server-side cache of the engine's S&P-500 universe (+ held-book symbols) used to validate extracted ticker entities. |
| `dashboard/src/services/rss-ingestion.ts` | RSS feed parser/ingester with ticker extraction. |
| `dashboard/src/services/scheduled-ingestion.ts` | Orchestrates the multi-step ingestion pipeline. |
| `dashboard/src/services/story-clustering.ts` | Story-graph clustering — Jaccard dedup, contradiction detection. |

## `data/` — data layer (Bloomberg-CSV provider + feature pipeline)

| File | Purpose |
|---|---|
| `data/__init__.py` | Package re-export hub for the data layer. |
| `data/bloomberg.py` | `BloombergConnector` — live Bloomberg Terminal connector via `blpapi` or Excel COM. |
| `data/bloomberg_import.py` | One-shot Bloomberg OHLCV CSV loader plus per-ticker feature computation. |
| `data/bloomberg_loader.py` | Per-ticker Bloomberg CSV parsers with column normalization. |
| `data/consolidated_loader.py` | `ConsolidatedBloombergLoader` — loads the consolidated `sp500_*.csv` panels. |
| `data/broad_pull_loaders.py` | `BroadPullLoader` — read-only Phase-0B loaders for the integrated broad-pull datasets under `data/bloomberg/broad_pull/` (gz handling, float32 downcast, logged winsorization of the manifest's outlier-flagged columns, lazy per-ticker access via normalized symbols). **Dormant — nothing consumes it (§2-safe)**; see `docs/WIRING_CAMPAIGN.md` Phase 0B. |
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
| `data/bloomberg/broad_pull/` | Integrated net-new broad-pull Bloomberg datasets (the ~25 logical / 27 files from `staging/` on branch `claude/bloomberg-broad-pull-2026-06-17`, mirroring its bucket structure): IV skew surface (`.gz`), macro calendar + releases, vol/rates/cross-asset wide series, per-name panels (returns+bid/ask, IV-term+RV `.gz`, beta/shares, fundamentals, estimates, valuation, options-sentiment), dividend-PIT, short interest, and the ratings/GICS snapshot. Read by `data/broad_pull_loaders.py`; **not yet consumed**. Census in `docs/DATA_INVENTORY.md` §6. |
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

## `docs/` — documentation set

| File | Purpose |
|---|---|
| `docs/DATA_ENGINE_AUDIT_2026-06-07.md` | Phase-1 data + engine audit findings (discovery; asserts nothing). Generated by `scripts/audit_data_engine.py` on `origin/main` @ frontier 2026-06-04: connector inventory, capability map, coverage matrix (referential gaps / 2026-03-23 reconstitution seam / survivorship / BK→BNY re-ticker), the frontier-pinned ranker probe (511→480 produced, 31 dropped, 0 silent), and the ranked weakness report (3 HIGH / 5 MEDIUM / 5 INFO-LOW). The Phase-1 checkpoint before tests. |
| `docs/BLOOMBERG_PULL_LIST.md` | No-code "what data we need" acquisition checklist for the Bloomberg terminal (compiled 2026-06-09): per-dataset universe, fields (FLDS-verify), history/frequency, and target file under `data/bloomberg/`, grouped Tier-0 (correctness blockers) then by category (volatility/options, equity pricing & microstructure, fundamentals & estimates, credit/ownership/positioning, macro/rates/FX/cross-asset, sentiment/ESG/news/events). Execution-spec companion to `docs/DATA_ACQUISITION_ROADMAP.md` (rationale + §2) and `docs/DATA_INVENTORY.md` (what we already hold). |
| `docs/BLOOMBERG_TERMINAL_NEXT_SESSION.md` | Operator action-list for the next logged-in Bloomberg Terminal session (authored 2026-06-27, post Phase-1): prerequisites; Priority-1 Phase-0A frontier bump (OHLCV via `scripts/pull_ohlcv.py` end_date edit + the producer-less ATM-IV monolith refresh that closes the #378 staleness gap) with the coupled re-baseline/frontier-bump tail; Priority-2 capability unlocks ranked by engine ROI (the Theta real-premium producer the skew surface is EV-inert without, macro calendar / short-interest already on main, IV-surface breadth); and the do-not-re-pull list. Focused next-session companion to `docs/BLOOMBERG_PULL_LIST.md` (full field checklist). |
| `docs/DATA_ACQUISITION_ROADMAP.md` | Bloomberg "pull-broadly" data-acquisition catalog (compiled 2026-06-09) for the wheel EV engine: a Tier-0 critical-gap list (skew reactivation, PIT dividend-yield, macro calendar, deep UST curve, corp actions) plus a 6-category exhaustive catalog (volatility/options, equity pricing & microstructure, fundamentals & estimates, credit/ownership/positioning, macro/rates/FX/cross-asset, sentiment/ESG/news/events) — 141 items with Bloomberg vehicle (FLDS-verify), engine consumer, benefit tier, feasibility, and §2 role — plus the no-pull wiring/implementation list. Companion to `docs/DATA_INVENTORY.md`. |
| `docs/DATA_INVENTORY.md` | Verified inventory (2026-06-08) of every dataset on disk — Bloomberg monolith CSVs (`data/bloomberg/`), the gitignored deep-history archive (`data/bloomberg/deep/`), Theta option/market data (`data_processed/theta/`), and derived stores — with type/title, file-read date ranges, and row/file/ticker counts. Regenerate via `scripts/inventory_data.py`. Records where the Google Drive `swe-deep-history/` archive maps on disk. |
| `docs/WIRING_CAMPAIGN.md` | Dependency-ordered plan to wire the ~27 staged broad-pull datasets into the EV engine: per-dataset rows (dataset · engine consumer · §2 role · EV-moving?→re-baseline-coupled · banked-at · roadmap/§9 ref) across Phase 0 (integration — 0A frontier-refresh tails EV-moving, 0B loaders plain), Phase 1 ((E) trio ceremony), Phase 2 (skew panel), Phase 3 (by-consumer), Phase R (single re-baseline). Flags the trio lane-claim ceremony / §2 panel / held gates. Grounded in `staging/BROAD_PULL_MANIFEST.md` + `docs/DATA_INVENTORY.md`. |
| `docs/DATA_POLICY.md` | Data tiers, provider matrix, what never enters git, point-in-time discipline, refresh procedures. |
| `docs/DATA_TEST_AUDIT_2026-06-09.md` | Phase-1 data-layer TEST-coverage audit (discovery; asserts nothing). Deeper round building on `DATA_ENGINE_AUDIT_2026-06-07` (W1–W13) + the #358/#366 Phase-2 suites: confirms the capability map (corrects two precedent claims — credit is OFF the EV path, R9's sector cap uses a hardcoded `DEFAULT_SECTOR_MAP` not `gics_sector_name`), maps real-data vs synthetic test coverage per data→engine path, reconciles W1–W13 (all closed/tracked), and registers 15 new weaknesses W14–W28 (13 (T) landable, 1 (E) / 1 (D) tracked) ranked into 5 surface-grouped test PRs. Evidence reproducible via `scripts/audit_data_engine.py` + `scripts/audit_data_tests.py`. |
| `docs/PHASE1_E_TRIO_EXECUTION_SPEC.md` | Turnkey, file:line-accurate implementation spec (verified vs `origin/main`) for the three (E) trio / risk-gate fixes — **#372** (R9 sector cap → real `gics_sector_name`, not `DEFAULT_SECTOR_MAP`), **#369** (extend the #363 IV gate to the fundamentals-fallback path), **#378** (IV-staleness gate on `_resolve_pit_atm_iv` + rate-fallback divergence): per-fix finding/fix/§2-role/CEREMONY, the exact characterization tests to flip + new tests to add, the §2-panel checklist, and the **#378-before-0A** ordering. Companion to `NEXT_DATA_SESSION_RUNBOOK.md` Phase 2 + `WIRING_CAMPAIGN.md` Phase 1. PLAN ONLY — no engine code. |
| `docs/PHASE2_SKEW_EXECUTION_SPEC.md` | Turnkey, file:line-accurate implementation spec (verified vs `origin/main` @ `21e489d`) for `WIRING_CAMPAIGN.md` Phase 2 — wiring the moneyness IV skew surface (`broad_pull/iv_surface`, 1.94M rows) into a new `data_connector.get_iv_surface` accessor → `skew_dynamics` (sizing `skew_mult`) + the `option_pricer`/`ev_engine` BSM-IV seam (`sigma=trade.iv` :376). Carries three adversarially-found campaign corrections: `vanna/charm/volga` already exist (scalar path), butterflies are absent (SVI `volatility_surface` only), and the surface has no true 25Δ column (5×5 `{90..110}`; `{90}`→put/`{110}`→call proxy). Includes the §2-panel checklist (lane-claim CI is silent → panel is sole gate), the ATM single-source reconciliation + `#378`-first ordering, and Phase R coupling. PLAN ONLY — no engine code. |
| `docs/IBKR_IMPORT.md` | IBKR PortfolioAnalyst PDF → read-only viewer artifacts (Phase 1): `scripts/ibkr_import.py` mapping, run-time reconciliation, and honest limitations (no per-fill dates, TWR-index monthly NAV, gross premium, null margin), plus scope/FX tagging. |
| `docs/DASHBOARD_TERMINAL.md` | Runbook for the role-dedicated **Dashboard** terminal (activated by "You are responsible for the Dashboard"): live IBKR portfolio-viewer ownership, command vocabulary ("update"/"update prices"/"update trades"/"status"/"show"/"probe"), the three read-only IBKR channels (cloud connector / IB Gateway Pro API / Flex Web Service incl. the query id), the refresh pipeline, guardrails, and gotchas. |
| `docs/IBKR_EV_CALIBRATION.md` | Phase 3: PIT, strike-matched, hold-to-expiry calibration of the engine's `prob_profit`/`prob_assignment`/`ev_raw` against the operator's 456 real S&P-500 wheel legs (175 puts + 281 covered calls) — per-leg + combined reliability + Brier/ECE + Wilson CIs; two opposite weaknesses (put top-bin over-confidence, call broad under-confidence); the Bloomberg-NFLX scale data-quality finding. Observational (§2/§3). |
| `docs/DATA_LAYER_ACTIVATION_ROADMAP.md` | Plan (2026-06-05) for activating the campaign's survivorship-free 1990–2026 layer: the on-the-bytes verified inventory (33 refresh CSVs + 13 deep panels @ refresh `6bb3399` / deep `e7818f4`), the gap (connector reads only 2018+ monoliths), and the prioritized R0–R7 roadmap with effort/risk/§2-touch/re-baseline + the R1 merge hazard. Safe prep done; merge/re-baseline/connector deferred. |
| `docs/DATA_LAYER_DEEP_READ_DESIGN.md` | Design (2026-06-05) for the connector deep-read (`_load` assembly of monolith ∪ deep ∪ delisted via a slice manifest, ticker/dedup precedence, memory/perf, opt-in flag) + the survivorship-aware backtest harness (PIT membership reuse) + Theta-chain cost-model fallback + the two audit code fixes. §2-safe by construction (assembly below `get_*`; trio untouched). Companion to the activation roadmap. |
| `docs/PREMIUM_CORRECTION_PILOT.md` | Observe-only pilot measuring real-mid − BSM(iv) premium correction (skew-driven under-pricing, NOT VRP) and the market-vs-engine tail-probability calibration gap; labeling discipline + what the 3-name post-split pilot can/cannot settle. |
| `docs/REBASELINE_D19_D21_RECAL_SCOPE.md` | Planning-only scope for the coordinated **D19** (exit-cost netting) + **D21** (forward-distribution horizon-units) + **probability-recalibration** re-baseline. Covers the entanglement (D21's over-long horizon deflates `prob_profit`, masking top-bin over-confidence; fixing it makes the measured gap worse), the dependency order, the full `prob_assignment`/`prob_profit` blast radius (backtests, calibration band, S-claims, premium-correction pilot risk axis, R1/R5/R11), the LOCO recalibration re-run on D21-corrected probabilities, and the decision-trio test + §2 plan. Draft for operator review; not yet executed. |
| `docs/SUPERVISED_BLOCK_WORKLIST.md` | Consolidated routing checklist (2026-06-11) for everything operator-gated: Block A (Terminal/data session — data queue #339/#354/#355/#357, reserved (E)s #369/#372/#378, NFLX mis-scale, option-volume capture, post-refresh hygiene) strictly BEFORE Block B (coordinated EV re-baseline — D19+D21+recalibration per `REBASELINE_D19_D21_RECAL_SCOPE.md`, brain-audit M2 widening coverage + M4 size-impact wiring, #402 re-pin, fingerprint gap). Strike items as they merge; archive when both blocks land. |
| `docs/REPO_MAP.md` | The single "where / what / authoritative" router: question→owning-doc map, the §2 authority block, the `src/` per-file truth table, and the layer→test lookup. Read this first to avoid opening 3 nav docs for one question. |
| `docs/REPO_EFFICIENCY_AUDIT.md` | Repo structure & reading-efficiency audit (2026-05-31): the evidence behind `REPO_MAP.md`, the tests/ dedup verdicts, and the phased execution plan. |
| `docs/LAPTOP_SETUP.md` | Machine bring-up — cloning, env, Theta Terminal, regenerating local data. |
| `docs/FRESH_LAB_BOX_SETUP.md` | Fresh/transient lab-box bring-up runbook for pulling Bloomberg from a machine that has a Terminal — the Bloomberg-pull counterpart to `LAPTOP_SETUP.md` (clone-before-orient ordering, the worked 2026-06-02 lab-box example, no-recall steps). |
| `docs/LAUNCH_READINESS.md` | The launch-blocker gate checklist consolidating the EV invariant, the four authoritative routes, and the dossier rules. |
| `docs/TESTED_SURFACE_MAP.md` | Per-module tested-surface map + top-N coverage-gap ranking, generated from `coverage.json` by `scripts/generate_tested_surface_map.py`. Answers "what is and isn't covered by the test suite" in one file. `coverage.json` is regenerated locally / in CI (gitignored, not committed); regenerate this doc after a meaningful coverage shift. |
| `docs/PRODUCTION_READINESS.md` | The real-money deployment gate. Consolidates findings from the S22 / S27 / S32 backtests + the four review docs (#194 / #195 / #197 + S32) into one answer to "should we deploy this engine against a real brokerage account?" Names three blockers (F4 tail-risk widening, D17 live-wire to `engine_api.py`, strategy capacity at >$100k), four caveats, and a deployment decision matrix. Complementary to `LAUNCH_READINESS.md` (code-quality merge gates). |
| `docs/THETA_INSTRUCTIONS.md` | Quick reference for refreshing every Theta-sourced dataset. |
| `docs/THETA_USAGE.md` | Theta Terminal v3 per-endpoint reference, tier behaviour, wire-format codes. |
| `docs/THETA_PULL_SESSION_NOTES.md` | Operational checklist and gotchas for a laptop Theta pull. |
| `docs/THETA_PULL_DATA_LOG.md` | Running prepend-only log of what the `pull_theta_option_history.py` larder has pulled — names, titles, and date spans only (no option data). One snapshot prepended ~4-hourly by the session-only Theta health-monitor loop. |
| `docs/THETA_ENTITLEMENT_RETEST_2026-06-17.md` | Live ThetaData v3 entitlement re-probe (2026-06-17) confirming greeks/IV history is 404/not-entitled, with the ranked next-pull decision (delisted survivor-bias, index GEX, universe expansion, BRKB). |
| `docs/THETA_ENRICH_RUNBOOK_2026-06-17.md` | Operational runbook for the 5-phase 2026-06-17 Theta enrichment (BRKB / deep365 / index-reference / delisted / universe expansion) producing the `data_processed/theta/option_history*` staging trees, with the verified on-disk outcome. |
| `docs/THETA_LARDER_SCOPE.md` | Scope + caveats for the `pull_theta_option_history.py` larder: top-150 by 2018→now turnover, 2018→now, all-strikes, 90-day lookback, SPY/QQQ reference-only. Documents the **survivor-bias caveat** (ranked on current 503 → backtests inherit survivor bias; delisted once-liquid names excluded until a PIT-membership source lands) and the deferred extensions (2016–17, >90d lookback, delisted backfill, IV/Greeks/tick phases). |
| `docs/TRADINGVIEW_INTEGRATION.md` | Parent guide for the two TradingView roles — engine bridge and analyst workspace. |
| `docs/IBKR_LIVE_BOOK_INTEGRATION.md` | Design doc for the read-only IBKR live-book feed (D24, gate-arming), the un-adopted exit-evaluator scope frontier (D25), and the read-only performance viewer (D26). Defines the point-in-time snapshot schema (§2.2), universe discipline (§2.3), and the viewer architecture (§6). |
| `docs/TRADINGVIEW_MCP_INTEGRATION.md` | Design contract for the MCP-driven chart provider. |
| `docs/GREEKS_UNIT_CONTRACT.md` | Canonical Greeks unit conventions. |
| `docs/CODE_REVIEW_2026-05-30.md` | Full-codebase code-review remediation ledger (2026-05-30): verified findings, fixes shipped vs deferred (D19/D21), and the per-item remediation log. |
| `docs/GOVERNANCE.md` | Model governance framework. |
| `docs/MODEL_CARDS.md` | Per-model documentation cards. |
| `docs/USAGE_TEST_LEDGER.md` | **FROZEN** (2026-05-29, D14 extension) — its S1–S46 entries were split verbatim into per-task fragments under `docs/worklog/`; now a banner + scenario→fragment map. New usage records are worklog fragments (`scripts/new_worklog.py`), indexed by `docs/worklog/INDEX.md`. |
| `docs/worklog/*.md` | Per-task **worklog fragments** (doc redesign, D14 extension): one file per task/scenario with front-matter + a fixed *what-we-tried / worked / didn't / fixed* body. Includes `README.md` (format spec), `_template.md`, the migrated `sNN-*.md` scenario records, and the generated `INDEX.md`. Generated/validated by `scripts/gen_worklog_index.py`. |
| `docs/PARALLEL_SESSIONS.md` | How the repo is worked by N parallel Claude Code terminals — roles, lanes, coordination board. |
| `docs/MAJOR_SESSION_PROMPT.md` | Reusable handoff prompt for a fresh Major Session (the allocator/coordinator role) — durable role contract + how to recover live state from board #113 / `git log`; pins no decaying snapshot. Companion to `docs/PARALLEL_SESSIONS.md`. |
| `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md` | S27 follow-up to PR #178's `ENGINE_BACKTEST_2022_2024.md`: re-runs the same 2022-2024 backtest against the post-fix engine (`claude/fix-ranker-iv-pit-aware` `d26a8d6`). Side-by-side ρ / quartile / per-year / tail-episode comparison. Verdict: signal preserved at ρ=0.22 (halved); 2022 bear actually stronger; F4 tail-risk gap confirmed real. |
| `docs/ENGINE_BACKTEST_S32_FRICTION.md` | S32 — $1M friction-modeled simulation closing S22 Caveat 3. Same window / universe / engine as S27 but 10× capital with three-layer friction overlay (bid/ask + commission + assignment slippage). Three parallel WheelTracker instances per friction level. Headline: friction drag is 0.27% NAV (much smaller than S22's "2-5% per leg" worst case); but capital deployment averages 10.8% at $1M, so engine returns +1.85% vs SPY +24% — the +27pp-over-SPY narrative inverts at scale. |
| `docs/ENGINE_BACKTEST_S34_UNIVERSE.md` | S34 — Universe expansion to 100 SP500 tickers at $1M (closes PRODUCTION_READINESS Blocker B3). Tests S32 F3's hypothesis: expand universe 24→100, leave everything else identical. Result: engine +35.61% NAV (full friction) vs SPY ~+24% = **+11.6pp OVER SPY** (vs S32's −22pp UNDER SPY). 34pp swing on universe size alone. ρ = 0.33 (higher than S32's 0.19); 22.1% capital deployment (2× S32); zero BP rejections. Universe expansion materially closes the capacity gap; multi-contract / strategy-stack remain candidates for further deployment but are not required. |
| `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` | S35 — 2018-2020 out-of-window cross-validation against S22 / S27. Same 24-ticker universe and $100k capital; only the time window changes. Headline: signal generalizes (ρ = 0.50 in 2020 — double S27's 0.22) but dollar-alpha does NOT (engine +3.57% vs SPY ~+45% = −41pp underperformance). The "+27pp over SPY" property turns out to be both $100k-specific (per S32) AND 2022-2024-window-specific. Plus the discovery of a 504-day OHLCV history gate. |
| `docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md` | S38 — 5-year multi-window backtest at $1M / 100 tickers / 2020-01-02 → 2024-12-31. Closes the gap S34 ($1M/100t/2022-2024 = +11.6pp) and S35 ($100k/24t/2018-2020 = −41pp) opened. Headline: engine returns +33.18% (full friction) vs SPY ~+85% = **−52pp UNDERPERFORMANCE** over the 5-year window. Confirms dollar-alpha is window-specific. Realized executed P&L is **negative** (−$28,647); all NAV growth is equity-beta on assigned positions (108.6%). Spearman ρ = 0.358 (signal generalizes; never negative per year). COVID refusal rate 97.8% (defensive behavior preserved at scale). |
| `docs/ENGINE_BACKTEST_S40_ROLLING_MULTIWINDOW.md` | S40 (PR #264) — rolling multi-window backtest at 100 tickers / $1M with 3 new start dates (2021/2022/2023, all ending 2026-02-06). Cross-referenced with S34 and S38 for 5 multi-year measurement points at $1M/100t spanning **−85pp to +10pp engine-vs-passive**. Headline: S38's −52pp is **NOT 2020-2024-specific** — it's a general property at $1M/100t scale, modulated by bull-year share of the window. The pattern is monotonic: pure-bull windows show −60 to −85pp underperformance; bear-included windows show parity or modest outperformance. The −52pp gap is structural to the strategy's limited deployment (15-23% NAV), not engine-defective. |
| `docs/ENGINE_BACKTEST_S43_ROLLING_MULTIWINDOW.md` | S43 (PR #270, Terminal C) — rolling 5-window backtest at $1M / 100 tickers on post-#260 engine. 4 runnable windows (W1=2018-2022, W2=2019-2023, W3=2020-2024 = S38 re-run, W4=2021-2025); 3 pre-COVID windows infeasible due to OHLCV starting 2018. Headline: engine never beats Univ-EW; range **−51pp to −104pp** across windows. ρ window-invariant 0.356-0.378. Per-year ρ positive in 16/16 cells. PR #260 signal-preserving on W3 ⟷ S38 comparison (Δρ −0.002). R10 would-fire 3.7-4.5% of executed opens; max single-name exposure reached 20-25% NAV vs the 10% cap. |
| `docs/ENGINE_BACKTEST_S44_S38_POSTF4_RERUN.md` | S44 (PR #271) — S38 re-run on post-F4 engine (SHA `56d8e5c`, post-PR #260 + #262). Tests whether F4 fix closes any of S38's −52pp engine-vs-passive gap. **Headline: hypothesis FALSIFIED.** Engine return +0.56pp (33.18% → 33.74%); ρ −1.0% (0.358 → 0.354); executed +0.7% (305 → 307); realized total −\$4,082 worse. The gap is structural to limited deployment, NOT to a missing tail-risk widening. §2 invariant CLEAN both pre- and post-F4. |
| `docs/ENGINE_BACKTEST_S34_REBASELINE_POST260.md` | S34 snapshot re-baseline against post-PR #260 + post-PR #262 engine (`56d8e5c`) + R10 firing analysis. Closes the slow-lane CI's `test_backtest_matches_snapshot[s34_universe_100t_1m]` assertion at $1M / 100 tickers / 2022-2024. Headline: snapshot row_count identical (10,911); ρ −1.18% relative (vs S27's −3.3% — wider universe absorbs the F4 reshuffle); NAV −$7,786 / −0.58%. **R10 fires 368× across 3 years on AZO (238) + BKNG (130) only**; 96 of 98 in-universe tickers never engage R10 — **inverts the S32 re-baseline's F5 ("R10 non-binding at $1M / 24t") at the 100t universe**. Wiring R10 in would cost $79,387 of realized executed P&L on this window (insurance vs optimization trade-off). Engine vs SPY: +11.6pp → +9.94pp (−1.66pp); engine vs same-universe Universe-EW: unchanged at +11.63pp. pytest -k s34 2/2 PASSED in 4h18m. Sn allocated at merge per `docs/PARALLEL_SESSIONS.md` rule 7. |
| `docs/verification_artifacts/s45_r10_firing_driver.py` | R10 firing probe — read-only client of `engine.portfolio_risk_gates.check_single_name_cap` that operates on the S34 rank_log to compute (a) per-ticker static R10 verdict (max_fcn vs 10% NAV cap) and (b) counterfactual harness-rule replay tracking R10 fires per ticker / per year / total refused notional. Re-runnable standalone. Filename preserves its pre-rule-7 working prefix; it's a tool artifact identified by its content. |
| `docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_dollar_impact_driver.py` | R11 dollar-impact A/B backtest driver — post-ship validation of the elevated-vol top-bin size-down (#306/#307, heavy-verify I11). Runs two arms over one shared daily rank: `suppressed` (the literal pre-R11 S38/S43 open policy) vs `active` (same + R11's exact gate — `prob_profit > R11_TOP_BIN_PROB` AND `vix(as_of) > R11_VIX_THRESHOLD` AND `ev > MIN_PROCEED_EV_DOLLARS`, all imported from `engine.candidate_dossier` so they cannot drift; VIX from `connector.get_vix_regime(as_of)["vix"]`, the exact wheel_runner call). Backfills the daily quota past R11-blocked opens. Reuses `backtests/regression/_common` helpers + `universes.UNIVERSE_100`. Emits per-window/per-regime/per-VIX-bucket Δ NAV + Sharpe + the averted-vs-forgone counterfactual P&L of the blocked set + §2 scan. Observation-only; §2-safe (R11 only removes opens). `--analyze` re-emits markdown tables from saved artifacts. |
| `docs/verification_artifacts/r11_dollar_impact_2026-06-01/R11_DOLLAR_IMPACT_FINDINGS.md` | R11 dollar-impact findings — full write-up. Verdict: R11 is targeted insurance for the 2022-style sustained grind-down (averts ~$165-269k CSP-leg loss, ~50% assignment) but lowers Sharpe in both windows, net-costs the book in a window with a 2020-style V-crash (W3 −$37.6k/−3.76pp), helps where the 2022 bear dominates (W4 +$21.7k/+2.17pp); the held-to-expiry "averted" number over-states full-wheel value because blocking entry forecloses the wheel recovery leg. |
| `docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_w3_analysis_RAW.txt` | Captured `--analyze` stdout for W3 (2020-01-02→2024-12-31). Companion observable for `R11_DOLLAR_IMPACT_FINDINGS.md`. |
| `docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_w3_summary.json` | W3 machine-readable summary (per-arm NAV/Sharpe/opens/assignments, blocked-set counterfactual, per-regime + per-VIX-bucket, §2 scan, fingerprint). |
| `docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_w4_analysis_RAW.txt` | Captured `--analyze` stdout for W4 (2021-01-04→2025-12-31). Companion observable for `R11_DOLLAR_IMPACT_FINDINGS.md`. |
| `docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_w4_summary.json` | W4 machine-readable summary (same schema as W3). |
| `docs/ENGINE_BACKTEST_S32_REBASELINE_POST260.md` | S32 snapshot re-baseline against post-PR #260 + post-PR #262 engine (`56d8e5c`). Closes the slow-lane CI's `test_backtest_matches_snapshot[s32_friction_24t_1m]` assertion at $1M / 24 tickers / 2022-2024. Headline: snapshot row_count identical (5,944); ρ −3.33% relative; NAV +$2,376 / +0.22% (marginal dollar-positive — refused trade was net-negative-realized). **F5: R10 structurally non-binding at $1M / 24t / 1-contract** — max position notional ~$20-70k vs $100k cap; the F4 fix's bundle is effectively a no-op at this scale (becomes binding at the wider universe, see companion S34 re-baseline). pytest -k s32 2/2 PASSED in 110m. Sn allocated at merge per `docs/PARALLEL_SESSIONS.md` rule 7. |
| `docs/ENGINE_REVERIFY_S46_POST_F4_R10.md` | S46 — re-verification of the closed tests (S27 / S30 / S31 / S33 / S35 / S36 / S37 + the realism + F4 baseline batteries) against the post-#260 / post-#262 engine (`56d8e5c`). Snapshot harness deferred (S27 1h45m no-completion, S35 follow-on Sn); on the cheaper evidence, all surfaces pass with two explainable drifts (S31 V1 `credit_multiplier` 0.80 → 1.0 from FRED adapter silent degradation; S37 crisis-day ρ +0.0734 consistent with #260's differential widening) and one sharp finding (FRED adapter empty-series crash swallowed by `wheel_runner.py`'s broad except — filed for follow-up). |
| `docs/HEAVY_R10_STRICT_SCALE.md` | HT-D — R10 strict-mode at $1M / 100t scale (2020-2024). First-ever backtest with `require_ev_authority=True` + attached `PortfolioContext`. A/B harness: loose tracker (S38/S44 baseline) vs strict tracker (D17 hard-blocks ON) sharing one daily SP rank call. Measures actual R10 / R9 / portfolio-delta / Kelly binding rates + NAV/return delta vs loose. Pilot finding: `portfolio_delta_breach` is the dominant binding gate (96.7% of refusals); R10 fires exactly on BKNG / AZO as S44 predicted (single-contract entry > 10% NAV). |
| `docs/F4_TAIL_RISK_DIAGNOSTIC.md` | Mechanical characterisation of why `prob_profit = 0.8333` stayed constant across COST's 31.5% drop in April-May 2022 (the F4 finding from S22 / S27 / S32). Verified live: at the engine's default `lookback_years=5.0` with `non_overlapping=True` at 35-day horizon, only ~30 samples populate the empirical forward distribution; advancing `as_of` by 14 days does not add new sample points. Proposes Fix A + B1 (shorter lookback + regime-conditioned distribution widening) with a definition-of-done checklist for closing `PRODUCTION_READINESS.md` §3 Blocker B1. |
| `docs/HEAVY_VERIFY_2026-06-27_DATA_WIRING_RELIABILITY.md` | Mac-terminal heavy-verify findings doc (#436): data-wiring accuracy + engine output-realism + `prob_profit` calibration + risk-free/PIT + tail/CVaR reliability (W1–W5). Validation-only (no engine-behaviour edits). Records the green baseline, the per-source pass/fail table, the confirmed BKNG/CVNA 2026-03-23 split-scale defect (D-W1-1) + the refuted NFLX ~10× claim, and the W2–W5 realism/calibration results. Each claim carries its data path + CI. Reproduced by `scripts/audit_data_wiring.py` (+ the W2–W5 drivers) → `docs/verification_artifacts/data_wiring_2026-06-27/`. |
| `docs/verification_artifacts/data_wiring_2026-06-27/*` | Machine-readable per-check JSON sidecars + drivers for the #436 Mac heavy-verify (W1–W5): `w1_ohlcv.json` (universe split-scale sweep), `w1_vol_iv.json` (served IV-band reconciliation), `w1_treasury.json` (coverage + decimal units), `w1_generic_sources.json`, `w1_summary.json`, and the W2–W5 realism/calibration/RFR/tail artifacts. Persisted before pretty-print so a console crash never loses compute. Re-emit with `python scripts/audit_data_wiring.py`. |
| `docs/verification_artifacts/README.md` | Conventions + index for the verification-artifact directory. Captures the doc → driver → raw-output relationship so future agents can re-run any verification and diff against the historically-captured output. |
| `docs/verification_artifacts/data_engine_audit_2026-06-07/audit.json` | Machine-readable sidecar for `docs/DATA_ENGINE_AUDIT_2026-06-07.md` — full inventory, capability map, coverage matrix, the raw per-ticker ranker drops (`{ticker,gate,reason}`), distribution-tier histogram, hygiene probes, and the ranked weaknesses. Re-emit with `python scripts/audit_data_engine.py --universe full`. |
| `docs/verification_artifacts/vnv_2026-06-01/funnel_2026-03-20.json` | Captured full-universe funnel + tier-distribution at as_of=2026-03-20 (503→423 survivors; drops event:68/history:11/premium:1; Wilson-CI coverage 98.6%). Companion observable for `docs/VNV_CAMPAIGN_2026-06-01.md` §1–2. |
| `docs/verification_artifacts/vnv_2026-06-01/calibration_full.json` | Captured full-universe prob_profit-calibration + ev_dollars-realism sweep (n=9,612 over 35 regime-spanning as_of dates). Companion observable for `docs/VNV_CAMPAIGN_2026-06-01.md` §3–4. |
| `docs/verification_artifacts/vnv_2026-06-01/profile_150ticker_2026-03-20.txt` | Captured cProfile (tottime) of a 150-ticker scan — connector `comp_method_OBJECT_ARRAY` 10.1s / 742 calls + `normalize_ticker` 2.4M calls + HMM E-step 7.9s. Companion observable for `docs/VNV_CAMPAIGN_2026-06-01.md` §6. |
| `docs/verification_artifacts/realism_verify_driver.py` | Driver for ENGINE_REALISM_VERIFICATION_2026-05-26: 5-ticker smoke + IV PIT match vs Bloomberg file + EV magnitude + F4 reproducibility + 3-anchor refusal check. Read-only client of the production ranker. Re-runnable from any worktree (edit `WORKTREE` constant). |
| `docs/verification_artifacts/realism_2026-05-26_raw_output.txt` | Captured stdout from `realism_verify_driver.py` against `origin/main` @ 9f0afaf. Companion observable for `docs/ENGINE_REALISM_VERIFICATION_2026-05-26.md`. |
| `docs/verification_artifacts/realism_2026-05-28_raw_output.txt` | Captured stdout from `realism_verify_driver.py` re-run against the post-#260 / post-#262 engine (`origin/main` @ `56d8e5c`). Byte-identical to the 2026-05-26 baseline on every probe (5-ticker smoke, IV PIT match, EV-sign, COST 2022-04-25 F4, refusal-behaviour 3-anchor). Companion observable for `docs/ENGINE_REVERIFY_S46_POST_F4_R10.md`. |
| `docs/verification_artifacts/f4_baseline_driver.py` | Multi-case F4 pre-fix baseline driver (COST 2022-04-04, UNH 2024-11-11, AAPL 2026-02-13 control). Captures `prob_profit` / `cvar_5` / `heavy_tail` + recomputed realised 35-day forward returns. Companion to Terminal A's incoming `claude/fix-f4-regime-conditioned-widening` branch — diff the post-fix output against the captured baseline to validate the structural fix. |
| `docs/verification_artifacts/f4_baseline_2026-05-26_raw_output.txt` | Pre-fix baseline output. Headline: COST 2022-04-04 reproduces `prob_profit=0.8333` exactly (F4 doc value is NOT stale; corrects the earlier realism-doc "drift" claim which was a date mismatch). UNH 2024-11-11 the cleanest F4 reproducer: `prob_profit=0.8571` vs realised −20.27%. `heavy_tail=False` on both F4 cases — POT-GPD not firing on realised-tail dates. |
| `docs/verification_artifacts/f4_baseline_2026-05-28_raw_output.txt` | `f4_baseline_driver.py` re-run against the post-#260 / post-#262 engine (`origin/main` @ `56d8e5c`). Headlines: UNH 2024-11-11 widens by `ev_dollars −$6.28 / cvar_5 −3.2%` (F4 fix mechanism reproduces); COST 2022-04-04 unchanged (rv30/rv252 ratio below the 1.30 widening threshold); AAPL 2026-02-13 byte-identical (calm-regime control). Companion observable for `docs/ENGINE_REVERIFY_S46_POST_F4_R10.md`. |
| `docs/verification_artifacts/s41_f4_validation_driver.py` | Driver for S41 — F4 fix validation (post-#260). Four probe sets: Layer 1 (3 named cases), Layer 1c (10 COST 2022-04 unfolding-event dates), Layer 3 (6 calm-regime controls), Layer 0 (direct `rv30/rv252` + widening factor on the PR #260 signal table). Read-only client of the production ranker. Re-runnable from any worktree; path bootstrap is `__file__`-relative. |
| `docs/verification_artifacts/pit_realism_driver.py` | HT-B driver (heavy-verify cycle 2026-05-30) — PIT-correct calibration test across 100 historical as-of dates × 100-ticker `UNIVERSE_100`. For each as_of calls `WheelRunner.rank_candidates_by_ev(as_of=...)` → captures `(strike, prob_profit, ev_dollars, ...)` per candidate → resolves realized 35-day close (via the Bloomberg CSV column-rename fix: CSV `high` is true close) → writes long-form CSV + emits per-window calibration tables. Computes BOTH OTM (matches `PROB_PROFIT_CALIBRATION_2026-05-28.md` convention) AND engine-exact `pnl > 0` calibration. Resumable mid-crash (skip-set on as_of). Re-runnable; raw CSV stays local per heavy-verify cycle spec. |
| `docs/verification_artifacts/s41_f4_validation_2026-05-28_raw_output.txt` | Captured stdout from `s41_f4_validation_driver.py` against `origin/main` @ 56d8e5c. Headlines: all 10 COST 2022-04 dates produce `prob_profit=0.8333, factor=1.0000` (F4 fix does NOT close the canonical case — calm pre-drawdown); UNH 2024-11-11 fires mildly (`factor=1.0121`, `ev_dollars +$108.25` — exact PR #260 match); 6 calm-regime control cells all `factor=1.0` (no spurious caution). Companion observable to `docs/ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md`. |
| `docs/verification_artifacts/r10_strict_driver.py` | HT-D driver. A/B harness: loose vs strict trackers sharing one daily SP rank call. Mirrors `backtests/regression/_common.run_backtest_multi_friction` structure but runs two trackers in parallel (loose: `require_ev_authority=False`; strict: `require_ev_authority=True` with token issuance via `consume_ranker_row` so D17 hard-blocks fire on every open attempt). Captures per-attempt outcome + reason from `tracker._ev_authority_log` delta. `--analyze` mode reads existing out-dir artifacts and emits markdown tables for the report. Read-only on `engine/`. Companion driver to `docs/HEAVY_R10_STRICT_SCALE.md`. |
| `docs/verification_artifacts/r10_pilot_2020-q1apr_raw_output.txt` | Pilot run stdout (2020-01-02 → 2020-04-30, 86 trading days, 26.8min). Captures the day-by-day NAV progression for both trackers + final DONE summary. Headlines: strict +7.7pp vs loose; 17 R10 refusals all on BKNG/AZO. Companion observable for `docs/HEAVY_R10_STRICT_SCALE.md` §3. |
| `docs/verification_artifacts/r10_pilot_2020-q1apr_summary.json` | Pilot `summary.json` with loose / strict / delta metrics, per-year breakdown, `put_refuse_by_reason` + `cc_refuse_by_reason`. Machine-readable companion to the pilot raw output. |
| `docs/verification_artifacts/r10_full_2020-2024_raw_output.txt` | Full 5y run trimmed stdout (1,258 trading days, 7.00h wall-clock). Per-50-day progress prints + final DONE summary. HMM `RuntimeWarning` lines stripped (bookkeeping noise from `engine/regime_hmm.py`, irrelevant to R10 measurement). Headlines: final NAV loose $1,405,794 (+40.6%) vs strict $1,247,668 (+24.8%) = −15.81pp delta; 571 R10 binds, all BKNG (331) + AZO (240). Companion observable for `docs/HEAVY_R10_STRICT_SCALE.md`. |
| `docs/verification_artifacts/r10_full_2020-2024_summary.json` | Full 5y `summary.json` machine-readable companion: loose / strict / delta metrics, per-year breakdown, `put_refuse_by_reason` (`portfolio_delta_breach: 5049, sector_cap_breach: 1, single_name_breach: 571`), `cc_refuse_by_reason` (`portfolio_delta_breach: 1655`), setup with OHLCV SHA256. |
| `docs/verification_artifacts/r10_full_2020-2024_analysis.txt` | Output of `r10_strict_driver.py --analyze --out-dir <full_dir>`. The §2 / §2.1 / §2.2 / §4 / §5 / §6 markdown tables used to populate `docs/HEAVY_R10_STRICT_SCALE.md`. Reproducible without re-running the 7h backtest by pointing `--analyze` at the existing artifacts. |
| `docs/verification_artifacts/persona_walkthrough_driver.py` | HT-A persona-walkthrough driver (heavy-verify cycle 2026-05-30). Single-file read-only client of the production engine; scripts a professional quant trader's end-to-end session at `as_of=2026-03-20` against four operator asks (rank 20, why filtered, size $250k book, downside-if-assigned) + Ask 1b pool-wide F-A1/F-A7 measurement + §2-invariant negative-control battery. Path bootstrap is `__file__`-relative so the driver runs on any clone. Forces UTF-8 stdout via `sys.stdout.reconfigure` so em-dashes / `§` survive a Windows redirect. Companion to `docs/HEAVY_PERSONA_WALKTHROUGH.md`. |
| `docs/verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt` | Captured stdout from `persona_walkthrough_driver.py` against `main @ 56c671d` at `as_of=2026-03-20`. ~380 lines covering Asks 1-4 + Ask 1b pool-wide measurement (336 positive-EV survivors, 316 trimmed by `top_n=20`, 262/336 (78%) percentile-collapse rate, listed non-collapsed names) + §2-invariant battery (D16 leg 1 + leg 2, dossier R1 + R1a, reviewer-never-upgrades-on-perfect-chart all observed upheld). The findings doc cites this file by line for every quantitative claim. |
| `docs/ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md` | S41 — F4 fix validation backtest (post-PR #260). Three concentric layers — unit reproduction of canonical F4 cases (COST 2022-04, UNH 2024-11, AAPL 2026-02-13), full S27 2022-2024 backtest re-run with COST cohort detail + 2022 worst-loss inventory, calm-regime preservation check on 2023-2024. Verdict: PR #260's named-case claims are mechanically reproducible (UNH ev_dollars +$114.53 → +$108.25 exact); fix fires on 12% of probed cells (matches PR #260's 14% claim within sampling noise); overall ρ preserved (+0.1881 → +0.1819); 2022 mean realized per ranked candidate drops 88% (composition shift, executed_trades −22%); calm-regime control cells all factor=1.00; named F4 dollar-damage NOT closed (and PR #260 admits this) — R10 single-name cap (PR #262) is the actual damage-bound. |
| `docs/HEAVY_PERSONA_WALKTHROUGH.md` | HT-A findings doc (heavy-verify cycle 2026-05-30). End-to-end walkthrough of the production engine through the eyes of a professional quant trader at `as_of=2026-03-20`. Asks 1-4 + Ask 1b pool-wide measurement + §2-invariant trace, all cited line-by-line against `verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt`. 9 findings (F-A1..F-A9): 7 SURFACE-class operator-surface gaps (silent `top_n` truncation [pool measured: 336 survivors / 316 trimmed]; `select_book` ranking footgun; `sector='Unknown'` phantom bucket; 4 of 5 multipliers structurally 1.0 on Bloomberg; EVT fields are dataclass defaults; dossier R7-R10 unreachable; `pnl_p25==p50==p75` collapse [pool measured: 262/336 = 78%]) + 2 §2-positive confirmations (D17 `sector_cap_breach` audit-log fidelity; D16/R1/R1a/reviewer-never-upgrades observed upheld). Cross-verified by a paired Verifier Session — top-6 EVs + tail-widening pattern + three sector_cap_breach percentages + D16 token-gate refusal all bit-identical. Read-only; defects are flagged for Major-Session triage, not fixed. |
| `docs/HEAVY_VERIFY_2026-05-31_INDEX.md` | Heavy-verify campaign 2026-05-31 (Major Session) — top-level synthesis answering "where can I trust this engine with money?". TL;DR verdict + trust/distrust tables across I1–I5 + the unifying mechanism (empirical forward distribution lags the regime at transitions → top-bin over-confidence + crash procyclicality). Observe-and-document; `engine/` never modified. |
| `docs/HEAVY_VERIFY_2026-05-31_I1_CALIBRATION.md` | I1 — full-503-universe PIT `prob_profit` calibration (16,005 rows, 74 monthly dates). Calibrated ±5pp in the 0.6–0.85 range; over-confident at the top (.95,1] says 96.5% → realizes 69.5%, Δ−27pp, worst in `crisis` regime); attribution dispute resolved (MISCAL under both conventions). `ev_raw` has ≈0 rank-correlation with realized $ P&L (tail-dominated; +0.16 in ROC space). Cites `verification_artifacts/campaign_2026-05-31/raw_output/i1_*`. |
| `docs/HEAVY_VERIFY_2026-05-31_I2_NET_OF_REALITY_PNL.md` | I2 — capital-constrained multi-regime wheel P&L with REAL Theta bid/ask fills + rf-on-collateral + dividend-adjusted index benchmark (the prior blind spot). Wheel beats passive +26.8pp (2022 bear) / +9.9pp (2020 crash) at ¼ drawdown; ≈matches in 2025; trails −19/−26pp in bulls (capped upside + cash drag). rf-carry is first-order (+8.4pp in 2023-24); within-path friction ~0.5–1.5%/yr. |
| `docs/HEAVY_VERIFY_2026-05-31_I3_STRESS_DISCIPLINE.md` | I3 — stress/discipline. Downgrade-only review contract sound; but (A) risk caps (R9/R10/delta/Kelly) DORMANT by default (`require_ev_authority=False`; 174/73-month book breaches); (B) sector-cap (option-only) vs delta-cap (incl. stock) asymmetry; (D) earnings-gate PIT look-ahead (conservative direction); (E) MATERIAL: procyclical at crash entry — 2020-03-02 flagged 89% positive-EV, realized −$1,305/contract at 82% assignment. |
| `docs/HEAVY_VERIFY_2026-05-31_I4_SECTION2_INVARIANT.md` | I4 — §2 invariant adversarial probe. HELD across all 6 attacks (sign-flip via multipliers, R6+R7–R10 composition, forged ev_row, token gate, R5 boundary, non-finite EV). Negative-EV cannot be made tradeable; `ev_dollars=ev_raw×regime_mult` is sign-preserving; chokepoint is the rank→token→consume wire, not the dossier. Lead-verified (`git diff engine/` empty). 5 residual non-breaching concerns listed. |
| `docs/HEAVY_VERIFY_2026-05-31_I5_PRIOR_CLAIMS.md` | I5 — re-verify three prior claims on the post-#294 engine. (A) MU CC negative-EV despite fat premium: −$812 (was −$1,058 pre-PIT-IV); (B) ranker IV is PIT — A/B vs forced stale-snapshot fallback shows up-to-2.6× EV swings; (C) Bloomberg IV no-skew: `put_iv==call_iv` 100.0000%. Confirms PR #294 left the EV-authority path byte-identical (D19/D21 reverted; CSP/committee fixes off the decision path). |
| `docs/HEAVY_VERIFY_2026-05-31_I6_DEEPENING.md` | I6 (Wave 2) — deepens I1/I3. (W2-A) HMM regime overlay directionally right for `crisis` (de-rated hardest, realizes worst) but over-penalizes `bear`; (W2-B) the engine's ranking adds real selection value — monthly top-10 by any signal beats random/all (+$166–206 vs −$26), `prob_profit` best risk-adjusted (reconciles I1's ≈0 dollar rank-corr: value is in tail-avoidance); (W2-C) out-of-sample recalibration (train 2020-23 → test 2024-26) cuts ECE 3.17→1.29pp and pulls the >0.90 forecast 0.924→0.806=realized → the over-confidence is learnable/fixable. Observe-only. |
| `docs/HEAVY_VERIFY_2026-05-31_I7_ROLL_ECONOMICS.md` | I7 (Wave 3) — roll/management economics: does WheelTracker.suggest_rolls beat hold-to-assignment on challenged short puts? When the engine offers a credit roll (26% of 636 challenged), rolling beats holding by +$195/contract (95% CI [+53,+345], 87% win, t=2.62), avoiding terminal assignment 80% of the time; positive in every regime (cleanest in bear/crash). Caveats: engine declines to roll 74% of the time (discipline), `recommend` flag not discriminative, and a HORIZON-MISMATCH confound (roll extends duration → benefits from recovery in a net-rising market). BSM-synthetic buyback. 1-in-6 stride sample. |
| `docs/HEAVY_VERIFY_2026-05-31_I8_DAILY_RISK.md` | I8 (Wave 3) — daily-marked NAV risk; CORRECTS I2's drawdown magnitudes. I2's monthly marks understated true intra-month drawdown by 1.6x-8x: daily-marked max DD is crash_2020 -20.56% (trough on the COVID bottom 2020-03-23; monthly showed -2.6%) and bear_2022 -10.31% (-6.6% monthly). Defensive edge holds in direction but is ~0.4-0.6x the index drawdown (daily, apples-to-apples), not 1/4. Sharpe halves under daily marks. Reuses I2's exact Sim (monthly reproduction matches). |
| `docs/HEAVY_VERIFY_2026-05-31_I9_FIX_GENERALIZATION.md` | I9 (B-verification gate) — does the calibration fix generalize to an UNSEEN crisis? Tests the recalibration-layer fix (I6-C) via walk-forward + leave-one-crisis-out + regime-holdout on the I1 realized rows. Result: it does NOT generalize — under-corrects crises (2020 crash top-bin 0.84-predicted vs 0.72-realized; regime-holdout 0.84 vs 0.67) and over-corrects benign transitions (2021); the crisis realized rate is unstable across crises (0.37–0.93, 56.5pp spread). Qualifies I6-C (its OOS demo was a benign window). Gate NOT cleared → needs a structural POT-GPD fix (untested), then bundle with D19+D21. Observe-only. |
| `docs/HEAVY_VERIFY_2026-05-31_I10_B1_VS_B2_SCOPING.md` | I10 — routes the over-confidence fix between B1 (probability fix) and B2 (behavioral transition gate) before any prototype. P1: crisis instability is REAL (well-powered 2020 0.57 [0.49,0.64] vs 2022 0.83 [0.73,0.89] = 26pp, CIs non-overlapping; 56pp headline inflated by thin cells). P2: NO simple PIT signal (rv30/rv252, drawdown, rv-accel) achieves the 3-way separation B2 needs — rv_ratio is HIGHEST at the 2020 recovery, the 2022 bear-onset looks calm by drawdown. P3: a crude LOCO gate conflates onset/recovery/sustained-bear. Verdict: neither naive fix clears the bar; route to a multi-feature onset detector (acceptance = 3-way LOCO separation) or a §2-clean risk-budget sizing rule. Observe-only. |
| `docs/HEAVY_VERIFY_2026-05-31_I11_RISK_BUDGET_STUDY_SPEC.md` | I11 (SPEC, observe-only design — execution pending operator nod) — parameter study for the R11 risk-budget reviewer routed to by I10. Picks the VIX-*level* size-down threshold (`sp500_vix_full.csv` `close`, NOT a ratio — I10's `rv_ratio` inverts onset/recovery; VIX level orders them right and was never tested) by leave-one-crisis-out cost/benefit: 2020-onset bleed averted vs 2022/2020-recovery premium forgone (2022 is a false-POSITIVE opportunity cost, not a miss — vol elevated all year). Coarse round-number θ (robust-not-optimal); honesty check scans for any low-level+fat-tail cell (term-structure backwardation as the secondary); **null-result exit → unconditional top-bin haircut, no detector**. R11 itself (downgrade-only, computed regime-matched payload) is a separate gated card; `engine/` untouched here. |
| `docs/HEAVY_VERIFY_2026-05-31_I11_RISK_BUDGET_STUDY.md` | I11 (RESULTS, observe-only) — executes the SPEC. NOT a null result: a VIX-*level* > 25 top-bin (`prob_profit>0.90`) size-down is favorably asymmetric in EVERY well-powered crisis fold (2020 net +$86,366 averting the −$1,305/ctr onset tail; 2022 net +$3,550). {20,22.5,25} survive every fold; θ≥27.5 FAILS the 2022 fold (selectively downgrades the profitable high-VIX months). Robust pick **VIX>25**; **conditional R11 supported** (not the null-result haircut). Spec-correction: top-bin 2022 netted −$14,445 (not the full book's +$5k), so the size-down helps 2022 too. §6 residual gap (named): VIX-level misses idiosyncratic calm-market single-name tails (30 at θ=25, mean −$1,611; 2024-25); backwardation flags only 27%; the high-notional ones are bounded by the armed R10 (#303) — R10+R11 complementary. Companion driver `i11_vix_budget_study.py`. |
| `docs/HEAVY_VERIFY_2026-05-31_REMEDIATION.md` | Actionable, sequenced remediation plan derived from the campaign (observe-and-document; no fix applied). Sorts the findings into 4 categories: A arm risk caps (R9/R10 now, delta/Kelly calibrate first; §2-safe, decision-adjacent — operator held); B over-confidence/procyclicality (EV-authority re-baseline, GATED by I9 — recalibration insufficient, needs structural POT-GPD, bundle with D19+D21); C reframe as a defensive sleeve + stop calling ev_dollars a profit forecast (doc/display); D daily-marking (done via I8); + I3-D earnings-gate PIT (harness). Corrects the triage's I4-vs-I1 mislabel (I4 §2 HELD, nothing to fix). |
| `docs/verification_artifacts/campaign_2026-05-31/` | Heavy-verify campaign 2026-05-31 artifacts (drivers + captured raw output). `campaign_lib.py` (shared data/realizer/benchmark lib), `rank_snapshots.py` (shared monthly full-universe ranked snapshots), `i1_calibration.py`, `i2_pnl.py`, `i3_*.py` (gate dormancy / cap asymmetry / compounding downgrades / earnings PIT / crash tail), `i4_section2_probe.py`, `i5_prior_claims.py`, `_grounding.py`, `_api_probe.py`, and `raw_output/*.{txt,json}` companions. Read-only clients of the production engine; reproducible per the INDEX. Regenerable `snapshots/` + `*.parquet` are gitignored. |
| `docs/REALISM_VERIFICATION_2026-05-28.md` | Live verification of F4 + R9 + R10 changes (PR #268) against `origin/main` @ 56d8e5c. 8 sections of verdict-emitting checks: launch-blocker subset 93/93, broader targeted pass 302/302, calm-regime 5-ticker EV smoke bit-identical to pre-F4 baseline, F4 diagnostic-case live replay (COST 2022-04-04 factor=1.0000, UNH 2024-11-11 factor=1.0121, AAPL 2026-02-13 calm control), R9 sector cap fires `sector_cap_breach` at >25% NAV, R10 single-name cap fires `single_name_breach` at >10% NAV with R10-beneath-R9 safety verified, R1 + R1a end-to-end (blocks `negative_ev` and `ev_non_finite` with distinct verdict_reasons), edge-case fail-closed battery (7/7 pass), engine determinism (bit-identical output across multiple calls). Verdict: engine is bulletproof at the §2 contract level; 8/8 surfaces green; 0 defects. |
| `docs/REAL_DATA_VERIFICATION_2026-05-28.md` | Real-data accuracy verification against raw Bloomberg historical data (PR #273). Five pre-declared external-anchor checks: (A) rv30/rv252 from raw OHLCV bit-identical to engine API (3/3 test cases, delta 0.0000); (B) prob_profit calibration on S38 17k rows — MIXED, top 2 bins (>0.90) MISCALIBRATED >10pp delta; (C) BSM put pricing within 5% of hand-coded textbook BSM (3.37% delta); (D) engine IV bit-identical to raw `sp500_vol_iv_full.csv` `hist_put_imp_vol`; (E) backtest regression S27/S32/S34/S35 reproducible byte-for-byte per A's PR #267. Documents the Bloomberg CSV column-rename quirk (CSV ships `open=HIGH, high=CLOSE, close=OPEN, low=LOW`; connector renames; external reproducers must use CSV `high` as true close). |
| `docs/REVERIFICATION_REPORT_2026-05-26.md` | Terminal A's executive read of the S1–S27 re-verification pass (engine SHA `8a17b0b`, rebased onto `46ddbd4`). Headlines: §2 invariant GREEN across 24 active scenarios; two §2 closures since the original entries (S19 C7b and one other) under PR #204's R1a guard. Persisted in-repo so future agents can recover the exact results and reasoning of the session. Companion sub-notes live inline under each `### Sn` entry in `docs/USAGE_TEST_LEDGER.md`. |
| `docs/VERIFICATION_INDEX_2026-05-28.md` | Master index for the 2026-05 verification campaign (PR #273). Three-section structure: (1) Tested surfaces table (15 rows mapping every verified claim to verification + status + artifact); (2) Campaign arc narrative (7 numbered steps S1→S44 + review PRs); (3) Repeat-the-verification appendix (concrete commands for each anchor check + drift indicator + launch-blocker subset). Plus known limitations + open recommendations + one-paragraph "is the engine real" summary. Built as the single reference for any future agent asking "what verification has been done on this engine?". |
| `docs/VNV_CAMPAIGN_2026-06-01.md` | Engine V&V campaign (efficiency / realism / reliability) on `origin/main` `1c69062`: funnel transparency (503→423, all drops auditable), Wilson-CI coverage 98.6% on the put ranker, prob_profit top-bin over-confidence (mild unconditionally, −0.345 in the 2022 rate-bear regime — reproduces the cockpit's 0.57 crisis ghost), `ev_dollars` ~0 linear corr but valid ranking/sign signal (confirms the cockpit), IV zero-skew (100%), connector ticker-filter efficiency hot spot. Drivers `scripts/vnv_*.py`; raw in `docs/verification_artifacts/vnv_2026-06-01/`. Read-only; no §2 issues. Independently reproduces `PROB_PROFIT_CALIBRATION_2026-05-28.md`'s top-bin finding + adds the regime split, EV-realism, and funnel/coverage. |
| `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` | Multi-backtest prob_profit calibration analysis extending PR #273's S38-only check across 10 configurations (S22, S27, S32, S34, S35, S38 pre/post-F4, S40 W1/W2/W3). Pre-declared standard: ≤5pp = calibrated, 5-10pp = slightly miscalibrated, >10pp = MISCALIBRATED. **Headline: 10 of 10 configs have ≥1 MISCAL bin; the top bin (0.95, 1.0] is MISCAL in 9 of 10 (Δ −5pp to −18pp); mean weighted MAD 6.16pp consistent with PR #197's 7.6%; F4 fix (PR #260) does NOT improve calibration.** Establishes top-bin miscalibration as structural to the empirical-distribution method, not S38-specific. Strengthens the F4 + R10 deployment-bundle framing (R10 is the load-bearing magnitude guard because the engine cannot self-correct top-bin over-confidence). |
| `docs/HEAVY_PIT_REALISM.md` | HT-B (heavy-verify cycle 2026-05-30) — fresh PIT-correct calibration test on the post-#249 / post-#260 / post-#262 engine across 3 sub-windows: in-sample 2022-2024 (matches S27/S32/S34 comparability) + OOS 2020 crisis + OOS fresh 2024-09→2026-02. Tests three competing hypotheses (H1 structural defect / H2 in-sample tuning / H3 post-#249 shift) and reports calibration tables using BOTH the OTM convention (matches `PROB_PROFIT_CALIBRATION_2026-05-28.md`) and the engine's exact prob_profit definition (P[spot ≥ strike − premium]). Companion driver `docs/verification_artifacts/pit_realism_driver.py`. |
| `docs/BACKTEST_REGRESSION_CAMPAIGN.md` | Campaign report for the backtest regression harness (4-PR series: scaffolding → snapshots → CI split → S35 re-baseline). Architecture, methodology, snapshot-vs-doc comparison tables for S27/S32/S34/S35, on-fail re-baseline workflow, compute profile, and known limitations. Reference doc for other agents picking up or auditing this work. |
| `docs/NEWS_REDESIGN_CAMPAIGN.md` | Campaign tracking doc for the 9-PR effort that severs verbal news from the EV decision path and replaces it with structured quantitative layers (earnings calendar, fundamentals, macro). Branch prefix `claude/lucid-davinci-pm15H`; coordination on board #113. Temporal doc — status table updated as each PR lands; structural decisions are in `DECISIONS.md` D18+. |
| `docs/EDGAR_EARNINGS.md` | Rationale + operational notes for the EDGAR earnings layer (campaign PR3/9). PIT story (8-K Item 2.02 is immutable; yfinance leaks lookahead), projection heuristic (median inter-filing delta), SEC User-Agent + 10 req/sec rate-limit, integration preview for follow-up PR3.5. Companion to `scripts/pull_edgar_earnings.py`. |
| `docs/bloomberg_refresh_runbook.md` | Point-in-time runbook for refreshing the Bloomberg connector CSVs. |
| `docs/CASY_BACKFILL_SPEC.md` | Exact Bloomberg pull spec for CASY's pre-2026 OHLCV/vol_iv/liquidity/earnings — the one Bloomberg-gated piece of #339 (everything else reconstructs from git); plus the post-pull integration + re-baseline plan. |
| `docs/NEXT_DATA_SESSION_RUNBOOK.md` | **The single authoritative checklist for the supervised Bloomberg re-baseline session.** Strict ordered scope: Phase 1 universe data (CASY + 10 blue-chip backfills → BK↔BNY collapse, dividends union, UNIVERSE_100 re-derive), Phase 2 the three (E) trio/risk-gate fixes (#372 R9→GICS HIGH, #369 #363 IV-gate fallback clean, #378 IV-staleness gate + rate-fallback) each lane-claimed + held for §2 review, Phase 3 re-baseline-all-4 (captures #363 ev_mean + the (E) frontier impact in one pass) + EXPECTED_FRONTIER/FRONTIER bumps + xfail flips + S34 provisional clear, Phase 4 frontier-coupled test re-picks (full-universe 480/31, W16/W30 JPM), Phase 5 fold the (D) producer pulls (#354/#355/#357/W28) into the same session. Governing rule: everything that moves the frontier/EV output lands before the snapshot re-pin. Companions: `CASY_BACKFILL_SPEC.md`, `DATA_TEST_AUDIT_2026-06-09.md`. |
| `docs/CONTRIBUTING.md` | Contributor workflow guide. |
| `docs/SECURITY.md` | Security policy and best practices. |

## `engine/` — quant + decision layer

| File | Purpose |
|---|---|
| `engine/__init__.py` | Package init re-exporting the legacy quant-layer symbols (pricing, risk, regime, signals, Monte Carlo, portfolio). |
| `engine/ev_engine.py` | `EVEngine.evaluate` — the authoritative probabilistic expected-value computation for short-option trades. |
| `engine/wheel_runner.py` | `WheelRunner` — the orchestrator and authoritative ranker (`rank_candidates_by_ev`, covered-call/strangle rankers, `select_book`, dossier builder); provider selection. |
| `engine/candidate_dossier.py` | The EV-plus-chart `CandidateDossier` artifact and `EnginePhaseReviewer` (the downgrade-only R1–R11 rules). |
| `engine/chart_context.py` | `ChartContext` dataclass and the `ChartContextProvider` protocol. |
| `engine/tradingview_bridge.py` | Pluggable TradingView chart-capture providers (filesystem, Playwright, MCP, chained) and the default-provider factory. |
| `engine/mcp_client.py` | `MCPCLIClient` — the `tv`-CLI subprocess client backing the MCP chart provider. |
| `engine/tv_signals.py` | Deterministic TradingView Pine-parity signal computation and `TVAlert` webhook parsing. |
| `engine/signal_context.py` | Builds the context dicts the signal framework consumes from the Bloomberg data loaders. |
| `engine/signals.py` | Signal-generation framework — IV-rank / trend / profit-target / stop-loss / DTE / event signals and the aggregator. |
| `engine/news_sentiment.py` | `NewsSentimentReader` — reads news-pipeline sentiment from disk. `sentiment_multiplier` is a constant-1.0 stub post-D18 (verbal news severed from the EV path); `get_ticker_sentiment` is preserved as an operator-transparency layer for the dashboard / row dict. |
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
| `engine/ibkr_portfolio_adapter.py` | D24/D26 read-only adapter (OUTSIDE the trio — imports nothing from `ev_engine`/`wheel_runner`/`candidate_dossier`). Turns the point-in-time IBKR artifacts on disk (`data_processed/ibkr/portfolio_snapshot.json` + `portfolio_history.json` + `wheel_ledger.json`; dir overridable via `SWE_IBKR_DATA_DIR`) into the engine types + every performance-viewer payload, reusing `portfolio_tracker`/`wheel_tracker`/`performance_metrics` for analytics and `portfolio_risk_gates` for the live R7–R11 overlay. Observational only — never ranks, never issues EV authority; out-of-universe names are exposure-only; CAD normalized to USD via the snapshot `fx_rates`. |
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
| `scripts/audit_api_smoke.py` | Standalone smoke-test client that hits a running `engine_api.py` and runs domain-grouped backend checks (was repo-root `audit.py`; renamed at the D27 move — it is an API smoke client, not the audit-cycle framework). |
| `scripts/pull_all.py` | Orchestrates every puller in dependency order, skipping steps whose upstream is unavailable. |
| `scripts/ibkr_ev_calibration.py` | Phase 3: PIT EV calibration — replays the operator's real short puts AND covered calls through `EVEngine.evaluate` at each trade's exact strike (reusing the ranker's PIT machinery) and compares `prob_profit`/`prob_assignment`/`ev_raw` to the realized hold-to-expiry outcome; per-leg + combined reliability/Brier/ECE + Wilson CIs + EV-sign split + a moneyness scale gate. Observational (§2/§3); never bypasses the engine. |
| `scripts/ibkr_flex_ledger.py` | Phase 4: re-keys `wheel_ledger.json` from the exact IBKR Flex 'Trades' export (two CSVs) — Open/Close-driven long/short average-cost stock + per-contract option realized with real dates; refreshes history `premium`; reconciles to the p6 book + MTM. Read-only/observational (§2/§3). |
| `scripts/ibkr_gateway_pull.py` | Headless read-only IB Gateway puller (morning-refresh Mode 2): connects `readonly=True`, reads `accountSummary` + `portfolio` + a Forex FX snapshot (no flaky account-update subscription), and feeds the SAME `ibkr_live_snapshot.build_snapshot` transform → `portfolio_snapshot.json`. For a Windows Task Scheduler pre-open run. Read-only/observational (§2/§3); never references an order method; day-change gated null. |
| `scripts/ibkr_import.py` | Read-only importer: IBKR PortfolioAnalyst since-inception PDF → `portfolio_snapshot`/`portfolio_history`/`wheel_ledger` JSON for `engine.ibkr_portfolio_adapter`. Observational (§2/§3); dedups the hidden 2× text layer; reconciles to Ending NAV / deposits / dividends at run time. |
| `scripts/ibkr_live_snapshot.py` | Morning-refresh core: builds the `schema_version:1` `portfolio_snapshot.json` straight from the live IBKR read-only connector (`get_account_summary` + `get_account_balances` + `get_account_positions`) — no PDF/Flex. Parses option `contract_description`, FX-normalizes, derives day-change from per-position `daily_pnl`, joins sector/name/in-universe from the constituents file. Shared by the agent (Mode 1) and headless puller (Mode 2). Read-only/observational (§2/§3). |
| `scripts/dashboard_refresh.py` | Dashboard-terminal one-command live refresh: regenerate the snapshot (from saved connector JSON via `ibkr_live_snapshot`), rebuild the Flex ledger incl. option expiries (via `ibkr_flex_ledger` + the gitignored Flex creds), sync the equity-curve live point, and verify. Backs up before overwriting; never writes the Flex token; READ-ONLY (§2/§3). See `docs/DASHBOARD_TERMINAL.md`. |
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
| `scripts/pull_edgar_earnings.py` | SEC EDGAR pull of Form 8-K Item 2.02 (earnings-release) filings → `data_processed/edgar/earnings_history.parquet`. Append-only by default; `--refresh` merges with the prior parquet so partial-refresh runs never silently drop prior data. Campaign PR3/9 — see `docs/EDGAR_EARNINGS.md`. |
| `scripts/pull_theta_indices_history.py` | Theta pull of VIX-family index OHLC history. |
| `scripts/pull_theta_iv_surface_history.py` | Theta pull of historical IV surfaces with strict partial-coverage rejection. |
| `scripts/pull_theta_option_history.py` | Per-(ticker,expiration) EOD + Open Interest option-history puller (STANDARD tier). `--all-strikes` uses the bulk per-expiration EOD endpoint (~100x fewer calls, all strikes in one request); `--lookback-days` sets the per-contract window (90 = the 0–90 DTE cross-section a wheel/skew fit reads); `--cadence weekly` keeps Friday expirations only (drops 0DTE bloat); `--out-dir` for a reference dir (index ETFs kept out of the ranker). Clamps `start_date` to the 2016-01-01 tier floor (below-floor → silent empty-range bug) and writes partitions atomically (tmp+rename, resume-safe). Scope + survivor-bias caveat in `docs/THETA_LARDER_SCOPE.md`; pinned by `tests/test_option_history_puller.py`. |
| `scripts/pull_theta_options_flow.py` | Theta pull of daily per-ticker options-flow aggregates. |
| `scripts/produce_option_premiums.py` | Distils the Theta option-history larder into the compact per-ticker **real EOD option-premium** parquet (`data_processed/option_premium/<T>.parquet`, gitignored) the connector's `get_option_premium*` accessors serve. Computes the EOD `mid = (bid+ask)/2`, normalizes `right`→`put`/`call`, parses the PIT `created` date axis, keeps two-sided uncrossed markets in a DTE belt (default 0–75), one row per `(date,strike,right)`. The data half of the real-premium producer that makes `edge_vs_fair` non-zero (skew/VRP EV-relevant — `docs/PHASE2_SKEW_EXECUTION_SPEC.md`); the EV-moving ranker wiring is a separate change. Output is gitignored ⇒ never enters `connector_data_sha256` (zero re-baseline). Pinned by `tests/test_option_premium_accessor.py`. |
| `scripts/pull_theta_corp_actions.py` | Theta pull of stock splits and dividends. |
| `scripts/pull_theta_option_tape.py` | Theta pull of intraday option trade+quote tape. |
| `scripts/pull_theta_vix_futures.py` | Theta pull of the VIX futures curve. |
| `scripts/theta_backfill.py` | Tier-aware Theta bulk-backfill CLI with subcommands. |
| `scripts/theta_health_check.py` | Theta Terminal health probe across every v3 endpoint the engine uses. |
| `scripts/probe_theta_capabilities.py` | Probes the Theta tier and writes the capability map. |
| `scripts/backfill_features.py` | Recomputes the feature store for every universe ticker in parallel. |
| `scripts/diagnose_candidates.py` | Read-only EV-ranker funnel report for zero-trade debugging. |
| `scripts/diagnose_iv_surface.py` | Fail-loud SVI IV-surface diagnostic (ROADMAP A2 / `DECISIONS.md` D9) — first production caller of `engine/volatility_surface.py`; reports per-expiry skew / term-structure and exits non-zero on any uncovered ticker. Pure core unit-tested in `tests/test_iv_surface_failloud.py`; connector path operator-first-run-verified. |
| `scripts/feature_smoke_test.py` | End-to-end smoke-test harness exercising the data layer, EV engine and API. |
| `scripts/quant_benchmark_gate.py` | Hard acceptance gate validating the quant engine against academic reference values. |
| `scripts/orchestrate.py` | Unified daily orchestrator (morning / intraday / evening / full). |
| `scripts/run_pipeline.py` | CLI front-end to the data `PipelineOrchestrator`. |
| `scripts/validate_environment.py` | Environment validation for CI — Python version, dependencies, env vars, directory structure. |
| `scripts/inventory_data.py` | Verified data-inventory scan: streams every Bloomberg CSV (incl. gzipped deep slices) for date-column min/max + row/ticker counts, and reads Theta parquet footers/filenames for per-dataset date coverage + totals. Writes `data_processed/_inventory_scan.json`; backs `docs/DATA_INVENTORY.md`. Reads files directly — no values from docs. |
| `scripts/audit_data_engine.py` | Reusable Phase-1 data + engine audit (discovery; read-only, additive). Logs the provider, auto-detects the data-supported frontier (not `date.today()`), inventories every connector CSV, builds the capability map + coverage matrix (referential / seam / survivorship / re-ticker), probes the production `WheelRunner.rank_candidates_by_ev` at the frontier (records produced/dropped/vanished via `frame.attrs["drops"]`, the `distribution_source` tier, non-finite outputs), runs data-hygiene checks, and emits the markdown findings doc + JSON sidecar. No §2 bypass; decision trio untouched. Produces `docs/DATA_ENGINE_AUDIT_*.md`. |
| `scripts/audit_tail_cvar.py` | W5 tail-fit / CVaR reliability audit (heavy-verify 2026-06-27, #436, Mac terminal, stretch; discovery, read-only, additive). Over the canonical grid: POT-GPD tail-fit status (fire rate, `tail_xi`, heavy-tail rate), empirical CVaR breach frequency (realized < `cvar_5`) vs nominal by VIX regime, and the forward-distribution cascade tier mix (`empirical_non_overlapping` → `empirical_overlapping`) by regime + sample thinness, confirming graceful degradation. No §2 bypass. Persists `w5_tail_cvar.json`. |
| `scripts/audit_risk_free_pit.py` | W4 risk-free-rate + PIT-correctness audit (heavy-verify 2026-06-27, #436, Mac terminal; discovery, read-only, additive). Confirms treasury coverage (1994→2026) makes the spurious-5% RFR fallback unreachable for any feasible as_of (refute-on-current-data); quantifies the latent EV impact via a forced-0.05 monkeypatch shim in-process (never the trio); and verifies the ranker's IV is point-in-time (`_resolve_pit_atm_iv`, no lookahead, moves with as_of). Persists `w4_risk_free_pit.json`. |
| `scripts/audit_prob_profit_calibration.py` | W3 prob_profit calibration stratified by VIX regime (heavy-verify 2026-06-27, #436, Mac terminal; discovery, read-only, additive). Reuses the canonical `vnv_prob_profit_calibration.py` harness verbatim (same grid / realized rule / DATA_END=2026-03-20, which pre-dates the W1 splice) and adds (VIX regime × prob bin) calibration with Wilson 95% CIs and n≥30 conclusiveness flags. Finds the top-bin over-confidence is calm/elevated-entry (~−16pp) but absent in crisis-entry (+0.008) — reconciling with the R11 prior as a VIX-at-entry vs followed-by-crisis difference (#442). No §2 bypass. Persists `w3_prob_profit_calibration.json`. |
| `scripts/audit_output_realism.py` | W2 engine output-realism audit (heavy-verify 2026-06-27, #436, Mac terminal; discovery, read-only, additive). Runs `WheelRunner.rank_candidates_by_ev` over the full universe at 5 historical `as_of` dates (calm / 2020 / 2022 regimes), checks premium/spot, served IV band, `ev_dollars` finiteness, `prob_profit`/`prob_assignment` ∈ [0,1], and recomputes per-candidate Greeks against `docs/GREEKS_UNIT_CONTRACT.md`. Buckets every non-finite/absurd value. No §2 bypass; decision trio untouched. Persists `w2_output_realism.json` before printing. |
| `scripts/audit_data_wiring.py` | W1 data-wiring accuracy audit (heavy-verify 2026-06-27, #436, Mac terminal; discovery, read-only, additive). Validates that `MarketDataConnector` serves every committed Bloomberg source faithfully: per-source row/coverage/monotonic checks, IV sentinel + sub-3 floor leakage, treasury coverage + decimal units, and the universe-wide OHLCV split-scale discontinuity sweep that confirms the BKNG (÷25) / CVNA (÷4.7) 2026-03-23 splice artifacts and refutes the NFLX ~10× mis-scale. No §2 bypass; decision trio untouched. Persists JSON to `docs/verification_artifacts/data_wiring_2026-06-27/` before printing. |
| `scripts/audit_data_tests.py` | Phase-1 data-layer TEST-coverage audit probes (discovery; read-only, additive). Companion to `audit_data_engine.py`: computes the deeper byte-evidence the 2026-06-09 round needs — #363 served-vs-raw IV band reconciliation, IV `(date,ticker)` uniqueness + no-tenor schema fact, OHLCV per-name depth + exact NaN-price rows + monotone dates, fundamentals `eqy_dvd_yld_12m` band + GICS-11 validity, credit `sp_rating` ladder + Altman-Z plausibility, dividend epsilon-negatives, treasury per-tenor coverage, and the data→`WheelRunner.rank_candidates_by_ev`→EVResult finite+sign check (no §2 bypass; decision trio untouched). Emits a JSON sidecar. |
| `scripts/check_manifest_coverage.py` | CI guard — fails the build when a tracked file is absent from FILE_MANIFEST.md (or vice versa), and ALSO when any tracked `.md` contains a committed git merge-conflict marker (`<<<<<<<`, `=======`, `>>>>>>>` at column 0, exact 7-char tokens — visual separators with more `=`/`<`/`>` are NOT flagged). Wired into the `FILE_MANIFEST Coverage` job in `.github/workflows/ci.yml`. |
| `scripts/sync_manifest.py` | Local sync helper — same scan as `check_manifest_coverage.py`, with `--fix` to append rows for missing files into a marked "Untriaged additions" section at the tail of `FILE_MANIFEST.md`. Orphans are flagged but never auto-deleted. Smooths the recurring CI failure where docs PRs forget manifest rows. |
| `scripts/check_doc_currency.py` | CI + hook guard against temporal-doc drift — checks `PROJECT_STATE.md`'s `Last updated` date and `CHANGELOG.md`'s newest month section; WARNs on mild staleness, FAILs only on egregious staleness / structural breakage (so normal PRs aren't blocked). Stdlib only (runs on Linux / Windows / CI); wired into the `FILE_MANIFEST Coverage` CI job and surfaced by the SessionStart hook. |
| `scripts/check_lane_claim.py` | CI guard (decision-layer lane gate, D15 2026-05 extension) — fails a PR whose diff touches the decision-layer trio (`engine/ev_engine.py` / `engine/wheel_runner.py` / `engine/candidate_dossier.py`) without a `lane-claim` block in the PR description naming the file. Stdlib + git only; the `decision-layer-claim` job in `.github/workflows/ci.yml` runs it PR-only. See `docs/PARALLEL_SESSIONS.md` §5. |
| `scripts/gen_worklog_index.py` | Generates `docs/worklog/INDEX.md` from worklog-fragment front-matter + the in-place dated reports (`ENGINE_BACKTEST_*` etc. — not moved; 243 inbound refs). `--check` fails CI when the index is stale. Stdlib only; replaces the hand-maintained `VERIFICATION_INDEX` (D14 extension). |
| `scripts/new_worklog.py` | Scaffolds a new `docs/worklog/<id>-<slug>.md` fragment from `_template.md` with front-matter filled. Stdlib only. |
| `scripts/s47_trader_session_2026_03_20.py` | S47 reproducer — observe-only "use the engine" wheel session at as_of=2026-03-20: ranks puts, interrogates strike/premium/prob realism (independent BSM recompute + skew estimate), exercises the earnings gate, R11 elevated-vol path, R9/R10 concentration caps, and the roll → assignment → covered-call lifecycle. Mirrors `docs/worklog/s47-live-wheel-session-2026-03-20-trust-audit-on-an.md`. Never modifies `engine/`. |
| `scripts/generate_tested_surface_map.py` | Reads `coverage.json` and writes `docs/TESTED_SURFACE_MAP.md` — per-module table + top-N gap ranking + module→test static-import map. Stdlib only; re-run after a meaningful coverage shift. |
| `scripts/setup-terminal.sh` | Parallel-session env loader for bash / Git Bash / WSL — source with a terminal letter (`source scripts/setup-terminal.sh a`) to export per-terminal `SWE_API_PORT`, `COVERAGE_FILE`, `PYTEST_CACHE_DIR`, `SWE_DATA_PROCESSED_DIR`, `SWE_MODELS_DIR`, `SWE_DATA_PROVIDER`. See `docs/PARALLEL_SESSIONS.md` "Env vars per terminal". |
| `scripts/setup-terminal.ps1` | PowerShell companion to `setup-terminal.sh` — dot-source (`. .\scripts\setup-terminal.ps1 a`) for native Windows shells. Sets the same six env vars. |
| `scripts/process_bloomberg_exports.py` | Cleans and validates Bloomberg-exported CSVs into the per-ticker layout. |
| `scripts/download_sp500_constituents.py` | Scrapes the current S&P 500 constituent list from Wikipedia. |
| `scripts/download_yf_ohlcv.py` | yfinance OHLCV downloader with multi-index header cleanup. |
| `scripts/download_yf_options.py` | yfinance option-chain downloader. |
| `scripts/bloomberg_smoke.py` | Bloomberg connectivity tester (a CLI tool, not a pytest file). |
| `scripts/transaction_costs_demo.py` | `print()`-driven walkthrough of the `engine.transaction_costs` round-trip (commissions, slippage, assignment) on a synthetic Wheel cycle — a demo, not a pytest file. |
| `scripts/vnv_funnel_tier_report.py` | Read-only V&V diagnostic — runs `rank_candidates_by_ev` over the universe at a fixed `as_of` and reports the candidate funnel (drops-by-gate from `df.attrs["drops"]`) + the `distribution_source` distribution (prob_profit Wilson-CI coverage). Companion to `docs/VNV_CAMPAIGN_2026-06-01.md` §1–2. Never modifies `engine/`. |
| `scripts/vnv_prob_profit_calibration.py` | Read-only V&V diagnostic — bins predicted `prob_profit` vs realized hold-to-expiry outcome (engine-EXACT breakeven) over a regime-spanning `as_of` grid, and measures `ev_dollars` predictive power (Pearson + quintile lift + sign gate). Companion to `docs/VNV_CAMPAIGN_2026-06-01.md` §3–4. Never modifies `engine/`. |
| `scripts/bloomberg_excel_extractor_v2.bas` | Bloomberg Excel VBA extractor — "fixed version with longer wait times" (V1 archived). |
| `scripts/bloomberg_export.vba` | Simplified Bloomberg VBA export macro. |
| `scripts/export_sheets_to_csv.vba` | Excel VBA macro exporting ticker worksheets to CSV. |
| `scripts/bloomberg_bql_pulls.md` | Copy/paste BQL query reference for pulling Bloomberg datasets. |
| `scripts/*_formulas.txt` | Generated per-ticker Bloomberg `=BDH(...)` formula lists for Excel paste (ohlcv / iv / earnings / dividends, plus the combined `bloomberg_formulas.txt`). |

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

## `studies/` — observe-only research studies (read-only over the engine)

| File | Purpose |
|---|---|
| `studies/premium_correction/__init__.py` | Package marker + study summary (real-mid vs synthetic-BSM premium correction). |
| `studies/premium_correction/splits.py` | Split-adjustment layer joining engine (split-adjusted) strikes to larder (raw) strikes; cumulative-factor table + adjusted↔raw strike/premium conversions. |
| `studies/premium_correction/pilot.py` | Observe-only harness: drives `WheelRunner.explore_ticker` (authoritative EV path) per (ticker, as_of), joins the real Theta EOD mid, computes the premium correction and the market-vs-engine calibration gap, emits records + summary + cross-plot. |

## `tests/` — test suite

| File | Purpose |
|---|---|
| `tests/__init__.py` | Test-package marker. |
| `tests/test_w5_tail_cvar.py` | W5 tail/CVaR methodology pins (heavy-verify 2026-06-27, #436). Pins the VIX-regime + breach-statistic helpers (breach = realized < `cvar_5`) and the engine tail-ordering contract that `cvar_5` (worst-5% mean P&L, signed) ≤ the 25th-percentile P&L. |
| `tests/test_w4_risk_free_pit.py` | W4 risk-free + PIT pins (heavy-verify 2026-06-27, #436). Pins that the served RFR is the real PIT decimal (ZIRP ~0 in 2021, ~5% in 2024) and the 0.05 fallback only fires before treasury coverage (pre-1994), and that the ranker's IV has no lookahead and moves with as_of (not a frozen present-day snapshot). |
| `tests/test_w3_calibration.py` | W3 calibration methodology pins (heavy-verify 2026-06-27, #436). Pins the Wilson helper (brackets p̂ in [0,1], known-value check), the VIX-regime bucketing, and the bin/gap/conclusiveness construction (gap = realized − forecast; n<30 → not conclusive; over-confidence ⇒ negative gap) — without re-running the 35-date grid in the per-PR lane. |
| `tests/test_w2_output_realism.py` | W2 output-realism pins (heavy-verify 2026-06-27, #436). At a calm + a stress `as_of` (universe_limit-bounded for the per-PR lane): all ranker outputs finite, `prob_profit`/`prob_assignment` ∈ [0,1], served IV in the decimal band (0.03, 5.0], premium a sane fraction of spot, and a 25-delta short put's Greeks honour `docs/GREEKS_UNIT_CONTRACT.md` (delta ∈ [-1,0], gamma≥0, vega≥0). |
| `tests/test_w1_data_wiring.py` | W1 data-wiring pins (heavy-verify 2026-06-27, #436). Verified-property tests (served IV band ∈ (3.0, 10000], no deep-IV sentinel leak, OHLC invariant, monotonic+positive OHLCV, treasury covers the feasible window) plus a strict-xfail that pins the confirmed BKNG/CVNA 2026-03-23 split-scale discontinuity to its behaviour (flips to XPASS when the OHLCV splice is regenerated). |
| `tests/test_deep_read_connector.py` | R2 deep-read connector tests — flag plumbing (default-OFF, `SWE_DEEP_HISTORY`), graceful degrade when deep slices absent (ON==OFF), and (local-only, gated on `SWE_DEEP_TEST_DATA`) assembly reaches 1994, default-OFF ignores deep when present, delisted Lehman returns its last bar, rotation invariant post-assembly, schema parity. |
| `tests/test_deep_read_assembly_synthetic.py` | R2 deep-read assembly unit tests on synthetic gz fixtures (CI-runnable, no committed data) — dedup precedence recent > deep-current > delisted, multi-slice span, delisted-only name presence, default-OFF monolith-only, missing-slice degrade, non-`_DEEP_SLICES` key not assembled. |
| `tests/test_survivorship_harness.py` | R3 survivorship harness tests (gated on `SWE_DEEP_TEST_DATA`) — PIT 2008 universe includes Lehman/WaMu + excludes post-2008 names + size ~500; `terminal_spot` delisting-aware; the `assert_data_window_available` deep-floor extension accepts a pre-2018 start. |
| `tests/test_survivorship_r6_lehman.py` | R6 survivorship proof (gated on `SWE_DEEP_TEST_DATA`) — a 2008 deep-history backtest where Lehman (LEHMQ) flows through the EV ranker and its post-delisting put loss is realized (non-NaN, < -$500), not silently dropped. |
| `tests/test_deep_iv_sentinel.py` | R7 deep-IV sentinel test (gated on `SWE_DEEP_TEST_DATA`) — the assembled vol_iv read nulls implied-vol values above the ~134217.7 sentinel floor (keeping the row, NaN IV) while preserving real distressed-name extremes (500-1000%). |
| `tests/test_credit_rating_population.py` | R0a regression guard — `analyze_ticker` populates `credit_rating` from the `get_credit_risk()` `sp_rating` key (not the raw `rtg_sp_lt_lc_issuer_credit` field). Pins the dead-read fix; documents the field is off the EV path. |
| `tests/test_premium_correction_pilot.py` | Validates the premium-correction pilot's split layer against known splits (AAPL 4:1, TSLA 5:1+3:1, NVDA 4:1+10:1) and pins the post-split pilot band as split-free — the guard against the raw↔adjusted strike mis-join. |
| `tests/quant_benchmarks.py` | Non-test helper — the quantitative tolerance registry used as release gates. |
| `tests/fixtures/theta_v3_*.csv` | Captured live Theta v3 SPY responses used as connector test fixtures. |
| `tests/fixtures/ibkr/*.json` | Frozen demo IBKR artifacts (snapshot + monthly history + closed-trade ledger) for the D26 performance-viewer tests; reproduce the design-doc Appendix A / `mock.ts` numbers. Also the out-of-box demo source the engine viewer reads via `SWE_IBKR_DATA_DIR`. |
| `tests/test_audit_invariants.py` | Launch-blocker invariants — Greeks unit contract, PIT safety, TV webhook HMAC, EV-engine invariants. |
| `tests/test_check_manifest_coverage.py` | Unit tests for the conflict-marker guard in `scripts/check_manifest_coverage.py` — pins exact 7-char marker shapes (with/without ref, separator alone) and the negative cases that drove the precision (visual `=========` separators, pytest section dividers, indented occurrences, prose substrings). |
| `tests/test_check_lane_claim.py` | Unit tests for `scripts/check_lane_claim.py` — the decision-layer lane gate's behaviour matrix (no-touch passes, unclaimed edit fails, claimed passes, partial claim fails on the remainder, no-claim-source skips as not-a-PR-context). |
| `tests/test_iv_surface_failloud.py` | Pins the A2 fail-loud SVI-surface contract (`DECISIONS.md` D9) — `require_surface` raises on an empty surface, `create_empirical_surface` raises on empty input, the diagnostic's pure core builds from a synthetic ATM term structure and fails loud on missing data, and a populated surface still works. |
| `tests/test_audit_viii_e2e.py` | Launch-blocker invariant — the end-to-end TV-webhook → EV-ranker → tracker authority chain. |
| `tests/test_audit_viii_unit_invariants.py` | Launch-blocker invariant — IV/rate percent-vs-decimal normalisation and the rolled-leg P&L accumulator. |
| `tests/test_audit_viii_real_data_smoke.py` | Real-Bloomberg smoke test of the EV ranker (module-level skip without the CSVs). |
| `tests/test_authority_hardening.py` | Launch-blocker invariant — TV / strangle / tracker route through the EV authority. |
| `tests/test_backtest_regression.py` | Backtest-regression harness — runs S27/S32/S34/S35 reproducers against the current engine and compares to committed snapshots in `backtests/regression/snapshots/`. Gated behind the `backtest_regression` marker (long-running). |
| `tests/test_ev_authority_log_schema.py` | Schema-closure regression for `WheelTracker._ev_authority_log` — pins the five D16 entry shapes (`issue`, `refuse_issue`, `consume`, `reject` × {unknown_token, missing_current_ev_dollars, stale_ev}) and detects drift (unknown action, missing required key, accidental extra key). |
| `tests/test_token_param_binding.py` | brain-audit M1 — EV-authority token parameter binding: cross-ticker/strike/dte/side mismatch refused (`token_param_mismatch`), unbound-token fail-closed (`unbound_token`), legacy snapshot log-rebuild, explicit-expiration escape hatch, persistence round-trip. |
| `tests/test_engine_api_port.py` | Unit tests for `engine_api._resolve_port()` — pins the `SWE_API_PORT` contract (default 8787, env override, loud failure on malformed / out-of-range). Closes D15 Unresolved per #154 C7. |
| `tests/test_engine_api_concentration.py` | Live-path proof for the `/api/concentration_preview` endpoint (`engine_api.build_concentration_preview`) — drives a synthetic concentrated batch through the REAL `consume_into_live_book` / `make_live_book_tracker` / `portfolio_risk_gates` and pins that R9 (sector) + R10 (single-name) caps refuse over-concentration on the operator path, that a non-positive-EV row is refused at the D16 token gate (no rescue), and that the response surfaces the armed cap %s + §2/§3 framing. Closes the "armed caps have zero live callers" gap (#154 / #343). |
| `tests/test_tv_nonce_register_lock.py` | Regression tests pinning the explicit `_TV_SEEN_NONCES_LOCK` around `_tv_seen_register` (S20 C3). Asserts lock existence + 64-worker contention behavior (exactly 1 worker accepts a duplicate digest, all 64 distinct digests accepted). |
| `tests/test_portfolio_risk_gates.py` | Unit tests for `engine/portfolio_risk_gates.py` (D17 / #154 C4 Phase 1) — pins the adapter (`take_snapshot`) per `WheelPosition` state plus the five gate functions' pass/fail/skip semantics against the locked D17 defaults. |
| `tests/test_ibkr_ev_calibration.py` | Phase-3 calibration-stats tests: Wilson CIs + reliability/Brier/ECE on synthetic inputs, and the S&P-500 universe loader. |
| `tests/test_ibkr_flex_ledger.py` | Phase-4 ledger tests on synthetic fills: Open/Close-driven long & short `StockBook` (round-trip realized, ACAT seed), helpers, and the two-file boundary dedup. |
| `tests/test_ibkr_gateway_pull.py` | Gateway-puller tests on pure helpers (ib_insync imported lazily, so no install needed): structured-contract → `contract_description` synthesis and its lossless round-trip back through the shared parser, the account-tag map, and §2/§3 guards (no trio import, no order method referenced). |
| `tests/test_ibkr_import.py` | Importer parser/helper tests (OCC parse, p6 open-position parse + FX/cash derivation, dividend dedup+net) on synthetic page text, plus adapter null-safety regression (`returns_view`/`risk_view` tolerate the present-but-null day-change/margin a real import carries). |
| `tests/test_ibkr_live_snapshot.py` | Live-connector snapshot-builder tests on synthetic IBKR payloads — `contract_description` parsing (incl. interposed venue token), FX normalization, closed-position exclusion, FX-normalized day-change, sector/name/in-universe joins, the `schema_version:1` contract, and the §2/§3 guard that the module imports nothing from the trio. |
| `tests/test_ibkr_portfolio_adapter.py` | D24/D26 adapter tests — snapshot→`PortfolioContext` fidelity, universe filter (CNQ/ENB exposure-only, never rankable), CAD→USD FX normalization, R9/R10 firing on the adapter-built context, the viewer payloads matching the approved numbers, and the guard that the adapter imports nothing from the trio + emits no tradeable verdict / EV-authority token. |
| `tests/test_portfolio_api_endpoints.py` | D26 endpoint tests — spins the stdlib server in-process and asserts each `GET /api/portfolio/{summary,positions,returns,income,risk,history}` returns the dashboard shape, plus the observational guard that no response carries a verdict / EV-authority field and an unknown sub-path 404s. |
| `tests/test_prob_profit_ci.py` | Small-sample honesty for `prob_profit` (2026-06-01) — pins the `_wilson_score_interval` math (known cells, edge/clamp cases, narrower interval at larger N) and that `EVResult` + the ranker core frame surface `n_scenarios` + `prob_profit_ci_low/high` bracketing `prob_profit`; additive, `prob_profit` itself unchanged. |
| `tests/test_trade_memo_ci.py` | Pins the Ollama trade memo's `_format_prob_profit_line` — renders `prob_profit` with its Wilson 95% interval + N and a small-sample caveat; omits the line when not evaluated; emits a clean point estimate (no false caveat) when CI/N absent; the prob block carries no EV/verdict/multiplier wording and the interval round-trips verbatim (display-only, additive). |
| `tests/test_production_tracker_caps.py` | Invariant pins for the production-armed concentration caps (heavy-verify 2026-05-31 Cat-A / #154 follow-up). Pins that `engine.wheel_runner.make_live_book_tracker` ARMS R9 sector + R10 single-name (refuses >25% sector / >10% single-name, token-free, reject-audit shape unchanged), that the library-default `WheelTracker` is unchanged (caps off), and that strict mode still arms all four D17 gates. Guards the decoupled `enforce_*` flags so production-arming can't be silently dropped. |
| `tests/test_data_connector_ticker_filter.py` | Output-equivalence pins for the connector ticker-filter speed-up: the cached-groupby `_filter_ticker` returns byte-identical frames to the prior `df[df["ticker"]==t]` mask (present / multi-occurrence / absent tickers; cache reuse; empty/tickerless passthrough), and `_load`'s unique-map normalization equals the prior `.apply(normalize_ticker)`. Guards that the perf change is §2-neutral (engine sees identical data). Synthetic data — no large-CSV dependency. |
| `tests/test_dossier_invariant.py` | Launch-blocker invariant — the downgrade-only `EnginePhaseReviewer` contract. |
| `tests/test_dossier_downgrade_property.py` | §2 downgrade-only lattice property — states the severity invariant (`proceed`<`review`<`skip`<`blocked`) ONCE over `EnginePhaseReviewer.review()`'s full R1–R11 chain instead of per-rule examples: 8 overlay firing scenarios × 6 `ev_dollars` probes assert `severity(with_overlay) ≥ severity(without)`, plus firing-cell teeth, a cannot-rescue-blocked check, a Hypothesis ev-fuzz, and five source-introspection meta-tripwires that force any future rule (R12+) into the matrix (overlay-guard count, no hard-returned `proceed`, verdict vocabulary, overlay-reason coverage, downgrade-to-`review`). Read-only against §2 (test-only). |
| `tests/test_r11_elevated_vol.py` | Pins R11 — the elevated-vol top-bin size-down reviewer (heavy-verify 2026-05-31 I11). Eight tests: fires (top-bin `prob_profit>0.90` + `vix_level>25` → review / `elevated_vol_top_bin`), no-ops (VIX≤25, not-top-bin, `vix_level` absent), never-rescues (negative-EV stays blocked by R1), strictly-greater-than boundaries, computed-not-hardcoded payload (candidate's own `cvar_5`), and `build_dossiers(vix_level=…)` threading. Read-only against §2 (downgrade-only). |
| `tests/test_dossier_r9_r10_audit.py` | S42 — systematic audit of R9 (sector_cap, PR #255) and R10 (single-name cap, PR #262). Six probe families (32 tests): R9 fires correctly, R10 fires correctly, downgrade-only invariant, fail-closed on missing context, cross-rule short-circuit ordering (R7 → R8 → R9 → R10), and edge cases (boundary semantics + four pinned sharp-edge findings — see `docs/USAGE_TEST_LEDGER.md` S42). Read-only against §2. |
| `tests/test_testing_md_taxonomy.py` | Taxonomy completeness gate — every `tests/test_*.py` must be named in `TESTING.md` (and every literal `tests/test_*.py` path there must exist). Mirrors the FILE_MANIFEST coverage gate; added at D27 after 89/144 files had drifted out of the taxonomy. |
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
| `tests/test_pricing_evaluate_invariants.py` | Quant audit round 2 (W63-W64): behaviour-pins the binomial tree's vega/theta/rho UNITS against analytic BSM (q=0 calls, American==European — a 100x unit slip would be invisible; only delta/gamma were validated) and EVEngine.evaluate finiteness at degenerate dte (<=0 → intrinsic + ev_per_day floor, no div-by-zero). Gamma left loose (tree-noise). §2 only asserted. |
| `tests/test_monte_carlo.py` | Quant-correctness — block bootstrap, jump diffusion, Longstaff-Schwartz. |
| `tests/test_advanced_quant.py` | Quant-correctness — third-order Greeks, BAW American pricing, multi-asset VaR. |
| `tests/test_quant_fixtures.py` | Quant-correctness — textbook-value regression for pricing and volatility estimators. |
| `tests/test_greeks_unit_invariants.py` | Launch-blocker invariant — the Greeks unit contract. |
| `tests/test_realized_vol.py` | Quant-correctness — the realised-volatility estimators. |
| `tests/test_forward_distribution_invariants.py` | Quant-layer test audit round 2 (W38-W43): behaviour-pins the forward-distribution cascade tier-SELECTION (exact tier per history depth, not membership), block-bootstrap/HAR-RV sampler determinism, the empirical min_samples/horizon/isfinite boundary, and the realized-vol non-positive/variance-floor guards (incomplete `_log` guard tracked as (E) #382 via xfail). The EV integrand's contracts; no §2 surface touched. |
| `tests/test_tail_risk.py` | Quant-correctness — POT-GPD tail estimation. |
| `tests/test_tail_copula_stress_invariants.py` | Quant audit round 2 (W44-W49): behaviour-pins CVaR>=VaR on the gpd xi<0 branch + the REAL copula simulators (existing copula tests monkeypatch the math away), `monte_carlo_stress` seed reproducibility + cvar95<=var95, and `run_scenario`'s expired-intrinsic branch. Risk-math feeding the EV heavy-tail penalty + reviewer R8; no §2 surface touched (t-copula df guard tracked as (E) #384). |
| `tests/test_f4_tail_risk_gap.py` | F4 regression-watch from PR #178 / #184 — pins that today's forward-distribution + POT-GPD pipeline does NOT widen tail metrics for the COST 2022-04 / UNH 2024-11 19-24% realized drops. Synthetic two-regime tests isolate the 504-day-lookback-dilution mechanism (H1) and demonstrate the fix direction. Two `xfail(strict=False)` tests track the `heavy_tail` flag until the gap is fixed. |
| `tests/test_f4_rv_widening.py` | F4 Fix B2 (post-rollback) — pins `realized_vol_ratio` / `realized_vol_widening_factor` / `realized_vol_widened_log_returns` in `engine/forward_distribution.py`. 18 tests: PIT-safety + edges on the ratio helper, calibration pins on the factor (threshold 1.30, slope 0.20, cap 1.15), sign-/mean-preserving invariants on the widened-returns transform, end-to-end ranker pins on COST/UNH/AAPL F4 cases. Documents the honest scope: rv-widening makes the engine ~2x more cautious on empirically-elevated-vol regimes but does NOT close named F4 cases (those are R10's job). See `docs/F4_TAIL_RISK_DIAGNOSTIC.md` §11. |
| `tests/test_portfolio_copula_coverage.py` | Quant-correctness — copula edge paths (PSD repair, Cholesky fallback, CVaR ladder). |
| `tests/test_risk_manager.py` | `RiskManager` — sizing, portfolio Greeks, VaR, sector exposure, HRP. |
| `tests/test_stress_testing.py` | `StressTester` — scenarios, sensitivity, Greeks stress ladder. |
| `tests/test_payoff_engine.py` | `PayoffEngine` — payoff diagrams, expected-move bands, strike recommendations. |
| `tests/test_regime_detector.py` | The rule-based regime classifier. |
| `tests/test_regime_hmm_invariants.py` | Quant audit round 2 (W50-W55): behaviour-pins the HMM regime-multiplier envelope sup/inf (0.2 crisis … 1.25 bull_quiet — existing test only checks a diffuse band), same-seed fit determinism (the cache + fingerprint depend on it), the unfit-guard RuntimeError, and the RegimeDetector degenerate realized-vol fallback. §2: the multiplier only scales a >=0 envelope. `fit`'s missing non-finite guard (returns NaN multiplier) tracked as (E) #386 via xfail. |
| `tests/test_dealer_positioning.py` | Dealer positioning — analyzer math, the clamped multiplier, reviewer rule R6. |
| `tests/test_dealer_positioning_invariants.py` | Quant audit round 2 (W60-W62): behaviour-pins the §2-critical `dealer_regime_multiplier` [0.70,1.05] clamp under ANY confidence (negative/>1/inf/nan) + the asymmetry (boost ≤+0.05, cut to 0.70), the `_classify_regime` boundaries (gex==0→neutral, sign branches, near-flip override), and `analyze()` flip-distance consistency. Asserts the clamp, never weakens it (wall-ordering needs a wall-fixture → deferred). |
| `tests/test_dealer_multiplier_evengine_integration.py` | Dealer-multiplier integration boundaries — pins that `[0.70, 1.05]` survives `EVEngine.evaluate` to `EVResult.dealer_multiplier`, the asymmetric-by-design clamp at the EVResult level, the `regime_mult *= dealer_mult` compounding, and the §2 "scales ev_dollars only" claim as proportionality. Companion to PR #185's blocked-path test. |
| `tests/test_extreme_numerics.py` | Numerical stability under near-expiry / near-zero-vol / deep-moneyness extremes. |
| `tests/test_edge_cases.py` | Engine edge cases across pricing, costs and the tracker state machine. |
| `tests/test_properties.py` | Hypothesis property-based invariants for pricing, RSI, IV-rank, Kelly. |
| `tests/test_point_in_time.py` | Anti-lookahead — rolling features and labels use only past data. |
| `tests/test_pit_leaks.py` | Launch-blocker invariant — news and credit-regime overlays honour a historical `as_of`. |
| `tests/test_asof_none_staleness.py` | M3 regression — `as_of=None` resolves to the universe data frontier so the staleness gate engages; index leavers dropped; fresh names byte-identical; drop-only invariant; explicit path untouched; CC + strangle siblings covered. |
| `tests/test_preflight_environment.py` | Preflight environment-invariant guard (automates CLAUDE.md §4 session-start checks): pins + logs that the default/`bloomberg` provider resolves to `MarketDataConnector` (silent provider selection is a recurring bug, §4.1) and that the bundled OHLCV reaches the pinned `EXPECTED_FRONTIER` — a loud, *diagnosing* failure ("OHLCV ends … expected ≥ … you may be on a STALE tree / wrong clone") that catches the stale-clone class (the "79-days-stale" premise + fingerprint false-positive from reading an older clone instead of main). Fast (date column only), deterministic (pinned frontier, not `today()`), self-skipping (skips on `SWE_DATA_PROVIDER=theta` or absent data). |
| `tests/test_contracts.py` | `engine.contracts` interface-validation helpers. |
| `tests/test_policy_config.py` | `TradingPolicyConfig` load/save/validate. |
| `tests/test_observability.py` | `engine.observability` trace context, decision journal, audit logger. |
| `tests/test_event_calendar.py` | `engine.event_calendar` queries, builder, ingestion manager. |
| `tests/test_event_gate.py` | `EventGate` lockout, buffer windows, candidate filtering. |
| `tests/test_reviewer_eventgate_invariants.py` | Quant audit round 2 (W65-W67): behaviour-pins the EXACT boundaries of the §2 downgrade-only reviewer rules — R5 inclusive `ev_dollars >= min_proceed_ev` (==threshold→proceed), R3 strict `diff > tol` (==tol→not-skip) + the `engine_spot>0` guard — and EventGate.is_blocked returning the EARLIEST in-window event across mixed wildcard-macro/ticker-specific. Asserts the §2 contract (downgrade-only), never weakens it. |
| `tests/test_event_gate_back_buffer.py` | Pin S23 F1 fix — `MarketDataConnector.get_recent_earnings` complements `get_next_earnings`; the three rankers register past earnings on the gate so the symmetric back-buffer fires. |
| `tests/test_corp_action_gate.py` | Pin #3A — the `engine.event_gate` `kind="corp_action"` lockout wired to `sp500_corporate_actions.csv` via `MarketDataConnector.get_corporate_actions` (excludes the 94% `Regular Cash` rows; PIT announcement filter) + the ranker helper `wheel_runner._register_corp_action_events` (registers disruptive splits/spinoffs/special-cash; no-op without the accessor / on error / gate=None; remove-only §2). Data-backed GE-spinoff / COST-special-cash end-to-end block. |
| `tests/test_earnings_drift.py` | `EarningsDriftAnalyzer` post-earnings drift statistics. |
| `tests/test_signals.py` | The signal-generation framework and aggregator. |
| `tests/test_skew_dynamics_invariants.py` | Quant audit round 2 (W56-W59): behaviour-pins the standalone skew-math in `skew_dynamics.py` — Nelson-Siegel fail-fast (iv_at/factor_loadings RuntimeError before fit) + degenerate-fit branches (n==1 level-only, n==0 → 0.20 sentinel), skew_momentum degenerate-history (empty→NaN, short→0 momentum), and the ivs_dislocation composite [-1,1] bound. (The live `skew_mult` clamp is dormant on Bloomberg + in the trio; not re-pinned.) |
| `tests/test_strangle_timing.py` | The strangle-timing engine — regime classification, entry scoring, IV overlay. |
| `tests/test_strangle_recommendation_gate.py` | The strangle phase/confidence downgrade-only recommendation gate. |
| `tests/test_data_connector.py` | `MarketDataConnector` query methods against synthetic CSVs. |
| `tests/test_theta_connector.py` | Live-Terminal smoke test of `ThetaConnector` (auto-skips). |
| `tests/test_theta_connector_v3.py` | HTTP-mocked `ThetaConnector` v3 endpoints and the per-endpoint failure contract. |
| `tests/test_theta_connector_coverage.py` | HTTP-mocked `ThetaConnector` edge-branch coverage. |
| `tests/test_data_integration.py` | `engine.data_integration` Bloomberg calendar/rate loaders. |
| `tests/test_data_integrity_bloomberg.py` | Phase-2(A) database-integrity contracts on the REAL bundled `data/bloomberg/*.csv` (distinct from the synthetic-fixture `test_data_connector.py`): per-file schema, OHLC positivity + the load-bearing rename invariant (0 viol/1.01M), vol_iv unit-consistency + sane band + zero-skew (W1/W8/W9), dividends `>= -1e-9` (W11), date hygiene + no-future bars, cross-file referential integrity with the 2026-03-23 seam membership split encoded structurally (W4), seam continuity + BK→BNY re-ticker, treasury band allowing brief negatives (W10), and the fast-CI fingerprint-completeness guard `set(connector_data_sha256().keys()) == set(_FILES)` (durable W3). `HAS_BLOOMBERG_DATA` skipif; 1 `xfail(strict)` for the 4 NaN-price rows (#357). |
| `tests/test_broad_pull_loaders.py` | Phase-0B loader tests for `data/broad_pull_loaders.py`: synthetic units (gz read, winsorization clamp+log+row-count preservation, float32 downcast, date parse, ticker normalization, PIT `series`/`category_series`/`snapshot_row`, registry consistency, and the §2 no-consumer guard) + real-data tests pinned to the byte-verified manifest (per-dataset rows/date-range/ticker-count/schema, both gz panels, winsorization vs the raw bytes, bid/ask ordering). `HAS_BROAD_PULL_DATA` skipif. |
| `tests/test_broad_pull_wiring_xfail.py` | Phase 1-3 wiring **acceptance scaffolds** — `xfail(strict)` behaviour pins (#366 discipline) for the supervised, EV-moving broad-pull wiring whose contract is explicit: #354 `get_fundamentals`/`get_credit_risk` point-in-time (`as_of`), #372 R9 sector grouping → real GICS (not `DEFAULT_SECTOR_MAP` `'Unknown'`). Fail today (wiring absent) → `xfail`; flip to strict-XPASS when each step lands, forcing the marker's removal. `HAS_BROAD_PULL_DATA` + `HAS_BLOOMBERG_DATA` skipif. |
| `tests/test_data_to_engine.py` | Phase-2(B) data→engine-output tests routed through `WheelRunner.rank_candidates_by_ev` / `rank_covered_calls_by_ev` (no §2 bypass), pinned to the frontier `2026-06-04`: output sanity (finite ev_dollars/ev_raw R1a, premium>0, decimal IV band, prob_profit∈[0,1], OTM strike<spot, Wilson-CI coherence), cascade-tier correctness (AAPL=empirical_non_overlapping), no-silent-drops accounting, thin/garbage graceful degradation, determinism, the dividends→CC ex-div mechanism (DIS, R1), a corrupt/truncated negative control via an injected fixture connector, and the 5-ticker smoke. `xfail(strict)` for W2 dateless-lookahead (#354) + the 11 W6 blue-chip backfills (#355). Full-universe produced/dropped pin behind `@pytest.mark.slow`. |
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
| `tests/test_wheel_backtest.py` | `src.backtest.wheel_backtest` run, scoring and metrics. |
| `tests/test_wheel_tracker_persistence.py` | `WheelTracker` JSON persistence round-trips. |
| `tests/test_wheel_tracker_suggest_rolls.py` | Launch-blocker invariant — `suggest_rolls` properties and roll-EV regression. |
| `tests/test_wheel_tracker_suggest_call_rolls.py` | Launch-blocker invariant — `suggest_call_rolls` properties and roll-EV regression. |
| `tests/test_suggest_rolls_drops.py` | Pin S22 F1 fix — `suggest_rolls` and `suggest_call_rolls` emit `.attrs["drops"]` mirroring the ranker drop-log pattern; per-filter-site drop entries with conformant schema. |
| `tests/test_suggest_rolls_defensive.py` | Pin S47 F-S47-1 fix — `include_defensive` surfaces credit-gate-failing (debit) rolls flagged `defensive=True` (each scored via `EVEngine.evaluate`); `.attrs["defensive"]` reports available/surfaced/suppressed so the silent-empty default is visible; default path + §2 (no rescue, one eval per row) preserved. |
| `tests/test_mark_to_market_iv.py` | `WheelTracker.mark_to_market` IV-staleness fix regression. |
| `tests/test_available_buying_power.py` | `WheelTracker.available_buying_power` CSP-collateral netting. |
| `tests/test_portfolio_tracker.py` | `PortfolioTracker` transactions, holdings, returns, snapshots. |
| `tests/test_transaction_costs.py` | `engine.transaction_costs` coverage — slippage tiers, OI penalties, sqrt-impact participation, round-trip cost composition. |
| `tests/test_tv_api.py` | The TradingView bridge HTTP endpoints in `engine_api.py`. |
| `tests/test_engine_api_hardening.py` | Error-path + security hardening for the `engine_api.py` network surface — loopback-bind/CORS helpers (R3), malformed-param → 400 (R18), generic-500 + correlation-id no-leak (R19), unknown-ticker → 404 on payoff/expected_move/strikes (R20), 4xx instead of 200 on no-data bodies (R21), unknown-path/oversized-body/invalid-JSON routing, and the R27 §2-adjacent negative/non-finite-EV verdict-LABEL alignment (`blocked`). |
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
| `tests/test_news_sentiment.py` | `NewsSentimentReader` — sentiment reading; the `TestSentimentMultiplier` class is rewritten post-D18 to assert the constant-1.0 contract across every band the old code derated/boosted. |
| `tests/test_news_severance.py` | DECISIONS.md D18 invariant — `sentiment_multiplier` is 1.0 across every `(sentiment, n_articles)` combination plus a side-effect test pinning that `get_ticker_sentiment` still returns the underlying data after the stub is called. |
| `tests/test_ev_engine_percentiles.py` | `EVResult.pnl_p25/p50/p75` invariants — monotone ordering, pre-multiplier invariance, `cvar_5 ≤ pnl_p25`, NaN guards on small distributions and event-lockout. |
| `tests/test_adversarial_news.py` | Adversarial news-scenario smoke test. |
| `tests/test_recovery_checkpoints.py` | `news_pipeline.recovery.checkpoints` save/load/resume. |
| `tests/test_recovery_fallbacks.py` | `news_pipeline.recovery.fallbacks` degraded-mode and fallback chains. |
| `tests/test_recovery_health.py` | `news_pipeline.recovery.health` provider health tracking. |
| `tests/test_dashboard.py` | The legacy `QuantDashboard` CLI surface. |
| `tests/test_infrastructure.py` | Repo-level infrastructure components — env validation, benchmarks, health, SLO. |
| `tests/test_iv_surface_history_puller.py` | Regression for the IV-surface-history puller. |
| `tests/test_option_history_puller.py` | Pins the option-history puller's correctness fixes: the 2016-01-01 history-floor clamp (below-floor start → empty whole range), the `--lookback-days` window, and atomic partition writes (no partial file a resume skips as "done"). |
| `tests/test_theta_indices_puller.py` | Regression for the Theta indices-history puller. |
| `tests/test_option_premium_accessor.py` | Pins the real EOD option-premium rail: the pure `distill_expiration_frame` transform (mid, right-normalize, DTE belt, crossed/empty-market drop, latest-snapshot dedup), the `produce_ticker` writer round-trip, and the connector accessors `get_option_premium` / `get_option_premium_chain` / `list_option_expirations` (PIT snapshot-on-or-before-`as_of`, staleness window, nearest-strike snap, `strike_tol`, missing-data → `None`/empty fallback). A guarded data-backed test distils a real AAPL partition when the larder is present. |
| `tests/test_real_premium_wiring.py` | Pins the real-premium ranker wiring: `_cumulative_split_factor` math + the connector's split-adjust-on-load (raw larder → engine split-adjusted frame), the `wheel_runner._resolve_real_premium` snap helper (expiry/strike tolerance, graceful `None`), and the end-to-end puts ranker (uses `premium_source="market_mid"` with the injected mid when the rail covers a name; byte-identical synthetic fallback when absent). |

## `tradingview/` — Pine indicator + analyst-workspace assets

| File | Purpose |
|---|---|
| `tradingview/README.md` | Hands-on setup checklist for the engine bridge (install the Pine indicator, wire the webhook). |
| `tradingview/CLAUDE.md` | Session-orientation contract for Claude acting as the TradingView analyst. |
| `tradingview/OVERVIEW.md` | Operating overview of the analyst function. |
| `tradingview/smart_wheel_signals.pine` | The Pine v5 indicator mirroring `engine/tv_signals.py`. |
| `tradingview/alert_payload_schema.json` | JSON Schema for the webhook payload. |
| `tradingview/launch-tradingview-cdp.sh` | Launches TradingView Desktop with CDP for the analyst workspace. |
| `tradingview/launch-tradingview-cdp.ps1` | Windows PowerShell companion to `launch-tradingview-cdp.sh` — locates the TradingView MSIX install and relaunches it with `--remote-debugging-port=9222` so the TradingView MCP can attach. See `docs/TRADINGVIEW_INTEGRATION.md` "Windows gotchas". |
| `tradingview/{models,pine,research}/.gitkeep` | Placeholders for analyst-workspace output directories. `models/` and `pine/` contents remain gitignored; tracked analyst research notes are covered separately below. |
| `tradingview/research/*.md` | Analyst research notes saved per the `tradingview/CLAUDE.md` workspace convention (`YYYY-MM-DD-<title>.md` — fallback for environments without a `.docx` writer). |

## `utils/` — shared utilities

| File | Purpose |
|---|---|
| `utils/__init__.py` | Re-exports the validation, dates, logging and metadata helpers. |
| `utils/data_validation.py` | Option/OHLCV validation, IV normalization, liquidity filtering. Live — consumed by `data/bloomberg_loader.py`. |
| `utils/dates.py` | Trading-day calendar, date normalization, DTE conversions. Dormant — no importers. |
| `utils/health.py` | `HealthChecker` — Kubernetes-style liveness/readiness checks. Test-only consumer (`tests/test_infrastructure.py`). |
| `utils/logging_config.py` | Logging setup helpers. Dormant — imported only by `utils/health.py`. |
| `utils/metadata.py` | Git-fingerprint and metadata-sidecar helpers for reproducibility. Dormant — no importers. |
| `utils/security.py` | Audit logging, input validation, secrets management, rate limiting. Dormant — no importers. |

## Untriaged additions (auto-appended by `scripts/sync_manifest.py`)

Rows below were added automatically because the file was tracked but absent from the manifest. Move each entry under the correct `## <directory>` section with a real purpose description, then delete it from here. Re-running `--fix` rebuilds this section from scratch.

| File | Purpose |
|---|---|
| `docs/HEAVY_NEWS_CALIBRATION_REVERIFY.md` | _TODO: describe (auto-added by `scripts/sync_manifest.py --fix`)._ |
| `docs/verification_artifacts/heavy_news_calibration_2026-05-30_raw_output.txt` | _TODO: describe (auto-added by `scripts/sync_manifest.py --fix`)._ |
| `docs/verification_artifacts/news_calibration_driver.py` | _TODO: describe (auto-added by `scripts/sync_manifest.py --fix`)._ |
| `docs/BRAIN_AUDIT_2026-06-11.md` | Overnight brain-audit campaign report (2026-06-11): 8-dimension probe-backed soundness review of the decision logic on 7f9dc10 — verdict, 4 new MEDIUMs, calibration truth table, full-suite 3,126-green evidence. Probe sidecars under docs/verification_artifacts/brain_audit_2026-06-11/. |
| `docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md` | Heavy-verify (E)-fix validation register (2026-06-09/10): 26-agent verification of #382/#384/#386, BIIB real-data determinism proof, 17-finding sweep, full backtest A/B PASS (580/580 leaves) + snapshot-drift evidence (issue #402). Payload sidecars under docs/verification_artifacts/efix_ab_2026-06-10/. |
| `docs/verification_artifacts/efix_ab_2026-06-10/main_s27_ivpit_24t_100k.json` | Raw build_payload(run()) backtest payload for the (E)-fix A/B (baseline origin/main@1985547 arm `main_*`, integration +#382/#384/#386 arm `integ_*`). Evidence for docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md §9 + issue #402 drift. |
| `docs/verification_artifacts/efix_ab_2026-06-10/main_s32_friction_24t_1m.json` | Raw build_payload(run()) backtest payload for the (E)-fix A/B (baseline origin/main@1985547 arm `main_*`, integration +#382/#384/#386 arm `integ_*`). Evidence for docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md §9 + issue #402 drift. |
| `docs/verification_artifacts/efix_ab_2026-06-10/main_s34_universe_100t_1m.json` | Raw build_payload(run()) backtest payload for the (E)-fix A/B (baseline origin/main@1985547 arm `main_*`, integration +#382/#384/#386 arm `integ_*`). Evidence for docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md §9 + issue #402 drift. |
| `docs/verification_artifacts/efix_ab_2026-06-10/main_s35_oos_24t_100k.json` | Raw build_payload(run()) backtest payload for the (E)-fix A/B (baseline origin/main@1985547 arm `main_*`, integration +#382/#384/#386 arm `integ_*`). Evidence for docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md §9 + issue #402 drift. |
| `docs/verification_artifacts/efix_ab_2026-06-10/integ_s27_ivpit_24t_100k.json` | Raw build_payload(run()) backtest payload for the (E)-fix A/B (baseline origin/main@1985547 arm `main_*`, integration +#382/#384/#386 arm `integ_*`). Evidence for docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md §9 + issue #402 drift. |
| `docs/verification_artifacts/efix_ab_2026-06-10/integ_s32_friction_24t_1m.json` | Raw build_payload(run()) backtest payload for the (E)-fix A/B (baseline origin/main@1985547 arm `main_*`, integration +#382/#384/#386 arm `integ_*`). Evidence for docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md §9 + issue #402 drift. |
| `docs/verification_artifacts/efix_ab_2026-06-10/integ_s34_universe_100t_1m.json` | Raw build_payload(run()) backtest payload for the (E)-fix A/B (baseline origin/main@1985547 arm `main_*`, integration +#382/#384/#386 arm `integ_*`). Evidence for docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md §9 + issue #402 drift. |
| `docs/verification_artifacts/efix_ab_2026-06-10/integ_s35_oos_24t_100k.json` | Raw build_payload(run()) backtest payload for the (E)-fix A/B (baseline origin/main@1985547 arm `main_*`, integration +#382/#384/#386 arm `integ_*`). Evidence for docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md §9 + issue #402 drift. |
| `docs/verification_artifacts/efix_ab_2026-06-10/efix_compare.py` | Stdlib A/B comparator: strict NaN-aware metric-leaf diff of the two arms + tolerance-banded drift check vs the committed snapshots. Re-runnable against the sibling payloads. |
| `docs/verification_artifacts/brain_audit_2026-06-11/d5_costs_probe.py` | Brain-audit probe — cost-model parameter extraction + cost-as-%-of-premium computation (dim 5). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/d6_probe1.py` | Brain-audit probe — §2 candidate-path enumeration — every ranker/API emission routes through evaluate (dim 6). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/d6_probe2.py` | Brain-audit probe — D16 token-binding gap reproduction — cross-ticker consume in strict mode (dim 6 MEDIUM). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/d8_probe1_connector.py` | Brain-audit probe — live IV-sentinel-gate verification on served connector reads (dim 8). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/d8_probe2_pit_fundamentals.py` | Brain-audit probe — #354 dateless-fundamentals lookahead live confirmation on the evaluate path (dim 8). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/d8_probe3_stale_leaver.py` | Brain-audit probe — as_of=None staleness-gate bypass reproduction — CTRA/LW rank on stale spots (dim 8 MEDIUM). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/dim4_msft_rv.py` | Brain-audit probe — MSFT realized-vol / widening spot probe (dim 4). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/dim4_ranker_probe.py` | Brain-audit probe — 20-ticker live-rank overlay-envelope probe — HMM/widening/dealer bounds + label mix (dim 4). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/dim4_unit_probe.py` | Brain-audit probe — unit-level GPD/HMM/dealer clamp + degenerate-input checks (dim 4). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/honesty_fields_smoke.py` | Brain-audit probe — Wilson-CI honesty stack live smoke on ranker output (dim 7). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p1_bsm_textbook.py` | Brain-audit probe — BSM put-call parity / Hull closed-form / Greek finite-difference cross-checks (dim 3). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p1_cascade_moments.py` | Brain-audit probe — forward-dist tier selection + distribution-moment sanity at the data frontier (dim 2). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p1_ev_hand_replication.py` | Brain-audit probe — EV integral hand replication — 21 EVResult fields match to <5e-9 incl. crash path (dim 1). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p2_iv_units_realism.py` | Brain-audit probe — IV percent->decimal contract proof + premium realism reconstruction (dim 3). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p2_multipliers_lockout_degenerates.py` | Brain-audit probe — multiplier order/clamps/anomaly tags + event lockout + degenerate inputs (dim 1). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p2_pit_cascade_jitter.py` | Brain-audit probe — PIT differential byte-identity probes + NOS re-phasing jitter measurement (dim 2). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p3_baw_sanity.py` | Brain-audit probe — BAW early-exercise premium sanity grid vs CRR binomial (dim 3). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p3_heavy_tail_fallback.py` | Brain-audit probe — GPD heavy-tail flag suppression on bounded short-put P&L + lognormal fallback behavior (dim 1/4). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p4_production_path.py` | Brain-audit probe — production ranker-path probe — ev_dollars == ev_raw x regime_mult per row, dealer neutral (dim 1). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/p4_smile_bias.py` | Brain-audit probe — zero-skew bias quantification — short-put conservative vs covered-call optimistic (dim 3). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/scale_corruption_check.py` | Brain-audit probe — NFLX/BKNG/CVNA OHLCV scale-corruption live check (dim 7/8). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/vnv_rederive.py` | Brain-audit probe — V&V calibration artifact re-derivation spot-check (dim 7). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
| `docs/verification_artifacts/brain_audit_2026-06-11/wilson_rederive.py` | Brain-audit probe — Wilson-CI committed-artifact re-derivation (dim 7). Executable evidence for docs/BRAIN_AUDIT_2026-06-11.md. |
