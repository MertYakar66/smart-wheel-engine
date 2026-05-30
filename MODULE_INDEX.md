# Module Index

> For the exhaustive per-file index — every tracked file, not just
> modules — see [`FILE_MANIFEST.md`](FILE_MANIFEST.md).

Per-module purpose and decision-layer relationship. The "Role" column
classifies each module against the EV decision contract from
`CLAUDE.md` §2:

| Role | Meaning |
|---|---|
| **authority** | Code path that decides whether a candidate is tradeable. Only `EVEngine.evaluate` lives here. |
| **runner** | Orchestrator that routes inputs into the authority. |
| **reviewer** | Can downgrade a verdict (proceed → review → skip → blocked). Cannot upgrade. |
| **multiplier** | Scales `ev_dollars` within a clamped range; never touches `ev_raw`. |
| **input** | Provides data, features, or a numerical signal consumed by the EV path or a reviewer. |
| **data** | Raw / processed market-data layer. |
| **tracker** | Position lifecycle bookkeeping; off the decision path. |
| **infra** | Logging, config, contracts, CI, hooks. |
| **display** | Reports, memos, payoff diagrams; not a decider. |
| **dormant** | Exported but no live caller. Wire-up requires an explicit decision. |

Status: `live` (production), `legacy` (still imported but superseded),
`research`, `experimental`, `phantom` (scaffolded, never populated),
`empty`.

---

## Top-level Python files

| File | Purpose | Status | Role |
|---|---|---|---|
| `engine_api.py` | HTTP API on `:8787` (32 endpoints) serving the Next.js dashboard. Top-of-file docstring lists every endpoint. | live | runner / display |
| `morning_run.py` | Browser-driven multi-LLM news pipeline (Claude / ChatGPT / Gemini paid sessions). Zero-API-cost. | live | input (news) |
| `audit.py` | Smoke-test client that hits `localhost:8787` and runs domain-grouped checks; used historically by audit-i through audit-viii. | live | infra |
| `conftest.py` | pytest fixtures + hypothesis profiles + custom markers. | live | infra |
| `requirements.txt` | runtime deps. | live | infra |
| `pyproject.toml` | packaging + tooling. **Stale**: `[project.scripts] wheel = "src.cli:app"` references a file that does not exist; `[tool.hatch] packages = ["src"]` excludes the live code. | partial | infra |
| `pull_branch.bat` / `pull.bat` / `fetch_data.bat` | Windows one-click launchers. | live | infra |

## `engine/` — quant + decision layer

### Authoritative decision layer

| Module | Purpose |
|---|---|
| `ev_engine.py` | `EVEngine.evaluate`. THE ranker. Runs event lockout → forward distribution → cost model → regime + dealer multipliers → returns `EVResult`. (**authority**) |
| `wheel_runner.py` | `WheelRunner.rank_candidates_by_ev`. The one public route into the EV path. Provider selection (`SWE_DATA_PROVIDER`) lives here. (**runner**) |
| `candidate_dossier.py` | EV + chart bundle + `EnginePhaseReviewer` (rules R1–R6). Reviewers downgrade only. (**reviewer**) |

### EV-path participants

| Module | Purpose |
|---|---|
| `event_calendar.py` | Earnings / dividend / FOMC calendar. |
| `event_gate.py` | Event-lockout window logic; consumed by `EVEngine`. |
| `forward_distribution.py` | Empirical, block-bootstrap, and HAR-RV cascade forward distributions. |
| `transaction_costs.py` | Commissions, slippage, assignment fees, sqrt impact, Reg-T margin. |
| `tail_risk.py` | POT-GPD tail estimation. |
| `portfolio_copula.py` | Student-t copula portfolio CVaR. |
| `regime_detector.py` | Rule-based regime: realised-vol vs implied-vol, trend, term-structure. (**multiplier input**) |
| `regime_hmm.py` | 4-state Gaussian HMM regime detector. Cached per-ticker by `WheelRunner._hmm_regime_cache` (audit-VIII P2). (**multiplier input**) |
| `dealer_positioning.py` | GEX / walls / gamma flip → `MarketStructure`. Optional `market_structure` kwarg on `EVEngine.evaluate`; multiplier clamped `[0.70, 1.05]`. (**multiplier**) |
| `skew_dynamics.py` | Nelson-Siegel skew dynamics. |
| `realized_vol.py` | RV estimators (close-to-close, Parkinson, Garman-Klass). |
| `earnings_drift.py` | Post-earnings drift adjustment. |
| `strangle_timing.py` | Strangle entry timing gate (the one timing-gated strategy permitted by `CLAUDE.md`'s NEVER list). |
| `data/quality.py` | Chain-quality gate on the EV path; drops candidates with stale / mispriced / low-liquidity option chains before `EVEngine.evaluate`. (Lives outside `engine/`.) |

### Reviewers (downgrade-only)

| Module | Purpose |
|---|---|
| `chart_context.py` | `ChartContext` dataclass + `ChartContextProvider` Protocol. |
| `tradingview_bridge.py` | `FilesystemChartProvider`, `PlaywrightChartProvider`, `ChainedChartProvider`, `MCPChartProvider`. `build_default_provider` chains them; MCP is opt-in via `SWE_USE_MCP_CHART` (see `docs/TRADINGVIEW_MCP_INTEGRATION.md`, `DECISIONS.md` D13). |
| `mcp_client.py` | `MCPCLIClient` — the tradingview-mcp `tv`-CLI transport backing `MCPChartProvider`. Subprocess client, no retries (see `DECISIONS.md` D12). |
| `tv_signals.py` | TradingView Pine signal parity for `/api/tv/signal` etc. |
| `signal_context.py` | Bloomberg-data wheel-opportunity scorer (`build_entry_context`, `build_exit_context`). |
| `signals.py` | Composite signal aggregator (legacy framework — `IVRankSignal`, `TrendSignal`, `ProfitTargetSignal`, `StopLossSignal`, `DTESignal`, `EventFilterSignal`). |
| `news_sentiment.py` | News ingestion + scoring. The only news-stack module on the EV path. |

### Data layer

| Module | Purpose |
|---|---|
| `data_connector.py` | `MarketDataConnector` — Bloomberg-CSV provider. Default when `SWE_DATA_PROVIDER` is unset. |
| `theta_connector.py` | Theta Terminal v3 connector. Tier-aware fallbacks; chunked history; EOD endpoint for unlimited windows. |
| `data_integration.py` | Provider-selection helpers; `get_current_risk_free_rate`. |
| `external_data/` | External data connector subpackage. |

### Core math

| Module | Purpose |
|---|---|
| `option_pricer.py` | Black-Scholes-Merton pricing, Greeks (1st/2nd/3rd order), IV solver (Newton-Raphson + Brent). |
| `binomial_tree.py` | Binomial-tree pricer. |
| `monte_carlo.py` | Block bootstrap, jump-diffusion, Longstaff-Schwartz American pricing. |
| `volatility_surface.py` | SVI calibration, `VolatilitySurfaceBuilder`. **live** (A2, 2026-05-30) — wired in fail-loud via `SurfaceDataUnavailable` / `require_surface`; first caller `scripts/diagnose_iv_surface.py` (see `DECISIONS.md` D9). |
| `model_validation.py` | Textbook + property tests for pricing models. |
| `shared_valuation.py` | Unified labeling — `simulate_option_trade`, `simulate_wheel_cycle`, `TradeOutcome`. |

### Risk + sizing

| Module | Purpose |
|---|---|
| `risk_manager.py` | Position sizing (Kelly, fractional Kelly), sector exposure manager, hierarchical risk parity, portfolio Greeks. |
| `stress_testing.py` | Historical + hypothetical scenarios; `StressTester`, `StressTestReport`. |

### Trackers

| Module | Purpose |
|---|---|
| `wheel_tracker.py` | Position-lifecycle bookkeeping: `WheelPosition`, `PositionState`. Audit-VIII fixed P&L double-count and orthogonalised the three ledgers (realized_pnl, transaction_costs, stock_basis). |
| `portfolio_tracker.py` | Portfolio-level holdings, transactions, returns; `PortfolioSnapshot`, `PerformanceMetrics`. |
| `portfolio_intelligence.py` | SEC / 13F portfolio context. |
| `performance_metrics.py` | Sharpe / Sortino / drawdown reports. |

### Infra / config / display

| Module | Purpose |
|---|---|
| `policy_config.py` | Runtime policy knobs. |
| `contracts.py` | Dataclasses for trade I/O. |
| `observability.py` | Structured logging. |
| `dependency_check.py` | Bootstrap utility. |
| `payoff_engine.py` | Payoff diagrams (display). |
| `trade_memo.py` | Ollama-driven memo / summary (72B / 32B local models). |

### `engine/__init__.py` re-exports

Currently exports the *legacy* quant layer (option_pricer, monte_carlo,
risk_manager, regime_detector, signals, stress_testing, transaction_costs,
volatility_surface, wheel_tracker, portfolio_tracker, etc.) but **does
not re-export the modern decision-layer entry points**.

To use the authoritative layer, import via full submodule paths:

```python
from engine.ev_engine import EVEngine
from engine.wheel_runner import WheelRunner
from engine.candidate_dossier import EnginePhaseReviewer
from engine.dealer_positioning import MarketStructure
from engine.tradingview_bridge import (
    FilesystemChartProvider,
    PlaywrightChartProvider,
    ChainedChartProvider,
)
```

Recommendation: extend the package `__all__` to include these modern
symbols. Held back from this review pass because touching
`engine/__init__.py` ripples through every import site.

---

## `advisors/` — investment committee

Status: live, advisory-only. Per CLAUDE.md §2 the committee cannot
upgrade a negative-EV verdict; audit-VIII added a
`tradeable_endpoint="/api/candidates"` and `ev_anchored: bool` to
prevent shadow synthetic trades from leaking into the committee
output.

| Module | Role |
|---|---|
| `committee.py` | Runs members; aggregates verdicts. |
| `buffett.py` / `munger.py` / `simons.py` / `taleb.py` | Per-investor heuristics. |
| `base.py` | Abstract advisor base class. |
| `schema.py` | Pydantic schemas for advisor I/O. |
| `scorecard.py` | Scorecard structure. |
| `integration.py` | Engine-side bridge. |

---

## `scripts/` — pulls + diagnostics

Two flavours: data pullers (`pull_*.py`) and diagnostics
(`diagnose_*.py`, `feature_smoke_test.py`, `*_health_check.py`,
`probe_theta_capabilities.py`). Plus Bloomberg Excel extractor assets
(`bloomberg_*.bas`, `bloomberg_*.vba`, `*_formulas.txt`,
`bloomberg_bql_pulls.md`) that drive the Bloomberg Terminal Excel
export workflow.

Key scripts:

| Script | Purpose |
|---|---|
| `pull_all.py` | Orchestrates every `pull_*.py` step; respects `--skip` for tier-blocked endpoints. |
| `backfill_features.py` | Rebuilds the `data/features/**` shards (1.2 GB total; AAPL is the in-git sample). |
| `diagnose_candidates.py` | Funnel report for zero-trade debugging. **Default `tickers=None` is full-universe and exceeds the 45 s Cowork bash timeout** — pass an explicit short list. See `docs/DATA_POLICY.md` §7 (sandbox-vs-laptop) and `CLAUDE.md`'s fresh-session bring-up. |
| `feature_smoke_test.py` | 127 checks across 26 sections (~107 PASS / 0 FAIL / ~20 SKIP on the laptop). |
| `theta_backfill.py` | Tier-aware bulk backfill with circuit breakers. |
| `theta_health_check.py` | Connectivity + Bloomberg fallback probe. |
| `probe_theta_capabilities.py` | Regenerates `data_processed/theta_capabilities.json`. |
| `quant_benchmark_gate.py` | Gate run for benchmark suite. |
| `validate_environment.py` | Local environment sanity (wired into CI). |
| `check_manifest_coverage.py` | CI guard — fails the build if any tracked file is absent from `FILE_MANIFEST.md` or vice versa (the gate behind `DECISIONS.md` D14's tiered layout). |
| `setup-terminal.{sh,ps1}` | Parallel-session env loader (bash / PowerShell) — sources per-terminal `SWE_API_PORT`, `COVERAGE_FILE`, `PYTEST_CACHE_DIR`, and three more vars. See `DECISIONS.md` D15. |

---

## `dashboard/` — Next.js v15

Live UI consumed by `engine_api.py`. Source under `dashboard/src/`;
`node_modules/` and built `.next/` are gitignored. The repo also
keeps a legacy Python CLI dashboard at `dashboard/quant_dashboard.py`
that the root README still references — it is not the primary UI.

Note: `dashboard/README.md` describes a "FinanceNews — AI Financial
News Platform". The directory was reused; the README was never
updated. See `PROJECT_STATE.md` §5.

---

## `tradingview/` — Pine + alert assets

| File | Purpose |
|---|---|
| `smart_wheel_signals.pine` | Pine indicator. |
| `alert_payload_schema.json` | Webhook payload schema. |
| `README.md` | Tradingview-side setup. |

---

## Other top-level dirs

| Dir | Purpose | Status |
|---|---|---|
| `financial_news/` | Standalone news platform — RSS connectors, clustering, processing, sources, storage, scheduler, UI. **Not on the EV path.** | research / parallel project |
| `news_pipeline/` | Browser-agent news pipeline that drives `morning_run.py`: scrapers, browser_agents, local_llm, orchestrator, publisher, recovery, security, slo. | live (operational), but not on the EV path |
| `local_agent/` | Local AI agent + Streamlit UI; agents, browser, mcp_server, memory, ui. | experimental |
| `ml/` | `wheel_model.py`, `earnings_model.py`, `model_governance.py`. | research |
| `backtests/` | `simulator.py`, `walk_forward.py`. | research |
| `tradingview/` | Pine indicator + webhook schema (above). | live |
| `tests/` | `test_*.py` files + `quant_benchmarks.py` shared fixtures. See `TESTING.md` for the taxonomy, launch-blocker subset, and live counts. | live |
| `data/`, `data_processed/`, `data_raw/` | See `docs/DATA_POLICY.md` §2 for the provider matrix and what is committed vs. regenerable. | live |
| `docs/` | The documentation set — operational, reference and design-contract docs. See `FILE_MANIFEST.md` for the full per-file listing. | live |
| `config/` | `settings.py`. | live |
| `utils/` | `data_validation.py`, `dates.py`, `health.py`, `logging_config.py`, `metadata.py`, `security.py`. | live |
| `notebooks/` | Exploration. | research |
| `src/` | **Phantom scaffold.** Empty `execution/`, `models/`, `risk/` packages; partial `data/` and `features/`. Do not extend. See `PROJECT_STATE.md` §4. | deprecated |
| `models/` | `ml/wheel_model.py`'s default model-output directory; empty in git. | live |
| `archive/` | Superseded / point-in-time artifacts; see `archive/README.md`. | reference |
