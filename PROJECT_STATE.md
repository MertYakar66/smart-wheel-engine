# Project State

**Last updated:** 2026-05-06 (against `origin/main` at `3754779`,
post-coverage-push merging PRs #63–#69).

This file records *temporal* state — what is authoritative now, what is
in progress, what is deprecated. It is the half-life partner of
`CLAUDE.md` (which is the *structural* contract). Update this file
when you finish a meaningful unit of work or discover that something
described here is no longer accurate.

---

## 1. Authoritative — do not bypass

| Module | Public entry | Locked by |
|---|---|---|
| `engine/ev_engine.py` | `EVEngine.evaluate` (line 237) | `tests/test_audit_invariants.py`, `tests/test_audit_viii_*` |
| `engine/wheel_runner.py` | `WheelRunner.rank_candidates_by_ev` (line 448) | `tests/test_authority_hardening.py`, `tests/test_audit_viii_real_data_smoke.py` |
| `engine/candidate_dossier.py` | `EnginePhaseReviewer` (line 109), rules R1–R6 | `tests/test_dossier_invariant.py` |
| `engine_api.py` | HTTP API on `:8787`; 32 endpoints listed in the file header | `tests/test_tv_api.py`, `tests/test_tv_dossier.py` |

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

- Theta `iv_surface/` (snapshot dir) coverage is **28/503 tickers**
  (5.6% — mega-caps + sector ETFs only). Unchanged.
- `iv_surface_history/` (history dir, distinct from the snapshot)
  is now **381/503 tickers** on disk after the 2026-05-04 pull (see
  §3.4). 122 tickers were rejected by strict mode (partial-coverage
  per PR #58 design — prefer loud failure to silent partial data).
- The SVI tools in `engine/volatility_surface.py`
  (`VolatilitySurfaceBuilder`, `create_empirical_surface`,
  `SVICalibrator`) are exported but **have zero non-test callers as of
  2026-04-25**. Audit pass: no live silent-fallback paths.
- `get_iv_surface()` returns an empty DataFrame on missing data, not a
  flat-IV stub.
- **Open decision:** before wiring SVI surfaces into a feature or the
  decision path, pick a missing-data contract — fail loudly on the
  ~122 uncovered tickers (snapshot: ~475), or use a clearly-named
  fallback (`flat_iv_fallback`, never silent).

### `.claude/` SessionStart hook

- `0e451f6 feat: .claude/settings.json SessionStart hook + dossier invariant test`
- `40d1ec4 fix: SessionStart hook — gate pip install behind venv-or-opt-in`
- The hook prints provider warning, dataset presence, theta manifest
  recency, missing-deps reminder, and connector class. Active and
  tested as of today.

### Theta data refresh — 2026-05-04 / 2026-05-05

**Pull session:** 8h 13min wall-clock via patched `pull_all.py`. Final
result: 5 OK / 3 FAIL / 1 SKIP. The three FAILs are loud strict-mode
rejections, not data loss — most data landed on disk before the FAIL
was raised. Smoke test after the pull: 127 total / **111 PASS / 0 FAIL
/ 16 SKIP** (all expected per CLAUDE.md §3 tier matrix).

**On-disk state of `data_processed/theta/` (new vs. prior session):**

| Directory | Tickers / Files | Δ vs 2026-04-23 manifest |
|---|---|---|
| `chains/` | 495 | refreshed |
| `stocks_eod/` | 493 | refreshed |
| `iv_history/` | 493 | refreshed |
| `iv_surface/` (snapshot) | 28 | unchanged (mega-caps + sector ETFs only) |
| **`iv_surface_history/`** | **381** | **NEW** (was absent; first-time backfill) |
| **`options_flow/`** | **499** | **NEW** (PCR/OI/unusual-volume per ticker) |
| `index_options_chains/` | 6 | SPX/SPXW/NDX/RUT/DJX/XSP |
| `index_options_surfaces/` | 6 | SPX/SPXW/NDX/RUT/DJX/XSP |
| `vix_family.parquet` | ~12y | refreshed |

**Persistent failed-symbol set** (don't reopen as bugs — root cause
known):

- `BF.B`, `BRK.B` — dotted-ticker symbols. Format is *already*
  normalized in `engine/theta_connector.py:134-160` (`_normalise_theta_symbol`
  maps `BRK-B` / `BRK/B` / `BRK B` → `BRK.B`). HTTP 472 on these
  means Theta has no historical data for them at this tier, **not** a
  format rejection. Confirmed by audit on 2026-05-05.
- `DAY` (Dayforce) — fails on both yfinance (`Quote not found for
  symbol: DAY`) and Theta v3 (HTTP 472 across all expirations).
  Symbol changed upstream. Will resolve when constituents
  list rotates.
- `NVR` — Theta v3 has no options_flow data. Per-tier coverage gap.

**Manifest staleness:** `_manifest.json` last entry is still
`2026-04-23T04:53:50` because `pull_all.py` invokes individual pullers
that don't always update the manifest. Disk state has clearly
advanced past the manifest. Don't trust the manifest's `ran_at` as
the freshness signal — compare directory mtimes instead.

### `pull_all.py` streaming visibility — shipped 2026-05-04

- **Problem:** `run_step()` previously used
  `subprocess.run(capture_output=True)`, buffering all child output
  until step exit. A 7-hour `theta_flow` step looked identical to a
  hung process for 7 hours.
- **Fix (PR #61, `6c0543d`):** switched to `subprocess.Popen` with
  line-buffered `stdout=PIPE` / `stderr=STDOUT`, iterate `proc.stdout`,
  re-print each line with `│` prefix to mark child output. Pass `-u`
  to children to defeat their own re-wrapped `TextIOWrapper` buffer.
  Also `write_through=True` on the parent's `TextIOWrapper` so headers
  flush immediately.
- **Behavioural impact:** every per-ticker progress line from each
  puller is now visible in real time. Verified on the 2026-05-04 run
  — saw `[ 250/503] INTC OK rows=20`-style lines stream live, caught
  the `DAY` upstream miss within 2 minutes instead of 2 hours.
- **No throughput change.**
- See also `docs/THETA_PULL_SESSION_NOTES.md` (PR #60, `cf92578`) —
  operational checklist for the laptop bring-up, captures HTTP 478
  / dotted-ticker / probe-timeout gotchas surfaced during this work.

### Foundation pass — 2026-05-05

Repository organisation + AI-agent-onboarding pass. **Zero
trading-logic changes.** Local was 3 commits behind `origin/main` at
start; fast-forwarded cleanly to `433231f`.

**Hygiene shipped:**

- `.gitignore` extended with explicit `Theta/` (installed software +
  creds.txt — see `DECISIONS.md` D6), `**/ThetaTerminal*.jar`,
  `*/creds.txt`, `tradingview/tradingview-mcp-jackson/` (vendored
  separate git repo + node_modules), and analyst-deliverable
  patterns (`tradingview/research/*.docx`,
  `tradingview/models/*.xlsx`, `tradingview/screenshots/`).
- `data/bloomberg/.DS_Store` untracked — the only stray
  Finder-metadata file in git.

**Docs added at repo root (AI-agent handoff path):**

| File | Purpose |
|---|---|
| `CHANGELOG.md` | Human-readable summary of meaningful changes; companion to `PROJECT_STATE.md` (current) and `ROADMAP.md` (next). |
| `DECISIONS.md` | 11 architectural decisions (D1–D11) with **Why** + **Rejected alternatives** + **Pinned by**. |
| `ROADMAP.md` | Tracks A (decision-layer correctness), B (documentation drift to repair), C (hygiene + governance follow-ups), D (out of scope). |
| `DATA_POLICY.md` | Three data tiers, provider matrix, what never enters git, point-in-time discipline, refresh procedures, drive-mount caveats. |
| `TRADINGVIEW_INTEGRATION.md` | Parent guide covering both engine bridge (Pine indicator + webhook → EV) and analyst workspace (Claude-driven TradingView Desktop via MCP). |
| `LAUNCH_READINESS.md` | Consolidated launch-blocker checklist: hard EV invariant, four authoritative routes, dossier R1–R6, pre-merge checklist. |
| `COMMIT_GUIDE.md` | `type(scope):` + `Changed/Why/Tested/Tried-but-rejected/Unresolved/AI-handoff` body format with worked example. |

**Existing docs touched (cross-refs / drift):**

- `AGENTS.md` — extended with a **Read on demand** section pointing at
  the new docs; updates to **What you can change** /
  **What requires explicit ask** / **Commit format**.
- `README.md` — top-of-file AI-agent banner pointing at AGENTS.md and
  the new doc set, plus an explicit note that the body below is
  partially out of date (the full repair is deferred to ROADMAP B1).
- `CONTRIBUTING.md` — top-of-file AI-agent banner; commit-format
  section now defers to `COMMIT_GUIDE.md`.
- `tradingview/README.md` — fixed broken link to a non-existent
  `TRADINGVIEW_INTEGRATION_REPORT.md`; now points at the new parent
  doc and `docs/TRADINGVIEW_MCP_INTEGRATION.md`.

**Untracked but ready to track on next branch** (ROADMAP C2):

- `tradingview/CLAUDE.md`, `tradingview/OVERVIEW.md`,
  `tradingview/launch-tradingview-cdp.sh`,
  `tradingview/{research,models,pine}/.gitkeep`.

**Not changed (deliberately):**

- `engine/`, `engine_api.py`, `advisors/`, dashboard runtime,
  `scripts/`, `tests/` — zero code edits this pass.
- `CLAUDE.md` — user-maintained per AGENTS.md contract.
- `pyproject.toml` — known-stale entries are tracked in ROADMAP B5.
- `data/bloomberg/sp500_*.csv`, `treasury_yields.csv` — local
  modifications from a yfinance refresh; the
  track-vs-gitignore decision is open (ROADMAP C1).

**Verification:** decision-layer modules
(`engine.ev_engine`, `engine.wheel_runner`,
`engine.candidate_dossier`, `engine.tradingview_bridge`,
`engine.tv_signals`, `engine.dealer_positioning`) all import
cleanly post-pass.

### Coverage push — 2026-05-05/06 (PRs #63–#69)

Six PRs landed cleanly into main, taking the test suite from
**1,106 → 1,580 tests (+474)** and the CI-scope coverage
(`src + engine + advisors + financial_news`) from baseline ~63%
to **82%**. Test runtime stable at ~1m50s. Zero behaviour
regressions. **CHANGELOG `2026-05` is the source of truth for
per-PR detail.**

**The `--cov-fail-under` gate moved 70 → 80** in this wave:
- `pyproject.toml [tool.coverage.report] fail_under = 80`
- `.github/workflows/ci.yml --cov-fail-under=80`
- 80 pins the floor we earned (82% baseline) with 2pp buffer for
  normal PR-to-PR noise. See `DECISIONS.md` D10 for the rejected
  alternatives (75 = too loose, 82 = brittle).

**Real bug surfaced and fixed:** `engine/event_gate.py`
`from_bloomberg_calendar` was admitting `pd.NaT` rows (only
filtered Python `None`), which then crashed `is_blocked()` on a
NaT-vs-`date` comparison. Three loops fixed in `#65`. Exactly the
class of latent bug D10's "coverage as forcing function" framing
predicts.

**ROADMAP Track E5b cancelled.** The remaining ~10pp to 90% lives
in `news_pipeline/{browser_agents,scrapers,orchestrator}.py` —
research-tier code (`MODULE_INDEX.md` "Other top-level dirs"), not
on the EV decision path, and would require ~hundreds of lines of
Playwright + aiohttp mock fixture infra. The engine consumes those
modules' outputs via files on disk; the consumers are already
covered. Pushing for higher % on the producers would be coverage
theater.

**Lint debt (44 errors)** still red on main as of `3754779`. PR
#64 closed 187/229 mechanical errors but the residual judgement-
required tail (UP038 unsafe-fixes, B904 raise-from, B023 closure
trap, F841 unused locals, B019 mutable defaults, F821 undefined
names, E741 ambiguous names) needs a follow-up PR. Tracked in
`ROADMAP.md` Track F.

**Open follow-ups (small, self-contained):**

| # | Item | Status |
|---|---|---|
| 1 | Lint debt cleanup (44 → 0 errors) | next |
| 2 | yfinance CSV refresh stash decision | next — real revisions, not noise; ROADMAP C1 |
| 3 | TRADINGVIEW_INTEGRATION.md MCP repo URL placeholder | needs user input |
| 4 | Original Theta walkthrough — `probe_theta_capabilities.py` step 2 | needs laptop run |
| 5 | `StrangleTimingWithIV.score_entry_with_iv` connector signature gap | pinned by strict xfail in `tests/test_strangle_timing.py::test_score_entry_with_iv_real_connector_signature_mismatch`; calls `connector.get_realized_vol`, `get_current_iv`, `get_vix_level`, `get_vix_contango` (none exist) and passes `as_of=` to `get_ohlcv` (not in signature). Resolution: extend `MarketDataConnector` with the IV/VIX accessors **or** rewrite the overlay to use `get_iv_history` / `get_vix` / `get_vol_risk_premium`. Remove the xfail in the same commit |

## 4. Deprecated / phantom — do not extend

- `src/` — phantom architecture from an earlier scaffold. Real
  decision-layer code is in `engine/`, not `src/`. The pyproject
  entrypoint `wheel = "src.cli:app"` targets a `src/cli.py` that does
  **not exist**; coverage and `[tool.hatch] packages = ["src"]`
  treat `src/` as load-bearing. Empty subpackages: `src/execution/`,
  `src/models/`, `src/risk/`. Partially populated: `src/data/` (only
  `schemas.py`, `validators.py`), `src/features/` (mirrors what
  `data/features/` consumes), and `src/backtest/` (`wheel_backtest.py`).
  Plan: either fully remove or fully repopulate. Until then, do not
  add new modules under `src/`.
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
