# Project State

**Last updated:** 2026-05-23 (against `origin/main` at `643f1af`,
post-D14 doc reorg and the parallel-session work campaign; see
`CHANGELOG.md` 2026-05 and the coordination board on GitHub
issue #113 for the per-PR detail).

This file records *temporal* state — what is authoritative now, what is
in progress, what is deprecated. It is the half-life partner of
`CLAUDE.md` (which is the *structural* contract). Update this file
when you finish a meaningful unit of work or discover that something
described here is no longer accurate.

> ⚠️ **Real-money deployment gate — read `docs/PRODUCTION_READINESS.md`
> before any decision to operate this engine against a real brokerage
> account.** The engine's predictive signal is verified (Spearman ρ ≈
> 0.22, scale-invariant), but three blockers remain unresolved as of
> `e504801`: F4 tail-risk widening, D17 live-wire to `engine_api.py`,
> and strategy capacity at scales above $100k. `docs/LAUNCH_READINESS.md`
> covers code-quality merge gates; `docs/PRODUCTION_READINESS.md`
> covers commercial deployment gates. They are complementary, not
> substitutes.

---

## 1. Authoritative — do not bypass

| Module | Public entry | Locked by |
|---|---|---|
| `engine/ev_engine.py` | `EVEngine.evaluate` | `tests/test_audit_invariants.py`, `tests/test_audit_viii_*`, `tests/test_evengine_event_lockout.py`, `tests/test_dealer_multiplier_evengine_integration.py` |
| `engine/wheel_runner.py` | `WheelRunner.rank_candidates_by_ev` | `tests/test_authority_hardening.py`, `tests/test_audit_viii_real_data_smoke.py`, `tests/test_f4_tail_risk_gap.py`, `tests/test_consume_ranker_row_anchor.py` |
| `engine/candidate_dossier.py` | `EnginePhaseReviewer`, rules R1–R6 | `tests/test_dossier_invariant.py` |
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
- **Implementation seam:** `engine/tradingview_bridge.py` —
  `MCPChartProvider(ChartContextProvider)` + `engine/mcp_client.py`
  `MCPCLIClient` (the `tv`-CLI transport).
- **Contract test:** `tests/test_dossier_invariant.py::test_mcp_provider_*`
  is active (the import guard auto-activated once `MCPChartProvider`
  shipped).
- **Status:** Stages 1–3 landed. Stage 1 (offline contract skeleton)
  and Stage 2 (`MCPCLIClient`) merged via PR #95; Stage 3 wires
  `MCPChartProvider` into `build_default_provider` behind the
  `SWE_USE_MCP_CHART` env var (opt-in; co-located transport — see
  `DECISIONS.md` D13). `MCPCLIClient` is still written against the
  *documented* `tv` CLI — `TODO(live-verify)` markers in
  `engine/mcp_client.py` flag fields to confirm against a live
  TradingView Desktop + tradingview-mcp server.

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
/ 16 SKIP** (all expected per the `docs/DATA_POLICY.md` §2 tier matrix).

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

**Lint debt — closed.** At commit `3754779` the residual was 44
judgement-required errors after PR #64 (`1fb2c33`) closed 187/229
mechanical fixes. The tail (B904 raise-from, B023 closure trap,
F841 unused locals, B019 lru_cache-on-method, F821 undefined names,
E741 ambiguous names + UP/I/F/C one-offs) was cleared by PR #79
(`9e15dbf`) on 2026-05-15 — 75 → 0 across the CI scope (additional
errors had accumulated between `3754779` and PR #79). ROADMAP Track
F is closed.

**Open follow-ups (small, self-contained):**

| # | Item | Status |
|---|---|---|
| 1 | Lint debt cleanup (44 → 0 errors) | done — PR #79 (`9e15dbf`) |
| 2 | yfinance CSV refresh stash decision | next — real revisions, not noise; ROADMAP C1 |
| 3 | TRADINGVIEW_INTEGRATION.md MCP repo URL placeholder | done — PR #100 (`ad1bbbc`) 2026-05-19; real URL + live-verified MCP shapes |
| 4 | Original Theta walkthrough — `probe_theta_capabilities.py` step 2 | needs laptop run |
| 5 | `StrangleTimingWithIV.score_entry_with_iv` connector signature gap | done — commit `210463d` 2026-05-20; overlay rewritten to use the real connector API (`get_ohlcv(end_date=…)`, `get_iv_rank`, `get_vol_risk_premium`, `get_vix_regime`); xfail replaced with the green `test_score_entry_with_iv_against_real_connector` regression test |

### Repository structure reorg (D14) — 2026-05-21

A structure-only documentation reorganisation (`DECISIONS.md` D14).
**Zero code or behaviour changes; no doc's substantive content was
rewritten.**

- The repo root now holds only the Tier-1 entry docs (`AGENTS.md`,
  `CLAUDE.md`, `README.md`) and the Tier-2 state/index docs
  (`PROJECT_STATE.md`, `MODULE_INDEX.md`, `TESTING.md`, `DECISIONS.md`,
  `COMMIT_GUIDE.md`, `FILE_MANIFEST.md`, plus `CHANGELOG.md` and
  `ROADMAP.md`).
- Operational and reference docs moved into `docs/`: `DATA_POLICY.md`,
  `LAPTOP_SETUP.md`, `LAUNCH_READINESS.md`, `THETA_INSTRUCTIONS.md`,
  `TRADINGVIEW_INTEGRATION.md`, `USAGE_TEST_LEDGER.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `Claude_Prompting_Master_Guide.md`.
- Three stale / superseded docs moved to `archive/2026-05/`:
  `OptionsEngine.txt`, `docs/ARCHITECTURE.md`,
  `docs/DATA_COLLECTION_REPORT.md` (see `archive/README.md`).
- Added `FILE_MANIFEST.md` (the exhaustive per-file index) and
  `.claude/commands/` (two thin slash-command wrappers). Removed the
  empty `validation/` placeholder directory.
- **Deferred to named follow-on PRs** so this PR's diff stays a pure
  move: a CLAUDE.md lean-rewrite, and a doc-truthfulness reconciliation
  pass for known-stale facts (code line numbers, counts, the §5 drift
  list below). This PR neither fixed nor propagated those.

## 4. Deprecated / phantom — do not extend

- `src/` — partial scaffold from an earlier layout. Real
  decision-layer code is in `engine/`, not `src/`. The
  `wheel = "src.cli:app"` pyproject entrypoint was removed under
  ROADMAP B5 (no `[project.scripts]` section today); `src/` remains
  a build target via `[tool.hatch.build.targets.wheel] packages = ["src"]`
  and a coverage source via `[tool.coverage.run] source = ["src", …]`
  (`pyproject.toml:68-72, 109`). Empty subpackages: `src/execution/`,
  `src/models/`, `src/risk/`. Partially populated:
  `src/data/` (only `schemas.py`, `validators.py`),
  `src/features/` (`technical.py` is **live** — imported by
  `engine/strangle_timing.py:31`, `engine/tv_signals.py:48`, and
  `engine_api.py:1161`; other modules dormant), and
  `src/backtest/` (`wheel_backtest.py` is a **heuristic** wheel
  backtester — explicitly §2-non-compliant per its top-of-file
  banner — distinct from the EV-driven path under
  `WheelRunner.rank_candidates_by_ev` + `WheelTracker.consume_ranker_row`;
  see `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`).
  Plan: either fully remove or fully repopulate. Until then, do not
  add new modules under `src/`.
- `models/` — retained, not deleted as a placeholder: `ml/wheel_model.py`
  names `models/wheel_entry_model.joblib` as its default model-output
  path, so `models/` is a *referenced* path. (`ml/wheel_model.py` does
  `mkdir(parents=True)` on save, so the directory is not strictly
  required to pre-exist — it was kept on the referenced-path basis, not
  a runtime need.) The genuinely zero-reference `validation/` placeholder
  was removed in D14.
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

_All previously-listed drift entries here are now closed — see the
closer paragraph below for the route to each fix._

The entries that previously lived here for `README.md`,
`docs/CONTRIBUTING.md`, and `dashboard/README.md` were closed by
the entry-doc repair pass — see `ROADMAP.md` Track B (B1, B2, B4).
The `pyproject.toml` drift entry (broken `wheel = "src.cli:app"`
script + wrong `packages = ["src"]`) was closed by ROADMAP Track B5
— see `CHANGELOG.md` 2026-05. The `engine/__init__.py`
modern-decision-layer re-export entry was closed by `ROADMAP.md`
Track A3 — also see `CHANGELOG.md` 2026-05.

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
