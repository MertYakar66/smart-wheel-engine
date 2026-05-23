# Changelog

A human-readable summary of meaningful changes to Smart Wheel Engine,
grouped by month and theme. For per-commit detail, run
`git log --oneline` or `git log --grep "<keyword>"`.

This file is the public-history companion to:
- `PROJECT_STATE.md` — current state of the world
- `DECISIONS.md` — the *why* behind structural choices
- `ROADMAP.md` — what is intentionally next

Format: `Added` / `Changed` / `Fixed` / `Deprecated` / `Docs` /
`Infra`. Each entry carries the commit SHA where the change shipped.

---

## 2026-05 — Coverage push + lint debt mechanical fix + foundation pass

### Added
- **Coverage suite expansion: 1,106 → 1,580 tests (+474)** across six
  PRs (`#63`–`#69`). Focus: pin invariants on EV-adjacent modules
  before chasing edge-case coverage. The CI-scope global went from
  baseline ~63% to **82%** (`src + engine + advisors + financial_news`).
  Per-PR breakdown:
  - **`#63` foundation pass** (`5c4f58b`) — 7 new root-level docs
    (CHANGELOG, DECISIONS, ROADMAP, DATA_POLICY, TRADINGVIEW_INTEGRATION,
    LAUNCH_READINESS, COMMIT_GUIDE), gitignore hygiene (Theta/,
    tradingview-mcp-jackson, analyst .docx outputs), .DS_Store untrack.
    Zero code edits.
  - **`#64` lint mechanical** (`1fb2c33`) — `ruff check --fix` +
    `ruff format` on all 9 CI-scope dirs. Closed 187 of 229 ruff
    errors (179 → 44 remaining). 62 .py files reformatted. Pre-existing
    lint failure on main reduced 76%, not eliminated.
  - **`#65` coverage E1** (`f395aa6`) — 7 EV-adjacent modules to
    88-100%: `policy_config` 100%, `event_gate` 97%, `observability`
    98%, `news_sentiment` 94%, `contracts` 94%, `earnings_drift` 88%,
    `tail_risk` 88%. Found and fixed an `event_gate.from_bloomberg_calendar`
    NaT crash (`is None` doesn't catch `pd.NaT` → later `is_blocked()`
    crashed on NaT-vs-date comparison).
  - **`#66` coverage E2** (`354f440`) — 4 external-data adapters
    (`cboe`, `edgar`, `fred`, `yfinance`) to 97-98% via `requests-mock`
    HTTP stubbing.
  - **`#67` coverage E3** (`10ddc9d`) — `engine/theta_connector.py`
    11% → **78%** (+67pp on a 470-stmt file). Mocked v3 endpoints
    via `requests-mock`; `tmp_path` for the Bloomberg fallback.
  - **`#68` coverage E4** (`526ce67`) — `event_calendar` 31% → **88%**
    (full new test file), `risk_manager` 63% → **83%** (extended from
    23 to 55 tests; covers `SectorExposureManager`,
    `HierarchicalRiskParity`, `run_stress_tests`).
  - **`#69` coverage E5a** (`3754779`) — `news_pipeline/recovery/*`:
    `checkpoints` 28% → 94%, `fallbacks` 0% → 92%, `health` 26% → 63%
    (sync paths only; async aiohttp paths intentionally deferred —
    see ROADMAP Track E5b cancellation).

### Changed
- **`pyproject.toml [tool.coverage.report] fail_under` 70 → 80**, and
  the matching CI workflow `--cov-fail-under` flag. Pins the floor
  earned by the coverage push (current baseline 82%, 2pp buffer for
  normal PR-to-PR noise). See `DECISIONS.md` D10.
- **`pyproject.toml` truth-up — closes ROADMAP Track B5.**
  Three concurrent fixes on the same file, all in the same commit:
  (1) removed `[project.scripts] wheel = "src.cli:app"` — the target
  `src/cli.py` does not exist (verified pre-edit); no consumer
  relied on the script. (2) Removed `prefect>=2.14.0` and
  `ib_insync>=0.9.86` from `[project.dependencies]` — both have
  zero imports in any tracked Python file (`git grep -l prefect`
  returns only `docs/CONTRIBUTING.md` and `pyproject.toml`; same
  for `ib_insync` plus two archived audit docs). `ib_insync` would
  also have violated the CLAUDE.md NEVER-rule "no broker
  integration." `streamlit` is retained — it is a real consumer
  (`local_agent/ui/streamlit_app.py`). (3) Expanded
  `[tool.hatch.build.targets.wheel] packages` from `["src"]` to
  `["engine", "advisors", "data", "financial_news", "news_pipeline",
  "src"]`. `src` is retained per DECISIONS.md D2 (still imported by
  tests + four production modules; the migration window has not
  closed). The built wheel now contains the live install surface
  instead of just the deprecated phantom.

### Fixed
- **Spread-penalty severe-tier never fired.** Inverted threshold ladder
  in `engine.transaction_costs.calculate_slippage` (lines 138-143) silently
  degraded the 2.0× "severe" multiplier to 1.5× for every option with
  `spread_pct > 0.50`. Branch order swapped; the 2.0× tier now applies
  correctly on the worst-spread tail. EV impact for wheel-sized orders
  is ~$4.50/contract on the affected tickers; large-cap liquid names
  (audit smoke list) are unaffected because their spread_pct stays
  below 0.30. Bug surfaced via #75's xfail; resolved in this PR.
  Refs: 2026-05-08 audit F7 follow-up, PR #75 chore commit.
- **`engine/event_gate.py`** — `from_bloomberg_calendar` defensive
  guard against `pd.NaT` rows (was only filtering Python `None`).
  Three loops fixed (earnings / macro / dividends). Found via the
  Phase 1 coverage tests in `#65` — exactly the kind of latent bug
  D10's "coverage as forcing function" framing predicts.
- **`engine/theta_connector.py`** (issue #71, `7a1ac38`) — silent
  Bloomberg CSV substitution on per-endpoint Theta failures. The
  connector treated 30s read-timeouts on per-symbol
  `/v3/option/history/eod` calls as "ThetaTerminal not reachable"
  and fell back to CSV, contaminating downstream features with
  mixed-provenance data while the Terminal itself was healthy
  throughout. Fix: a per-instance probe + raise contract.
  `_fetch` now catches `(ConnectionError, RetryError, Timeout)`
  and routes to `_handle_network_failure`, which probes
  `is_terminal_alive` (5s GET on
  `/v3/option/list/expirations?symbol=SPY`). Probe healthy → raise
  `PerEndpointFailure` (typed exception carrying a `FailureRecord`).
  Probe also fails → set `self._terminal_down` (per-instance flag),
  return empty DataFrame, and the existing empty-df → super
  Bloomberg fallback handles the carve-out. Subsequent failures
  within the same instance short-circuit on the flag without
  re-probing. `get_failures()` returns + clears a per-instance
  accumulator that pullers will write to a JSON sidecar at
  end-of-run (C4). Per-puller wiring lands separately in this
  PR. See `DECISIONS.md` D11.

### Infra
- **`pyproject.toml [project.optional-dependencies] dev`** — declared
  `requests-mock>=1.11` and `responses>=0.25`. Phase 2/3 tests had
  been installed locally with `--break-system-packages` but never
  declared, causing CI's clean install to fail pytest collection on
  the `test_external_data_*.py` and `test_theta_connector_v3.py`
  files. (`770dfd1`)

### Docs
- **`DECISIONS.md` D10 rewritten** — "Tests pin invariants; coverage
  is secondary" → "Invariants first, then 80% line coverage as a
  forcing function for edge-case discovery." Records the rationale
  for the gate value and the 2026-05 coverage push.
- **`PROJECT_STATE.md` §3.7** added — records the coverage push
  outcome, the four still-open follow-ups (lint debt cleanup,
  yfinance stash decision, MCP repo URL, Theta walkthrough), and
  the explicit E5b non-pursuit rationale.
- **`ROADMAP.md` Track E** updated — E1–E5a moved to "shipped" with
  CHANGELOG cross-refs; E5b struck out with rationale; E6 (gate
  bump) shipped here.

### Deferred / not pursued
- **ROADMAP Track E5b** (browser_agents + scrapers + orchestrator
  coverage). Per `DECISIONS.md` D10's exclusion note: those modules
  are research-tier (`MODULE_INDEX.md` "Other top-level dirs"),
  not on the EV decision path, and would require ~hundreds of lines
  of Playwright + aiohttp mock fixture infrastructure to test plumbing
  the engine consumes via files on disk. Decision: don't chase the
  remaining ~15pp via browser-mock harness — keep focus on the EV
  contract. The 80% gate is the floor; doesn't pretend 90% is the
  target.

---

## 2026-05 (early) — Theta data refresh + tooling visibility

### Fixed
- **`scripts/pull_all.py`** now streams subprocess output in real time
  (`subprocess.Popen` + line-buffered + `write_through`). A long puller
  no longer looks identical to a hung process. (`6c0543d`, PR #61)
- **`scripts/pull_theta_iv_surface_history.py`** — shared connector
  across buckets; strict rejection of partial coverage (loud failure
  preferred to silent gaps). Per-bucket fallback to the next-nearest
  bucket with data when the requested bucket is empty. (`3f6fad1`,
  `a9ddb00`, PRs #58 / #59)
- **`scripts/pull_theta_indices_history.py`** — chunk requests to
  365-day windows; same-day incremental skip; rc=0 when everything
  is already up to date. (`1df5552`, `ecbe195`, PRs #56 / #57)

### Docs
- **`docs/THETA_PULL_SESSION_NOTES.md`** — laptop bring-up checklist
  for daily Theta refresh: HTTP 478 trap, AAPL smoke test, dry-run
  → live pull_all → smoke-test verify. Complements `LAPTOP_SETUP.md`.
  (`cf92578`, PR #60)
- **`PROJECT_STATE.md` §3.4 / §3.5** — records the 2026-05-04 / -05
  Theta refresh (8h13m, 5 OK / 3 FAIL / 1 SKIP, smoke test 111 PASS /
  0 FAIL / 16 SKIP) and the `pull_all.py` streaming fix. (`433231f`,
  PR #62)

---

## 2026-04 — Documentation foundation + AI-agent contract

### Added
- **`AGENTS.md`** — canonical read order for any AI agent entering the
  repo (Claude, Codex, Cursor, Copilot, Aider). Pins the hard EV rule
  from `CLAUDE.md` §2. (`348ebef`)
- **`PROJECT_STATE.md`** — temporal-state companion to the structural
  contract in `CLAUDE.md`. (`348ebef`)
- **`MODULE_INDEX.md`** — per-module purpose with role classification
  (authority / runner / reviewer / multiplier / input / data / tracker
  / infra / display / dormant). (`348ebef`)
- **`TESTING.md`** — test taxonomy + launch-blocker subset + "what to
  run when you change ___" map. (`348ebef`)
- **`docs/THETA_USAGE.md`** — consolidated Theta v3 reference
  (per-endpoint, tier matrix, Bloomberg fallbacks). (`22aa086`)
- **`docs/TRADINGVIEW_MCP_INTEGRATION.md`** — design contract for the
  pending MCP chart provider; pins the four hard invariants and M1
  scope. (`c064652`)
- **`.claude/settings.json` SessionStart hook** — every fresh Claude
  session now validates dataset presence, Theta manifest recency, and
  connector class. (`0e451f6`, refined in `40d1ec4`)

### Fixed
- **`.gitattributes`** pins LF line endings to stop CRLF/LF noise
  caused by Drive's sync. (`32c1c6d`)
- **`scripts/pull_theta_iv_surface_history.py`** uses the history
  endpoint (not snapshot) for back-series pulls. (`c2b1c29`, PR #55)
- **`engine/theta_connector.py`** — match v3 API contract: interval
  enum, history chunking, EOD endpoint for unlimited windows.
  (`85a026b`)
- Chain-quality issues drop the dealer overlay rather than dropping
  the whole ticker. (`a006c09`)

### Infra
- **Tier-aware Theta pullers** + yfinance Bloomberg fallbacks +
  feature smoke test. (`4af072c`)

---

## 2026-03 — Audit cycles consolidate the EV invariant

This series of audits hardened the rule that **no tradeable candidate
bypasses `EVEngine.evaluate`** (`CLAUDE.md` §2). Each audit shipped
behavioral changes and the tests that pin them; see `PROJECT_STATE.md`
§2 for the detailed table.

### Audit-VIII (`e4c30e1`)
- **Fixed** EV-path unit bugs (IV / risk-free-rate percent↔decimal
  normalisation).
- **Fixed** roll/close P&L double-count in `wheel_tracker.py`.
- **Fixed** committee authority leak (committee could shadow-rank
  synthetic trades).
- **Tests:** `test_audit_viii_unit_invariants.py`,
  `test_audit_viii_e2e.py`, `test_audit_viii_real_data_smoke.py`
  (20 new tests). Suite: 1087 passing / 0 failing (down from
  1067+1 / 578).

### Audit-VII (`506b348`)
- **Added** unified orchestrator + HMM regime wiring + Grok/X agent +
  news API + ML guard.

### Audit-VI (`7e1bda7`)
- **Closed** authority leaks across `tv` webhook / analyze / strangle
  / strikes / wheel_tracker. **Tests:** `test_authority_hardening.py`.

### Audit-V (`4afe7ea`, `48fe29b`)
- **Added** market-level dealer positioning (GEX, walls, gamma flip,
  regime). P0/P1 unify decision authority. Survivorship + chain
  quality + stress residual gates. **Tests:**
  `test_dealer_positioning.py`.

### Audit-IV (`2440891`)
- **Added** TradingView visual-layer bridge + candidate dossier
  (Mode B). **Tests:** `test_tv_dossier.py`.

### Audit-III (`81a42b1`)
- **Added** POT-GPD CVaR; 4-state Gaussian HMM; Nelson-Siegel skew
  dynamics; Student-t copula; event gate.

### Audit-II (`3be3f2a`)
- **Added** EV engine wired into runner; forward distributions;
  empirical surface; early-assignment-div; survivorship audit;
  calibration gate; sqrt impact.

### Audit (`8ca561c`)
- **Added** institutional EV engine. PIT bug fixes. TV webhook
  hardening.

---

## How to maintain this file

When you ship a meaningful change:

1. Add an entry to the **current month** section under the right
   bucket (`Added` / `Changed` / `Fixed` / `Deprecated` / `Docs` /
   `Infra`).
2. Lead with the file or feature, then a one-line description, then
   the commit SHA in backticks. PR number if relevant.
3. Don't restate every bug fix — only the ones that change behavior
   a future agent could be surprised by.
4. If the change retired a piece of the system, update
   `PROJECT_STATE.md` §4 (deprecated) at the same time.
5. New month? Start a new section header. Don't bury history under
   "current".
