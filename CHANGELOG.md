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

## 2026-05 — Theta data refresh + tooling visibility

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
