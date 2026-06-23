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

## 2026-06-22 — Phase 0B broad-pull loaders (additive, dormant)

The §2-safe plumbing slice of the wiring campaign (`docs/WIRING_CAMPAIGN.md`
Phase 0B; branch `claude/phase0b-broad-pull-loaders-2026-06-22`). Makes the
broad-pull data *loadable* but wires it into **no** consumer — anything that
moves `EVEngine.evaluate` (Phase 1-3) is supervised, re-baseline-coupled, and
out of scope here.

### Added
- `data/broad_pull_loaders.py` — `BroadPullLoader`, read-only loaders for the
  27 net-new broad-pull datasets (gz handling, float32 downcast, logged
  winsorization of the manifest's outlier-flagged columns, lazy per-ticker
  access via normalized symbols). Dormant: not imported by the decision trio,
  any risk gate/reviewer, or `ConsolidatedBloombergLoader.load_all`.
- `data/bloomberg/broad_pull/` — the 27 net-new datasets integrated to their
  connector-read location (byte-identical to `staging/` on the broad-pull
  branch; the 0A frontier-refresh tails are deliberately excluded as EV-moving).
- `tests/test_broad_pull_loaders.py` — 53 tests: synthetic units for the loader
  logic + real-data tests pinning every dataset to the byte-verified manifest,
  plus a structural guard that no decision-path module consumes the loader.

### Docs
- `docs/DATA_INVENTORY.md` §6 is the byte census; `FILE_MANIFEST.md` +
  `docs/worklog/` updated. Coverage-omit adds the new loader (D10: research-tier
  ETL off the decision path).

## 2026-06-09 — D27 repository restructure for agent navigability

Full-repo structural pass (`DECISIONS.md` D27;
`docs/worklog/d27-repo-restructure-for-agent-navigability.md` is the task
record). Every folder read; changes were truth-restoration, not relocation.

### Docs
- Root index docs reconciled to the 2026-06 state; ROADMAP gained the
  "Open work" router; `.env.example` rewritten around the verified-real
  env surface; three superseded docs archived to `archive/2026-06/`;
  four doc status banners truth-synced (the IBKR design doc had claimed
  "none adopted" while D24/D26 shipped); REPO_MAP + verification-artifacts
  router bugs fixed; data-doc cluster gained reading-order headers.
- TESTING.md taxonomy completed 55 → 144 test files (five new sections);
  README's launch-blocker command regained `test_r11_elevated_vol.py`.
- MODULE_INDEX truth pass: the stale `engine/__init__.py` section (A3
  shipped), four engine dormancy reclassifications, `config/` + `utils/`
  status notes.

### Added
- `tests/test_testing_md_taxonomy.py` — two-way taxonomy gate (suite ⊆
  TESTING.md; literal taxonomy paths exist), the manifest-gate pattern
  applied to the test map.

### Changed
- `audit.py` → `scripts/audit_api_smoke.py` (git mv; nine refs updated;
  D14's rejected-alternative annotated — its lint-debt reason dissolved
  with Track F).

### Infra
- Four `.gitkeep`s removed from populated dirs (closes parked ROADMAP C3);
  empty-by-design placeholders retained.

---

## 2026-06 (early) — W-series verification campaign + IBKR live book + data refresh R1

~58 commits, 2026-06-03 → 2026-06-09 (PRs #317–#394). Four themes.

### Added
- **IBKR read-only live-book surface (D24 + D26).** `engine/ibkr_portfolio_adapter.py`
  + six `GET /api/portfolio/*` endpoints + the `/portfolio` Next.js viewer
  (**#344**); real IBKR history import + exact-fill wheel ledger (**#359**);
  PIT EV calibration vs the operator's real CSPs (**#362**); live morning
  refresh via cloud transform + headless Gateway puller (**#368**); flat
  per-leg positions (**#390**); unified Cockpit/Terminal design + honest KPI
  provenance (**#391**). Runbook: `docs/DASHBOARD_TERMINAL.md` (**#388**).
- **Armed production rank→book entry path.** `WheelRunner.consume_into_live_book`
  builds its tracker via `make_live_book_tracker` (R9 sector + R10 single-name
  caps ON), closing the "canonical armed constructor has zero non-test callers"
  gap from the 2026-06-01 reliability sweep (**#343**). R9/R10 also wired onto
  `GET /api/concentration_preview` (**#351**).
- **prob_profit small-sample honesty** — Wilson CI + `n_scenarios`, tier-gated
  to the IID forward path (**#317**), rendered in the cockpit (**#318**).
- Defensive (debit) rolls surfaced on `suggest_rolls` / `suggest_call_rolls` (**#342**).
- Deep-read assembly + survivorship harness, default-OFF (**#335**).

### Changed
- **Bloomberg data refresh R1** — 16 monolith CSVs refreshed + S27/S32/S34/S35
  snapshots re-baselined (**#338**); backtest-regression fingerprint now pins
  vol_iv + treasury SHAs (**#334**) and the **full connector input set**
  (`connector_data_sha256`, **#346**) so any data drift fails fast in the
  per-PR lane.

### Fixed
- `wheel_runner` credit_rating dead-read — reads the `sp_rating` key (**#333**).
- `data_connector` gates implausible vol_iv IV at the source — sub-3% floor +
  monolith sentinel (**#363**); deep-IV `134217.7` sentinel nulled on the
  assembled vol_iv read (**#336**).
- `mcp_client` classifies a live CDP-down `tv`-CLI error as `mcp_unavailable` (**#341**).
- Dashboard: `/terminal` per-panel crash isolation (**#349**), recharts
  first-paint + a11y fixes (**#352**), in-app ChartPanel handed off to
  TradingView (**#350**), honest disabled state on the unwired Ask bar (**#348**).

### Tests — the W-series campaign (W10–W67)
Two-phase data + engine audit (**#353** discovery → **#358** integrity suites),
then eight data-test PRs (**#370–#379**: IV-surface coverage, EV sign controls,
fundamentals/GICS, OHLCV/dividends hygiene, credit ladder, covered-call
real-data coverage, realism-at-scale + VIX-R11, cross-file date consistency)
and six quant-invariant PRs (**#383–#394**: forward-distribution cascade,
tail-risk/copula/stress, HMM regime multiplier, skew dynamics,
dealer-positioning clamp, binomial Greek units, reviewer R3/R5 boundaries).
Register: `docs/DATA_TEST_AUDIT_2026-06-09.md` (**#380**). Plus an
environment-invariant preflight guard (**#364**).

### Docs
- Data layer: activation roadmap + deep-read/survivorship design (**#332**),
  verified `docs/DATA_INVENTORY.md`, `docs/DATA_ACQUISITION_ROADMAP.md`,
  `docs/BLOOMBERG_PULL_LIST.md`, CASY backfill spec (**#347**), fresh lab-box
  bring-up (`docs/FRESH_LAB_BOX_SETUP.md`, **#345**).
- **`docs/NEXT_DATA_SESSION_RUNBOOK.md` (#381)** — the single authoritative
  re-baseline-session runbook (data queue + three (E) fixes + re-baseline).
- `TESTING.md` flags the bare-`pytest tests/` backtest_regression slow-lane
  trap (**#367**).

---

## 2026-05-30 — full-codebase code-review remediation (branch `claude/code-review-fixes`)

A multi-agent read-only review of the whole repo, then a fix pass on `main` for
the findings verified to be live on `main` (15 findings were already fixed on
main and skipped). Full detail + the verified-findings ledger:
`docs/CODE_REVIEW_2026-05-30.md`. Decisions: **D20** (treasury rate is percent →
÷100 unconditionally) shipped; **D19** (EV nets expected exit costs) and **D21**
(forward-distribution horizon calendar/trading-day mismatch) are confirmed +
fix-ready but **DEFERRED** — both change the EV-authority output and trip the
byte-identical-to-main baselines, so they land together in a coordinated backtest
+ calibration re-baseline rather than in this bug sweep.

### Fixed (EV decision path)
- Risk-free rate accessors (`data_connector`, `data_integration`) divide the
  percent treasury value by 100 unconditionally; the old `>1` heuristic mis-read
  every sub-1% rate (the 2011-2022 ZIRP era) 100× too high. **D20**. (Only the
  sub-1% ZIRP era is affected, so 2026-dated baselines are unchanged.)
- `dealer_positioning.analyze` anchors time-to-expiry to `as_of` (was wall-clock
  `now()` → collapsed in backtests); `theta_connector.get_vol_risk_premium`
  normalizes IV percent→decimal; `regime_hmm.fit` refuses degenerate input.

### Fixed (off path: trackers, risk, validation, dashboard, scripts)
- `portfolio_tracker` partial option close (P&L on closed qty only) +
  cash-flow-adjusted risk metrics/drawdown; `risk_manager` VaR vega ×100;
  `stress_testing` IV-shock now in the full repricing; `payoff_engine` CSP EV
  downside-aware; `engine_api` committee `p_otm`/`p_profit`; `model_validation`
  LSM tier kwargs (was dead); `news_sentiment` tz-aware compare; dashboard IV
  ×100 display; theta-puller `PerEndpointFailure` handling; iv-surface ticker
  filter; `pull_earnings_yf` merge-on-partial-fetch.

### Fixed (hardening)
- NaN/inf and edge guards across `ev_engine`, `realized_vol`, `contracts`,
  `regime_detector`, `dealer_positioning`.

### Deferred (confirmed + fix-ready, not shipped — bundle into one re-baseline)
Both change the EV-authority output and trip the team's byte-identical-to-main
backtest baselines, so they are documented and ready but not shipped in this bug
sweep; apply them together with a backtest + prob_profit re-baseline.
- **D19** EV omits the expected exit-leg transaction cost (~$1-4/contract, EV
  mildly overstated). Fix authored; does NOT touch prob_profit. The exact site is
  marked with a DEFERRED note in `ev_engine.evaluate`.
- **D21** forward-distribution horizon calendar/trading-day mismatch. Fix-ready
  (`calendar_days_to_trading_bars`), but applying it shifts every EV/prob_profit
  value (e.g. prob_profit 0.833→0.886) and would de-calibrate the published
  prob_profit matrix + all backtest snapshots. Left as a documented, unit-tested
  helper the orchestrator does not yet call.

### Docs
- Reconciled `GREEKS_UNIT_CONTRACT` (vega finite-diff example), `MODEL_CARDS`
  (parametric-VaR formula, live regime models), `MODULE_INDEX` (5 RV estimators),
  and code docstrings (`ev_engine` omega cap, `dealer_positioning` no-rescue
  mechanism, `pull_earnings_yf` gate claim) with the actual code.

---

## 2026-05-30 — merge-prep cycle: news-architecture redesign PRs 1–3 + MP-D + S42 close

The 2026-05-30 "merge-prep" parallel-session cycle landed five PRs onto
`main` (board #113). Decision-log entry for the news work: **D18**.
Headline reframing: **verbal news** (qualitative narrative) is severed
from the EV path (operator-only); **numbered news** (earnings dates,
fundamentals, macro) replaces it as separate structured layers.

### Added
- `EVResult` gains `pnl_p25 / pnl_p50 / pnl_p75` — raw P&L distribution
  percentiles (pre regime/dealer multipliers), surfaced through the
  ranker row and `/api/candidates`. Operator framing shifts from a point
  estimate to a distribution. **PR #248** (`tests/test_ev_engine_percentiles.py`).
- `EDGARAdapter.recent_8k_filings / earnings_history / project_next_earnings`
  — PIT-correct earnings-release history + projection from SEC EDGAR
  Form 8-K Item 2.02; drop-in compatible with
  `MarketDataConnector.get_next_earnings`. Integration into the EV path
  deferred to a follow-up. **PR #251** (+22 tests; `docs/EDGAR_EARNINGS.md`).
- `scripts/pull_edgar_earnings.py` — CLI puller for the EDGAR earnings
  store (append-only, `--refresh` merges with the prior parquet). **PR #251**.

### Changed
- **D18: verbal news severed from the EV decision path.**
  `engine/news_sentiment.py::sentiment_multiplier` is now a constant-1.0
  stub. `get_ticker_sentiment` is preserved so the dashboard, the row
  dict (`news_sentiment` / `news_n_articles`), and the morning brief
  still surface the underlying score for operator transparency. **PR #249**
  (`tests/test_news_severance.py` added; `TestSentimentMultiplier` +
  `test_multiplier_is_pit` rewritten).
- `MODULE_INDEX.md`, `PROJECT_STATE.md`, `README.md`, `AGENTS.md`,
  `ROADMAP.md`, `FILE_MANIFEST.md`, `scripts/pull_news_sentiment.py`
  aligned with the D18 severance (Major-Session cycle-close reconciliation;
  folds the descriptive-doc sweep originally drafted as PR #252).

### Fixed
- **MP-D / D9 follow-up:** `VolatilitySurface.get_iv` / `get_skew`'s three
  internal `return 0.20` missing-data fallbacks now `raise
  SurfaceDataUnavailable`, extending the fail-loud contract #286 set on
  the public surface down into the surface object's own methods. **MP-D**
  (squash `a77d61d`; `tests/test_iv_surface_failloud.py` +6).
- **S42 Findings #1–4 (dossier defensive guards):** drop the `or 1`
  truthy fallback on `contracts` in R9/R10; new `_filter_bsm_safe_positions`
  skips rows that would crash BSM (`strike<=0` / `contracts<=0` / missing
  symbol) in `check_var` + `check_stress_scenario`; `SectorExposureManager`
  uses defensive `.get()`. §2-preserved (defensive only; reviewers stay
  downgrade-only). **PR #275** (`tests/test_dossier_r9_r10_audit.py`
  updated; `TestFilterBSMSafePositions` +15).

### Docs
- `docs/NEWS_REDESIGN_CAMPAIGN.md` (PR #250, prior) is the canonical
  campaign state; PRs 4–9 (quality score / R9 reviewer / FRED
  `credit_mult` / backtest re-baseline / dashboard panes / override log)
  are not started.

---

## 2026-05 (late) — Open-question closures: A2 (iv_surface) + C1 (CSV tracking)

### Added
- **A2 — SVI iv_surface tooling wired in, fail-loud** (ROADMAP A2 / `DECISIONS.md`
  D9). `engine/volatility_surface.py` gains `SurfaceDataUnavailable` +
  `require_surface()`; the previously-dormant SVI calibrator now has a first
  production caller, `scripts/diagnose_iv_surface.py` (operator diagnostic —
  per-expiry skew / term-structure, **exits non-zero** on any uncovered ticker,
  never a fabricated flat IV). `create_constant_surface` stays the only opt-in
  flat surface. Pinned by `tests/test_iv_surface_failloud.py`.

### Decided
- **C1 — bloomberg yfinance CSVs: keep tracking as data commits** (ROADMAP C1).
  The point-in-time "what data did we run on?" audit trail outweighs the
  commit-per-refresh history noise; zero migration. Recorded in
  `docs/DATA_POLICY.md` §5.

## 2026-05 (late) — Repo-efficiency + coordination cycle

### Added — coordination overhaul + documentation redesign (PR `#285`)
- **Coordination: Major-Session task-card model + decision-layer CI gate.**
  `docs/PARALLEL_SESSIONS.md` rewritten — a single persistent Major Session
  allocates **disjoint task cards** (one per terminal), so two terminals can't
  be handed the same file (duplicate self-selected work, e.g. the `select_book`
  #107/#109 double-build, is designed out). New `scripts/check_lane_claim.py`
  + the `decision-layer-claim` CI job fail any PR that edits the decision-layer
  trio without a `lane-claim` block, replacing the manual "checked the board"
  prose. New `.github/pull_request_template.md`. Extends `DECISIONS.md` D15.
- **Docs: per-task worklog fragments replace the 490 KB ledger monolith.**
  `docs/USAGE_TEST_LEDGER.md` (8,600 lines / 42 `Sn`) split **verbatim** into
  `docs/worklog/*.md` fragments; monolith frozen to a banner + scenario→fragment
  map. New `scripts/gen_worklog_index.py` generates `docs/worklog/INDEX.md`
  (CI-checked via `--check`) over the fragments + the dated reports indexed
  **in place** (243 inbound refs ⇒ not moved), retiring the hand-maintained
  `VERIFICATION_INDEX`. `scripts/new_worklog.py` scaffolds a fragment. Each task
  now writes its own file — no shared "magnet" doc to collide on. Extends
  `DECISIONS.md` D14.

### Changed
- **Coordination contract hardened** (PR `#282`). `docs/PARALLEL_SESSIONS.md`
  rule 7: `Sn` / `D`-numbers are allocated at MERGE (serialised), not at
  work-start — kills the parallel-counter race (three `Sn` collisions on
  2026-05-28). New rule 8: one `FILE_MANIFEST.md` owner per cycle; the later
  branch rebases, never `checkout --theirs`.

### Added
- **`docs/TESTED_SURFACE_MAP.md` + CI coverage artifact** (PR `#281`). One
  file answers "what's covered" (per-module map + top-N gap ranking,
  generated from `coverage.json`, which is now gitignored and uploaded by CI
  as an artifact rather than committed).
- **`scripts/sync_manifest.py`** (PR `#279`). `--fix` appends missing
  `FILE_MANIFEST.md` rows; closed four pre-existing manifest gaps.
- **Conflict-marker CI guard** (PR `#283`). `scripts/check_manifest_coverage.py`
  now fails on committed `<<<<<<<` / `=======` / `>>>>>>>` markers in any
  tracked `.md` (plus a 16-test unit suite) — catches the rebase trap
  automatically instead of by luck.

### Docs
- **Verification corpus consolidated** (PR `#280`). 12 dated review snapshots
  archived to `archive/2026-05/`; `docs/VERIFICATION_INDEX_2026-05-28.md` is
  the canonical living index; `PROJECT_STATE.md` refreshed.
- **S46 re-verify** of the closed tests on the post-#260/#262 engine
  (PR `#278`).
- **`PRODUCTION_READINESS.md` B1 framing** reconciled — "shipped as the
  #260 + #262 bundle (residual structural limit)", no longer self-contradictory
  (PR `#283`).

### Infra
- Closed PR #253 (rolled-back F4 Fix-B1+C research record; findings preserved
  in `docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10–11). Deleted eight stale branches.

---

## 2026-05 (late) — D17 B2 closure + F4 fix v2 + engine audit

### Added
- **R9 sector_cap dossier soft-warn + D17 wire on `/api/tv/enrich`**
  (PR `#255`, `f3a4fa8`). Closes D17 B2 part 2. Adds the per-sector
  exposure cap as a downgrade-only rule on `EnginePhaseReviewer`
  alongside the existing R7 (VaR) and R8 (stress + dealer regime)
  soft-warns. Default 25% NAV per sector via
  `engine.portfolio_risk_gates.check_sector_cap`. Mirrors the
  tracker's HARD refusal at `open_short_put` time when
  `require_ev_authority=True`. `/api/tv/enrich` now constructs and
  threads a `PortfolioContext` through `build_dossiers` so the
  soft-warn fires live for any candidate that reaches the endpoint.
- **R10 single-name (per-underlying) exposure cap** (PR `#262`,
  `45ca861`). F4 damage-bounding closure — bounds the
  idiosyncratic-drawdown style of failure that no market-wide
  regime detector can predict (see `docs/F4_TAIL_RISK_DIAGNOSTIC.md`
  §10). 10% NAV cap on aggregated SHORT option notional per
  underlying. Sits BENEATH R9: a ticker concentrated as the only
  name in its sector could pass R9 at 25% but trip R10 at 10%.
  Soft-warn on dossier (downgrade `proceed → review`,
  `verdict_reason="single_name_breach"`); HARD refusal on the
  tracker when `require_ev_authority=True`. Pure-function gate at
  `engine.portfolio_risk_gates.check_single_name_cap` shared by
  both surfaces.

### Fixed
- **F4 tail-risk widening v2 — realized-vol-ratio
  (RV30 / RV252)** (PR `#260`, `0dddf76`). Replaces the rolled-back
  HMM-based widening from PR `#253` which over-fired on `K=4`
  crisis labels (98% of 2022-2024 dates, inverting S27 ρ from
  +0.188 to −0.094 — see memory `f4-widening-overfires-on-hmm-
  labels`). The shipped form uses a continuous regime-conditioned
  multiplier driven by the realized-vol ratio: when recent
  volatility exceeds the longer-horizon baseline by enough, the
  forward distribution is widened proportionally. Sign-preserving
  (factor ≥ 1.0; never narrows tail risk), capped at 1.5×,
  routed through the forward distribution → `EVEngine.evaluate`
  (never an overlay on final `ev_dollars`). S27 ρ gate held
  (≥ +0.15 required); COST 2022-04 + UNH 2024-11 anchor cases
  resolved.
- **D17 portfolio-context live wire on `/api/tv/dossier`**
  (PR `#233`, `b55a59a`). Closes D17 B2 part 1. The R7 (VaR)
  and R8 (stress + dealer regime) soft-warns now fire live for
  any candidate that reaches the `/api/tv/dossier` endpoint —
  previously the wire existed but no production endpoint
  constructed a `PortfolioContext`, so the soft-warns were
  silent on the network surface.

### Docs
- **Systematic engine audit of `engine/` + `advisors/`**
  (PR `#232`, `8a17b0b`). Six subsystem audits — decision-layer,
  risk-management, advisors-scorecard, dealer positioning,
  event gate, news sentiment — produced as separate
  `docs/AUDIT_<subsystem>_2026-05.md` files. Findings catalogued
  by severity with closure owners; high-priority items routed to
  PROJECT_STATE WIP and ROADMAP follow-ups.
- **Engine realism + reliability verification** (PR `#244`,
  `70fdb78`). Six observable tests against `origin/main` @
  `9f0afaf`: §2 launch-blocker subset (93/93), 5-ticker smoke
  (all realistic), IV PIT realism vs Bloomberg-direct (all
  within 0.015% rel-diff), EV magnitude regime-multiplier
  dominance (corr(iv, ev_dollars) = 0 — regime sensitivity
  dominates raw IV per D17), F4 reproducibility (drifted to
  prob_profit=0.903, closed by `#260`), refusal behaviour at
  3 anchor dates. `docs/verification_artifacts/` ships the
  driver + raw output for re-runnable verification.
- **F4 baseline doc pre-#260** (PR `#245`, `b2cce25`). Captures
  the COST 2022-04 / UNH 2024-11 / AAPL anchor cases' pre-fix
  `prob_profit` values against the post-IV-PIT engine. Provides
  the diffable baseline that `#260` later resolved.
- **PRODUCTION_READINESS B3 sync + B2 / R10 closures captured**
  (PR `#257`, `79a6b88`). Refreshes the deployment-matrix to
  reflect both S34's +11.6pp at $1M/100t/2022-2024 AND S38's
  −52pp at $1M/100t/2020-2024 — the window-specificity finding
  (engine vs passive delta correlates with bear-year share of
  the measurement window). Notes B2 (D17 wire) and R10
  (single-name cap) as closed; carries the honest reframe per
  Terminal B's S38 finding.
- **Five doc-drift one-liners from audit §2a** (PR `#254`,
  `b956cde`). Targeted corrections across five Tier-1/2 docs to
  resolve issues surfaced by the systematic audit (`#232`)
  before higher-effort rewrites.

### Chore / refactor
- **Layout audit §1 (PR `#247`, `56d8e5c`).** Read-only
  structural audit ran five Explore agents + a synthesis pass;
  this PR applies seven pure rename / delete / move
  cleanups from §1 plus a SESSION_HANDOFF banner refresh.
  Zero behaviour change; the eighth flagged item (scorecard
  enum dedup) split into PR `#258` per single-concern rule.
- **`advisors/scorecard.py` enum dedup** (PR `#258`,
  `6aa9609`). `ConfidenceLevel` and `JudgmentType` deduped to
  `advisors/schema.py` — single source of truth for the two
  enums; behaviour-preserving import rewires across consumers.
  Split from PR `#247` because it's a behaviour change
  (`Enum` re-pointing affects identity checks).

---

## 2026-05 (late) — Backtest regression harness

### Tests
- **S42 R9 + R10 reviewer audit** (`tests/test_dossier_r9_r10_audit.py`,
  32 tests). Systematic audit of the two new dossier downgrade rules
  shipped in PR `#255` (R9 sector_cap) and PR `#262` (R10 single-name
  exposure cap). Six probe families pin behavioural correctness, the
  downgrade-only invariant, fail-closed-on-missing-data semantics,
  rule-order short-circuit (R7 → R8 → R9 → R10), and cap-boundary
  semantics (both R9 and R10 use strict `>` — exact 25% sector / 10%
  single-name passes). Surfaced four low-severity sharp edges as
  documented findings — see `docs/USAGE_TEST_LEDGER.md` S42 for
  detail. No engine math changed; read-only against §2.

### Fixed
- **Dossier reviewer defensive guards (S42 Findings #1-4 closed).**
  Four sharp edges surfaced by the S42 audit are now closed by a
  stacked hardening PR (`claude/fix-dossier-defensive-guards`):
  - `engine/risk_manager.py`
    `SectorExposureManager.calculate_sector_exposures` now uses
    defensive `.get()` calls — skips malformed rows rather than
    raising `KeyError`. (Closes Finding #1.)
  - `engine/portfolio_risk_gates.py` — new pure-function filter
    `_filter_bsm_safe_positions` drops rows that would crash BSM
    (`strike ≤ 0`, `contracts ≤ 0`, missing symbol). Wired into
    `check_var` and `check_stress_scenario`. R8 no longer
    pre-empts R9/R10 for malformed rows; R10's defensive
    try/except is now reachable via the dossier path. (Closes
    Findings #2 + #4.)
  - `engine/candidate_dossier.py` — the R9 and R10 paths drop
    the `or 1` truthy fallback on contracts so an explicit
    `contracts=0` is honoured (`proposed_notional` becomes 0;
    existing `if nav > 0 and proposed_notional > 0` guard
    catches it). (Closes Finding #3.)
  - `tests/test_dossier_r9_r10_audit.py` — F4.3, F4.4, F6.3,
    F6.4, F6.5 updated from `pytest.raises(...)` pins to assert
    the new graceful verdicts. Audit value preserved — tests
    still pin behaviour, just the now-correct behaviour. The
    audit acted as the forcing function it was designed to be.
  - §2 invariant preserved: changes are defensive (skip
    malformed rows / honour explicit 0). Reviewers remain
    downgrade-only. No `ev_dollars` / `ev_raw` / multiplier
    code edited.

### Added
- **Backtest regression harness** (4-PR series). Converts the four
  committed ledger backtests (S27 / S32 / S34 / S35) from human-
  curated Markdown into executable pytest assertions against the
  current engine. Snapshots in `backtests/regression/snapshots/`
  are the regression baseline; the slow-lane test is gated behind
  `@pytest.mark.backtest_regression` and a dedicated workflow.
  - **PR1** `claude/backtests-regression-scaffolding` — `backtests/
    regression/` scaffolding (4 reproducers + universes + driver),
    `pyproject.toml` adds `backtests` to wheel packages.
  - **PR2** `claude/backtests-regression-snapshots-tests` — locked
    snapshots for S27 / S32 / S34 plus `tests/
    test_backtest_regression.py`, `backtest_regression` marker,
    `.claude/commands/backtest-regression.md` skill, TESTING.md +
    LAUNCH_READINESS.md §10 updates. CC wheeling added to the driver
    (without it, assigned tickers locked out of rotation; S27 v1
    executed 15 trades vs documented 50; v2 with CC wheeling executed
    51). `run_backtest_multi_friction` shares one SP rank call per
    day across N friction levels (~3× faster than naive sequential).
    Comprehensive campaign report at `docs/BACKTEST_REGRESSION_CAMPAIGN.md`.
  - **PR3** `claude/backtests-regression-ci-split` — `ci.yml` `Test
    Suite` excludes the marker; new
    `.github/workflows/backtest-regression.yml` workflow_dispatch
    entry (cron deferred until CSV hydration in CI is solved).
  - **PR4** `claude/backtests-regression-s35-rebaseline` — S35
    re-baseline against the current driver. Finding: S35's headline ρ
    is **driver-invariant** (0.4970 doc vs 0.4998 snapshot) but
    execution count doubled (19 → 40) — the post-PIT-fix engine
    surfaces more positive-EV opportunities in 2018–2020 than the
    original throwaway harness captured.

### Docs
- `docs/BACKTEST_REGRESSION_CAMPAIGN.md` — campaign report
  (architecture, methodology, snapshot-vs-doc tables for all four,
  on-fail re-baseline workflow). Entry point for any agent picking
  up or auditing the harness.
- `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` amended with a
  `## Re-baseline 2026-05-26 (regression harness lock)` section
  preserving the original numbers and documenting the driver-
  implementation divergences.
- `TESTING.md` — new `backtest_regression` marker row + "Backtest
  regression — re-baseline workflow" section.
- `docs/LAUNCH_READINESS.md` — new §10 "Backtest regression gate"
  (per-release blocker, not per-merge) + pre-merge checklist bullet
  for the five engine files (`ev_engine`, `wheel_runner`,
  `forward_distribution`, `dealer_positioning`, `tail_risk`) whose
  changes mandate running the harness.

### Snapshot-vs-doc divergences (campaign-wide)

| Backtest | ρ match | NAV match | Notes |
|---|---|---|---|
| S27 | within 0.03 | −16 % | systematic strike-rounding gap; executed count 51 vs 50 ✓ |
| S32 | within 0.006 | +5.8 % | "BP not binding at $1M with 24 tickers" ✓ |
| S34 | **within 0.002** | within 2.5 % | strongest match (large sample averages out per-ticker noise) |
| S35 | within 0.003 | +11.8 % | doubled execution count is the meaningful finding |

The harness captures **current engine behavior** as the regression
baseline. The documented numbers serve as a sanity check on signal
direction and ordering, not as an exact target. Future engine drift
fails the test loudly; intentional methodology changes force an
explicit re-baseline + doc amendment.

### Fixed
- **`_common.py`** replaces deprecated `datetime.utcnow()` with the
  modern `datetime.now(UTC)` (PR `#259`). Closes 3 deprecation
  warnings per reproducer run; future-proofs against Python 3.13.
  Output format shifts from `"...Z"` to `"...+00:00"` (both ISO 8601
  UTC). `generated_at` is metadata and skipped by the regression
  comparison, so existing snapshots remain valid.

### Docs
- **`docs/BACKTEST_REGRESSION_CAMPAIGN.md`** Known-limitations section
  rewritten as a consolidated catalog with three categories
  (harness-internal H1–H6, engine-realism E1–E6, production-readiness
  P1–P3), each item carrying status + owner. Cross-references the
  existing `SOUNDNESS_REVIEW`, `F4_TAIL_RISK_DIAGNOSTIC`, and
  `PRODUCTION_READINESS` docs. Single source of truth for "what the
  harness does and doesn't claim."

### Validated end-to-end (overnight)
All four backtests' `@pytest.mark.backtest_regression` cases passed
against committed snapshots on `main`:
- **S27** ✓ ~50 min
- **S32** ✓ 1 h 42 min
- **S34** ✓ 6 h 54 min (longest; multi-friction forward-replay over 30 K rank rows)
- **S35** ✓ ~50 min

Harness mechanism is operationally proven; what remains unverified is
real-drift detection — the snapshots were self-generated, so the test
passes trivially today. First real engine change is the actual
validation. Documented as H3 in the campaign report.

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
- **`engine/__init__.py` re-exports the modern decision layer.** Adds
  `EVEngine`, `EVResult`, `ShortOptionTrade`, `WheelRunner`,
  `EnginePhaseReviewer`, `CandidateDossier`, and `MarketStructure` to
  `__all__` (the seven names CLAUDE.md §1 calls the authoritative
  surface), and a docstring line pointing back at CLAUDE.md §1.
  A fresh agent can now do `from engine import EVEngine` instead of
  discovering the four submodule paths. Closes `ROADMAP.md` A3 —
  the parking concern ("ripples through every import site") was
  verified false by a pre-edit grep: every existing import uses the
  full submodule path, so additions to `__all__` can only enable new
  imports, never break existing ones. `PROJECT_STATE.md` §5 drift
  entry closed in the same PR.

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
  end-of-run. Per-puller wiring lands separately in this PR.
  See `DECISIONS.md` D11.
- **Lint debt closed — 75 ruff errors → 0 across the CI scope.**
  PR #79 (`9e15dbf`, 2026-05-15) cleared the judgement-required tail
  left after PR #64's mechanical pass: B904 raise-from, B023 closure
  trap, F841 unused locals, B019 lru_cache-on-method, F821 undefined
  names, E741 ambiguous, plus UP/I/F/C one-offs. Rule-per-commit pass;
  CI scope (`src/ engine/ data/ advisors/ financial_news/ tests/
  scripts/ utils/ news_pipeline/ dashboard/`) verified clean
  post-merge. **Two real latent bugs surfaced:** F821 in
  `engine/ev_engine.py` — the unconventional `if False:  # TYPE_CHECKING`
  guard hid the bug from ruff but `typing.get_type_hints(EVEngine)`
  would have raised `NameError`; fix uses the canonical
  `if TYPE_CHECKING:` and adds the missing `EventGate`,
  `MarketStructure`, `datetime.date` imports. B023 closure traps in
  `engine/wheel_runner.py` (put-delta solver) and
  `engine/earnings_drift.py` (return calculator) — both rebound via
  default-arg capture; latent if either nested function is ever
  stored or deferred. ROADMAP Track F closed.
- **`engine/mcp_client.py` live-verified against TradingView Desktop +
  `tv` CLI (LewisWJackson/tradingview-mcp-jackson fork).** PR #100
  (`ad1bbbc`, 2026-05-19). Closes the placeholder URL and the
  `TODO(live-verify)` markers left from MCP Stage 2: Windows `cmd /c`
  invocation for the npm-linked `tv` shim, the `success` status field,
  and the `file_path` screenshot key all confirmed. `tv state` carries
  no price (`visible_price` deferred to a future `tv quote` call —
  operator decision). Only the per-mode error strings in
  `MCPCLIClient._classify` remain unverified; no live error path was
  exercised. `tests/test_mcp_client.py` grew 32 → 39 tests (still all
  subprocess-mocked). Closes `PROJECT_STATE.md §3` follow-ups row 3.
- **`engine.strangle_timing.StrangleTimingWithIV.score_entry_with_iv`
  — Layer-2 IV overlay repaired.** Commit `210463d` (2026-05-20).
  The previously-dead method called four nonexistent connector
  methods (`get_realized_vol`, `get_current_iv`, `get_vix_level`,
  `get_vix_contango`) and passed `as_of=` to `get_ohlcv` (also
  missing). Rewritten to use the real `MarketDataConnector` API:
  `get_ohlcv(end_date=…)`, `get_iv_rank`, `get_vol_risk_premium`,
  `get_vix_regime`. The strict-xfail
  `test_score_entry_with_iv_real_connector_signature_mismatch` was
  replaced with the green regression test
  `test_score_entry_with_iv_against_real_connector`, which exercises
  the overlay end-to-end against the real connector using AAPL's
  committed Bloomberg CSVs. Closes `PROJECT_STATE.md §3` follow-ups
  row 5.

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
- **`tradingview/` analyst workspace staged** — PR #78
  (`claude/audit-fixes-no-coverage`), merged to main as `4e9c3f3` on
  2026-05-15. Tracked the six analyst-workspace files left untracked
  by the foundation pass: `tradingview/CLAUDE.md` (workspace contract
  for Claude as Mert's financial analyst — pre-flight checklist,
  deliverable conventions, known coverage gaps), `tradingview/OVERVIEW.md`
  (operating-overview narrative), `tradingview/launch-tradingview-cdp.sh`
  (executable launcher with `--remote-debugging-port=9222`), and
  `.gitkeep` placeholders for `models/`, `pine/`, `research/`. The
  `tradingview-mcp-jackson/` nested repo and `*.docx` deliverables
  remain gitignored. Closes ROADMAP Track C2.
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
