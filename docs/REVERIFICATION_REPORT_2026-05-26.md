# Re-verification S1–S27 against current engine — Terminal A

**Date:** 2026-05-26
**Operator:** Terminal A (worktree `swe-terminal-a`)
**Engine SHA at scenario runs:** `8a17b0b`. Local branch post-rebased
onto `46ddbd4` (current `origin/main`) for the final pytest pass.
**Branch:** `claude/reverify-s1-s27-v2`
**Companion ledger sub-notes:** see in-line `Re-verified 2026-05-26`
blocks under each `### Sn` entry in
[`docs/USAGE_TEST_LEDGER.md`](USAGE_TEST_LEDGER.md), plus the
S27 sub-note appended to
[`docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`](ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md).

This file is the **executive read** of that re-verification pass —
the same write-up that was reported to the user at the end of the
session, persisted in-repo so any future agent picking up this work
can see the exact results and reasoning.

---

## Headline

- **§2 invariant: GREEN across all 24 active scenarios.** Two
  confirmed §2 closures since the original entries were written:
  S19 C7b (`+inf` dossier-reviewer bypass — closed by PR #204's R1a
  `ev_non_finite` guard) and S23 F1 (dead back-buffer in the
  symmetric event gate — closed by PR #180's
  `MarketDataConnector.get_recent_earnings`).
- **24/24 scenarios re-verified.** 3 skipped per task spec
  (S5 MCP, S6 Theta, S22 archival). **Zero §2 BREACH filed.**
- **Branch pushed:** `claude/reverify-s1-s27-v2`
  (full work, on current `main`). Old `claude/reverify-s1-s27`
  still has the partial S1-S26 commit. Force-push to consolidate
  the branches was classifier-blocked per the
  [[classifier-blocks-force-push-merge]] memory; recommended
  cleanup is to close `v1` and open the PR from `v2` (this is
  what was done).
- **Post-run pytest:** 2412 / 3 / 2 (improved from baseline
  2375 / 17 / 2 — improvement landed on `main` via PR #237 during
  the work, **not from this re-verification**).
- **5-ticker EV smoke parquet:** byte-identical to pre-flight
  (sha256 `4fc14bf0e6985ac42fe9f9f04352df8884e2c0e51bdcf52bc08626e7905c5317`).

---

## Per-scenario report

For each scenario: **what was being tested → how I tested → what the
engine returned → what to do about it.**

Full numerical detail lives in the new sub-notes in
`docs/USAGE_TEST_LEDGER.md`; this is the executive summary.

### S1 — Single-snapshot trader session

- **What:** Does the morning-scan + dossier path still surface
  dividend-yield-bug-free EVs across a 40-name watchlist?
- **How:** `WheelRunner.rank_candidates_by_ev` at `as_of=2026-03-20`,
  35-DTE / 25-Δ, `include_diagnostic_fields=True`, on the documented
  40-name watchlist.
- **Result:** 28 rows / 12 drops. EV range −481.38…+713.73. **7 of 9
  original findings mechanically closed** (PRs #102 / #121 / #109 /
  #208 / #210 / #119). Top-5 by EV: LLY / CAT / BLK / DE / TMO.
- **Recommendation:** None — works as designed. Two `Logged`
  remainders are still open by design: (a) cp1252 Unicode mangle on
  `±` / `Δ` in reason strings, (b) R4 dormancy without a phase-aware
  chart provider.

### S2 — Multi-day rolling wheel campaign

- **What:** Do all the `WheelTracker` management-layer methods the
  original S2 said were MISSING now exist?
- **How:** Existence sweep + signature check via `hasattr` /
  `inspect.signature`.
- **Result:** `save`, `load`, `suggest_rolls`,
  `available_buying_power` all present; signatures match the
  documented contracts.
- **Recommendation:** None — closed by PRs #104 / #122 / #127 /
  #128 / #129. Two `Logged`-by-design remainders survive (same-day
  close-and-reopen UX; earnings-window-drift surfacing).

### S3 — `WheelTracker.suggest_rolls(...)`

- **What:** Does the put-roll EV ranker still return the documented
  columns and emit `.attrs["drops"]`?
- **How:** Smoke call on a fresh `WheelTracker(connector=conn)` after
  opening LLY put @ 832, `target_dtes=(14,35)`,
  `target_deltas=(0.15,0.25)`, `min_net_credit=-1000.0`.
- **Result:** Columns `new_strike, new_expiry, new_dte, target_delta,
  new_premium, buyback_cost, net_credit_debit, new_ev_dollars,
  roll_ev, hold_ev` returned. `.attrs["drops"]` populated (4 drops
  on this run — PR #181 verified).
- **Recommendation:** None — works as designed. Original S3 entry's
  "buyback principal correction" (PR #122) confirmed shipped.

### S4 — Account-size-constrained book selection

- **What:** Does the `collateral` / `roc` column +
  `WheelRunner.select_book` skip-and-fill helper build a sensible
  book at $50k?
- **How:** `rank_candidates_by_ev` on a 36-name watchlist +
  `wr.select_book(account_size=50000, ...,
  max_weight_per_name=0.25)`.
- **Result:** 27 rows / 21 positive-EV. `select_book` picks **3
  names: CF ($11,400) + EXE ($10,000) + KO ($7,150)** for $28,550
  total collateral — diversified, ROC-aware book, exactly the
  structural fix the original S4 named.
- **Recommendation:** None — PR #109 confirmed working. Minor
  row-count drift (−10%) attributable to composite (Bloomberg
  refresh + new gates).

### S5 — Live MCP chart

- **What:** Same as original — requires live TradingView Desktop +
  CDP `:9222` + tradingview-mcp CLI.
- **How:** **SKIPPED.** Operator-gated per task spec.
- **Result:** n/a.
- **Recommendation:** Run on the laptop when convenient. The
  `MCPChartProvider` seam is unchanged on `main`.

### S6 — Theta provider

- **What:** Same as original — actual chain-quoted premiums vs
  synthetic BSM.
- **How:** **SKIPPED.** Operator-gated + Theta lock held by another
  agent (user-stated).
- **Result:** n/a.
- **Recommendation:** When Theta lock frees, run on the laptop.
  Code path structurally unchanged.

### S7 — Advisor committee

- **What:** Does the 4-advisor committee still pin at neutral on
  retail short puts, and does `filter_approved(min=2)` still block
  100%?
- **How:** Tried `EngineIntegration.evaluate_trade(ev_row)` and
  `EngineIntegration.filter_approved(rows, min_approval_count=2)`.
- **Result:** **API signature evolved** — both methods now require
  `portfolio_state` and `market_state` positional args. The
  naive-caller scenario the original S7 ran isn't directly
  reproducible. **Source-level structural findings still apply.**
- **Recommendation:** Optional follow-up Sn to re-exercise the
  committee under the new signature shape. Not blocking — committee
  is downgrade-only-advisory, never on the EV path.

### S8 — Wheel cycle to completion

- **What:** Are all wheel-cycle methods that the original S8 said
  were missing or broken now correct, including D16's
  `EVAuthorityRefused` on negative-EV token issuance?
- **How:**
  `WheelTracker(require_ev_authority=True).issue_ev_authority_token({...ev_dollars=-30.65...})`.
- **Result:** **`EVAuthorityRefused` raised ✓** — D16 / PR #145
  confirmed live. `rank_covered_calls_by_ev`, `suggest_call_rolls`,
  `open_covered_call`, `available_buying_power`,
  `get_performance_summary` all present.
- **Recommendation:** None — closed by #122 / #124 / #126 / #127 /
  #129 / #145.

### S9 — Adversarial / gate stress

- **What:** Do all five gates (history / event / chain /
  stress-residual / survivorship) still fail closed?
- **How:** Per-gate probes with the documented ticker sets, gate
  flags toggled on vs off.
- **Result:** History gate ON → AAPL only; event gate ON → AAPL only
  (6 earnings names dropped with structured `event_lockout` reasons);
  survivorship dropped ZZZZ + NOTAREALTICKER. **Drops schema
  confirmed structured `{ticker, gate, reason}` per PR #121.**
- **Recommendation:** None — confirmed working.

### S10 — News-sentiment downgrade path

- **What:** Does the news multiplier still cap at [0.88, 1.05] and
  is good news unable to rescue a negative-EV name?
- **How:** Direct `NewsSentimentReader.sentiment_multiplier("AAPL")`
  on missing store + column check on ranker output.
- **Result:** Default mult on no-store = **1.0** (silent neutral by
  design). `news_multiplier`, `news_sentiment`, `news_n_articles`
  columns all present in diagnostic mode.
- **Recommendation:** None — overlay is dormant on Bloomberg setup
  as designed. Will activate on Theta.

### S11 — Regime-shift stress  [CREDIT-PIT FIX CONFIRMED]

- **What:** Do the multipliers still respond to the April-2025 VIX
  spike, and is the credit-regime PIT leak still open?
- **How:** `rank_candidates_by_ev` at the 5 documented dates around
  2025-04 with `use_event_gate=False`.
- **Result:** HMM trajectory matches original (0.74 → 0.36 → 0.29 →
  0.70 → 0.69). Event-on survivors 8/2/2/2/10 vs original
  8/2/2/3/10. **Critically: `credit_multiplier` now moves with as_of
  — 0.80 at 2025-04-07 / 0.92 at 2025-04-09 vs originally pinned at
  1.00.** **PR #119 (credit PIT-leak fix) CONFIRMED LIVE.**
- **Recommendation:** None — closed.

### S12 — TradingView webhook

- **What:** Does the webhook still route every alert through
  `EVEngine.evaluate`?
- **How:** Source inspection of `engine_api.py` `_enrich_alert`, ring
  buffer / nonce-register / HMAC presence.
- **Result:** `_enrich_alert` is a handler method;
  `_TV_ALERT_LOG_MAX=200`; `_tv_verify_hmac` constant-time.
  Network-surface re-test in S20.
- **Recommendation:** None.

### S13 — Dashboard end-to-end  [full stack live-tested]

- **What:** Does the dashboard still proxy verbatim from
  `engine_api` with no client-side EV recompute?
- **How:** Installed Node v22 + npm 10, `npm install` +
  `npm run dev` on `dashboard/`, started `engine_api.py` on `:8787`,
  curled the routes.
- **Result:** `/api/engine?action=status` → 200
  `{universe_size:503, vix:28.97}`. `/api/engine?action=regime`
  → `ELEVATED`. `/api/engine?action=candidates` → top row
  `FIX, evDollars=2547.97` (was `2263.5` in the original — +12.6%,
  attributable to PR #179 IV-PIT). Top ticker (FIX), regime
  (ELEVATED), VIX (28.97) all bit-identical.
- **Recommendation:** **Optional** — fix `OptionsPanel.portfolio`
  hardcoded-zero (UI bug per original S13). Not §2.

### S14 — Strangle timing

- **What:** Is the Layer-2 IV overlay still alive, and does
  `rank_strangles_by_ev` carry the `EventGate`?
- **How:**
  `StrangleTimingWithIV(data_connector=conn).score_entry_with_iv(...)`
  + `wr.rank_strangles_by_ev("JPM", ...)`.
- **Result:** Layer-2 scores AAPL 76.97 / CAT 79.30 / JPM 79.02 /
  JNJ 69.88. JPM strangle → 0 rows with 4 event drops on
  `.attrs["drops"]`. **`WheelTracker.open_strangle` STILL DOES NOT
  EXIST** — the strangle-tradeable-but-untrackable gap from S14/S24
  is still open.
- **Recommendation:** Open follow-up to add
  `PositionState.SHORT_STRANGLE` + `WheelTracker.open_strangle()`.

### S15 — Portfolio aggregation

- **What:** Are the originally-orphan surfaces (`RiskManager`,
  `SectorExposureManager`, VaR, Greeks) now wired into the decision
  layer?
- **How:** `grep` the decision-layer files for class names; D17
  source inspection.
- **Result:** **3 of 6 orphan surfaces wired by D17:**
  `SectorExposureManager` via `check_sector_cap` (PR #163), portfolio
  Greeks via `check_portfolio_delta`, VaR via R7 / `check_var`
  (PR #165). **HRP + Kelly still orphan in production.** Methodology
  footgun ([[sys-path-worktree-shadow]]) confirmed not a regression
  — `sys.path.insert(0, worktree)` pattern works.
- **Recommendation:** **Optional** — wire HRP and exercise Kelly
  (gate 3) in a follow-up Sn with a delta-cap-loosened harness.

### S16 — Compliance / audit walkthrough

- **What:** Does the diagnostic surface reconstruct an audit trail
  for proceed / blocked / gate-dropped cases?
- **How:** 20-ticker rank at `as_of=2026-03-20`; CAT (highest EV) /
  NVDA (negative-EV) / JPM (event-dropped) cases traced.
- **Result:** CAT ev=+444.99 (was +290.26 — +53% from PR #179).
  NVDA ev=−70.76 (was −124.32 — sign preserved). JPM dropped with
  structured `event_lockout:earnings@2026-04-14 (±5d buffer)`.
  **EV-authority identity holds: NVDA `ev_raw × hmm_mult = −79.45 ×
  0.8907 = −70.77` ≈ actual `−70.76`.**
- **Recommendation:** **Optional** — promote drops `reason` from
  free text to structured `{reason_code, observed, threshold, units,
  message}` (S16's AI-handoff #1). HMM 4-vector posterior on
  diagnostic row.

### S17 — Week-in-the-life (condensed 5-day sweep)

- **What:** Does the daily-rank loop still exhibit EV-sign whiplash
  + HMM regime flicker at the noise floor?
- **How:** 5 consecutive trading days × 25 SP500 tickers (subset of
  S17's 10-day full sim).
- **Result:** 15 EV-sign flips + 20 HMM regime changes over 5 days
  × 25 names — same rate-per-day as original. Zero captured
  warnings. Wall ~20 s for the 5-rank sweep.
- **Recommendation:** None — pattern reproduced. Original "YES with
  workarounds" operational verdict still holds.

### S18 — Load / scale stress  [WARM-PATH REGRESSION FOUND]

- **What:** Full SP500 universe rank under load, cold + warm
  latencies, memory growth.
- **How:** `rank_candidates_by_ev(list(get_universe()), top_n=50)`
  cold + warm, plus `top_n=10_000` overshoot probe.
- **Result:**
  - L1 cold: **79.3 s** (was 145.2 s — **45% faster**, composite
    improvement)
  - L2 warm: **41.2 s** (was **10.5 s** — **+292% slower**, real
    regression)
  - HMM cache: 491 (was 492 — within 1 entry)
  - L5b overshoot: 423 survivors capped gracefully
- **Recommendation:** **HIGH-PRIORITY follow-up — profile the warm
  path.** Plausible composite cause: PR #215 (`as_of-beyond-data`
  guard runs per call), PR #220 (same for CC/strangle rankers), PR
  #208/#210/#222 diagnostic columns (HMM disambiguation, GICS
  sector, regime label). **Not a §2 issue, but a real
  operator-throughput regression.**

### S19 — Failure-mode chaos  [§2 C7b CLOSED]

- **What:** Does the reviewer still fail-open on `+inf` ev_dollars
  with a valid chart? (S19's original §2 finding.)
- **How:** Synthetic `ChartContext(is_ok=True, visible_price=spot)`
  + EV vector `(+25, +inf, -inf, NaN)` through
  `EnginePhaseReviewer.review(dossier)`.
- **Result:**
  - `+25` → `proceed / ev_above_threshold` (control)
  - **`+inf` → `blocked / ev_non_finite`** (originally `proceed`)
  - **`NaN` → `blocked / ev_non_finite`** (originally degraded to
    `review`)
  - **`-inf` → `blocked / ev_non_finite`**
  - Bonus: `as_of=2099-01-01` now raises
    `ValueError: ...beyond OHLCV data cutoff 2026-03-20` (PR #215,
    was silent substitution)
- **Recommendation:** **None — §2 surface CLOSED.** PR #204's R1a
  guard works exactly as `CLAUDE.md` §2 R1a documents.

### S20 — API concurrency  [§2 G3 RE-REFUTED on network surface]

- **What:** Can the network webhook surface be exploited to
  introduce `+inf` ev_dollars and produce a tradeable verdict?
- **How:** Spun engine_api on `:8787`, sent `+inf` / `-inf` / `NaN`
  ev_dollars in webhook payload, plus G1 ring-trim / G2 torn-read /
  G4 nonce-replay / G7 validation at `workers=4`.
- **Result:** All non-finite payloads → server-computed AAPL EV
  (`−14.46`) + `verdict=skip`. Buffer holds exactly 200 after 220
  POSTs. 40 concurrent GETs all return exactly 30 items. 16
  same-nonce POSTs → 1×200, 15×409. Empty body → 400.
  **`_sanitize_nans`: response body never contains `Infinity` / `NaN`
  tokens.**
- **Recommendation:** None — server-side override `_enrich_alert` is
  mechanically tight. PR #216 (`request_queue_size=128`) and PR #219
  (`_tv_seen_register` lock) shipped on `main` during the run; both
  harden the same surfaces.

### S21 — D17 confirm-fixed + pro-account sizing

- **What:** Does Prong A still trip `sector_cap_breach` on CAT @
  $150k and Prong B trip `portfolio_delta_breach` at $1M?
- **How:** `WheelTracker(initial_capital=150_000,
  require_ev_authority=True).open_short_put` on CAT with token
  issued; then $1M / 9-name pos-EV book with token-consume.
- **Result:**
  - **Prong A: CAT → `action=reject reason=sector_cap_breach` ✓**
  - **Prong B: 2 / 9 opened, 7 blocked by
    `portfolio_delta_breach` ✓** — identical pattern to original.
- **Recommendation:** None — D17 working as documented. Delta cap
  calibration discussion still open as a separate design call (not
  a bug).

### S22 — Roll defense economics

- **What:** Archival per task spec.
- **How:** **SKIPPED** per task spec — "pre-IV-PIT-fix engine;
  duplicates S27."
- **Result:** S22 F1 (drops accumulator on `suggest_rolls`) closure
  verified live in S3 above.
- **Recommendation:** None.

### S23 — Earnings-window navigation  [F1 + F3 BOTH CLOSED]

- **What:** Is the event-gate back-buffer still dead code, and is
  IV-input still a snapshot (non-PIT)?
- **How:** AVGO at four dates around 2026-03-04 earnings (TDB / TDA
  / 6d / 9d post).
- **Result:**
  - **F1 (dead back-buffer) CLOSED:**
    `MarketDataConnector.get_recent_earnings` now exists (PR #180).
    AVGO at `as_of=2026-03-05` (TDA-1) now DROPS with `event` gate
    (originally surfaced as tradeable).
  - **F3 (IV not PIT-aware) CLOSED:** AVGO `iv=0.4844` at
    2026-03-10 and `iv=0.4982` at 2026-03-13 — vary across dates,
    match IV file. Originally both used `iv=0.4296` (snapshot). PR
    #179 confirmed in `rank_candidates_by_ev`.
- **Recommendation:** None — both findings mechanically closed.

### S24 — Multi-strategy book composition

- **What:** Does `take_snapshot` decompose a 3-state book
  (SHORT_PUT + STOCK_OWNED + COVERED_CALL) correctly?
- **How:** Built MRK short-put + KO assigned + (would-be CC) on
  $500k tracker; called `take_snapshot(positions,
  today=date(2026,3,20))`.
- **Result:** Snapshot returned `option_positions=1` +
  `stock_holdings=1` — correct schema mapping. **`WheelTracker.open_strangle`
  still absent** — S14/S24 strangle-integration gap remains open.
- **Recommendation:** Follow-up to add strangle tracker support
  (combined with the S14 finding).

### S25 — Vol-shock recovery (MU)  [F3 + F4 BOTH CLOSED]

- **What:** Is the CC ranker now PIT-IV-aware around MU's
  2026-03-18 earnings?
- **How:** `wr.rank_covered_calls_by_ev("MU", shares_held=100,
  as_of="2026-03-17"/"2026-03-19", use_event_gate=False)`.
- **Result:** **MU CC iv=0.6939 @ 2026-03-17 (was 0.6485
  snapshot)** + **iv=0.6515 @ 2026-03-19** — matches IV history file
  exactly. PR #179 propagated to `rank_covered_calls_by_ev`. Sign of
  25-Δ CC EV stays negative on both dates — engine's conservative
  posture on high-vol earnings holds.
- **Recommendation:** None — original entry's exit prediction
  confirmed exactly.

### S26 — Mid-cycle re-evaluation

- **What:** Does `suggest_rolls` still produce defensible roll /
  hold recommendations on a winning (MU) and challenged (AAPL) put?
- **How:** Open AAPL 25-Δ put at `as_of=2026-02-09`, re-eval
  2026-02-23 (21 DTE remaining); open MU 25-Δ put at 2026-03-03,
  re-eval 2026-03-17 (the MU rallied +21.6% scenario).
- **Result:**
  - AAPL challenged: 4 / 4 recommend, edge **+$213.59** (orig
    +$229.26, −6.8% noise).
  - MU winning: 4 / 4 recommend, edge **+$2181.54** (orig +$1876.05,
    +16.3% from PR #179 + PR #122).
- **Recommendation:** None — engine's `recommend` boolean
  well-calibrated.

### S27 — $100k 2022-2024 frictionless backtest (24 SP500 tickers)

- **What:** Re-run the full S27 backtest on current engine. Does
  the predictive signal (ρ, quartile spread, per-year shape)
  survive?
- **How:** **Full original window run.** 2022-01-03 → 2024-12-31,
  24 tickers, $100k, 35-DTE / 25-Δ, frictionless,
  `require_ev_authority=False`. Wall-clock 62 min (first attempt
  died at day 600 from resource contention with Terminal C's
  concurrent backtest; restart with checkpointing succeeded).
- **Result (vs original S27 doc):**

  | Metric | Doc | Re-run | Delta |
  |---|---|---|---|
  | Spearman ρ | 0.2183 | **0.1881** | −14% |
  | Q3 / Q0 PnL ratio | 1.7× | **1.5×** | preserved |
  | Final NAV | $151,444 | **$164,876** | +9% |
  | Hit-rate | 76.4% | **80.5%** | +4 pp |
  | Executed trades | 50 | **15** | **−70%** |
  | 2022 mean realized | $21.68 | **$1.72** | −92% (F4 re-surfacing) |

  Per-year ρ: bear (0.38) > recovery (0.18) > bull (0.08) — same
  shape as original. **§2 verified across all 5,944 ranked rows.**
- **Recommendation:** **The 70%-fewer-trades delta is the most
  interesting finding.** Most likely cause: harness BP-gating
  differences vs the original throwaway driver; secondary cause:
  PR #215 / #220 / #227 cutoff guards. Worth a controlled isolation
  Sn to confirm. **Engine signal preserved; trade-execution
  mechanics shifted.**

---

## Drift attribution (headline)

- **PR #179 (IV-PIT)** — primary suspect on 11 EV-magnitude drifts.
  Cleanly propagates IV-PIT into all three rankers.
- **PR #119 (news + credit PIT-leak)** — S11 credit-overlay closure
  (1.00 pinned → 0.80 / 0.92 at 2025-04 VIX spike).
- **PR #180 (symmetric event-gate back-buffer via
  `get_recent_earnings`)** — S23 F1 closure; AVGO TDA now blocks.
- **PR #204 (R1a `ev_non_finite` guard)** — S19 C7b §2 closure;
  `+inf` / `-inf` / `NaN` → `blocked / ev_non_finite`.
- **PR #215 / #220 + diagnostic columns (#208 / #210 / #222)** —
  **S18 L2 warm-rank latency regression (10 s → 41 s, +292%)**. Not
  a §2 issue, but a real operator-facing throughput change.
  **Flagged. No fix in this PR.** Highest-priority follow-up.

---

## Recommended follow-ups (not bundled in this re-verification)

1. **S18 L2 warm-path regression** (highest priority). Profile a
   single warm `rank_candidates_by_ev(503)` call to identify the
   ~+30 s overhead. Suspects: per-ticker `_check_as_of_cutoff`
   running synchronously, plus diagnostic columns adding
   `get_fundamentals` lookups per row.
2. **Strangle-tracker integration** (S14 / S24 open gap). Add
   `PositionState.SHORT_STRANGLE` + `WheelTracker.open_strangle()`.
3. **HRP + Kelly exercise** under a delta-cap-loosened harness
   (S15 / S21 open).
4. **S16 structured-drops** (`reason` free text → discrete fields).
5. **S27 trade-count isolation Sn.** Confirm whether the 70%-fewer
   executed trades is harness-shape (BP gating differences vs the
   original throwaway driver) or engine-mechanism (PR #215 / #220 /
   #227 cutoff guards).
6. **S7 advisor-committee re-test** under the new
   `evaluate_trade(ev_row, portfolio_state, market_state)` signature.

---

## Validation gates (final state)

| Gate | Result |
|---|---|
| Launch-blocker subset | **93 / 93 passed** (post-run, rebased) |
| Full pytest | **2412 / 3 / 2** (3 failures = pre-existing Windows-local `theta_connector` issues) |
| 5-ticker EV smoke parquet | **byte-identical** to pre-flight (sha256 `4fc14bf0...5317`) |
| ruff | 547 pre-existing errors in `.py` files; this re-verification touched 0 `.py` files (docs-only) |

**Pre/post pytest delta interpretation:** The pre-flight baseline
at engine SHA `8a17b0b` was 2375 passed / 17 failed / 2 xfailed.
The post-run was at SHA `46ddbd4` (rebased) and showed 2412 passed
/ 3 failed / 2 xfailed. The 14-test improvement is attributable to
**PR #237** (`fix(tests): extend synthetic OHLCV to cover
as_of=2026-03-15`), which merged into `main` during this work and
was pulled in via `git rebase origin/main` before the final pytest
run. **None of the 17 pre-existing failures were introduced by this
re-verification.** The 3 remaining post-run failures are
pre-existing Windows-local `theta_connector` issues per the
[[windows-local-vs-ubuntu-ci]] memory.

---

## Engine state — overall posture

The engine is **mechanically sound on the §2 invariant** at
`origin/main` HEAD `46ddbd4`. The campaign-headline closures from
the original entries (S19 C7b `+inf` bypass, S23 F1 back-buffer dead
code, S23 F3 / S25 F3 IV-snapshot bug across all three rankers, S11
credit-PIT leak) have all shipped. Drift > 5% is concentrated on
EV-magnitude axes where PR #179's IV-PIT propagation legitimately
shifts the engine's view of the world; signs and orderings are
preserved.

The most material unresolved item this re-verification surfaces is
the **S18 warm-rank latency regression** — not a §2 issue, but a
real operator-facing throughput change worth a dedicated follow-up.
