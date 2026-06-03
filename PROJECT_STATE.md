# Project State

**Last updated:** 2026-05-31.

> **Live sources of truth — don't duplicate them here, they decay.** The
> current `main` HEAD and exact test count are in `git log origin/main` and
> the latest CI run; in-flight work is on GitHub issue #113 (the
> coordination board); per-PR history is in `CHANGELOG.md`; the canonical
> verification index is `docs/VERIFICATION_INDEX_2026-05-28.md`. This file
> records *durable* temporal state (what's authoritative vs deprecated), not
> a pinned commit SHA or test count — last time those were pinned here they
> went stale within a day.

This file records *temporal* state — what is authoritative now, what is
in progress, what is deprecated. It is the half-life partner of
`CLAUDE.md` (which is the *structural* contract). Update this file
when you finish a meaningful unit of work or discover that something
described here is no longer accurate.

> ⚠️ **Real-money deployment gate — read `docs/PRODUCTION_READINESS.md`
> before any decision to operate this engine against a real brokerage
> account.** Predictive signal is verified (Spearman ρ ∈ [0.19, 0.55]
> across 14+ window×year cells; never negative; window- and capital-
> invariant within ~0.05). All three historical deployment blockers
> are now closed at the code level: **B1** (F4 tail-risk) shipped as
> a defense-in-depth bundle via PR #260 (realized-vol-ratio widening
> — the frequency guard) + PR #262 (R10 single-name notional cap —
> the magnitude guard); **B2** (D17 live-wire) shipped via PR #233 +
> #255; **B3** (capacity) is structurally closed via S34. The
> *structural* finding from S38 + S40 + S44 remains: the engine
> systematically underperforms passive in bull-dominated multi-year
> windows due to limited deployment (15-23% NAV) — this is not
> fixable engine-side. `docs/LAUNCH_READINESS.md` covers code-
> quality merge gates; `docs/PRODUCTION_READINESS.md` covers
> commercial deployment gates. They are complementary, not
> substitutes.

> **What the engine is / isn't (defensive-sleeve framing — heavy-verify
> 2026-05-31, Category C).** The engine is a **defensive premium sleeve, not a
> bull-market growth substitute**: it earns its keep in down/sideways/high-rate
> tape (+27pp vs passive in the 2022 bear, +10pp in the 2020 crash, at ~0.4–0.6×
> the index drawdown) and structurally lags strong bulls (−19 to −26pp) — size it
> as a complement to long equity beta, not a replacement. And **`ev_dollars` is a
> tail-risk-adjusted *ranking* score, NOT a dollar-profit forecast** — it has ≈0
> rank-correlation with realized $ P&L (I1); its value is in *selection* (top-K
> beats random, I6-B). Use `prob_profit` / `ev_roc` for ranking and read
> `ev_dollars` only as a tail-aware score. Basis:
> `docs/HEAVY_VERIFY_2026-05-31_INDEX.md` (Category C of
> `docs/HEAVY_VERIFY_2026-05-31_REMEDIATION.md`).

---

## 1. Authoritative — do not bypass

| Module | Public entry | Locked by |
|---|---|---|
| `engine/ev_engine.py` | `EVEngine.evaluate` | `tests/test_audit_invariants.py`, `tests/test_audit_viii_*`, `tests/test_evengine_event_lockout.py`, `tests/test_dealer_multiplier_evengine_integration.py` |
| `engine/wheel_runner.py` | `WheelRunner.rank_candidates_by_ev` | `tests/test_authority_hardening.py`, `tests/test_audit_viii_real_data_smoke.py`, `tests/test_f4_tail_risk_gap.py`, `tests/test_consume_ranker_row_anchor.py` |
| `engine/candidate_dossier.py` | `EnginePhaseReviewer`, rules **R1–R11** (R1 ev-non-finite / negative-EV; R2 chart-missing; R3 spot-mismatch; R4 phase-contradiction; R5 ev-threshold; R6 short-gamma / dealer-flip; R7 portfolio VaR; R8 stress + dealer-regime; **R9 sector cap**; **R10 single-name cap**; **R11 elevated-vol top-bin**) | `tests/test_dossier_invariant.py`, `tests/test_portfolio_risk_gates.py`, `tests/test_dossier_r9_r10_audit.py`, `tests/test_r11_elevated_vol.py` |
| `engine_api.py` | HTTP API on `SWE_API_PORT` (default `:8787`; per-terminal in worktrees per D15); endpoint header in the file | `tests/test_tv_api.py`, `tests/test_tv_dossier.py`, `tests/test_engine_api_port.py` |

These four routes are the only sanctioned paths from raw inputs to a
tradeable verdict. Reviewers (chart provider, advisor committee,
dealer positioning, **R7-R10 portfolio-context gates**) can
downgrade outputs — never upgrade. As of D18 (2026-05-26), news
sentiment is severed from the EV path —
`engine/news_sentiment.py::sentiment_multiplier` is a constant-1.0
stub; the operator dashboard still consumes the underlying score for
transparency. R1 (negative or non-finite EV →
blocked) is the hard CLAUDE.md §2 invariant; R7-R10 are conditional
soft-warns that fire only when a `PortfolioContext` is attached.
**The token gate (D16) re-checks R1 at fire time** — see `DECISIONS.md` D16.

> **Data currency (point-in-time).** The committed Bloomberg CSVs are
> point-in-time as of **2026-03-20** (the freshest cut the `xbbg`
> pullers' hardcoded `end_date` reaches). A full refresh is **partially
> blocked**: only 3 of the 9 connector CSVs have a reproducible in-repo
> producer; the other 6 — including the core IV file
> `sp500_vol_iv_full.csv` — have no repo producer. See
> `docs/DATA_POLICY.md` §5 and `docs/bloomberg_refresh_runbook.md`.

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

After audit-VIII the suite reported 1087 passed / 0 failed and 287
deprecation warnings (down from 1067+1 / 578).

**Post-audit-viii shipped work (2026-05).** Tracked as PR-level
entries in `CHANGELOG.md` rather than the `audit-<N>` series:

| Bundle | Shipped via | Tests added |
|---|---|---|
| **D16** EV-authority token verdict-binding (issuance + consume both re-check `ev_dollars > 0`) | PR #128 + audit-viii follow-on | `tests/test_authority_hardening.py` (D16 block), `tests/test_audit_viii_e2e.py`, `tests/test_wheel_tracker_persistence.py::test_persisted_token_consume_round_trip_d16` |
| **D17** portfolio-risk gates (sector / single-name / delta / Kelly / VaR / stress / dealer-regime) — tracker hard-blocks + R7-R10 dossier soft-warns; live-wired on `/api/tv/dossier` and `/api/tv/enrich` | PR #154 + #163 + #233 + #255 + #262 | `tests/test_portfolio_risk_gates.py`, `tests/test_authority_hardening.py::TestD17HardBlocks`, `tests/test_dossier_r9_r10_audit.py`, `tests/test_ev_authority_log_schema.py` |
| **F4 tail-risk widening v2** — realized-vol-ratio (RV30/RV252) regime-conditioned widening; replaces rolled-back HMM v1 from #253 | PR #260 | `tests/test_f4_rv_widening.py` |
| **Backtest regression harness** — S27/S32/S34/S35 + S43 rolling multi-window pinned against current engine | PR #196 / #220 / #270 | `tests/test_backtest_regression.py` (gated by `@pytest.mark.backtest_regression`) |
| **Verification battery wrap-up** — live R1-R10 check + real-data anchor checks + master canonical index | PR #268 / #270 / #271 / #273 | (docs + drivers; observable-test pattern, not pytest) |

The suite is **~2,500 tests** — run `pytest --collect-only -q` for the
live count — up from 1087 at audit-viii. A small number of Windows-local
Theta-tier tests skip/flake off the laptop; they are not engine defects.

## 3. Work in progress

> ⚠️ **PROJECT_STATE divergence note (2026-05-28):** prior to this
> refresh, the file was last touched on 2026-05-23 against `origin/main`
> @ `643f1af`. Five days and ~15 merged PRs have intervened (D17
> closures, F4 v2, R9-R10, S40-S46 backtests, verification campaign);
> the older subsections below (TradingView MCP integration, iv_surface
> decision, Theta data refresh, etc.) are preserved verbatim as the
> historical record. Newest-first additions for the 2026-05 late
> campaign appear above them.

### D17 portfolio-risk-gates closure — 2026-05-26 → 2026-05-27

- **B2 part 1** (PR `#233`, `b55a59a`) — D17 portfolio-context live
  wire on `/api/tv/dossier`; R7 (VaR) and R8 (stress + dealer-regime)
  soft-warns now fire live on the network surface for any candidate
  that reaches the endpoint.
- **B2 part 2** (PR `#255`, `f3a4fa8`) — R9 `sector_cap` dossier
  soft-warn added on top of R7/R8 + D17 wired on `/api/tv/enrich`.
  Mirrors the tracker's HARD refusal at `open_short_put` time when
  `require_ev_authority=True`. Default 25% NAV per sector via
  `engine.portfolio_risk_gates.check_sector_cap`.
- **R10 single-name cap** (PR `#262`, `45ca861`) — F4 damage-bounding
  closure. 10% NAV per-underlying short-option notional cap. Sits
  BENEATH R9 (a ticker concentrated as the only name in its sector
  could pass R9 at 25% but trip R10 at 10%). Soft-warn on dossier
  (`verdict_reason="single_name_breach"`); HARD refusal on tracker
  when `require_ev_authority=True`. See `docs/F4_TAIL_RISK_DIAGNOSTIC.md`
  §10 for the F4 motivation.
- `DECISIONS.md` D17 documents the full design (gates, defaults,
  Q3 missing-data semantics, rejected alternatives).

### D17 cap adoption status — verified 2026-06-01

A reliability sweep against `origin/main` found the R9/R10 concentration
caps are correctly **implemented and unit-tested** but **not active on any
default operator path** — "documented protection ≠ active protection".
Source-verified specifics:

- **Hard refusals (tracker, at `open_short_put`):** armed when
  `enforce_sector_cap` / `enforce_single_name_cap` is `True` (decoupled
  from `require_ev_authority` since D22 / PR #303 —
  `engine.wheel_tracker._d17_gate_enabled`, `wheel_tracker.py:1805`), or
  when `require_ev_authority=True`. All three default `False`
  (`WheelTracker.__init__`, `wheel_tracker.py:278-280`). The canonical
  armed constructor `engine.wheel_runner.make_live_book_tracker()` (sets
  both `enforce_*_cap=True`) has **zero non-test callers** — every non-test
  `WheelTracker(...)` site uses the bare default (`backtests/simulator.py`,
  `backtests/regression/_common.py`, `scripts/s47_trader_session_2026_03_20.py`,
  `scripts/transaction_costs_demo.py`). The hard refusals are therefore
  dormant on every backtest / reproduction / demo path. (Arming them on the
  regression baselines is intentionally avoided — D22 rejected flipping the
  library default because it moves pinned snapshots + mechanics tests.)
- **Soft-warns (dossier R7-R10, downgrade proceed→review):** fire only when
  a populated `PortfolioContext` is attached to `build_candidate_dossiers()`.
  On the network surface that happens on `/api/tv/dossier` + `/api/tv/enrich`
  **only when the caller supplies `nav`** — `_build_portfolio_context_from_params`
  returns `None` otherwise (`engine_api.py:284-285`, the Q3 "don't fire on
  absent evidence" rule). The default `rank_candidates_by_ev` attaches no
  context, and the tracker's own `portfolio_context_snapshot()` is never
  auto-fed to the reviewer (only test + S47 + verification-artifact callers).
- **Net:** the caps guard correctly *when armed*, but there is **no
  default-armed live path today** — consistent with the §3 design that the
  engine has no broker / OMS / execution surface. Closing the gap is a
  forward step: route a future live path (or the S47-style operator harness)
  through `make_live_book_tracker()`, and/or have the tracker auto-feed its
  book to the dossier reviewer (an engine behaviour change that needs a
  backtest re-baseline). This note supersedes the "HARD refusal … when
  `require_ev_authority=True`" framing in the B2 / R10 bullets above, which
  predates the D22 decoupling.

> **Known stale code comment (flagged, not fixed — trio files).** The inline
> `# Gate 1 (R9): sector cap — armed by enforce_sector_cap (default on)` at
> `wheel_tracker.py:1923` reads "default on", but the bare-constructor default
> is OFF — accurate only inside `make_live_book_tracker()`. Likewise that
> factory's docstring says production "MUST be constructed through this
> factory" though no production caller exists yet. Both are trio-file edits
> deferred to an operator-greenlit decision-layer touch.

### F4 tail-risk widening v2 — 2026-05-27

- **HMM v1 rolled back** (PR `#253` draft research-record). The
  K=4 HMM "crisis" label over-fired on the calm-bull plurality
  (98% of 2022-2024 dates) and inverted S27 ρ from +0.188 to
  −0.094 — see memory `f4-widening-overfires-on-hmm-labels` for
  the diagnosis. Branch `claude/fix-f4-regime-conditioned-widening`
  carries the research record; the fix was abandoned.
- **Realized-vol-ratio widening shipped** (PR `#260`, `0dddf76`).
  Uses RV30 / RV252 as a continuous regime-conditioned multiplier
  (cap 1.5×; sign-preserving — factor ≥ 1.0; never narrows tail
  risk). Routed through the forward distribution →
  `EVEngine.evaluate` (never an overlay on final `ev_dollars`).
  S27 ρ gate held (`ρ ≥ +0.15` required). COST 2022-04 and UNH
  2024-11 anchor cases resolved at the input-signal level.
- **B1 closure framing:** F4 deployment is the **bundle** of #260
  (frequency guard, RV widening) + #262 (magnitude guard, R10
  single-name cap). Neither alone is sufficient — S41 (PR #267) +
  S44 (PR #271) backtests independently confirmed that #260 alone
  is signal-preserving but not value-creating in dollars; R10 is
  the load-bearing magnitude guard.

### Backtest + verification campaign — 2026-05-25 → 2026-05-28

Backtest evidence sequence at $1M / 100t scale (deployment matrix in
`docs/PRODUCTION_READINESS.md` §1):

| Sn | Window | Scale | Engine vs passive | Doc |
|---|---|---|---|---|
| S34 | 2022-2024 | $1M / 100t | **+11.6pp** | `docs/ENGINE_BACKTEST_S34_UNIVERSE.md` |
| S35 | 2018-2020 OOS | $100k / 24t | **−41pp** | `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` |
| S38 | 2020-2024 | $1M / 100t | **−52pp** | `docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md` |
| S40 | rolling 3-start | $1M / 100t | **−85pp to +10pp** | `docs/ENGINE_BACKTEST_S40_ROLLING_MULTIWINDOW.md` (PR #264) |
| S41 | 2022-2024 (F4 fix probe) | $100k / 24t | F4 fix signal-preserving (ρ +0.188 → +0.182), value-neutral | `docs/ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md` (PR #267) |
| S43 | rolling 4-window | $1M / 100t post-#260 | **−51pp to −104pp**; ρ window-invariant | `docs/ENGINE_BACKTEST_S43_ROLLING_MULTIWINDOW.md` (PR #270) |
| S44 | S38 re-run post-F4 | $1M / 100t | **F4 hypothesis FALSIFIED** — +0.56pp delta | `docs/ENGINE_BACKTEST_S44_S38_POSTF4_RERUN.md` (PR #271) |
| S46 | re-verify closed tests post-F4 + R10 | various | completed | `docs/ENGINE_REVERIFY_S46_POST_F4_R10.md` |
| S47 | live wheel session trust-audit (`as_of=2026-03-20`, VIX 28.97, HMM bear) | $- / 5t | TRUST entry, DISTRUST management; R11 size-down behaves correctly | `docs/worklog/s47-live-wheel-session-2026-03-20-trust-audit-on-an.md` |

The current usage-test high-water is **S47** (the worklog ledger
`docs/worklog/INDEX.md` is the source of truth — don't hardcode a count;
it drifts every session).

Verification campaign wrap-up artifacts (all merged to `origin/main`
during the 2026-05-28 → 2026-05-29 wave; see `CHANGELOG.md`):

- PR #268 — `docs/REALISM_VERIFICATION_2026-05-28.md` (live R1-R10
  battery, 8/8 sections green, 0 defects);
- PR #270 — S43 rolling backtest + R10 post-hoc audit;
- PR #271 — S44 S38 post-F4 re-run;
- PR #273 — `docs/REAL_DATA_VERIFICATION_2026-05-28.md`
  (rv30/rv252 bit-identical; prob_profit calibration MIXED; BSM ≤5%
  textbook; IV bit-identical to raw CSV) + multi-config calibration
  follow-up `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md`;
- **Canonical wrap-up index:** `docs/VERIFICATION_INDEX_2026-05-28.md`
  (marked CANONICAL 2026-05-28 in this PR). Twelve dated review docs
  were archived to `archive/2026-05/` in this PR; the index carries
  forward each headline finding.

### prob_profit calibration — open structural finding

The **top bin (0.95, 1.0]** of `prob_profit` is over-optimistic by
10-18pp across all 10 backtest configurations measured (S22, S27,
S32, S34, S35, S38 pre/post-F4, S40 W1/W2/W3). 9 of 10 configs have
the top bin MISCAL > 10pp; F4 fix (#260) does NOT improve calibration.
Established as **structural to the empirical-distribution method**
in `engine.forward_distribution`, not S38-specific. **R10 (#262) is
the load-bearing magnitude guard** precisely because the engine
cannot self-correct its top-bin over-confidence in the rank-time
EV. Doc: `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md`. Open research
direction: wire POT-GPD tail extension (machinery in
`engine/tail_risk.py`) into the `prob_profit` computation path —
the engine has the machinery but does not apply it to `prob_profit`.

### TradingView MCP integration

- **Design contract:** `docs/TRADINGVIEW_MCP_INTEGRATION.md`
  (commit `c064652`). Pins the seam, the canonical `ChainedChartProvider`
  ordering, the four hard invariants (including no-quiet-substitution
  on MCP failure), and the M1 scope (5 MCP tools).
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
- **Resolved (2026-05-30, A2 — `DECISIONS.md` D9):** chose **fail loudly**.
  The SVI tooling is wired in behind `SurfaceDataUnavailable` +
  `require_surface`; first caller `scripts/diagnose_iv_surface.py` (non-zero
  exit on uncovered tickers). `create_constant_surface` is the only opt-in
  flat surface. Pinned by `tests/test_iv_surface_failloud.py`.

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
| `LAUNCH_READINESS.md` | Consolidated launch-blocker checklist: hard EV invariant, four authoritative routes, dossier R1–R11, pre-merge checklist. |
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
  `engine/news_sentiment.py` (operator-only transparency layer — was
  a downgrade-only reviewer until D18 severed it from the EV path),
  `scripts/pull_news_sentiment.py` (one-shot puller, still writes the
  parquet that the dashboard consumes). Post-D18 (2026-05-26), **no
  news subsystem feeds the EV authority** — verbal news is operator-
  layer only. Replacement quantitative layers (EDGAR earnings dates,
  FRED macro, EDGAR-XBRL fundamentals quality score) are in flight
  via the news-architecture redesign campaign — see
  `docs/NEWS_REDESIGN_CAMPAIGN.md`.

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

**This 2026-05-28 PR** (`claude/docs-consolidate-verification`)
closes the latest drift batch: it archives 12 superseded review
docs to `archive/2026-05/`, marks `docs/VERIFICATION_INDEX_2026-05-28.md`
as the single CANONICAL living verification index (with an
"Archived snapshots" map + "Deferred (locked by open PRs)" table),
and refreshes §1-§6 of this file to reflect the 2026-05 late
campaign (D17 closures, F4 v2, R9-R10, S34→S46, verification
wrap-up). No code changes; the engine SHA and the §2 invariant are
unchanged.

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
- **Parallel sessions (D15).** Each executor terminal runs in its own
  git worktree (`../swe-terminal-<x>`); the primary clone
  (`smart-wheel-engine/`) is reserved for Sessions and orchestration.
  Per-terminal env (`SWE_API_PORT`, `COVERAGE_FILE`, `PYTEST_CACHE_DIR`)
  is sourced from `scripts/setup-terminal.{sh,ps1}` or — preferred on
  Windows — from a per-worktree `.claude/settings.local.json` (gitignored)
  picked up by the Claude Code harness on launch. Coordination is via
  GitHub issue #113. See `docs/PARALLEL_SESSIONS.md`.
