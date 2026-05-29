# Session Report — Deployment-Readiness Verification Campaign (2026-05-26)

**Owner:** Terminal B
**Branch baseline:** `origin/main` @ `3b67055` (post-merge)
**Window:** 2026-05-26, single continuous session
**Companion docs:** [`docs/PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) (deployment gate), [`docs/LAUNCH_READINESS_ANALYSIS_2026-05-26.md`](LAUNCH_READINESS_ANALYSIS_2026-05-26.md) (verdict), [`docs/SOUNDNESS_REVIEW_2026-05-26.md`](SOUNDNESS_REVIEW_2026-05-26.md) (math re-verification)

This is the **machine-readable session ledger** — what the user asked,
what we tested, how we tested it, what we found, and what the next
agent should pick up. Other agents reading this should be able to
reconstruct the campaign without re-deriving anything.

---

## 1. User intent (chronological)

1. "Make sure our options engine is sound and delivering realistic outputs. Verify whether our product is working effectively under stress. Make necessary implementations. Be accurate and professional."
2. "Please proceed if there is something needs to be done. I will leave the computer open. Do not ask for permission. Proceed with what you think is best for the project."
3. "Review the results again to make sure our options engine is sound."
4. "Fix those findings."
5. "Let's focus on the other deployments, fix all."
6. "Consider merging the PRs if you are done with them."
7. "Beware that there are other agents working on the same project — make no mistakes."

The instructions decoded to four phases:

- **Phase A** — verify soundness (re-verify earlier findings, find new gaps).
- **Phase B** — close the three named deployment blockers (B1 F4 tail-risk, B2 D17 live-wire, B3 capacity at >$100k).
- **Phase C** — stress test at scale (multi-window backtest).
- **Phase D** — land the work (merge cascade).

---

## 2. What we tested

### 2.1 Soundness verification (Phase A)

| Surface | Test type | Question |
|---|---|---|
| §2 invariant (decision-layer ranker route) | Static read + targeted test re-run | Does any code path convert a non-tradeable candidate into a tradeable one without `EVEngine.evaluate`? |
| Realized vs predicted P&L | Numeric re-derivation across 532 executed put rows from S22/S27/S32 backtests | Does the formula `realized = max(strike - spot_at_expiry, 0) - premium` match what the tracker actually credits/debits? |
| Spearman ρ statistical significance | Recomputed ρ + p-value from raw ledger | Is the ranking signal genuinely better than random, or could ρ ≈ 0.22 be sampling noise? |
| Engine refusal mechanism | Counted refusal rate during 2020-02-15 → 2020-05-15 (COVID) | Does the engine sit out crises correctly? |
| Equity-beta vs put-selection alpha | Decomposed NAV gain into "realized P&L from executed trades" vs "unrealized appreciation on assigned stocks" | Is the engine's "+27pp over SPY" headline actually put-selection alpha, or is it equity beta on bull-market-favored single names? |
| BKNG concentration | Removed BKNG from S27 backtest and re-computed both ρ and net realized P&L | Is the signal robust to removing the dominant single contributor? |

### 2.2 Deployment-blocker remediation (Phase B)

| Blocker | Originating finding | Fix attempted |
|---|---|---|
| **B1 — F4 tail-risk widening** | `prob_profit = 0.8333` stayed constant across COST's 31.5% drop in April-May 2022 (S22 / S27 / S32 F4) | **Fix A:** reduce `lookback_years` default from 5.0 to 2.0 / 3.0; raise `min_empirical_samples` from 20 to 40 |
| **B2 — D17 live-wire** | `EnginePhaseReviewer` R7+R8 soft-warns require a `PortfolioContext` that is silently dropped by `/api/tv/dossier` (which has no portfolio params) | **Fix:** added opt-in `portfolio_context` query param to `/api/tv/dossier` that parses `nav` + `holdings_csv` + `puts_held_csv` + `regime_map_csv` into a `PortfolioContext` |
| **B3 — Capacity at >$100k** | S32 (closed earlier) found 10.8% capital deployment at $1M with 24 tickers → engine returns +1.85% vs SPY +24% (−22pp) | **Fix:** expand universe from 24 to 100 SP500 tickers (first alphanumeric) |

### 2.3 Engine-API hardening (Phase B side branch)

| Surface | Question |
|---|---|
| `_tv_seen_register` thread safety (S20 C3) | Is the check-then-set protected by anything other than CPython's GIL? |
| HTTP listen-queue depth (S20 G5 / C2) | At what concurrency does the engine API start dropping connections? |

### 2.4 Multi-window stress (Phase C)

| Scenario | Question |
|---|---|
| **S38** — $1M / 100 tickers / 2020-01-02 → 2024-12-31 (5 years) | Does S34's "+11.6pp over SPY at $1M / 100t / 2022-2024" generalize across a longer multi-window window that includes COVID + 2021 bull + 2022 bear + 2023-2024 recovery? |

---

## 3. How we tested

### 3.1 Soundness verification

- **§2 invariant** — read `engine/wheel_runner.py` end-to-end, traced every callable that returns a `DataFrame` row destined for `WheelTracker.open_short_put_v2`. Re-ran the launch-blocker subset (93 tests across `test_audit_invariants.py`, `test_dossier_invariant.py`, `test_authority_hardening.py`, `test_audit_viii_*`, `test_launch_blockers.py`). Confirmed reviewer rules R1-R8 are downgrade-only.

- **P&L formula consistency** — extracted 532 executed put rows from S22 / S27 / S32 `rank_log.csv` and `closed_position.csv` outputs (already in `%TEMP%` from earlier backtests). Re-computed realized P&L from `strike`, `spot_at_expiry`, `premium`, `multiplier` and compared to the tracker's logged value. Match within rounding error (< $0.01) on all 532 rows.

- **Spearman ρ** — recomputed via `scipy.stats.spearmanr(predicted_ev_dollars, realized_pnl)` from the raw ledger. p-values < 1e-100 across all scenarios; signal is genuinely statistically real.

- **Refusal during COVID** — counted `drops_summary` and `n_executed` rows for 2020-02-15 → 2020-05-15 in S35 (and later S38). Compared mean realized P&L of all candidates vs only executed.

- **Equity-beta decomposition** — for each S27 / S34 / S35 / S38 backtest, partitioned final NAV change into:
  - Realized P&L from closed put + CC trades.
  - Unrealized appreciation on assigned-and-held stock positions (mark-to-market at backtest end).
  Computed the ratio `realized / (realized + unrealized)` to surface what fraction of NAV gain came from the wheel strategy's option leg vs equity beta on assignments.

- **BKNG concentration test** — re-ran S27 ranker output filtered to exclude BKNG rows; recomputed ρ and aggregated realized P&L. ρ drift was 0.327 → 0.324 (signal robust); net realized P&L flipped sign (BKNG was 110% of the original net).

### 3.2 Blocker remediation

- **B1 F4 fix** — created branch `claude/fix-f4-tail-risk-lookback`. Tried multiple variants:
  - `lookback_years=5.0 → 2.0` (default)
  - `lookback_years=5.0 → 3.0`
  - `min_empirical_samples=20 → 40`
  - All combinations of the above.
  Re-ran `test_f4_tail_risk_gap.py`'s COST 2022-04-25 + UNH 2024-11-21 fixtures after each variant; measured `prob_profit` before vs after.

- **B2 D17 wire** — added `_build_portfolio_context_from_params(nav, holdings_csv, puts_held_csv, regime_map_csv)` helper to `engine_api.py`. Parses CSV-encoded params into `WheelPosition` / `StockHolding` / `dealer_regime_by_ticker` and constructs a `PortfolioContext`. Wired into `/api/tv/dossier` request handler. Added 13 tests in `tests/test_tv_dossier_d17_wire.py`:
  - 4 tests on the helper alone (null nav → None, well-formed CSV → context, mismatched columns → None, …).
  - 6 tests on the route (no params → unchanged behavior, valid context → R7 soft-warn fires when VaR exceeds 5%, R8 soft-warn fires on vol-spike OR short-gamma, …).
  - 3 tests on token / verdict propagation.

- **B3 universe expansion (S34)** — separate Sn backtest, ran by Terminal C earlier, results re-verified by Terminal B and amended with equity-beta + concentration framing.

### 3.3 Engine-API hardening

- **TV nonce lock (#219)** — added module-level `_TV_SEEN_NONCES_LOCK = threading.Lock()` and wrapped the entire body of `_tv_seen_register` in `with _TV_SEEN_NONCES_LOCK:`. Added `tests/test_tv_nonce_register_lock.py`:
  - Structural regression: `_TV_SEEN_NONCES_LOCK` exists, is a `threading.Lock`, acquire-release works.
  - **64-worker race-the-window**: `ThreadPoolExecutor(64) + threading.Barrier(64)` firing `_tv_seen_register("same-digest")` concurrently; exactly 1 worker must accept.
  - 64 concurrent distinct digests: all 64 must accept.
  - Sequential calls must release the lock cleanly.

- **Listen-queue depth (#216)** — introduced `_EngineHTTPServer(ThreadingHTTPServer)` subclass with `request_queue_size = 128` (matches uvicorn / gunicorn defaults; stdlib default was 5). Added 4 regression tests in `tests/test_engine_api_port.py::TestEngineHTTPServerListenQueueDepth`.

### 3.4 S38 multi-window backtest

- **Harness:** `%TEMP%\s38_backtest\run.py` — identical to S34 except `START_DATE=2020-01-02` and `END_DATE=2024-12-31`. Same 100 alphanumeric SP500 tickers. $1M starting capital. 35-DTE / 25-delta short puts. Wheel into CC on assignment, hold to expiry. `require_ev_authority=False`. Three parallel `WheelTracker` instances per friction level (frictionless / bid_ask / full).
- **Execution:** ~3 hours wall clock on dev box. 1,258 trading days. 17,360 candidates per friction level (52,080 total ledger rows). 0 rank failures.
- **Outputs:** `rank_log.csv` (52,080 rows), `closed_position.csv`, `summary.txt`. All analysis derived from these flat files; no engine code modified during the run.

---

## 4. Results

### 4.1 Soundness verification

| Question | Verdict | Evidence |
|---|---|---|
| Does any tradeable candidate bypass `EVEngine.evaluate`? | **No.** §2 verified. | Read-through + 93/93 launch-blocker tests pass |
| P&L formula consistent? | **Yes**, < $0.01 rounding error across 532 rows | Re-derived `realized = max(strike - spot, 0) - premium` matches tracker |
| Is ρ ≈ 0.22 statistically real? | **Yes**, p < 1e-100 on N ≈ 5k-17k samples | scipy.stats.spearmanr |
| Does engine sit out crises correctly? | **Yes**, 97.8% refusal in COVID (S38) | 847 candidates → 19 executed |
| Is "+27pp over SPY" put-selection alpha? | **No.** **92% of S34's NAV gain is equity beta on assigned stocks** | Realized executed P&L is negative (−$3,421 in S27, −$28,647 in S38); NAV growth comes from holding assigned stock through 2023-2024 bull |
| Is the signal robust to removing BKNG? | **ρ yes** (0.327 → 0.324), **dollar P&L no** (BKNG drove 110% of S27's net) | Concentration risk is real |

### 4.2 Blocker remediation

| Blocker | Outcome | PR |
|---|---|---|
| **B1 — F4 tail-risk widening** | ❌ **Fix A NEGATIVE RESULT.** Every lookback variant tried (5→2y, 5→3y, +`min_empirical_samples=40`) made the engine MORE optimistic on COST 2022-04, not less. Root cause: the 2-year window from 2022-04 is dominated by COVID recovery + 2021 bull → empirical distribution is right-skewed. Reverted. Documented in [F4_TAIL_RISK_DIAGNOSTIC.md §9](F4_TAIL_RISK_DIAGNOSTIC.md). | #234 (negative result doc) |
| **B2 — D17 live-wire** | ✅ **CLOSED.** Opt-in `portfolio_context` parsing wired into `/api/tv/dossier`. R7+R8 soft-warns now fire when called with context. 13 tests pin the contract. | #233 |
| **B3 — Capacity at >$100k** | ✅ **STRUCTURALLY CLOSED.** Universe expansion 24 → 100 tickers raises deployment from 10.8% to 22.6% at $1M. Capacity gap closable; the question of whether deployment translates to dollar-alpha is separate (see §4.4). | #226 (S34) |

### 4.3 Engine-API hardening

| Surface | Outcome | PR |
|---|---|---|
| TV nonce lock | ✅ Explicit `threading.Lock` added; 64-worker contention test asserts exactly 1 accept | #219 |
| Listen-queue depth | ✅ Bumped 5 → 128 via `_EngineHTTPServer` subclass; 4 regression tests | #216 |

### 4.4 S38 multi-window backtest

**Headline:** at $1M / 100 tickers over the full 5-year 2020-2024 window, **the engine UNDERPERFORMS SPY by ~52pp**.

| Friction | Final NAV | Return | Short puts | Hit-rate | ρ |
|---|---|---|---|---|---|
| Frictionless | $1,348,704 | +34.87% | 305 | 77.0% | 0.362 |
| bid_ask | $1,332,071 | +33.21% | 305 | 77.0% | 0.359 |
| **Full friction** | **$1,331,764** | **+33.18%** | 305 | 77.0% | **0.358** |

**SPY over the same window:** ~+85% (price + dividends).
**Engine vs SPY:** **−52pp.**

**Comparative dollar-alpha across the 5 completed $-scale backtests:**

| Sn | Capital | Universe | Window | Engine | SPY | Engine vs SPY |
|---|---|---|---|---|---|---|
| S27 | $100k | 24 | 2022-2024 | +51% | +24% | **+27pp** |
| S32 | $1M | 24 | 2022-2024 | +1.8% | +24% | **−22pp** |
| S34 | $1M | 100 | 2022-2024 | +35.6% | +24% | **+11.6pp** |
| S35 | $100k | 24 | 2018-2020 | +3.6% | ~+45% | **−41pp** |
| **S38** | **$1M** | **100** | **2020-2024** | **+33.2%** | **~+85%** | **−52pp** |

Five configurations, five different deltas spanning −52pp to +27pp. **Dollar-alpha is a multi-dimensional function of (capital × universe × window). There is no single number that represents the engine's edge.**

**Critical secondary findings from S38:**

- **Realized executed P&L is NEGATIVE** (−$28,647 over 305 puts + 168 CCs). All NAV growth (+$331,764) came from equity-beta on assigned positions (108.6% attributable to equity-beta residual). Same shape as S27 (NAV +$51,444 / realized −$3,421) and S35 (NAV +$3,566 / realized −$48,326). **The engine's put-selection alpha is consistently negative on average across all multi-window backtests.**
- **Signal generalizes (ρ = 0.358 across N = 17,192).** Higher than S34's 0.327. Per-year ρ: 2020 0.55, 2021 0.21, 2022 0.37, 2023 0.31, 2024 0.31. **Never negative per year.**
- **COVID refusal 97.8%** (847 candidates, 19 executed). Mean realized of refused candidates would have been −$254 per trade if blindly executed (~−$215k of losses avoided). **This is the engine's strongest defensible property.**
- **Concentration amplified at scale.** Top 5 tickers contributed +$23,127; the other 57 traded tickers lost −$51,774 net. Aggregate net is −$28,647.
- **Quartile monotonicity at extremes, broken in middle.** Q0 realized +$28; Q1 +$3; Q2 −$67; Q3 +$206. Q3 beats Q0 by 7.4× but Q2 is negative.
- **Capital deployment 22.6%.** ~77% of $1M sits in cash earning nothing during a multi-year bull market. This is the dominant explanation for the −52pp gap.

### 4.5 §2 invariant verification across the cascade

No PR in this campaign touched `engine/ev_engine.py`, `engine/wheel_runner.py`, or `engine/candidate_dossier.py` directly. The two interface-layer fixes (#233 D17 wire, #219 TV nonce lock) and the queue-depth fix (#216) maintain the §2 contract: they downgrade or harden, never upgrade.

**Launch-blocker test subset: 93/93 passing on `origin/main` @ `3b67055`.**

---

## 5. Deliverables shipped

| # | PR | Title | Status |
|---|---|---|---|
| 1 | [#237](https://github.com/MertYakar66/smart-wheel-engine/pull/237) | `fix(tests)` — extend synthetic OHLCV + lint cleanup (CI unblocker) | MERGED |
| 2 | [#234](https://github.com/MertYakar66/smart-wheel-engine/pull/234) | `docs(F4)` — §9 Fix A negative result | MERGED |
| 3 | [#218](https://github.com/MertYakar66/smart-wheel-engine/pull/218) | `docs(production_readiness)` — real-money deployment gate | MERGED |
| 4 | [#225](https://github.com/MertYakar66/smart-wheel-engine/pull/225) | `docs(launch_readiness_analysis)` — comprehensive readiness verdict | MERGED |
| 5 | [#233](https://github.com/MertYakar66/smart-wheel-engine/pull/233) | `fix(engine_api)` — D17 portfolio-context live wire (closes B2) | MERGED |
| 6 | [#219](https://github.com/MertYakar66/smart-wheel-engine/pull/219) | `fix(engine_api)` — TV nonce check-then-set lock | MERGED |
| 7 | [#235](https://github.com/MertYakar66/smart-wheel-engine/pull/235) | `docs(backtest)` — S38 multi-window backtest | MERGED |
| 8 | [#216](https://github.com/MertYakar66/smart-wheel-engine/pull/216) | `fix(engine_api)` — listen-queue depth 5 → 128 | MERGED |
| 9 | [#226](https://github.com/MertYakar66/smart-wheel-engine/pull/226) | `docs(backtest)` — S34 universe expansion (closes B3) | MERGED |
| 10 | [#224](https://github.com/MertYakar66/smart-wheel-engine/pull/224) | `docs(backtest)` — S35 out-of-window cross-validation | MERGED |
| - | [#236](https://github.com/MertYakar66/smart-wheel-engine/pull/236) | `chore(lint)` — pandas import (superseded by #237) | CLOSED |

**`origin/main` HEAD = `3b67055`** at end of session.

---

## 6. What should be done next

### 6.1 Critical (blocks autonomous deployment)

- **B1 — F4 tail-risk widening, structural fix.** Fix A (lookback compression) failed. The next attempt should be one of:
  - **Regime-conditioned distribution widening** — when HMM labels the current bar as `crisis` or `bear`, widen the empirical distribution by mixing in tail observations from a long-history vol-conditional bucket. Spec at [F4_TAIL_RISK_DIAGNOSTIC.md](F4_TAIL_RISK_DIAGNOSTIC.md) Fix B.
  - **POT-GPD on the rolling distribution** instead of the static one. The engine already has POT-GPD; the gap is that `prob_profit` is computed from the empirical distribution, not the POT-GPD tail extension.
  - **Hybrid Bayesian update** that pulls `prob_profit` toward a wide prior when sample size < 30.
  Test cases already exist in `tests/test_f4_tail_risk_gap.py` (currently XFAIL). Unxfail when the fix lands.

- **Decision on the deployment matrix.** S38's −52pp result over a 5-year multi-window directly contradicts S34's window-favored "+11.6pp." The `PRODUCTION_READINESS.md` matrix currently lists "Supervised $500k-$1M, universe ≥ 100 tickers — Conditional ✅" — this should be revised to "Conditional ⚠ with explicit underperformance acknowledgment." A small PR amending §5 is needed.

### 6.2 High-priority (closes known gaps)

- **Rolling-multi-window backtest.** S38 was a single 5-year window. A rolling-5-year-window study (2015-2019, 2016-2020, 2017-2021, 2018-2022, 2019-2023, 2020-2024) would surface whether the −52pp is itself a feature of the 2020-2024 specific window or a general property at scale. ~5 backtests × 3 hours each ≈ 1 working day of compute.

- **SPY-included tracker.** S38 reports "engine vs SPY" using an external ~+85% reference. A direct SPY-in-tracker simulation with the same friction model would let us compare dollar-for-dollar on identical assumptions. Small change to `engine/wheel_tracker.py`: add a `benchmark_ticker` kwarg that opens a position at `as_of=START_DATE` and marks it to market at every step.

- **Out-of-sample HMM/POT-GPD parameter re-fit.** The forward-distribution and regime classifier parameters were fit on data overlapping the backtest periods. A true OOS run would re-fit on pre-2018 data only and re-run S22/S27/S32/S34/S35/S38. Would either confirm or reveal in-sample bias in the dollar-alpha results.

### 6.3 Medium-priority (deferred follow-ons)

- **`ranking_key` kwarg** (S37 F5 follow-on). Add `ranking_key: str = "ev_dollars"` to `rank_candidates_by_ev` accepting `"ev_dollars" | "roc" | "hybrid"`. Backwards-compatible default. Single-PR.
- **`hmm_argmax_prob` diagnostic column** (S36 F5 follow-on). Surface the probability of the labeled HMM state alongside `hmm_regime` so a trader inspecting one row can see when a label is marginal (e.g., KO at Mar 2026 = crisis 0.544 vs normal 0.423).
- **Documentation: ranking philosophy.** New `docs/RANKING_PHILOSOPHY.md` explaining the ev_dollars-vs-roc trade-off for the trader-facing audience.

### 6.4 Long-deferred (physically blocked)

- **S6 — Theta provider with real chains.** Cross-verifies engine output against actual chain-quoted premiums vs the Bloomberg synthetic BSM premium. Blocked on Theta Terminal access (the user's laptop runs the puller; other agents are working on this — DO NOT collide with `engine/theta_*`, `data/theta_*`, `scripts/pull_theta_*`, `models/theta_*`).
- **Multi-contract / strategy-stack expansion.** S37's recommendation to deploy the idle cash buffer via additional strategies (timing-gated strangles, calendar-spread overlays, …). Would address the 77% idle cash in S38 but requires non-trivial product design.

### 6.5 Marketing / framing (operator-facing)

- **Reposition the engine as a conservative income strategy, not an alpha strategy.** +33% over 5 years = ~5.9% annualized + correct crisis refusal is a defensible value proposition. SPY-beating is not.
- **Any external pitch citing "+27pp over SPY" or "+11.6pp over SPY" must be qualified as window-specific.** The honest forward expectation in a bull-dominated 3-5 year window is engine underperformance of 20-50pp vs SPY.

---

## 7. AI handoff (procedural notes for future agents)

- **Branch hygiene:** All 10 PRs were squash-merged with branch deletion. Local divergent branches that were rebased mid-session (e.g., `claude/docs-f4-fix-a-attempt-failed`) were left untouched; they're harmless and will be reaped by future fetches.

- **Force-push is hard-blocked.** Per the Claude Code classifier with the user's "do not do anything irreversible" rule, `git push --force` and `git push --force-with-lease` are denied on branches with open PRs. Use `gh pr update-branch <PR#>` (non-destructive merge commit) or, on conflicts, local `git merge origin/main --no-edit` + `git commit --no-edit` + plain `git push`. See [memory note: gh-pr-update-branch-pattern](../.claude/projects/...). The classifier does NOT block plain `git merge`; only the force-push.

- **The CI unblocker pattern.** PR #215's `max_as_of_staleness_days=30` gate tripped pre-existing tests using `as_of="2026-03-15"` against synthetic OHLCV ending ~2021-05-13. If you see test failures with messages like `as_of YYYY-MM-DD is N days beyond latest data (YYYY-MM-DD); max_as_of_staleness_days=30`, the fix is to either:
  1. Bump `default_days` in the connector stub (1400 → 3000 covers up to 2027).
  2. Pass `max_as_of_staleness_days=99999` in the test call.

- **Three terminals, one repo.** Terminals A, B, C work in separate worktrees:
  - A: `C:\Users\merty\Desktop\swe-terminal-a`
  - B: `C:\Users\merty\Desktop\swe-terminal-b` (this session)
  - C: primary clone (or another worktree)
  Claim work on board [#113](https://github.com/MertYakar66/smart-wheel-engine/issues/113) before `git checkout -b`. Derive the terminal identifier from the worktree path, not the task type.

- **Theta Terminal access is a hard lock.** Other agents are pulling Theta data; DO NOT touch `engine/theta_*`, `data/theta_*`, `scripts/pull_theta_*`, `models/theta_*`, or hit `127.0.0.1:25503`. Verified safe surfaces for this session: `engine_api.py`, `engine/forward_distribution.py`, `engine/candidate_dossier.py` (read), `docs/`, `tests/`.

---

## 8. Method appendix

**Engine version under test:** `origin/main` @ `2da76ff` at session start (post-IV-PIT-fix). Final session HEAD `3b67055` after all merges.

**Data provider:** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector` (Cowork sandbox). Bloomberg CSVs only; no Theta access during this session.

**Backtests run from scratch this session:**
- S38 (this session, this report's primary contribution).
- Earlier ones (S22 / S27 / S32 / S34 / S35) were re-analyzed but not re-run — outputs were already in `%TEMP%` from prior sessions.

**Throwaway harness pattern:** all Sn drivers live under `%TEMP%\sN_backtest\` and are NOT committed (per the established Sn convention). Outputs (`rank_log.csv`, `closed_position.csv`, `summary.txt`) stay local; the committed deliverable is the analysis doc + ledger entry + manifest row.

**Tests added this session:**
- `tests/test_tv_dossier_d17_wire.py` (13 tests, PR #233)
- `tests/test_tv_nonce_register_lock.py` (6 tests, PR #219)
- 4 new tests in `tests/test_engine_api_port.py::TestEngineHTTPServerListenQueueDepth` (PR #216)

**Tests modified for CI unblock (PR #237):**
- `tests/test_event_gate_back_buffer.py` (bumped `default_days` 1400 → 3000)
- `tests/test_ranker_iv_pit.py` (two stubs, same bump)
- `tests/test_pit_leaks.py` (removed unused `import pandas as pd`)

**§2 floor: 93/93 launch-blocker tests passing.**
