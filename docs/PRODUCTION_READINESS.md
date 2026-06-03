# Production readiness — real-money deployment gate

**Last updated:** 2026-05-28 (post-S34/S38/S40 multi-window evidence,
post-PR #255 B2 closure, post-PR #260 F4 realized-vol-ratio widening
+ PR #262 R10 single-name cap = the F4 deployment bundle that closes
§3 B1, post-S41 (PR #267) + S44 (PR #271) honest scope-limits validation;
this revision amends §1 headline + §5 deployment matrix to honestly
reframe the engine-vs-SPY claim AND syncs §1/§3/§6 to reflect F4 fix
(PR #260) having shipped — complements PR #257's earlier §2/§3/§6
sync that predated PR #260 + #262).
**Owner:** anyone deploying this engine against a real brokerage
account is the owner of this doc.

This file is the single source of truth for the question:

> *Should we run this engine against real money today?*

It is the commercial-deployment companion to `docs/LAUNCH_READINESS.md`
(which is the code-quality gate for *merging*) and to `PROJECT_STATE.md`
(which is the structural temporal state). LAUNCH_READINESS answers
"is this PR safe to merge?"; this file answers "is this engine safe to
operate?".

---

## Headline status

| Question | Honest answer |
|---|---|
| Does the engine produce **realistic, EV-correct outputs**? | **Yes.** Spearman ρ ranges 0.19–0.50 across (capital × universe × window) configurations; the ranking quality is scale-invariant AND universe-invariant AND window-invariant. Statistically overwhelming across every measurement (p < 1e-48 in all cases). |
| Does it **survive operational stress** (load, chaos, concurrency)? | **Yes.** 2,374 of 2,378 tests pass; the 2 failing tests are documented Windows-local Theta-tier flakes, not engine defects. S18 / S19 / S20 reliability arc (PR #194) verified. |
| Is its **§2 invariant** intact ("no tradeable candidate bypasses EVEngine.evaluate")? | **Yes.** Verified across S18 load, S19 chaos, S20 concurrency, S22 / S27 / S32 / S34 / S35 backtests, and the audit-of-audit review (PR #195). |
| Does it **beat SPY at meaningful capital scales**? | **Window-dependent across the entire range.** Measured (capital × universe × window) engine-vs-SPY deltas span **−52pp (S38: $1M / 100t / 2020-2024) to +27pp (S22/S27: $100k / 24t / 2022-2024)**. S34's "+11.6pp at $1M/100t" was 2022-2024-window-specific; **the subsequent S38 multi-window result demonstrates window-specificity** — same universe / capital over the longer 2020-2024 window (which includes COVID + 2021 mega-bull + 2022 bear + 2023-2024 recovery) returned **−52pp**. **No single number represents the engine's forward edge.** |
| **Where does the dollar alpha come from?** (post-soundness-review) | **Mostly from equity beta on assigned stocks, not put-selection skill.** S34 backtest: of $356,128 NAV gain at $1M, only $28,571 (8%) came from realized put trades; the other $327,557 (92%) came from STOCK APPRECIATION on assigned positions during the 2023-2024 bull market. S27 was even more pronounced: realized executed P&L was −$3,421 (NEGATIVE); all $51,444 NAV gain came from equity beta. **S38 reinforced and intensified this pattern**: realized executed P&L over 305 puts + 168 CCs across 2020-2024 was **−$28,647** (also NEGATIVE); all NAV growth (+$331,764) came from equity-beta-on-assignments (108.6% attributable). **The "engine beats SPY" framing is largely a levered SPY-subset bet via wheel assignments**, not a pure put-premium edge claim. See `archive/2026-05/SOUNDNESS_REVIEW_2026-05-26.md` and `docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md` §"Alpha decomposition". |
| **Is the dollar alpha concentration-resilient?** (post-soundness-review) | **No — one ticker can dominate.** In S34, BKNG alone contributed $31,576 of $28,571 total executed realized P&L (110% of net). Without BKNG, the engine's realized executions are slightly negative (−$3,004). **However, the ranking SIGNAL is robust** — ρ moves 0.327 → 0.324 when BKNG is removed. Ranking quality is genuine; the dollar outcome at scale is concentration-dependent. |
| Should we **deploy it autonomously with real money today**? | **No** — but for reasons no longer dominated by an open B1. **B1 (F4 tail-risk widening) shipped 2026-05-27/28 as a deployment bundle**: PR #260 (realized-vol-ratio widening, the **frequency guard**) + PR #262 (R10 single-name cap, the **magnitude guard**). Honest scope-limits documented by S41 (PR #267, A) + S44 (PR #271, B): PR #260 alone is signal-preserving but not value-creating; PR #262 alone caps notional but doesn't widen distributions in elevated-vol regimes; **together they form the F4 defense-in-depth pair**. **D17 live-wire shipped 2026-05-26** (B2 closed, PR #233 + #255). **Multi-window confirmation also ran** (S38 PR #235; S40 PR #264 extended to 5 measurement points spanning −85pp to +10pp engine-vs-passive at $1M/100t) — the −52pp pattern is now established as a general property at $1M/100t scale, not 2020-2024-specific. **The remaining barrier to autonomous deployment is the structural finding itself**: engine systematically underperforms passive in bull-dominated multi-year windows due to limited deployment (15-23% NAV), regardless of F4 fix status. See §3 below. |
| Should we **use it as a research / decision-aid signal**? | **Yes, with supervision and explicit window-sensitivity + alpha-decomposition caveats.** The signal is real (verified ρ + robust to concentration); the refusal mechanism is the engine's strongest property (98%+ refusal in adverse periods, correct in aggregate); the dollar outcome at scale comes mostly from equity beta on assignments. |

**One-sentence verdict:** the engine is a *research-grade ranker
with verified predictive signal and a strong crisis-refusal mechanism*
(S38 measured 97.8% refusal during 2020-02-15 → 2020-05-15 COVID,
avoiding ~$215k of would-have-been losses on the refused candidates);
the *dollar alpha at scale is window-dependent across a wide range
(−85pp to +10pp engine-vs-passive at $1M/100t across 5 measured windows
per S40 PR #264) — there is no single forward estimate*. As an
autonomous trading system it is not yet deployable: **B1 (F4 fix) is
shipped 2026-05-27/28 via the #260 + #262 deployment bundle, B2
(D17 live-wire) shipped 2026-05-26 via #233 + #255, B3 (capacity) is
structurally closed via S34 — the deployment gates are mechanically
closed**, but the structural finding (S40 + S44) that engine
systematically underperforms passive in bull-dominated multi-year
windows due to limited deployment is not fixable by any engine-side
change. As a supervised decision-aid + crisis-refusal layer
over a wheel strategy, it has demonstrable value — but the honest
headline value proposition is **conservative income generation
(+33% over 5y in S38 ≈ +5.9% annualized) with strong crisis
refusal**, NOT SPY-beating dollar alpha.

---

## 1. What's been verified

The product has been stress-tested across four arcs in 2026-05. Every
finding below is reproducible from on-disk artifacts (rank logs in
`%TEMP%\s{22,27,32}_backtest\`, pytest in CI, code on `origin/main`).

| Arc | Sn / PR | What was verified | Where it lives |
|---|---|---|---|
| **Predictive validity** | S22 / S27 / PR #197 | Engine `ev_dollars` ranks realized P&L with Spearman ρ ≈ 0.22 (post-IV-PIT-fix). Quartile monotonicity clean. Top quartile beats bottom by 1.7×–2× in mean realized. | `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`, `archive/2026-05/PREDICTIVE_VALIDITY_REVIEW.md` |
| **Reliability** | S18 / S19 / S20 / PR #194 | 503-ticker load runs (S18). 27 hostile / malformed input vectors fail-closed (S19). HTTP API concurrency holds at default-thread-count (S20). | `docs/USAGE_TEST_LEDGER.md` §S18–§S20, `archive/2026-05/RELIABILITY_ARC_REVIEW.md` |
| **Audit-of-audit** | PR #195 | All 22 Terminal A campaign PRs independently re-verified at the code level. 0 §2 breaches missed. 1 cosmetic attribution note. | `archive/2026-05/TERMINAL_A_AUDIT.md`, `archive/2026-05/AUDIT_OF_AUDIT_REVIEW.md` |
| **Friction at scale** | S32 / PR #213 | Friction drag at $1M = **0.27% NAV** (much smaller than S22's "2-5% per leg" worst case). Signal preserved under friction (ρ moves 0.194 → 0.192). | `docs/ENGINE_BACKTEST_S32_FRICTION.md`, `docs/USAGE_TEST_LEDGER.md` §S32 |

What this validates:

- **Engine quality is real.** The ranker output is mechanically correct,
  EV-coherent, friction-robust, and statistically defensible.
- **Reliability is solid.** The engine handles malformed input, load,
  and concurrency cleanly.
- **The §2 invariant has been pinned multiple ways.** Defence-in-depth
  is structurally enforced.

---

## 2. What's **not** production-ready (the headline gap)

S32, S34, and S38 between them define the dollar-alpha story
across (capital × universe × window) configurations:

| Metric | $100k 24t 2022-2024 (S22/S27) | $1M 24t 2022-2024 (S32) | $1M 100t 2022-2024 (S34) | $1M 100t 2020-2024 (S38) |
|---|---|---|---|---|
| Final NAV | $151,444 | $1,018,514 | $1,356,128 | $1,331,764 |
| Return | +51.4% | +1.85% | **+35.6%** | +33.18% |
| SPY same window | ~+24% | ~+24% | ~+24% | ~+85% |
| Engine vs SPY | **+27pp** | **−22pp** | **+11.6pp** | **−52pp** |
| Capital deployment | ~50–100% | 10.8% | 22.1% | 22.6% |
| Spearman ρ | 0.218 | 0.192 | 0.327 | 0.358 |

Three findings from the matrix:

1. **The "+27pp over SPY" headline from S22/S27 was a $100k-capital
   artifact**, not a property of the engine. At $1M with the
   original 24-ticker universe, the engine UNDERPERFORMS SPY by 22pp.
2. **Universe expansion to 100 tickers (S34) flips the $1M result
   to +11.6pp over SPY** — the capacity gap is structurally closeable.
3. **But the +11.6pp is 2022-2024-window-specific.** S38 ran the
   same 100-ticker universe at $1M over 2020-2024 and the engine
   underperformed SPY by 52pp. The structural finding (universe
   expansion uncaps capacity) is robust; the dollar alpha is
   window-favored, not a property of the engine.
4. **Spearman ρ is window-INVARIANT and capital-INVARIANT.** Signal
   quality holds at 0.19–0.36 across every configuration; the
   ranker is structurally good. The dollar alpha is mostly equity
   beta on assigned stocks, as the §1 / §4 alpha decomposition
   makes explicit.

This is **not** a marketing problem; it is a capacity / window /
alpha-source problem. The engine's strongest demonstrable property
is the refusal mechanism + signal robustness, not dollar
outperformance of SPY at scale. Detail and remediation in §3 and §4.

---

## 3. Production-readiness blockers (must be resolved before real money)

These three items are the difference between "research tool" and
"autonomous deployment." **Status as of 2026-05-27:** B2 shipped
(PR #255), B3 structurally shipped via S34 (window-favored dollar
alpha caveat per S38), B1 rolled back after S27 ρ inversion (the
naive Fix B1+C attempt destroyed the broader signal — see
`docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10). The new R1+
single-name exposure cap (PR #256) is the orthogonal-by-design
damage-bounding response to B1's remaining gap.

### Blocker 1 — F4 tail-risk widening — **SHIPPED 2026-05-27/28 as the #260 + #262 deployment bundle**

**Status:** ✅ **Closed.** F4 widening shipped via PR #260 (realized-
vol-ratio widening). F4 damage-bounding shipped via PR #262
(R10 single-name notional cap). **Together they close B1**;
neither alone is sufficient. Honest scope-limits validated by
S41 (PR #267, Terminal A) + S44 (PR #271, Terminal B).

**What was the original problem:** The forward-distribution +
POT-GPD tail-estimation pipeline in `engine/forward_distribution.py`
produced **`prob_profit = 0.8333` constant across COST's 31.5% drop in
April–May 2022** — same engine output for every one of 10 candidates
during a peak-to-trough move of $608 → $416 on a single name. A
single concentrated tail event on a held position cost $5–15k per
position that the engine said wouldn't happen.

**Fix design (PR #260, the "frequency guard"):**
`realized_vol_widening_factor` in `engine/forward_distribution.py`
thresholds the ratio `rv30/rv252` at ≥ 1.30 and widens the empirical
forward-distribution sample by up to 1.15× when the ratio crosses
the threshold. Fires on ~12% of probed cells per S41's calibration
(23.0% in 2022 bear, 2.6% in 2023 recovery, 11.0% in 2024 mixed) —
concentrates impact in the regime years where tail risk is empirically
elevated. **Calm-regime no-op verified**: factor = 1.0000 on every
cell in S41's six calm-control probes (AAPL/MSFT 2023-2024) and on
the 5-ticker EV smoke (XOM/JPM/MSFT/UNH/AAPL at 2026-03-20).

**Fix design (PR #262, the "magnitude guard"):**
`check_single_name_cap` in `engine/portfolio_risk_gates.py` caps
per-underlying short-option notional at 10% NAV
(`_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10`). Wired both as
`EnginePhaseReviewer.R10` (soft-warn preview, downgrades
`proceed → review`) and `WheelTracker._evaluate_d17_hard_blocks`
(HARD refusal at `open_short_put` time when
`require_ev_authority=True`). Sits BENEATH R9 (25% sector cap) —
catches single-name concentration that a permissive sector cap can't
reach. Verified live in PR #268 (realism battery): a 14%-NAV AAPL
position with R9 still passing has R10 firing `single_name_breach`.

**Why both are required:**
- **PR #260 catches the SECOND-and-subsequent vol-cluster events**
  (e.g., UNH 2024-11 fires factor=1.0121 mildly). But it cannot
  catch first-event idiosyncratic drawdowns where pre-event
  `rv30/rv252 < 1.30` (the COST 2022-04 case had rv30/rv252 = 0.96
  throughout the unfolding event). The fix is structurally LAGGED
  by the 30-day RV window.
- **PR #262 bounds dollar damage on the first-event idiosyncratic
  cases** that #260 cannot catch. At $100k NAV the 10% cap bounds
  any single COST position at $10k notional; at $1M / 100t the cap
  rarely binds (S40 measured 0 BP-binding skips in W1/W2 and 126
  in W3) but provides a hard floor.
- Per S41 + S44, **PR #260 alone is signal-preserving but not value-
  creating** on the S27 backtest (ρ −3.3%, NAV −12.1%, executed −22%)
  nor on the S38 setup (ρ −1.0%, NAV +0.4%, executed +0.7%). The
  value proposition is the binary refusal of catastrophic single-
  trade losses *via the EV ranker's frequency guard*; the dollar
  damage bound is *via* R10.

**Tests:**
- `tests/test_f4_rv_widening.py` (18 tests, PR #260): unit reproduction
  of named F4 cases, calibration pins, sign- and mean-preserving
  properties, end-to-end ranker behaviour.
- `tests/test_portfolio_risk_gates.py::TestCheckSingleNameCap`
  (9 tests, PR #262): cap arithmetic + missing-data semantics.
- `tests/test_dossier_invariant.py::TestD17DossierR10SingleNameCap`
  (5 tests): R10-fires-when-R9-passes safety property.
- `tests/test_authority_hardening.py::test_d17_single_name_breach_via_injected_snapshot`:
  tracker hard-block integration.
- Live re-verification: `docs/REALISM_VERIFICATION_2026-05-28.md`
  (PR #268) confirmed all the above on `origin/main` @ `56d8e5c`
  with 8/8 verification surfaces green.

**Backtest evidence on F4 fix's impact:**
- S41 (S27 24t / $100k / 2022-2024, PR #267): F4 alone signal-
  preserving (ρ −3.3%, NAV −12.1%, executed −22%). PR #260's
  honest scope confirmed: does NOT close the named F4 cases.
- S44 (S38 100t / $1M / 2020-2024, PR #271): F4 has near-zero
  impact (ρ −1.0%, NAV +0.4%, executed +0.7%). Hypothesis from
  S40's AI-handoff (that F4 widening would close 5-10pp of the
  −52pp engine-vs-passive gap) is **FALSIFIED** — the gap is
  structural to limited deployment, not a missing widening
  mechanism.

### Blocker 2 — D17 hard-blocks are not wired to the live HTTP endpoint

**What:** `engine/portfolio_risk_gates.py` ships six pure-function gates
(sector cap, portfolio delta, Kelly per-trade NAV, parametric VaR,
stress drawdown, dealer regime). `WheelTracker._evaluate_d17_hard_blocks`
consumes them in strict mode (`require_ev_authority=True` +
`PortfolioContext` attached). PR #205 wired this on the *ranker* side
(`consume_ranker_row` + `portfolio_context_snapshot`). **But the HTTP
endpoint at `engine_api.py` does not yet call `consume_ranker_row` or
attach a `PortfolioContext` to the dossier path.** The live API
endpoint a real trader or dashboard hits today routes through `EVEngine.evaluate`
but not through the D17 surface.

**Why it matters:** S22 reported `+200/trade` mean executed; S27
reported `−72/trade` under the same backtest. The difference was
**accidental BP saturation refusing the worst trades** in S22 — an
accident that the post-IV-PIT-fix engine no longer reproduces. Without
D17 wired live, production has neither (a) the original accidental
BP protection nor (b) the explicit D17 protection. The worst trades
WILL fire.

**Evidence:** `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md` §F5,
`archive/2026-05/PREDICTIVE_VALIDITY_REVIEW.md` P6, `docs/ENGINE_BACKTEST_S32_FRICTION.md`
§F5. PR #205 added the helpers; the HTTP-endpoint hookup is open.

**Required fix:** Modify `engine_api.py._handle_tv_dossier` (and the
TradingView webhook `_enrich_alert` path) to:
1. Call `WheelTracker.portfolio_context_snapshot(...)` for the current
   account state.
2. Attach the resulting `PortfolioContext` to the `build_candidate_dossiers`
   call.
3. Default `require_ev_authority=True` for any execution-routing
   endpoint.

Regression test: drive the API with a payload that should trigger
`sector_cap_breach` and assert the endpoint returns the refused
verdict with `action="reject"` + `reason="sector_cap_breach"`.

**Without this fix:** D17 protection exists only in tests, not in the
production code path.

### Blocker 3 — Strategy capacity at $1M — **STRUCTURALLY CLOSED by S34; dollar alpha window-dependent per S38**

**What (original):** With 24 tickers × one-position-per-name × 35-DTE
× hold-to-expiry × `top_n=10` × `MAX_NEW_PER_DAY=3`, S32 measured the
strategy deploys only 10.8% of $1M starting capital. 89% of NAV sits
idle. S34 tested the natural fix (universe expansion to 100 tickers).

**S34 result (`docs/ENGINE_BACKTEST_S34_UNIVERSE.md`):**

| Run | Capital | Universe | Engine NAV | Engine vs SPY | Deployment |
|---|---|---|---|---|---|
| S27 | $100k | 24 | +51% | **+27pp** | ~50-100% (BP saturated) |
| S32 | $1M | 24 | +1.85% | **−22pp** | 10.8% |
| **S34** | **$1M** | **100** | **+35.6%** | **+11.6pp** | **22.1%** |

**The capacity gap is LARGELY closed by universe expansion alone.**
At $1M with 100 alphanumeric SP500 tickers, the engine beats SPY by
+11.6pp (vs S32's −22pp at 24 tickers). A 34pp swing on universe
size alone. Multi-contract and strategy-stack remain candidates for
further deployment but are NOT required for $1M-class trading.

**Why it still matters:**
- Deployment at 22.1% is still not full — ~78% of NAV idle at $1M
  even with 100 tickers. Further capacity gains via multi-contract
  or strategy stack would push deployment higher.
- Universe shape matters: S34's first-100-alphanumeric cut excludes
  COST (the F4 test case). Sector-balanced or SP100 cuts would
  produce different absolute numbers; the structural finding
  (capacity gap closable) is robust but the exact "+11.6pp" is
  universe-shape-specific.
- **S35 window-sensitivity still applies — and S38 measured the
  window cost.** S34's +11.6pp is a 2022-2024 result. S38
  (`docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md`) ran the same 100-ticker
  universe at $1M over 2020-2024 (5y) and found **the engine
  UNDERPERFORMED SPY by ~52pp** (engine +33.18% vs SPY ~+85%). The
  structural "universe expansion closes capacity gap" finding
  holds (deployment never hit a BP wall over 5y either), but
  **dollar alpha is window-dependent.** The engine's value at
  scale is *conservative income generation with strong refusal in
  crises*, not SPY-beating dollar alpha. Operators picking the
  S34 result for deployment expectations should explicitly
  disclaim the window dependence.

**Updated required fixes (re-prioritised by S34 + S38):**
- ✅ **Universe expansion to 100+ tickers — VALIDATED structurally
  by S34** (capacity-gap closure) **and challenged on dollar alpha
  by S38** (window-dependence). Engineering work is complete; the
  remaining call is a deployment-thesis one (income generation vs
  SPY-beating).
- 🔬 **Per-universe-shape audit.** S34's first-100-alphanumeric cut
  excluded COST and many notable post-`CNP` names. A sector-balanced
  or SP100-by-market-cap cut is the next universe-shape question if
  alpha extraction at scale becomes a deployment requirement.
- 📋 **Multi-contract per position** — still scoped (helps for
  $5M+ but not required at $1M).
- 📋 **Strategy stack** (wheel + strangle + covered-call) — still
  scoped, increases deployment further.

**Without this fix:** $1M+ deployments to this engine will
underperform a passive index hold. The engine is a $100k-class
strategy with current parametrization.

---

## 4. Soft caveats (not blockers, but worth knowing)

### Caveat 2 — in-sample parameters

The HMM regime thresholds, dealer-multiplier clamp, dropout
thresholds, forward-distribution method-selection logic, and POT-GPD
tail calibration were all tuned with full visibility into 2018–2026
data. **The 2022-2024 backtest window is in-sample for those
parameters**, even though per-day data is PIT-disciplined.

Real out-of-sample would require parameter-freeze-then-replay
infrastructure (snapshot the HMM / POT-GPD parameters as they would
have been on 2022-01-01 and replay) — not present today.

Practical implication: "ρ = 0.22 in 2022-2024" carries a soft
look-ahead bias on the *parameters*. Future periods may not see
the same signal magnitude.

### Caveat — Bloomberg snapshot is single-point

The Bloomberg `sp500_vol_iv_full.csv` is a single-snapshot IV surface.
The `_resolve_pit_atm_iv` helper (PR #179) correctly resolves PIT IV
via `get_iv_history(end_date=as_of)`, but that history covers a
limited window. For dates far outside the snapshot vintage, the IV
may be unreliable. Verified on UNH 2024-01-15: snapshot IV 0.4323 vs
PIT IV 0.1712 — 60% relative correction. The PIT path is the correct
one; the snapshot path is the documented fallback.

### Caveat — no skew on Bloomberg connector

S29 documented that 100% of 1.35M IV rows in `sp500_vol_iv_full.csv`
have `put_iv == call_iv` exactly. Bloomberg connector cannot supply
skew-aware IV. Theta connector can, but Theta-tier-data access on
the dev box is limited (see the 2 Theta-Windows-local flake tests).
Production deployments needing skew-aware IV must run with
`SWE_DATA_PROVIDER=theta` and a working Theta Terminal subscription.

### Caveat — alpha decomposition (added 2026-05-26 post-soundness-review)

The engine's NAV-level outperformance against SPY in S22 / S27 / S34
comes primarily from **equity beta on assigned stocks**, not from
put-selection skill per se. The wheel strategy by design captures
equity beta via assignments: when a put is assigned, the seller takes
delivery of the stock at the strike; that stock then participates in
subsequent equity moves.

The mechanical evidence from `archive/2026-05/SOUNDNESS_REVIEW_2026-05-26.md`:

| Backtest | NAV gain | Realized P&L from executed | Equity-beta residual |
|---|---|---|---|
| S27 ($100k 24t 2022-2024) | +$51,444 | **−$3,421** (NEGATIVE) | +$54,865 |
| S34 ($1M 100t 2022-2024) | +$356,128 | +$28,571 | **+$327,557 (92% of gain)** |
| S35 ($100k 24t 2018-2020) | +$3,566 | −$48,326 (NEGATIVE) | +$51,892 |

**Implication:**
- The engine's value as a *premium-selling alpha engine* is small per
  trade (the ranker is statistically robust but each trade earns only
  a modest expected dollar amount above breakeven).
- The engine's value as a *wheel-strategy executor with strong
  refusal* is substantial: refuses 97-99.8% of candidates in adverse
  regimes, correctly identifies which stocks to take assignment on
  (those stocks then appreciate during bull markets).
- **In a regime where equity beta on the engine's preferred
  assignments does NOT outperform SPY** (e.g., 2018-2020 per S35, or
  a future regime where wheel-favored single names lag), the engine
  would likely underperform SPY by 10-40pp.

This is **honest framing**, not a defect. The wheel strategy is
*designed* to capture equity beta. The earlier session docs framed
the "beat SPY" headline as ranker-driven; this caveat clarifies it
is mostly assignment-driven equity exposure with a smaller
ranker-driven edge on top.

### Caveat — single-ticker concentration risk (added 2026-05-26 post-soundness-review)

In S34, **BKNG alone contributed $31,576 of $28,571 net executed
realized P&L (110% of net)**. Without BKNG, the engine's realized
executions across 268 other trades summed to **−$3,004**.

Mechanism: BKNG's stock rose from ~$2,336 (Jan 2022) to ~$5,000+
(end of 2024). High-priced tickers carry proportionally larger
premiums (premium scales with strike); the engine systematically
ranked BKNG high; BKNG kept the puts OTM as it rose; each trade
captured $3-4k of premium income.

The engine's **within-ticker ρ on BKNG was modest (0.156)** — this
wasn't unusual skill at timing BKNG entries; it was systematic
exposure to a single-name uptrend.

**The signal IS robust to BKNG removal** (full-set ρ = 0.3273
→ ex-BKNG ρ = 0.3244, delta 0.0029). The ranking quality doesn't
depend on the concentration. But the **dollar outcome is
concentration-dependent**: a deployment that doesn't see a BKNG-style
single-name bull run would not reproduce the S34 dollar result.

**Implication:** any dollar-alpha forecast from the engine's
historical backtests should be discounted for concentration risk.
The +11.6pp S34 result is not a "minimum" or "expected" — it's a
specific realization that depended materially on a single name's
trajectory.

---

## 5. Deployment decision matrix

| Use case | Capital | Engine state | Verdict |
|---|---|---|---|
| **Research signal / paper-trading the ranker** | Any | Today | ✅ **Go.** Signal is real (ρ ≈ 0.22); pair it with human review on every candidate. |
| **$100k account, supervised** | ≤ $100k | Today | ⚠ **Conditional.** The BP-saturation pattern accidentally limits damage; this is exactly the scale where S22 / S27 reported the +27pp-over-SPY result. Acknowledge F4 tail risk and review every entry; supervise rolls. |
| **$100k account, autonomous** | ≤ $100k | Today | ❌ **No.** Engine-side mitigations are now in place (F4: #260 RV-ratio widening + #262 R10 single-name cap = B1 deployment bundle; D17 wired to `engine_api.py` via #233 + #255 = B2 closed), but autonomous mode at this scale still falls foul of the **structural finding**: named-case events like COST 2022-04 stay below the `rv30/rv252 ≥ 1.30` widening trigger throughout the drawdown, so `prob_profit` can stay at 0.83+ through a 30%+ realised drop. The empirical-distribution method has no self-correction surface for this class of mis-calibration; supervised mode is what catches it. |
| **$500k–$1M supervised, universe ≥ 100 tickers** | $500k–$1M | After S34 + S38 (2026-05-26) | ⚠ **Conditional with explicit underperformance acknowledgment.** S34 (PR #226) showed +11.6pp over SPY at $1M / 100t / **2022-2024** — that result was **window-specific**. **Subsequent S38 (PR #235) ran the same universe / capital over the longer 2020-2024 window and returned −52pp** (engine +33.18% vs SPY ~+85%). Honest forward expectation at $1M scale spans **−52pp to +11.6pp** across measured multi-year windows. D17 live-wire shipped 2026-05-26 (B2 closed, PR #233 + #255); **F4 fix shipped 2026-05-27/28 as the #260 + #262 bundle (B1 closed)**. Defensible **only with strict supervision and the explicit understanding that the engine is a conservative income strategy with crisis refusal — not a SPY-beating alpha strategy**. S44 (PR #271) verified F4 fix is signal-preserving but does NOT close the engine-vs-passive gap at this scale. The +33% / 5y in S38 ≈ 5.9% annualized is a defensible income-tier value proposition; the "+11.6pp over SPY" framing is not. |
| **$500k–$1M, universe ≤ 24 tickers** | $500k–$1M | Today | ❌ **No.** S32 measured −22pp underperformance. Use the 100-ticker universe (per S34), with the S38 caveat above. |
| **$1M+ autonomous deployment** | $1M+ | Today | ❌ **No.** **F4 fix shipped via PR #260 + #262 bundle (B1 closed)**, but S44 (PR #271) confirmed F4 fix has near-zero impact at $1M/100t scale (ρ −1.0%, NAV +0.4%) — the −52pp gap was never F4-bound; it's structural to limited deployment. The multi-window evidence (S38 + S40 5 measurement points spanning −85pp to +10pp) shows engine systematically underperforms passive in bull-dominated 3-5y windows at $1M scale (engine +33% vs SPY ~+85% over 2020-2024). Autonomous deployment under these conditions runs a strategy that deploys ~15-23% of capital, produces **near-zero realized put P&L over 5y** (S38 pre-F4: −$28,647; S44 post-F4: −$32,729), and relies entirely on equity-beta-on-assignments for NAV growth. Plus Caveat 2 (parameters in-sample). |
| **Any production deployment** | Any | **All three named blockers shipped 2026-05-26/27/28** | Conditional ⚠ with structural underperformance acknowledgment. **B1 F4 fix shipped via #260 + #262 bundle** (frequency + magnitude guards; S41 + S44 validated honest scope-limits); **B2 D17 live-wire shipped** via #233 + #255; **B3 capacity structurally shipped** via S34 (100-ticker universe at $1M). **The post-F4 re-run question is answered** (S44 PR #271): F4 fix has near-zero impact on the −52pp engine-vs-passive gap (NAV +0.4%); the gap is structural to limited deployment, not F4-bound. **The rolling multi-window question is answered** (S40 PR #264 + C's S43 PR #270): engine-vs-passive at $1M/100t spans −85pp to +10pp across 5 measured multi-year windows; the −52pp pattern generalises. **The remaining barrier to autonomous deployment is the structural finding itself** — the engine systematically underperforms passive in bull-dominated multi-year windows because the wheel strategy deploys only 15-23% of NAV; no engine-side change addresses this. Deployment decisions must explicitly acknowledge the honest value proposition: **conservative income (+5-10% annualized depending on regime mix) with strong crisis refusal**, not bull-market alpha. |

---

## 6. Required work before real money

Below is the punch list. Each item has a defined "done" and a defined
test. Order is rough priority (highest first); items are mostly
independent and can be picked up in parallel.

| # | Item | Definition of done | Test |
|---|---|---|---|
| **B1** | **F4 tail-risk widening fix.** **(Shipped 2026-05-27/28 as #260 + #262 bundle.)** PR #260 (`realized_vol_widening_factor` in `engine/forward_distribution.py`) provides the frequency guard: thresholds `rv30/rv252 ≥ 1.30` and widens the empirical sample up to 1.15× when the regime is vol-clustered. Fires on ~12% of probed cells (S41 calibration). PR #262 (`check_single_name_cap`, R10) provides the magnitude guard: caps per-underlying short-option notional at 10% NAV. **Honest scope-limits validated**: PR #260 does NOT close the named F4 cases (COST 2022-04 had rv30/rv252=0.96 throughout — the fix is structurally lagged by the 30-day RV window); R10 catches the dollar damage on first-event idiosyncratic cases via notional capping. Backtest impact: S41 reported PR #260 alone signal-preserving (ρ −3.3%, NAV −12.1%); S44 reported near-zero impact at $1M/100t scale (ρ −1.0%, NAV +0.4%) — confirming the gap is structural to limited deployment, not F4-bound. | `tests/test_f4_rv_widening.py` (18 tests) for PR #260; `tests/test_portfolio_risk_gates.py::TestCheckSingleNameCap` (9 tests) + `tests/test_dossier_invariant.py::TestD17DossierR10SingleNameCap` (5 tests) for PR #262. Live re-verified in `docs/REALISM_VERIFICATION_2026-05-28.md` (PR #268). | ✅ shipped + tests pass + live verification green. |
| **B2** | **Wire D17 strict mode to `engine_api.py`.** **(Shipped — 2026-05-27.)** `_handle_tv_dossier` already wired D17 via query params (`nav`/`holdings`/`puts_held`/`regime_map` → `_build_portfolio_context_from_params` → `build_candidate_dossiers`). PR `claude/wire-d17-engine-api` closes the remaining surfaces: (1) adds **R9 sector_cap_breach** to `EnginePhaseReviewer` as the documented soft-warn preview of the tracker's hard refusal, (2) wires the same D17 portfolio-context engagement into `_enrich_alert` / `/api/tv/enrich` (the pull-enrichment surface; the POST webhook does NOT accept these params since Pine cannot know the operator's book). The documented sector_cap_breach integration test in `tests/test_tv_dossier.py` is now live. | New integration test in `tests/test_tv_dossier.py`: drive endpoint with a payload that should trigger `sector_cap_breach`; assert response carries the refused verdict. | targeted integration test passes; manual smoke against running `engine_api.py`. |
| **B3** | **Universe expansion to 100+ tickers.** **(Largely shipped — 2026-05-26.)** S34 (`docs/ENGINE_BACKTEST_S34_UNIVERSE.md`) validated the universe-expansion hypothesis at $1M with 100 first-alphanumeric SP500 tickers: engine returned **+35.6% NAV over 2022-2024 vs SPY ~+24% = +11.6pp**, vs S32's −22pp at 24 tickers. Average deployment **22.1%** at $1M (vs S32's 10.8%; original hypothesis was 40–60%, exceeded only the lower bound). The reproducer + snapshot live at `backtests/regression/s34_universe_100t_1m.py` + `.json`. **Remaining caveat:** S38 (`docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md`) showed the +11.6pp is 2022-2024-window-specific — over 2020-2024 the engine returns +33.18% vs SPY ~+85% = **−52pp** at the same universe / capital. The structural finding ("universe expansion closes capacity gap") is robust; the +11.6pp dollar alpha is window-favored. A 2020-2024 100-ticker run is the documented next step in §3. | S34 reproducer pinned via `tests/test_backtest_regression.py::test_backtest_matches_snapshot[s34_universe_100t_1m]`. | ✅ S34 ran + documented; S38 multi-window caveat documented. Full closure requires deploying decision on 2022-2024-window claim vs S38's longer-window result. |
| C1 | **Out-of-sample parameter freeze.** Implement HMM / POT-GPD parameter freeze + replay infrastructure. | snapshot parameters at a cutoff date, run a held-out backtest. | new harness; new test that asserts the frozen parameters reproduce the snapshot's classifier output. |
| C2 | **413 Payload Too Large on engine_api.py.** S20 found the 10 MB request causes `ConnectionAbortedError` instead of a proper 413. Add `Content-Length` check at top of `do_POST`. | targeted test in `tests/test_tv_api.py`. | targeted test passes. |
| C3 | **Unify the two verdict paths in `engine_api.py`.** `_handle_tv_dossier` uses `EnginePhaseReviewer`; `_enrich_alert` runs an inline R1/R5 clone. Both should route through `EnginePhaseReviewer`. | both endpoints produce identical verdicts on the same `ev_row` input. | new regression test driving identical input through both endpoints. |
| C4 | **`_tv_seen_register` thread lock.** S20 found the check-then-set in `_tv_seen_register` is lock-free; protected today only by CPython GIL + dict-op atomicity. Add an explicit `threading.Lock`. | wrap the function body in `with _TV_SEEN_NONCES_LOCK:`. | high-concurrency stress test that asserts no >1 accept under contention. |
| **R1+** | **Single-name (per-underlying) exposure cap.** **(Shipped — 2026-05-27.)** F4 damage-bounding follow-up to B1's rollback. Caps per-symbol short-option notional at `max_single_name_pct × NAV` (default 10%). Wired both as `EnginePhaseReviewer.R10` (soft-warn preview) and `WheelTracker._evaluate_d17_hard_blocks` (HARD refusal at `open_short_put` time when `require_ev_authority=True`). Bounds the F4 idiosyncratic-drawdown damage that the rolled-back widening fix couldn't address. See `engine.portfolio_risk_gates.check_single_name_cap` + DECISIONS.md D17 R10. | `tests/test_portfolio_risk_gates.py::TestCheckSingleNameCap` (9 unit tests), `tests/test_dossier_invariant.py::TestD17DossierR10SingleNameCap` (5 dossier tests, including the R10-fires-when-R9-passes safety property), `tests/test_authority_hardening.py::TestD17HardBlocks::test_d17_single_name_breach_via_injected_snapshot`, `tests/test_ev_authority_log_schema.py` extended with `_SHAPE_REJECT_SINGLE_NAME`. | shipped + tests pass. |

**Blockers** = B1, B2, B3. **Caveats / cleanups** = C1–C4. **Hardening** = R1+.

**Status snapshot (2026-05-28):**
- **B1** — F4 tail-risk fix: ✅ **shipped as the #260 + #262 deployment
  bundle.** PR #260 = realized-vol-ratio widening (frequency guard,
  fires on ~12% of cells per S41 calibration). PR #262 = R10 single-
  name cap (magnitude guard, 10% NAV per underlying). Together they
  close B1 because neither alone is sufficient: #260 does NOT close
  the named F4 cases (COST 2022-04 had rv30/rv252 < 1.30 throughout)
  and #262 doesn't widen distributions in elevated-vol regimes. Honest
  scope-limits + S41/S44 backtest evidence in `docs/ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md`
  (PR #267) + `docs/ENGINE_BACKTEST_S44_S38_POSTF4_RERUN.md` (PR #271).
  Note: the Fix B1+C attempt on PR #253 was rolled back after S27
  Spearman ρ inversion (static-multiplier widening based on HMM
  `crisis` label cannot satisfy both "close named F4 cases" +
  "preserve calm-period ρ"); PR #260 replaced it with the
  realized-vol-ratio approach that satisfies both.
- **B2** — D17 wire on `engine_api.py`: ✅ shipped (PR #233 dossier
  wire + PR #255 R9 sector_cap added to `EnginePhaseReviewer` + D17
  wired to `/api/tv/enrich` mirroring the dossier path).
- **B3** — Universe expansion: ✅ **structurally shipped** via S34
  (100-ticker universe + reproducer/snapshot/test). Window-favoured
  dollar alpha caveat per S38 remains; S40 extended to 5 measurement
  points spanning −85pp to +10pp engine-vs-passive across multi-year
  windows.
- **R1+ (R10)** — Single-name exposure cap (F4 damage-bounding pair
  to #260): ✅ shipped (PR #262 — R10 + tracker hard-block on
  `check_single_name_cap`). Part of the B1 closure bundle.

The MINIMUM bar for "deploy real money at any scale" is **B1 + B2**.
**Both shipped 2026-05-26/27/28** (B1 = #260 + #262 bundle; B2 = #233 + #255).
B3 is required only for "deploy real money at scales above $100k" —
also structurally shipped via S34. **The deployment gates are
mechanically closed.** The remaining barrier to real-money deployment
is the **structural finding** (S40 + S44) that engine vs passive at
$1M/100t scale is window-dependent across −85pp to +10pp due to
limited deployment (15-23% NAV) — fixing F4 / D17 / capacity does
not change this. Deployment decisions must explicitly acknowledge
the engine's **honest value proposition: conservative income +
crisis refusal, not bull-market alpha**.
**Current state: B1 mitigated — #260 + #262 deployment bundle shipped
(RV-ratio widening + R10 single-name notional cap), residual structural
limitation documented; B2 shipped; B3 structurally shipped (window
caveat).** The three named blocker-gates are mechanically in place.
**Named-case closure for B1 is structurally impossible at the engine
layer** (e.g. COST 2022-04 had `rv30/rv252 < 1.30` throughout, so the
widening fix never fires; the empirical-distribution method cannot
self-correct top-bin over-confidence — see
`docs/PROB_PROFIT_CALIBRATION_2026-05-28.md`), and the structural
finding (S38 / S40 / S44 evidence) that engine vs passive at
$1M/100t spans −85pp to +10pp across multi-year windows due to
limited deployment (15-23% NAV) is not a blocker the engine layer
can remove. **Deployment is now a value-proposition decision** —
"conservative income + crisis refusal at this scale, not bull-market
alpha" — **not a blocker-removal decision.**

---

## 7. Operational guidance (if running anyway)

If despite the above you do run this engine against real money today,
the minimum supervision protocol is:

1. **Cap account size at $100k.** Above this, F4 + capacity gaps cost
   more than the engine earns vs a passive hold.
2. **Review every candidate before execution.** Use the engine's
   ranked output as a *decision aid*, not as an autonomous executor.
3. **Cap single-name exposure at 25% of NAV.** This is what D17's
   `_DEFAULT_MAX_SECTOR_PCT` enforces in strict mode — wired to
   `engine_api.py` via PR #233 + #255; PR #262's R10 single-name cap
   also hard-blocks at 10% NAV per underlying at `open_short_put`
   time when `require_ev_authority=True`. Supervised mode still
   benefits from a visual sanity check on each candidate's
   `verdict_reason` — the gates are now mechanical, but the engine
   has no way to flag *why* a name became concentrated.
4. **Watch concentration tail-risk by hand.** When a held position
   drops more than 10% from entry, override the engine's
   `prob_profit` reading and consider closing. F4 widening (PR #260)
   is live but only fires on the ~12% of cells where
   `rv30/rv252 ≥ 1.30`; named-case events like COST 2022-04 stay
   below that threshold throughout the drawdown so the empirical
   distribution does NOT widen. The engine's `prob_profit` can stay
   high (0.83+) through a 30%+ realised drop; human supervision is
   the only check that catches this class of mis-calibration.
5. **Daily session restart.** The S18 finding of `+5 handles per
   ranker call` means a process running for >7 days accumulates
   thousands of file handles. Restart at session-close.

This is **operational hygiene only**, not a replacement for the
B1 / B2 / B3 fixes.

---

## 8. Sources of truth

- **CLAUDE.md** §2 — the §2 invariant (no candidate bypasses
  `EVEngine.evaluate`).
- **`docs/worklog/INDEX.md`** — the running record of every Sn usage
  test / backtest / verification (per-task fragments under
  `docs/worklog/`; the pre-2026-05-29 monolith `docs/USAGE_TEST_LEDGER.md`
  is now frozen and covers only S1–S32). The worklog ledger is the
  source of truth for the current Sn high-water — do not hardcode a
  count here; it drifts every session.
- **`docs/LAUNCH_READINESS.md`** — code-quality merge gates (R1–R11,
  the four authoritative routes, the launch-blocker test subset).
  Complement, not substitute, for this doc.
- **`docs/ENGINE_BACKTEST_2022_2024.md`** — S22 pre-IV-PIT-fix backtest
  (closed PR #178). Historical record; numbers superseded.
- **`docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`** — S27 post-IV-PIT-fix
  backtest. Headline ρ = 0.22.
- **`docs/ENGINE_BACKTEST_S32_FRICTION.md`** — S32 $1M friction
  simulation. Headline +1.85% vs SPY +24%.
- **`docs/ENGINE_BACKTEST_S34_UNIVERSE.md`** — S34 universe expansion
  (24 → 100 tickers) at $1M / 2022-2024 (PR #226). Headline +11.6pp
  over SPY — **window-specific; see S38 for the multi-window correction**.
- **`docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md`** — S35 out-of-window
  cross-validation 2018-2020 (PR #224). Headline −41pp under SPY at
  $100k / 24t; demonstrated dollar-alpha is window-dependent.
- **`docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md`** — S38 multi-window
  backtest at $1M / 100t / 2020-2024 (PR #235). Headline **−52pp under
  SPY**; falsifies the S34-only "+11.6pp" alpha claim and provides the
  multi-window evidence cited in §1 and §5 of this doc.
- **`archive/2026-05/SESSION_REPORT_2026-05-26.md`** — full campaign ledger for
  the deployment-readiness verification session that produced S34 /
  S35 / S38 + D17 live-wire (B2) + engine-API hardening.
- **`archive/2026-05/LAUNCH_READINESS_ANALYSIS_2026-05-26.md`** — parallel
  launch-readiness verdict doc (snapshot at SHA `c07b265`, amended
  2026-05-27 post-S38 to keep matrix in sync with this file).
- **`archive/2026-05/PREDICTIVE_VALIDITY_REVIEW.md`** — independent meta-review
  of S22 + S27 (PR #197). Headline: 8 VERIFIED, 1 VERIFIED-WITH-NOTE.
- **`archive/2026-05/RELIABILITY_ARC_REVIEW.md`** — independent review of S18 /
  S19 / S20 (PR #194). Headline: 2 CONFIRMED, 2 CONFIRMED-WITH-NOTE.
- **`archive/2026-05/AUDIT_OF_AUDIT_REVIEW.md`** — meta-verification of the
  Terminal A campaign audit (PR #195). 6 VERIFIED, 1 VERIFIED-WITH-NOTE.
- **GitHub issue #113** — coordination board; per-PR claims with
  branch + files + verdicts.
- **`PROJECT_STATE.md`** §1 — the four authoritative routes from
  inputs to a tradeable verdict.

---

## 9. When this file changes

Update this file when:

- A blocker (B1 / B2 / B3) ships → mark it done, update the deployment
  matrix.
- A new backtest produces a result that materially changes the
  deployment matrix (e.g., a $1M run with 100+ tickers).
- A new defect is discovered that crosses the "could lose real money"
  threshold (add as a new blocker or caveat).
- The brokerage connector / live execution surface changes.

This file is the **single doc** to read before any decision about
running the engine against a real account. If it's stale, that
itself is the blocker — pause deployment decisions until it's
refreshed.
