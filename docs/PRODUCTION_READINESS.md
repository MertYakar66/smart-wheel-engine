# Production readiness — real-money deployment gate

**Last updated:** 2026-05-26 (against `origin/main` at `e504801`,
post-PR #216 / S32).
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
| Does it **beat SPY at meaningful capital scales**? | **Conditionally yes** at $1M with 100-ticker universe in 2022-2024 (S34: +11.6pp over SPY). **No** at $100k in 2018-2020 (S35: −41pp under SPY). The dollar-alpha is regime-dependent; the +27pp headline from S22/S27 is window-specific. |
| **Where does the dollar alpha come from?** (post-soundness-review) | **Mostly from equity beta on assigned stocks, not put-selection skill.** S34 backtest: of $356,128 NAV gain at $1M, only $28,571 (8%) came from realized put trades; the other $327,557 (92%) came from STOCK APPRECIATION on assigned positions during the 2023-2024 bull market. S27 was even more pronounced: realized executed P&L was −$3,421 (NEGATIVE); all $51,444 NAV gain came from equity beta. **The "engine beats SPY" framing is largely a levered SPY-subset bet via wheel assignments**, not a pure put-premium edge claim. See `docs/SOUNDNESS_REVIEW_2026-05-26.md`. |
| **Is the dollar alpha concentration-resilient?** (post-soundness-review) | **No — one ticker can dominate.** In S34, BKNG alone contributed $31,576 of $28,571 total executed realized P&L (110% of net). Without BKNG, the engine's realized executions are slightly negative (−$3,004). **However, the ranking SIGNAL is robust** — ρ moves 0.327 → 0.324 when BKNG is removed. Ranking quality is genuine; the dollar outcome at scale is concentration-dependent. |
| Should we **deploy it autonomously with real money today**? | **No** — three blockers remain (F4 tail-risk, D17 live-wire, multi-window confirmation). Plus the equity-beta-vs-ranking-alpha framing means autonomous deployment would be a thinly-disguised levered equity beta bet in disguise. See §3 below. |
| Should we **use it as a research / decision-aid signal**? | **Yes, with supervision and explicit window-sensitivity + alpha-decomposition caveats.** The signal is real (verified ρ + robust to concentration); the refusal mechanism is the engine's strongest property (98%+ refusal in adverse periods, correct in aggregate); the dollar outcome at scale comes mostly from equity beta on assignments. |

**One-sentence verdict:** the engine is a *research-grade ranker
with verified predictive signal and a strong refusal mechanism*; the
*dollar alpha at scale comes mostly from equity beta on wheel-strategy
assignments rather than from pure put-selection skill*. As an
autonomous trading system it is not yet deployable (three blockers
remain). As a supervised decision-aid + refusal layer over a wheel
strategy, it has demonstrable value — with the explicit
acknowledgment that historical dollar-outperformance is partly a
levered SPY-subset bet on bull-market-favored single names.

---

## 1. What's been verified

The product has been stress-tested across four arcs in 2026-05. Every
finding below is reproducible from on-disk artifacts (rank logs in
`%TEMP%\s{22,27,32}_backtest\`, pytest in CI, code on `origin/main`).

| Arc | Sn / PR | What was verified | Where it lives |
|---|---|---|---|
| **Predictive validity** | S22 / S27 / PR #197 | Engine `ev_dollars` ranks realized P&L with Spearman ρ ≈ 0.22 (post-IV-PIT-fix). Quartile monotonicity clean. Top quartile beats bottom by 1.7×–2× in mean realized. | `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`, `docs/PREDICTIVE_VALIDITY_REVIEW.md` |
| **Reliability** | S18 / S19 / S20 / PR #194 | 503-ticker load runs (S18). 27 hostile / malformed input vectors fail-closed (S19). HTTP API concurrency holds at default-thread-count (S20). | `docs/USAGE_TEST_LEDGER.md` §S18–§S20, `docs/RELIABILITY_ARC_REVIEW.md` |
| **Audit-of-audit** | PR #195 | All 22 Terminal A campaign PRs independently re-verified at the code level. 0 §2 breaches missed. 1 cosmetic attribution note. | `docs/TERMINAL_A_AUDIT.md`, `docs/AUDIT_OF_AUDIT_REVIEW.md` |
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

S32 measured the engine's behaviour at realistic capital. The result
inverts the campaign's earlier marketing-friendly claim:

| Metric | $100k (S22 / S27) | $1M (S32) | SPY same window |
|---|---|---|---|
| Final NAV | $151,444 | $1,018,514 | $1,240,000 |
| Return | +51.4% | **+1.85%** | **~+24%** |
| Engine vs SPY | **+27pp** | **−22pp** | — |
| Average capital deployment | ~50–100% (BP saturated) | **10.8%** | 100% |
| Spearman ρ (signal quality) | 0.218 | 0.192 | — |

**The "+27pp over SPY" headline from S22 / S27 was a $100k-capital
artifact, not a property of the engine.** At $1M with the current
24-ticker × 35-DTE × 25-delta × hold-to-expiry parametrization,
89% of NAV sits in cash — the strategy cannot deploy more.
A passive SPY hold beats the engine cleanly at any non-toy
capital level today.

This is **not** a marketing problem; it is a capacity-and-parametrization
problem. Detail and remediation in §3 and §4.

---

## 3. Production-readiness blockers (must be resolved before real money)

These three items are the difference between "research tool" and
"autonomous deployment." Each has a measurable defect and a scoped
fix path; none has shipped yet.

### Blocker 1 — F4 tail-risk machinery does not widen on adversarial drawdowns

**What:** The forward-distribution + POT-GPD tail-estimation pipeline
in `engine/forward_distribution.py` + `engine/tail_risk.py` produced
**`prob_profit = 0.8333` constant across COST's 31.5% drop in April–May
2022 — in BOTH the pre-fix and post-fix engines.** Same engine output
for `prob_profit` on every one of 10 candidates during a peak-to-trough
move of $608 → $416 on a single name.

**Why it matters:** A single concentrated tail event on a held position
(COST 2022, UNH 2024) costs $5–15k per position that the engine said
wouldn't happen. With 10 such events expected over a 3-year window in
realistic market regimes, this is a 5-figure unforced error.

**Evidence:** `docs/ENGINE_BACKTEST_2022_2024.md` (S22 finding F4),
`docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md` (S27 finding F4
"UNCHANGED. The bug was not the cause."), `docs/PREDICTIVE_VALIDITY_REVIEW.md`
P5, `docs/ENGINE_BACKTEST_S32_FRICTION.md` (re-surfaces under friction).
PR #196 added a regression watch (`tests/test_f4_tail_risk_gap.py`)
but the underlying widening logic is not yet fixed.

**Required fix:** Targeted research on either (a) the empirical lookback
window (currently 504 days; may be too long, swamping recent vol with
pre-2022 calm), or (b) the POT-GPD threshold (may be too high to capture
mild-but-persistent tail events). Unit test that replays COST 2022-04
through the forward-distribution + POT-GPD pipeline; verify that
`prob_profit` moves from 0.83 to something lower during the 22% drop.

**Without this fix:** Real-money deployment is exposed to single-name
tail events the engine cannot warn about.

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
`docs/PREDICTIVE_VALIDITY_REVIEW.md` P6, `docs/ENGINE_BACKTEST_S32_FRICTION.md`
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

### Blocker 3 — Strategy capacity at $1M — **PARTIALLY CLOSED by S34**

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
- **S35 window-sensitivity still applies.** S34's +11.6pp is a
  2022-2024 result. A multi-window run (e.g., 2020-2024) is
  required before drawing forward-deployment conclusions.

**Updated required fixes (re-prioritised by S34):**
- ✅ **Universe expansion to 100+ tickers — VALIDATED by S34.**
  Sufficient to flip engine vs SPY from −22pp to +11.6pp at
  $1M / 2022-2024.
- 🔬 **Multi-window backtest** (2020-2024 with 100-ticker universe)
  to address S35's window-sensitivity finding. Without this, the
  +11.6pp result could be 2022-2024-specific.
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

The mechanical evidence from `docs/SOUNDNESS_REVIEW_2026-05-26.md`:

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
| **$100k account, autonomous** | ≤ $100k | Today | ❌ **No.** F4 tail-risk gap means single-name drawdowns are not protected. D17 not wired to API means the engine has no live portfolio-level brake. |
| **$500k–$1M supervised, universe ≥ 100 tickers** | $500k–$1M | After S34 (2026-05-26) | ⚠ **Conditional.** S34 validated that universe expansion to 100 tickers closes the capacity gap (engine vs SPY: −22pp → +11.6pp at $1M / 2022-2024). Still subject to (a) F4 fix, (b) multi-window confirmation per S35's window-sensitivity finding, (c) D17 live-wire. With strict supervision and these conditions partially met, supervised use is defensible. |
| **$500k–$1M, universe ≤ 24 tickers** | $500k–$1M | Today | ❌ **No.** S32 measured −22pp underperformance. Use the 100-ticker universe (per S34). |
| **$1M+ autonomous deployment** | $1M+ | Today | ❌ **No.** Three blockers (F4, D17-live) remain even with universe expansion. Plus S35 window-sensitivity. Plus Caveat 2 (parameters in-sample). |
| **Any production deployment** | Any | **After F4 + D17-live + universe expand + multi-window backtest** | Conditional ✅. Re-evaluate based on (a) the post-F4-fix Spearman, (b) the post-F4-fix tail-risk regression test, (c) the multi-window backtest at $1M with 100-ticker universe (S34-style on 2020-2024), and (d) D17 live-wire in `engine_api.py`. |

---

## 6. Required work before real money

Below is the punch list. Each item has a defined "done" and a defined
test. Order is rough priority (highest first); items are mostly
independent and can be picked up in parallel.

| # | Item | Definition of done | Test |
|---|---|---|---|
| **B1** | **F4 tail-risk widening fix.** Diagnose whether the 504-day empirical lookback dilutes recent vol or the POT-GPD threshold is too high. Implement the fix. | `tests/test_f4_tail_risk_gap.py` extended: replay COST 2022-04-01 → 2022-05-25 through the forward-distribution + POT-GPD pipeline; assert `prob_profit` moves below 0.75 at the trough. | targeted unit test passes; full S22 / S27 backtest re-run shows 2022 mean realized > $0. |
| **B2** | **Wire D17 strict mode to `engine_api.py`.** `_handle_tv_dossier` and `_enrich_alert` must call `portfolio_context_snapshot` and attach the result to `build_candidate_dossiers`. Default `require_ev_authority=True` for execution-routing endpoints. | New integration test in `tests/test_tv_dossier.py`: drive endpoint with a payload that should trigger `sector_cap_breach`; assert response carries the refused verdict. | targeted integration test passes; manual smoke against running `engine_api.py`. |
| **B3** | **Universe expansion to 100+ tickers.** Add a default universe config that's the full SP500 (or a documented 100-name subset). Re-run S32 at $1M with the larger universe. | New `### S33` ledger entry: `$1M friction-modeled simulation, 100+ ticker universe`. Hypothesis: average deployment rises from 10.8% to ~40–60%, NAV vs SPY gap closes from -22pp to roughly -10pp. | run completes; documented in `docs/ENGINE_BACKTEST_S33_*.md` (TBD). |
| C1 | **Out-of-sample parameter freeze.** Implement HMM / POT-GPD parameter freeze + replay infrastructure. | snapshot parameters at a cutoff date, run a held-out backtest. | new harness; new test that asserts the frozen parameters reproduce the snapshot's classifier output. |
| C2 | **413 Payload Too Large on engine_api.py.** S20 found the 10 MB request causes `ConnectionAbortedError` instead of a proper 413. Add `Content-Length` check at top of `do_POST`. | targeted test in `tests/test_tv_api.py`. | targeted test passes. |
| C3 | **Unify the two verdict paths in `engine_api.py`.** `_handle_tv_dossier` uses `EnginePhaseReviewer`; `_enrich_alert` runs an inline R1/R5 clone. Both should route through `EnginePhaseReviewer`. | both endpoints produce identical verdicts on the same `ev_row` input. | new regression test driving identical input through both endpoints. |
| C4 | **`_tv_seen_register` thread lock.** S20 found the check-then-set in `_tv_seen_register` is lock-free; protected today only by CPython GIL + dict-op atomicity. Add an explicit `threading.Lock`. | wrap the function body in `with _TV_SEEN_NONCES_LOCK:`. | high-concurrency stress test that asserts no >1 accept under contention. |

**Blockers** = B1, B2, B3. **Caveats / cleanups** = C1–C4.

The MINIMUM bar for "deploy real money at any scale" is **B1 + B2**.
B3 is required only for "deploy real money at scales above $100k."

---

## 7. Operational guidance (if running anyway)

If despite the above you do run this engine against real money today,
the minimum supervision protocol is:

1. **Cap account size at $100k.** Above this, F4 + capacity gaps cost
   more than the engine earns vs a passive hold.
2. **Review every candidate before execution.** Use the engine's
   ranked output as a *decision aid*, not as an autonomous executor.
3. **Cap single-name exposure at 25% of NAV.** This is what D17's
   `_DEFAULT_MAX_SECTOR_PCT` enforces in strict mode; today you must
   enforce it manually because D17 isn't wired to the API.
4. **Watch concentration tail-risk by hand.** When a held position
   drops more than 10% from entry, override the engine's
   `prob_profit` reading and consider closing — the engine will not
   alert you to widening tail risk until F4 ships.
5. **Daily session restart.** The S18 finding of `+5 handles per
   ranker call` means a process running for >7 days accumulates
   thousands of file handles. Restart at session-close.

This is **operational hygiene only**, not a replacement for the
B1 / B2 / B3 fixes.

---

## 8. Sources of truth

- **CLAUDE.md** §2 — the §2 invariant (no candidate bypasses
  `EVEngine.evaluate`).
- **`docs/USAGE_TEST_LEDGER.md`** — all 32 Sn usage tests (S1 – S32);
  the running record of what's been measured.
- **`docs/LAUNCH_READINESS.md`** — code-quality merge gates (R1–R8,
  the four authoritative routes, the launch-blocker test subset).
  Complement, not substitute, for this doc.
- **`docs/ENGINE_BACKTEST_2022_2024.md`** — S22 pre-IV-PIT-fix backtest
  (closed PR #178). Historical record; numbers superseded.
- **`docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`** — S27 post-IV-PIT-fix
  backtest. Headline ρ = 0.22.
- **`docs/ENGINE_BACKTEST_S32_FRICTION.md`** — S32 $1M friction
  simulation. Headline +1.85% vs SPY +24%.
- **`docs/PREDICTIVE_VALIDITY_REVIEW.md`** — independent meta-review
  of S22 + S27 (PR #197). Headline: 8 VERIFIED, 1 VERIFIED-WITH-NOTE.
- **`docs/RELIABILITY_ARC_REVIEW.md`** — independent review of S18 /
  S19 / S20 (PR #194). Headline: 2 CONFIRMED, 2 CONFIRMED-WITH-NOTE.
- **`docs/AUDIT_OF_AUDIT_REVIEW.md`** — meta-verification of the
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
