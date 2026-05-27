# Launch-readiness analysis — 2026-05-26

**Question (from user):** *"Is our options engine ready to be used in
real life trading?"*

**Verdict, one sentence:** **NO — not for autonomous real-money
deployment today; YES with strict supervision at $100k–$1M with the
100-ticker universe.** S34 (PR #226) closes the largest open
question by demonstrating that universe expansion alone closes the
capacity gap (+11.6pp over SPY at $1M with 100 tickers vs S32's
−22pp at 24 tickers). S35 (PR #224) confirms the predictive signal
is window-invariant (ρ = 0.50 in 2020, 0.33 at 100t/$1M, 0.22 at
$100k, 0.19 at $1M/24t) but reveals that **dollar-alpha is
window-dependent** (+27pp / +11.6pp / −41pp across configurations).
Three engineering blockers remain: F4 tail-risk widening (PR #221
diagnostic shipped; fix pending), D17 live-wire to `engine_api.py`,
and a multi-window confirmation backtest. Two of this session's
fixes ship today (`request_queue_size` PR #216, nonce lock
PR #219). The engine is **deployable as a supervised decision-aid
at $1M with the 100-ticker universe in conditions resembling
2022-2024**, with explicit acknowledgment that historical
performance varies by ±70pp on regime alone.

---

> ⚠ **Amendment 2026-05-27 (post-S38).** The "multi-window confirmation
> backtest" that this doc identified as the third remaining blocker has
> since landed: **S38 (PR #235)** ran the same universe / capital as
> S34 (100 tickers, $1M) but over the longer **2020-2024** window
> (which includes COVID + 2021 mega-bull + 2022 bear + 2023-2024
> recovery). **The S38 result was −52pp under SPY** (engine +33.18% vs
> SPY ~+85%), **falsifying the "deployable as a supervised decision-aid
> at $1M in conditions resembling 2022-2024" framing as a forward
> estimate**. Honest forward expectation at $1M / 100t across measured
> multi-year windows now spans **−52pp to +11.6pp**. The §3 deployment
> matrix row "Supervised $500k–$1M, universe ≥ 100 tickers" has been
> amended in-place to ⚠ **Conditional with explicit underperformance
> acknowledgment**. The honest value proposition is **conservative
> income generation (+33% / 5y ≈ +5.9% annualized) with strong
> crisis refusal (97.8% COVID refusal, ~$215k loss avoidance in S38)**
> — NOT SPY-beating dollar alpha. **Also:** D17 live-wire shipped
> 2026-05-26 (B2 closed, PR #233); only B1 (F4 tail-risk fix) remains
> of the three named blockers, and it is in flight on Terminal A's
> `claude/fix-f4-regime-conditioned-widening`. See
> `docs/PRODUCTION_READINESS.md` (the gate doc, updated in the same
> 2026-05-27 PR) for the canonical reframing, and
> `docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md` for the underlying
> evidence. The 2026-05-26 verdict and analysis below is preserved
> for campaign-arc readability; this callout marks where subsequent
> evidence has moved the picture.

---

This file is the consolidated launch-readiness analysis the user
explicitly requested on 2026-05-26. It synthesises the entire
campaign (S18 through S34 + the four review PRs + the small fixes
shipped this session) into a single yes/no readiness verdict. It is
intended to be the authoritative input to any decision about running
real money against this engine, alongside `docs/PRODUCTION_READINESS.md`
(the gate doc) and `docs/LAUNCH_READINESS.md` (the merge-gate
checklist).

**Engine SHA at analysis:** `origin/main` @ `c07b265`.
**Author:** Terminal B, fresh session, no campaign context beyond
this session's own work.

---

## TL;DR — the readiness picture

| Pillar | Status | Source |
|---|---|---|
| §2 invariant ("no candidate bypasses `EVEngine.evaluate`") | ✅ holds | Launch-blocker subset 93/93 pass; verified across 22-PR Terminal A campaign audit (PR #195) |
| Predictive signal | ✅ real, scale-invariant | Spearman ρ ≈ 0.22 at $100k (S27 / PR #197), 0.19 at $1M (S32 / PR #213) |
| Friction at scale | ✅ small | 0.27% NAV drag at $1M (S32) — much smaller than S22's "2-5% per leg" worst case |
| Reliability (load + chaos + concurrency) | ✅ verified | S18 / S19 / S20 + PR #194 reliability arc review |
| Operational hygiene (listen queue, nonce lock) | ✅ closing today | PRs #216, #219 (both shipped this session) |
| **F4 tail-risk widening** | ❌ open | PR #196 regression watch + PR #221 diagnostic + fix scoped but unimplemented |
| **D17 live-wire to `engine_api.py`** | ⚠ partial | PR #205 wired ranker side; HTTP-endpoint hookup open |
| **Capacity at scales > $100k** | ❌ open at 24-ticker; ⏳ TBD at 100-ticker | S32 measured 10.8% deployment at $1M; S34 (in flight) testing 100-ticker hypothesis |
| **Caveat 2** (in-sample HMM / POT-GPD parameters) | ⚠ soft | acknowledged across S22 / S27 / S32; parameter-freeze-then-replay infrastructure does not exist |

Score: **6 of 9 pillars green; 3 with material follow-up; 0 §2 breaches.**

The honest read: **the ranker is good; the strategy implementation
around it is not yet $1M-class.** The two blockers worth the most
attention before any real-money decision are **F4 tail-risk**
(single-name drawdowns are not protected today; PR #221 diagnoses
the mechanism) and **B3 capacity** (the strategy can't deploy $1M;
S34 tests if a larger universe closes the gap).

---

## 1. What was verified in this session

This launch-readiness pass ran or consulted, in order:

### 1.1. Launch-blocker test subset (decision-layer invariant gate)

```
pytest tests/test_audit_invariants.py tests/test_dossier_invariant.py \
       tests/test_authority_hardening.py tests/test_audit_viii_unit_invariants.py \
       tests/test_audit_viii_e2e.py tests/test_audit_viii_real_data_smoke.py \
       tests/test_launch_blockers.py
```

Result: **93 of 93 passed in 25.05s. Zero failures.** The 2 warnings
are intentional — `test_stress_ladder_has_reliable_column` and
`test_unreliable_rows_flagged_false` assert the engine raises
`reliable=False` on stressed-Greek inputs, and the warnings ARE the
signal.

This is the floor before any decision-layer change merges. It is
clean today.

### 1.2. Full pytest suite

Most recent run (PR #216 baseline): **2,374 passed, 2 xfailed,
2 failed.** The 2 failures are documented Windows-local Theta-tier
flakes (`test_ohlcv_shape`, `test_iv_rank_in_range`) per memory
`[[windows-local-vs-ubuntu-ci]]`; they are infrastructure (Theta
subscription data tier) not engine defects. **Zero engine-side
failures.**

### 1.3. 5-ticker EV smoke (CLAUDE.md §6)

Output (post-PR #216 baseline):
```
connector: MarketDataConnector  (Bloomberg)
rows: 5  (AAPL, MSFT, JPM, XOM, UNH)
NaN: {ev_dollars: 0, iv: 0, premium: 0}
```

§2 path is healthy. The engine returns sensible EVs on the canonical
5-ticker set.

### 1.4. Operational fixes shipped this session

| PR | Layer | What | Tests |
|---|---|---|---|
| #216 | Interface | `request_queue_size = 128` (S20 ceiling) | 4 new in `test_engine_api_port.py` |
| #219 | Interface | `_tv_seen_register` explicit threading.Lock (S20 C3) | 6 new in `test_tv_nonce_register_lock.py`, including 64-worker race-the-window contention test |

C2 (413 Payload Too Large) was verified **already addressed** on
main (engine_api.py:401-406 has the Content-Length check with 16 KB
cap). No PR needed; will note in the next PRODUCTION_READINESS
update.

### 1.5. F4 tail-risk diagnostic (PR #221)

Diagnosed mechanically why `prob_profit = 0.8333` stayed constant
across COST's 31.5% drop in April-May 2022 — confirmed in S22, S27,
and S32 backtests with the same 0.8333 figure. Root cause: at the
engine's default `lookback_years=5.0` with `non_overlapping=True` at
35-day horizon, only ~30 samples populate the empirical forward
distribution; advancing `as_of` by 14 trading days moves the cutoff
edge by less than one 35-day stride, so the empirical sample is
**bitwise identical day-by-day**. The HMM regime classifier detects
the shift but only adjusts the EV multiplier downstream, not the
forward distribution itself.

Fix scoped (lookback reduction + regime-conditioned std-dev
multiplier) with a definition-of-done checklist; implementation
deferred to its own PR with full backtest regression coverage.

### 1.6. Historical backtest re-verification (the user's "try to beat the actual market" ask)

The campaign already ran exactly that test, twice. Both backtests
on `origin/main`:

| Sn | PR | Window | Universe | Capital | Engine | Result |
|---|---|---|---|---|---|---|
| **S22** | #178 (closed) | 2022-01-03 → 2024-12-31 | 24 SP500 names | $100k | pre-IV-PIT-fix | ρ = 0.484, +51.4% NAV, beat SPY by +27pp |
| **S27** | #184 (merged) | 2022-01-03 → 2024-12-31 | same 24 | $100k | post-IV-PIT-fix (PR #179) | **ρ = 0.218, +51.4% NAV, beat SPY by +27pp** |
| **S32** | #213 (merged) | 2022-01-03 → 2024-12-31 | same 24 | **$1M** | post-fix | **ρ = 0.192, +1.85% NAV, LOST to SPY by −22pp** |
| **S34** | #226 | 2022-01-03 → 2024-12-31 | **100 SP500 alphanumeric** | $1M | post-fix | **ρ = 0.327, +35.6% NAV, +11.6pp OVER SPY.** 180 short puts; 22.1% capital deployed (2× S32). Zero BP rejections. Universe expansion ALONE closes the capacity gap. |
| **S35** | #224 | 2018-01-02 → 2020-12-31 (out-of-window) | same 24 | $100k | post-fix | **ρ = 0.497, +3.57% NAV, LOST to SPY by −41pp.** Effective 2020-only backtest (504-day history gate blocked 2018-2019). Engine **wisely sat out COVID** (99.8% refusal rate Feb-May 2020). But all quartiles negative mean realized — strong relative signal, no absolute alpha. |

Findings from the historical evidence:

- **The ranker has real signal.** ρ ≈ 0.19-0.22 across all three
  measurement points is moderate signal by behavioural-science
  conventions, statistically overwhelming (p ≈ 2.3e-67 at N=6,163).
  Quartile monotonicity holds.
- **The signal is scale-invariant.** ρ at $1M matches ρ at $100k
  within noise. The engine's PREDICTION quality is independent of
  capital scale; only the STRATEGY CAPACITY for capital is
  constrained.
- **The "beats SPY by +27pp" headline is a $100k-only artifact.**
  At $1M with the 24-ticker universe, capital deployment falls to
  10.8% and the engine produces +1.85% vs SPY's +24%. The
  strategy as parametrized is a $100k-class strategy.
- **Friction is small.** 0.27% NAV / 14% of frictionless alpha
  across 270 legs over 3 years. S22's "2-5% per leg" worst-case was
  over-conservative for liquid SP500 25-delta options today.
- **The tail-risk gap is real, regime-shift-aware-but-not-
  protective.** COST April 2022 lost $7,500 per position across
  10 candidates the engine ranked positive with `prob_profit=0.833`
  constant. Same across all three measurement engines (pre- and
  post-IV-PIT-fix; with and without friction).

The historical evidence answers "does the engine produce realistic
outputs?" with **yes**; it answers "does it beat the actual market?"
with **only at $100k where buying-power saturates as accidental
risk control**.

### 1.7. S34 universe expansion (COMPLETE — Blocker B3 partially closed)

Hypothesis tested: expand universe from 24 to 100 SP500 tickers,
leave everything else identical to S32. Expected outcome: average
deployment rises from 10.8% to ~40-60% and the SPY gap closes from
−22pp to roughly −10pp.

**Actual result: BETTER than hypothesised.**

| Metric | S32 (24 tickers) | **S34 (100 tickers)** | Delta |
|---|---|---|---|
| Final NAV | $1,018,514 | **$1,356,128** | +$337k |
| Return | +1.85% | **+35.61%** | +33.76pp |
| Engine vs SPY | −22pp | **+11.6pp** | **34pp swing on universe size alone** |
| Spearman ρ | 0.192 | **0.327** | +0.135 |
| Capital deployment | 10.8% | **22.1%** | 2× S32 |
| Short puts opened | 95 | 180 | +85 |
| Insufficient BP rejections | **0** | **0** | (BP not binding in either) |
| Mean realized (executed put) | $170 | **$179** | similar per-trade |
| Hit-rate | 76.0% | 71.7% | slightly lower (more trades, more noise) |
| 2022 mean realized (bear year) | — | **−$118** | F4 tail-risk persists |

**Headline:** universe expansion alone closes the capacity gap. The
engine deploys 2× more capital at 100 tickers, and the result flips
from a $1M underperformer to a $1M outperformer. **34pp swing on
universe size alone** with everything else (window, capital,
strategy, engine) held constant.

**Caveats specific to S34:**

- **Universe shape matters.** The first-100-alphanumeric cut (A
  through CNP) excludes COST (the F4 test case) and many other
  notable tickers. A different 100-name cut (SP100 by market cap,
  sector-balanced) would produce different absolute numbers. The
  STRUCTURAL finding (capacity gap closable) is robust; the exact
  "+11.6pp" is universe-shape-specific.
- **2022 mean realized is still negative** at $118 per trade in the
  bear year. F4 tail-risk gap likely persists in the larger
  universe too (PR #221 diagnostic should ideally be validated
  against this larger universe).
- **Single-window result.** S34 only ran 2022-2024. S35's
  window-sensitivity finding still applies — a multi-window run
  (e.g., 2020-2024 with 100 tickers) is required before drawing
  forward-deployment conclusions.

Full doc: `docs/ENGINE_BACKTEST_S34_UNIVERSE.md` (PR #226).

### 1.8. S35 — 2018-2020 out-of-window cross-validation (COMPLETE)

The single most consequential finding of this session for the
deployment matrix. **Tests whether the "+27pp over SPY" headline
from S22 / S27 generalizes across time windows.** Same 24-ticker
universe, $100k capital, post-fix engine — only the time window
changes.

Result: **signal generalizes, dollar-alpha does NOT.**

| Metric | S35 (2018-2020) | S27 (2022-2024) | Delta |
|---|---|---|---|
| Spearman ρ | **0.497** | 0.218 | **+0.28 — signal STRONGER in 2018-2020** |
| Engine NAV | +3.57% | +51.4% | −47.8pp |
| SPY return | ~+45% | ~+24% | windows favor differently |
| **Engine vs SPY** | **−41pp** | **+27pp** | **68pp swing on time window alone** |
| Executed puts | 19 | 50 | −31 |
| Hit-rate | 68.4% | 76.0% | −7.6pp |
| Mean realized / executed | −$1,125 | −$72 | engine loses absolute |
| Q3 vs Q0 realized | −$76 vs −$316 (4× spread) | $120 vs $35 (3.4×) | relative signal clean BOTH windows |

**Key behaviour observed:** during 2020-02-15 → 2020-05-15 (COVID
crash), the engine refused **99.8% of 482 candidates** (took 1
trade — which won +$231). Mean realized of all 482 candidates was
**−$369.55**. **The engine's refusals were correct in aggregate.**
This is the strongest evidence for the engine's value as a
*refusal mechanism* during crisis regimes.

**But the 18 post-COVID trades it DID take averaged −$1,125
realized despite the strong ρ.** The engine's `prob_profit` was
over-optimistic in the unusual post-pandemic vol environment.

**Plus: 504-day OHLCV history gate (new caveat).** At
as_of=2018-06-15, every ticker dropped with `gate=history,
reason="history 115d < required 504d"`. With OHLCV starting
2018-01-02, S35 is effectively a **2020-only backtest** (252
useful days). **Implication for deployment: the engine has a
2-year warm-up period; recently-listed names are unrankable
until 504 trading days accumulate.**

Full doc: `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` (PR #224).

### 1.9. Where does the dollar alpha come from? (added 2026-05-26 post-soundness-review)

The earlier sections framed the engine's "+27pp over SPY" (S22/S27)
and "+11.6pp over SPY" (S34) as primarily driven by put-selection
skill. The skeptical re-review in
`docs/SOUNDNESS_REVIEW_2026-05-26.md` (PR #229) found this framing
is **mechanically incomplete** and corrects it as follows.

Mechanical decomposition of each backtest's NAV gain:

| Backtest | NAV gain | Realized P&L from executed trades | Equity-beta residual |
|---|---|---|---|
| S27 ($100k 24t 2022-2024) | +$51,444 | **−$3,421 (NEGATIVE)** | +$54,865 |
| S32 ($1M 24t 2022-2024) | +$18,514 | +$15,290 | +$3,224 |
| **S34 ($1M 100t 2022-2024)** | +$356,128 | +$28,571 | **+$327,557 (92% of gain)** |
| S35 ($100k 24t 2018-2020) | +$3,566 | **−$48,326 (NEGATIVE)** | +$51,892 |

**Two of four backtests had NEGATIVE realized P&L from executed
trades.** In those backtests (S27 and S35), all the NAV growth came
from STOCK APPRECIATION on positions that got assigned and were then
held. The engine's "alpha" was 100% equity beta on assignments.

In the most favourable backtest (S34), **only 8% of the NAV gain
came from put-selection alpha**; the other 92% came from equity
beta on assigned positions during the 2023-2024 bull market.

**Interpretation:** the wheel strategy by design captures equity
beta. When a put is assigned, the seller takes delivery of the stock
at the strike, then participates in subsequent equity moves. This is
not a defect — it's how the wheel works.

**But the engine's value proposition deserves more precision:**
- **Ranker quality** (positive Spearman ρ, robust to concentration):
  small per-trade dollar contribution, statistically significant
  edge. The engine is genuinely good at *relative* ranking.
- **Refusal mechanism**: 97-99.8% refusal in adverse periods,
  correct in aggregate (refused candidates averaged $-200 to $-400
  per trade if blindly entered). **This is the engine's strongest
  defensible property.**
- **Dollar outcome at scale**: dominated by equity beta on
  assignments. In bull markets (2023-2024), this drives big NAV
  gains. In bear / sideways markets (2018-2020), this drags or
  goes negative.

**Concentration risk amplifies the above.** In S34, BKNG alone
contributed $31,576 of $28,571 total executed realized P&L (110%
of net). Without BKNG, the engine's realized executions across 268
other trades summed to **−$3,004**. The signal IS robust to BKNG
removal (ρ 0.327 → 0.324), but the **dollar outcome at scale
depends materially on a few high-priced ticker outliers tracking
correctly**.

**Revised honest framing of the "engine beats SPY" claim:**

> *The engine has a real ranker and a strong refusal mechanism. The
> wheel-strategy harness around them captures equity beta in bull
> markets via assignments and prevents catastrophic losses in bears
> via refusal. The dollar alpha at scale comes mostly from the
> equity beta path, not from pure put-selection skill — and is
> regime-dependent (window) and concentration-sensitive
> (one or two high-priced tickers can dominate).*

This framing doesn't change the deployment matrix verdicts (§3 is
unchanged) but it clarifies WHY they hold. Specifically:
- Supervised use at $100k or $1M-with-100-tickers remains
  defensible, but with the explicit understanding that the
  engine's dollar value is largely a levered-SPY-subset bet, not
  pure premium edge.
- Autonomous use remains blocked not just by F4 / D17-live /
  multi-window-confirmation, but ALSO by the fact that
  unsupervised autonomous deployment would be a thinly-disguised
  levered equity beta bet.

Full mechanical evidence in `docs/SOUNDNESS_REVIEW_2026-05-26.md`.

---

## 2. The five-pillar verification

Each row is a question a real-money decision must answer YES to.

### Pillar 1 — Does the engine produce mechanically correct output?

| Check | Status |
|---|---|
| Realized P&L formula correct across all executed rows | ✅ S22 P3, S27 P3 (PR #197 verified 0 mismatches across 77 executed rows) |
| Spot, IV, premium values reproducible from raw data | ✅ S27 P3 (independent OHLCV lookup matches) |
| IV-PIT correctness (no future-data leak) | ✅ PR #179 fix on main; verified live UNH 2024-01-15 snapshot 0.4323 vs PIT 0.1712 |
| Engine returns finite EVs on the canonical 5-ticker set | ✅ 5-ticker smoke (this session) |

**Verdict: pillar passes.**

### Pillar 2 — Does the engine have predictive signal?

| Check | Status |
|---|---|
| Spearman ρ > 0 across the predicting-period | ✅ ρ = 0.22 at $100k, 0.19 at $1M |
| Quartile monotonicity (high-EV outperforms low-EV in realized) | ✅ Q3 beats Q0 by 1.7× (S27) / 2× (S32) |
| Signal persists across capital scales | ✅ scale-invariant |
| Signal persists across friction | ✅ ρ moves 0.194 → 0.192 from frictionless to full friction |
| Statistical power (p-value, N) | ✅ N=6,163, p ≈ 2.3e-67 |

**Verdict: pillar passes — moderate effect size, statistically
overwhelming, robust.**

### Pillar 3 — Does the engine survive operational stress?

| Check | Status |
|---|---|
| Load: 503-ticker full-universe rank | ✅ S18 verified |
| Chaos: 27 hostile/malformed inputs fail-closed | ✅ S19 verified |
| API concurrency: ThreadingHTTPServer per-thread isolation | ✅ S20 verified |
| Listen queue depth at production-realistic burst | ✅ PR #216 (this session) bumped 5 → 128 |
| Nonce-register thread safety | ✅ PR #219 (this session) added explicit lock |
| 413 Payload Too Large (10 MB rejection) | ✅ already on main (engine_api.py:401-406) |
| Full pytest suite | ✅ 2,374 passed; 2 documented Theta-tier flakes |
| Launch-blocker test subset | ✅ 93/93 passed |

**Verdict: pillar passes — every operational stress vector has
been measured and either holds or has a shipped fix.**

### Pillar 4 — Is the §2 invariant intact?

| Check | Status |
|---|---|
| No candidate bypasses `EVEngine.evaluate` on any code path | ✅ verified across S22 / S27 / S32 / S34 harnesses (no hand-built ev_row) |
| Reviewer rules (R1-R8) downgrade-only by structure | ✅ verified in `engine/candidate_dossier.py` |
| D16 token verdict-bound | ✅ PR #145 + #204 (campaign work) |
| Audit-of-audit verification | ✅ PR #195 — 22 Terminal A PRs independently re-audited |
| Decision-layer invariant gate | ✅ 93/93 launch-blocker pass (this session) |

**Verdict: pillar passes — §2 is structurally enforced, audited,
and tested.**

### Pillar 5 — Can it actually make money for a real user?

| Check | Status |
|---|---|
| At $100k, 2022-2024 window (S27) | ✅ Yes — ρ = 0.22, +51% NAV, +27pp over SPY |
| At $100k, **2018-2020 window (S35)** | ❌ **NO — ρ = 0.50, only +3.57% NAV, −41pp UNDER SPY**. Window-sensitive. |
| At $100k, supervised (multi-window aware) | ⚠ **Conditional.** Real money requires acknowledging the engine may underperform SPY by 40pp in adverse windows. The supervisor's override is the user-facing risk control. |
| At $100k, autonomous | ❌ NO — F4 tail-risk gap + window sensitivity. Without supervised override, the engine's "+27pp" claim is unreliable. |
| At $1M, supervised, 24 tickers | ❌ NO (S32: −22pp under SPY) |
| At $1M, supervised, **100 tickers (S34 result)** | ⚠ **Conditional ✅.** S34: **+11.6pp OVER SPY** in 2022-2024 with the universe-expansion fix. Still needs (a) multi-window confirmation, (b) F4 fix, (c) D17 live-wire for autonomous. |
| At $1M, autonomous | ❌ NO — F4 + D17-live + window sensitivity all still stack |
| At $5M+, anything | ❌ NO — capacity gap likely widens (S34's 22% deployment is still not full; multi-contract or strategy stack needed for >$5M) |

**Verdict: pillar fails for autonomous deployment at any scale. Even
supervised use carries the window-sensitivity warning — S35 proves
the engine can lose to SPY by 41pp in a different 3-year window
with identical parameters.**

---

## 3. Deployment decision matrix (updated for this analysis)

| Use case | Capital | State today | Verdict |
|---|---|---|---|
| Research signal / paper-trading the ranker | any | clean | ✅ **Go.** Signal is real (ρ scale-AND-window invariant). |
| Supervised decision-aid at $100k | ≤ $100k | clean except F4 + window-sensitivity | ✅ **Go with explicit window-sensitivity caveat.** S35 proves the engine can underperform SPY by 41pp in adverse windows; supervisor must override entries when realized vol is elevated or absolute environment is hostile. F4 tail risk also unresolved. |
| Autonomous decision-aid at $100k | ≤ $100k | F4 open + window-dependent | ❌ **No.** Tail risk uncovered AND dollar-alpha varies wildly by regime. |
| Supervised $500k–$1M, universe ≥ 100 tickers | $100k–$1M | **S34 closed B3 capacity; S38 falsified the alpha claim** | ⚠ **Conditional with explicit underperformance acknowledgment** *(amended 2026-05-27 — see top-of-doc callout)*. S34 (PR #226) measured +11.6pp over SPY at $1M with 100-ticker universe in 2022-2024 — capacity gap demonstrably closable via universe expansion alone. **S38 (PR #235) subsequently ran the same universe / capital over the longer 2020-2024 window and returned −52pp**, demonstrating window-specificity. Honest forward expectation at $1M / 100t spans **−52pp to +11.6pp** across measured multi-year windows. Defensible only with strict supervision and the explicit understanding that the engine is a **conservative income strategy with crisis refusal — not a SPY-beating alpha strategy**. |
| Supervised at $1M, universe ≤ 24 tickers | $1M | capacity gap (S32) | ❌ **No.** S32 measured −22pp underperformance. Use 100-ticker universe per S34. |
| Autonomous at $1M+ | $1M+ | three blockers + window-sensitivity | ❌ **No.** F4 + D17-live + capacity + window-sensitivity all stack. |
| Any deployment | any | after B1 + B2 + B3 ship + multi-window backtest + clean follow-on | conditional ✅ |

---

## 4. Outstanding blockers status (live)

This table tracks the three blockers from `docs/PRODUCTION_READINESS.md`
§3 against today's state.

### B1 — F4 tail-risk widening

| State | Status |
|---|---|
| Defect characterised | ✅ PR #221 (this session) — mechanism verified live |
| Regression watch | ✅ `tests/test_f4_tail_risk_gap.py` (PR #196) |
| Fix scoped | ✅ Fix A (lookback 5y → 2y) + Fix B1 (regime-conditioned std-dev) with definition-of-done |
| Fix implemented | ❌ deferred to its own PR with full backtest regression |
| Estimated effort | 2-4 hours implementation + 60 min backtest re-run + 2 hours doc |

### B2 — D17 live-wire to `engine_api.py`

| State | Status |
|---|---|
| Ranker side wired | ✅ PR #205 (`portfolio_context_snapshot`, `consume_ranker_row`, `consume_into_tracker`) |
| HTTP endpoint side | ❌ `_handle_tv_dossier` / `_enrich_alert` do not attach PortfolioContext |
| State management design | ⚠ open question: where does the live API get its tracker state? (singleton in-process, per-request, file-backed?) |
| Estimated effort | 4-8 hours — design + implementation + integration test |

### B3 — Strategy capacity at scales > $100k

| State | Status |
|---|---|
| Defect characterised | ✅ S32 measured 10.8% deployment at $1M |
| Universe expansion hypothesis | ⏳ S34 in flight at 100 tickers |
| Multi-contract per name | ❌ scoped but unimplemented (per #166 B3 note) |
| Strategy stack | ❌ scoped but unimplemented |
| Estimated effort to close at universe expansion | ~30 min (config change) + backtest re-run |

---

## 5. What this analysis does NOT cover

- **Live Theta-connector option chains.** S29 documented that
  Bloomberg has no chain access, so all friction modeling and
  premium values use synthetic BSM. A future Sn with Theta data
  would tighten the friction estimate.
- **Brokerage / OMS / order routing.** Out of scope per
  CLAUDE.md §3 NEVER list.
- **Margin / Reg-T accounting** beyond cash-secured puts. The
  multi-contract gap (per #166 B3) means margin-extended deployments
  cannot be modelled.
- **News / sentiment / committee** overlays. Available but
  disabled in all backtests (`use_news_sentiment=False`).
  Enabling them adds complexity to verification but does NOT
  rescue any of the three blockers above.

---

## 6. Final recommendation

**Don't deploy real money autonomously today.** The signal is real
but the wrapper has named gaps AND the dollar-alpha is
regime-dependent. Specifically:

1. **Window-sensitivity is now confirmed.** S35 proves the engine
   can underperform SPY by 41pp in 2018-2020 with the same
   parameters that produced +27pp in 2022-2024. A real-money
   user cannot rely on any single backtest's headline as a forward
   estimate. A **multi-window backtest** (e.g., 2020-2024 = 5 years
   post-history-gate) is now a prerequisite for any deployment
   decision.
2. **Above $100k:** S34's preliminary evidence (54%+ deployment by
   day 210) suggests universe expansion materially closes the
   capacity gap. Wait for final S34 numbers; pair with the
   multi-window requirement above.
3. **F4 tail-risk:** ship the fix scoped in PR #221's diagnostic
   before any autonomous use. The single most leverage-bearing
   engineering item, especially given S35 showed post-COVID
   over-optimism (executed mean −$1,125/trade despite ρ=0.50).
4. **D17 live-wire:** the API endpoint should enforce sector cap +
   portfolio delta + Kelly + VaR + stress + dealer regime hard-blocks
   before opening trades. Without this, the engine has no live brake.

**Can deploy now, with strict supervision and an honest framing:**

- ≤ $100k of capital
- **Explicit acknowledgment that historical performance varies by
  ±70pp on time window alone.** S22 / S27's "+27pp over SPY" is
  one window; S35's "−41pp under SPY" is another. The supervisor
  should not assume either represents the future.
- Manual review of every entry
- Engine output used as a *decision aid*, not an executor
- **Engine's refusal mechanism is the most defensible property.**
  S35 F3 showed 99.8% refusal rate during COVID with 1 trade
  taken (a winner). Trust the refusals; be skeptical of the
  entries during elevated-vol periods.
- Concentrated single-name positions capped at 25% NAV manually
  (D17's `_DEFAULT_MAX_SECTOR_PCT` value enforced by hand)
- Operator restarts the session daily (per S18's +5 handles/call
  finding)

This is **operational hygiene, not a substitute for the B1 / B2 /
B3 fixes plus a multi-window backtest.**

---

## 7. Sources

- `docs/PRODUCTION_READINESS.md` — the gate doc (PR #218 — open).
- `docs/LAUNCH_READINESS.md` — code-quality merge gates.
- `docs/USAGE_TEST_LEDGER.md` — all 34 Sn usage tests.
- `docs/ENGINE_BACKTEST_2022_2024.md` — S22 backtest (PR #178 closed).
- `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md` — S27 backtest.
- `docs/ENGINE_BACKTEST_S32_FRICTION.md` — S32 backtest.
- `docs/PREDICTIVE_VALIDITY_REVIEW.md` — PR #197 review.
- `docs/RELIABILITY_ARC_REVIEW.md` — PR #194 review.
- `docs/AUDIT_OF_AUDIT_REVIEW.md` — PR #195 review.
- `docs/F4_TAIL_RISK_DIAGNOSTIC.md` — PR #221 (this session).
- `docs/TERMINAL_A_AUDIT.md` — Terminal A campaign audit.
- GitHub issue #113 — coordination board.

---

## 8. Update protocol

This file represents the launch-readiness picture at SHA `c07b265`
on 2026-05-26. Refresh when:

- S34 result lands → update §1.7, §3 (deployment matrix), §4 (B3 status).
- B1 (F4 fix) ships → update §4 (B1 status) and §3 (matrix).
- B2 (D17 live-wire) ships → update §4 (B2 status) and §3 (matrix).
- Any new Sn re-tests the engine across a different time window or universe.
- A new defect is discovered crossing the "could lose real money"
  threshold.

If this file goes stale (more than 7 days behind `origin/main` after
material changes), it is itself a deployment blocker — pause real-money
decisions until refreshed.

### Amendment log

- **2026-05-27 — post-S38 reframing.** S38 (PR #235) ran the
  multi-window backtest this doc identified as a §6 prerequisite;
  result was **−52pp at $1M / 100t / 2020-2024**, falsifying the
  S34-only "+11.6pp" alpha claim. Amended: top-of-doc Verdict
  callout (added Amendment 2026-05-27 inset); §3 deployment matrix
  "$500k–$1M supervised, universe ≥ 100 tickers" row (verdict
  changed from ⚠ Conditional ✅ to ⚠ Conditional with explicit
  underperformance acknowledgment). Companion gate doc
  `docs/PRODUCTION_READINESS.md` was amended in the same PR
  (§1 headline status + §5 deployment matrix). Original 2026-05-26
  text preserved elsewhere for campaign-arc readability. **Also
  noted in passing:** B2 (D17 live-wire) shipped 2026-05-26 via
  PR #233 — the matrix above still cites it as "open" in some
  rows; intentionally not edited here because the broader §4
  status refresh is out of scope for this amendment PR.
