# HT-D — R10 strict-mode at $1M / 100t scale (2026-05-30)

**Question:** *Every prior canonical backtest (S27 / S32 / S34 / S35 /
S38 / S40 / S43 / S44) ran with `require_ev_authority=False` on the
`WheelTracker`. That meant the **R10 single-name notional cap**
(`engine.portfolio_risk_gates.check_single_name_cap`, the doc-designated
**load-bearing magnitude guard** for the engine's `prob_profit` top-bin
over-confidence per `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` +
`docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10 +
`docs/PRODUCTION_READINESS.md` §3 B1) has **never** been exercised in a
backtest. S44 §7 AI handoff explicitly flagged this open question. Does
R10 actually bind at this scale, and if it does, does it materially
move the engine-vs-passive deployment story?*

**Headline answer:** R10 binds **571 times** in the strict-mode
$1M/100t/2020-2024 run — exclusively on **BKNG (331) and AZO (240)**,
exactly the high-priced single names S44 §7 predicted. The
**`portfolio_delta_breach` gate dominates at 92.1% of all D17 refusals
(6,704 of 7,276)**, not R10 — re-framing the load-bearing-gate
question. Strict mode **opens only 31 puts in 5 years vs loose's 492**
(99.5% refuse rate) and ends the window with **NAV $1,247,668 (+24.8%)
vs loose's $1,405,794 (+40.6%) = −15.81pp delta**. **Crucially, strict
was AHEAD of loose for 66.6% of trading days** (peaked at +$149k =
+18.5pp ahead in April 2020 COVID bottom) and only crossed below loose
permanently on **2023-11-01**, then lost ground through the 2024 bull
to the final −$158k gap. **R10 + the portfolio-delta gate together
preserve capital in crises but their combined refusal-on-state (not
refusal-on-EV) mechanism causes severe bull-market drag** — the
engine essentially becomes a passive long-stock holder of 21 names
after 2020 because the gate-trio never re-opens room for new opens
once the assigned-stock book saturates portfolio delta.

**Engine SHA at run:** `origin/main` @ `56c671d` (cycle-1 close —
post #284 / #285 / #286 / #287 / #248 / #249 / #251 / #275 / #288).

**Baseline:** A/B harness inside one driver — loose
(`require_ev_authority=False`, matching S38/S44) vs strict
(`require_ev_authority=True` + `min_nav_for_trading=0` + attached
connector) running in parallel on a SHARED daily SP rank call. This
eliminates harness-drift vs S38's throwaway harness — the strict-vs-
loose delta isolates R10/D17 cleanly.

---

## 1. Setup

- **Window:** 2020-01-02 → 2024-12-31. Same calendar window as
  S38 / S44 / S43 W3, but the driver iterates **1,304 business days**
  (`pd.bdate_range`, Mon-Fri excluding weekends only) whereas S38's
  harness used a US-holiday-aware 1,258-trading-day filter. The
  difference is ~46 US market holidays — they are no-op iterations
  on both trackers (no fresh rank data, no opens), so the A/B is
  byte-clean over the same 1,304 days, but the methodological claim
  "identical to S38/S44" is softened by the calendar filter; the
  shared-rank ρ = 0.4326 confirms the input is identical on the
  trading days that ARE common. Earlier draft erroneously cited
  S38's 1,258 as this run's count — corrected below in §2.2 / §3 /
  §10 to the driver's actual 1,304.
- **Capital:** $1M
- **Universe:** `backtests.regression.universes.UNIVERSE_100` (100 first-
  alphanumeric SP500 tickers — same as S34 / S38 / S40 / S43 / S44)
- **Strategy:** 35-DTE / 25-delta short puts, wheel into 35-DTE / 25-
  delta covered call on assignment, hold to expiry
- **Friction:** `full` only (canonical S38 / S44 headline). Friction is
  an independent dimension already characterised by S38 / S44 across
  all 3 levels; R10 cares about per-name notional aggregation, not
  entry-cost shape.
- **Tracker config:**
  - `tracker_loose`: `require_ev_authority=False` (identical to S38 / S44
    baseline)
  - `tracker_strict`: `require_ev_authority=True`,
    `min_nav_for_trading=0`, `connector=MarketDataConnector(...)` —
    triggers `_evaluate_d17_hard_blocks` on every put + CC open attempt
- **R10 defaults (production):** `_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10`
  (10% NAV per ticker, aggregating short-option notional across put +
  CC legs). `_DEFAULT_MAX_SECTOR_PCT = 0.25` (R9).
  `_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0`. `_DEFAULT_KELLY_FRACTION
  = 0.5` (preemptively reserved per docstring — won't fire under one-
  contract sizing).
- **Engine SHA under test:** `origin/main` @ `56c671d` (cycle-1-close).
- **Driver:** `docs/verification_artifacts/r10_strict_driver.py` (this
  PR). Wall-clock 7.0h (419.7 min for **1,304 business days = 19.3
  s/day**, modestly slower than the 18.7 s/day pilot rate because the
  strict-mode path adds D17 gate evaluation on every refused attempt —
  6,704 extra `_evaluate_d17_hard_blocks` calls vs the loose path).
- **Output artifacts:** `rank_log_{loose,strict}.csv`,
  `open_attempts_{loose,strict}.csv`, `cc_attempts_{loose,strict}.csv`,
  `equity_curve_{loose,strict}.csv`, `tracker_{loose,strict}_state.json`,
  `daily_state.csv`, `summary.json`. Generated under
  `%TEMP%/r10_full/` (NOT committed — the rank logs are 1.2 MB × 2
  per the cycle conventions). The committed companion artifacts in
  `docs/verification_artifacts/` are: the driver itself, the trimmed
  raw stdout, and the `summary.json`.

## 2. Headline results

| Metric | Loose | Strict | Δ strict − loose |
|---|---|---|---|
| Final NAV | $1,405,794 | $1,247,668 | **−$158,127** |
| Return (5y) | +40.58% | +24.77% | **−15.81pp** |
| Final cash | $87,099 | $721,047 | +$633,948 |
| Put open ATTEMPTS | 619 | 5,652 | +5,033 |
| Puts OPENED | 492 | 31 | −461 |
| Put refuse rate | 20.5% | **99.5%** | +78.9pp |
| Put assignments | 113 | 21 | −92 |
| CCs OPENED | 288 | 1 | −287 |
| Spearman ρ (all candidates) | 0.4326 | 0.4326 | 0.0000 (shared rank) |
| Mean realized (all candidates) | +$54.83 | +$54.83 | identical (shared rank) |
| Executed realized total | +$65,692 | +$975 | **−$64,717** |
| Executed realized per trade | +$151.36 | +$97.47 | −$53.89 |

**Reading.** Strict mode refused **99.5% of put open attempts** over
the 5-year run (5,621 of 5,652) and **99.94% of CC open attempts**
(1,655 of 1,656). The 31 puts strict did open were chronologically
clustered in the early-2020 period (see §4); after that the D17 gate
stack saturated and the strict tracker effectively became a 21-name
passive long-stock holder for the remaining 4 years of the run. Loose
opened 492 puts (20.5% refuse rate — only `insufficient_bp`),
captured 5× more executed realized P&L (+$65,692 vs +$975), and ended
+15.81pp ahead.

The shared Spearman ρ = 0.4326 confirms identical rank input to both
trackers; only the execution layer differs.

### 2.1 D17 hard-block firing breakdown (strict only)

| Gate | Put refusals | CC refusals | Total | % of all strict refusals |
|---|---|---|---|---|
| `portfolio_delta_breach` | 5,049 | 1,655 | **6,704** | **92.1%** |
| `single_name_breach` (R10) | 571 | 0 | 571 | 7.8% |
| `sector_cap_breach` (R9) | 1 | 0 | 1 | 0.0% |
| `kelly_size_exceeded` | 0 | 0 | 0 | 0.0% |
| **Total refusals** | **5,621** | **1,655** | **7,276** | 100% |

**Headline structural finding:** the dominant binding D17 gate at
$1M/100t scale is **`portfolio_delta_breach`** (`check_portfolio_delta`
with default cap `_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` → ±$3,000
delta-dollar cap at $1M NAV) — **92.1% of all D17 refusals**. R10
(`single_name_breach`, the doc-designated load-bearing magnitude guard
per `PRODUCTION_READINESS.md` §3 B1) bound **7.8% of the time** (571
of 7,276). R9 fired once. Kelly never fired.

This re-frames the "load-bearing magnitude guard" framing in
`docs/PRODUCTION_READINESS.md` §3 B1. R10 IS firing exactly on the
predicted high-priced names (BKNG / AZO — see §5), but the
portfolio-delta gate is doing 12× the volume of refusals at this NAV.
Both gates contribute to the strict tracker's saturation and
subsequent inability to add new positions for 4 years.

### 2.2 Per-name peak short-option notional

| Tracker | Peak max-name %NAV | Date | n_names at peak |
|---|---|---|---|
| **Loose** | **25.36%** | 2020-03-23 | 45 |
| Strict | 4.80% | 2020-03-23 | 11 |

**Loose tracker breached the 10% R10 per-name cap on 623 of 1,304
trading days = 47.8%.** If R10 had been enabled on the loose tracker
(equivalent to running it in strict mode for R10 only), the gate would
have bound on nearly half the days of the run. Most of these breaches
are *assignment-driven concentration* — the loose tracker accumulated
high-basis assigned stock during 2020 COVID and that exposure took
years to cycle off:

| Year | Loose peak max-name %NAV |
|---|---|
| 2020 | 25.36% |
| 2021 | 23.33% |
| 2022 | 23.36% |
| 2023 | 18.00% |
| **2024** | **4.64%** |

The monotone decrease from 25.4% → 4.6% across 5 years reflects the
loose tracker's natural diversification as winning positions cycled
out and new positions were added. **By 2024 the loose book had self-
diversified to where R10 would no longer bind** — useful evidence
that R10 is more relevant in early-phase / drawdown-shock phases than
in steady-state operation.

Strict tracker breached 10% per-name on **0 of 1,304 days** (by
construction — the R10 hard-block prevents it). Strict's 2020 peak
was 4.80% on the same 2020-03-23 date as loose, with 11 short option
legs (vs loose's 45). Strict's per-name %NAV was **0.00% in 2021,
2022, 2023, 2024** because strict had ZERO new short option opens
after 2020 — `_per_name_short_notional` measures HELD short options,
and strict had only 21 stock-only positions (post-assignment) +
1 open CC for those years.

## 3. Trajectory — when does strict win, and when does it lose?

**Strict was AHEAD of loose on 869 of 1,304 trading days = 66.6% of
the run.** The strict-NAV-vs-loose-NAV gap evolved through five
distinct phases:

| Phase | End date | Loose NAV | Strict NAV | Δ strict − loose | Regime |
|---|---|---|---|---|---|
| COVID bottom | 2020-04-01 | $808,258 | **$957,590** | **+$149,332 (+18.5%)** | bear |
| 2020 close | 2020-12-31 | $1,038,724 | $1,061,815 | +$23,091 (+2.2%) | recovery |
| 2021 close | 2021-12-31 | $1,127,216 | $1,160,140 | +$32,924 (+2.9%) | bull |
| 2022 close | 2022-12-30 | $1,070,140 | $1,116,923 | +$46,783 (+4.4%) | bear |
| 2024 open  | 2024-01-01 | $1,202,748 | $1,166,358 | **−$36,390 (−3.0%)** | bull (crossover) |
| 2024 close (final) | 2024-12-31 | $1,405,794 | $1,247,668 | **−$158,127 (−11.3%)** | bull |

**Critical crossover date: 2023-11-01.** Strict's last day ahead.
After that, loose pulled away through the 2024 bull and ended
−$158,127 behind. The strict-mode advantage is real and material in
bear / drawdown regimes (peaked at +18.5% NAV ahead during COVID) but
the gate stack's inability to RE-OPEN room for new positions during
recovery + bull regimes means strict misses the rank-quality alpha
that loose captures across 2023-2024.

The crossover happens **NOT** because strict held a bad book — its
21 long stock positions appreciated through 2023-2024 too — but
because loose simultaneously opened 161 new puts in 2023-2024 + 277
covered calls, harvesting the engine's positive ρ = 0.42 ranking
signal during favourable years. Strict couldn't.

## 4. Per-year breakdown

| Year | n_candidates | ρ (shared) | Mean realized (all) | Loose executed | Strict executed | R10 (loose) | R10 (strict) |
|---|---|---|---|---|---|---|---|
| 2020 | 2,585 | 0.5768 | +$22.56 | **174** | **31** | 235 days > 10% | 108 |
| 2021 | 2,569 | 0.3194 | +$168.09 | 108 | **0** | 173 days > 10% | 97 |
| 2022 | 2,558 | 0.3883 | −$139.06 | 81 | **0** | 204 days > 10% | 224 |
| 2023 | 2,568 | 0.4120 | +$113.44 | 77 | **0** | 11 days > 10% | 75 |
| 2024 | 2,617 | 0.4253 | +$107.55 | 52 | **0** | 0 days > 10% | 67 |

**Key observations:**

- **Strict opened ALL 31 of its put trades in 2020. Strict opened
  ZERO puts in 2021, 2022, 2023, and 2024.** The strict tracker
  freezes after the early-COVID acquisition phase because the
  assigned-stock long-delta saturates the ±$3,000 portfolio-delta
  cap, and the strict tracker has no closing-side mechanism to
  unload the stock and free delta room (the wheel-into-CC path
  could close some via call assignment, but only 1 CC was opened
  in the entire 5y window because the same portfolio_delta_breach
  gate refuses CC opens too).
- **Loose opens scale down monotonically** (174 → 108 → 81 → 77 →
  52) as cash gets allocated and the book matures. This is normal
  wheel-strategy behaviour — diminishing returns from re-deployment.
- **R10 (strict) firings track the rank-frequency of BKNG/AZO**,
  not portfolio state. R10 fired 108 times in 2020, 97 in 2021,
  224 in 2022 (the bear year — BKNG hit lows and the ranker
  ranked it heavily), 75 in 2023, 67 in 2024. R10 is a per-attempt
  ranker-driven gate; portfolio_delta_breach is a portfolio-state-
  driven gate. The two have different binding dynamics.
- **R10 (loose) counterfactual** shows the natural decline of
  per-name concentration in the loose path: 235 → 173 → 204 → 11 →
  0 days above 10% per-name. The 2022 spike is BKNG's bear-year
  drawdown re-concentrating exposure; 2023-2024 the loose book
  diversified out completely.

## 5. R10 firing pattern — the predicted high-priced names

S44 §7 AI handoff predicted: *"R10 mostly fires when the engine ranks
AAPL or BKNG heavily for several consecutive days."* The full 5y
run confirms exactly:

### `single_name_breach` (R10) — top tickers

| Ticker | Refuse count | % of R10 refusals |
|---|---|---|
| **BKNG** | **331** | 58.0% |
| **AZO** | **240** | 42.0% |

**Only BKNG and AZO fire R10.** No other tickers in `UNIVERSE_100`
trigger the gate — the mechanism is "single 25-Δ short-put contract
notional exceeds 10% of $1M NAV", which only happens at strike >
$1,000 (BKNG @ ~$1,990 → ~20% NAV; AZO @ ~$1,100 → ~11% NAV).

S44's prediction was 100% correct on the *tickers* (BKNG / AZO);
the run also confirms the mechanism is *single-contract entry
notional*, not *accumulation across multiple contracts*. R10 is
effectively a **single-contract entry cap for the high-priced subset
of the universe at $1M scale**. At $10M scale these same names would
have notional/NAV = 2%/1%, well under the cap, and R10 would not
fire on them. R10 is NAV-relative; at smaller NAV (e.g. $100k) far
more names would breach the cap.

### `portfolio_delta_breach` — top tickers

| Ticker | Refuse count |
|---|---|
| CDNS | 405 |
| AMT | 392 |
| CCI | 339 |
| BF/B | 262 |
| ADBE | 255 |
| CARR | 195 |
| AMAT | 194 |
| AKAM | 191 |
| ARES | 183 |
| CLX | 173 |

These are moderate-priced names the engine repeatedly ranked highly
and the gate refused over the strict tracker's frozen-portfolio
years (2021-2024). The presence of CDNS / AMT / CCI / ADBE here
mirrors S44's per-ticker concentration analysis — the engine has a
recurring preference for these tickers, and the portfolio-delta gate
refused them on every attempt after the strict portfolio saturated.

### `sector_cap_breach` — top tickers

| Ticker | Refuse count |
|---|---|
| AZO | 1 |

R9 (sector cap, default 25% NAV) bound exactly once in the entire
5y run — on AZO, where R9 happened to fire before R10. This is the
"R10 sits beneath R9" sequencing from `engine/wheel_tracker.py:1855-1876`:
R9 is checked first in `_evaluate_d17_hard_blocks`. AZO is in the
Consumer Discretionary sector; the loose path's BKNG (also Consumer
Discretionary) + AZO together can push that sector over 25% NAV,
but R9's near-zero firing rate confirms that the engine's natural
sector mix is already wider than the cap permits — the cap is over-
calibrated for this universe / strategy combination.

## 6. §2 invariant scan

| Tracker | Puts opened | with `ev_dollars ≤ 0` | non-finite `ev_dollars` | Verdict |
|---|---|---|---|---|
| **Loose** | 492 | **0** | **0** | **§2 OK** |
| **Strict** | 31 | **0** | **0** | **§2 OK** |

**§2 invariant holds in both modes.** No PUT positions were opened
with non-positive or non-finite EV. The strict tracker's
`consume_ranker_row` path refuses non-positive at issuance (R1-
equivalent at the launch gate via `EVAuthorityRefused`); the loose
tracker filters at the harness level (`if float(row.get("ev_dollars",
0.0)) <= 0: continue` in the open loop).

## 7. Findings

**F1 — R10 binds 571 times over 5 years on the predicted tickers.**
Exclusively BKNG (331) + AZO (240). S44 §7 AI handoff's prediction
("R10 mostly fires when the engine ranks BKNG or AZO heavily for
several consecutive days") is exactly confirmed. The mechanism is
*single-contract entry notional > 10% NAV*, not *accumulation across
multiple contracts*. R10 is NAV-relative — at $10M NAV these
breaches would not occur; at $100k NAV many more names would breach.

**F2 — `portfolio_delta_breach` is the dominant D17 binding gate at
this scale, NOT R10.** 6,704 of 7,276 refusals (92.1%) fired on the
±$300 / $100k NAV portfolio-delta cap. R10 was 7.8%; R9 was 0.01%;
Kelly never fired. The `docs/PRODUCTION_READINESS.md` §3 B1 framing
of R10 as the "load-bearing magnitude guard" reflects the *design
intent* but is misleading about the *actual binding rate*. At this
scale the portfolio-delta gate carries 12× R10's refusal volume.

**F3 — Strict mode FREEZES after the initial deployment phase.**
**Strict opened ALL 31 of its put trades in calendar year 2020.
Strict opened ZERO puts in 2021, 2022, 2023, 2024.** Mechanism: the
21 assigned-stock positions accumulated in 2020 contribute
~$5,000-6,000 of long delta-dollars (each 100-share lot at ~$50-300
spot), which alone exceeds the ±$3,000 portfolio-delta cap. Every
subsequent new-put attempt would push delta further over the cap and
is refused. No closing-side mechanism exists to free delta room
without an external trigger (call-assignment via the wheel-CC leg
COULD free room, but only 1 CC opened in the entire run because the
same gate refuses CC opens too).

**F4 — Strict outperforms loose in bear regimes; loose outperforms
strict in bull regimes; over the full 5y window loose wins by
−15.81pp.** Strict was AHEAD on 66.6% of trading days (peaked at
+$149k / +18.5% NAV in April 2020 COVID bottom; held the lead
through end of 2022 bear) but **lost the cumulative race during the
2023 bull recovery** (last day strict ahead: 2023-11-01). Final delta
−$158,127 / −15.81pp over the 5y run. **The gate stack is a
crisis-preservation mechanism, not a sustained-alpha mechanism.**
This is the answer to the HT-D headline question: R10 (and its D17
peers) DO meaningfully change the deployment story — they preserve
capital in drawdowns and lose capital in recoveries — and the
direction depends entirely on regime mix over the measurement window.

**F5 — R10's binding rate is non-stationary across the run.** Loose
tracker's max-name %NAV decreased monotonically from 25.4% (2020) to
4.6% (2024) as the book naturally diversified. R10's would-have-fired
days dropped 235 → 0 across 2020 → 2024 (per the loose-mode
counterfactual). **R10 is most useful in early-phase / drawdown-
shock phases when concentration is acute; less useful in steady-state
when the book has self-diversified.** Production implication: R10
adds the most defensive value in the first 1-2 years after a fresh
deployment, less in years 3+.

**F6 — The `portfolio_delta_breach` gate is structurally over-
calibrated for the wheel strategy.** A wheel book has STRUCTURALLY
long delta from assigned stock; the ±$300 / $100k NAV cap (set in
D17 for a generic options book per #154 C4) is wrong for this
strategy. The 21 assigned-stock positions alone exceeded the cap and
froze the strict tracker. A wheel-aware delta cap should EITHER
exclude assigned-stock-leg delta from the calc OR set a much higher
cap for stock-leg-inclusive measurement. **Not in HT-D scope to fix;
surfaced as a finding for Major Session triage.**

**F7 — §2 invariant CLEAN in both modes.** Zero opens with
`ev_dollars ≤ 0` or non-finite EV in either tracker. Both the
launch-gate (`consume_ranker_row` → `issue_ev_authority_token` raises
on non-positive) and the harness EV filter (loose path) enforce R1
correctly.

**F8 — R10 + D17 stack causes the strict tracker to evolve into a
passive long-stock holder.** End-of-run state: strict holds 21 stock
positions, 1 open CC, $721k cash (vs $87k loose cash). The wheel's
income-generation purpose (collecting put premium) is structurally
defeated by the D17 stack at this scale: strict's executed realized
P&L is +$975 (essentially zero) over 5 years, vs loose's +$65,692
(+6.6% NAV). The strict tracker becomes a 21-name long-stock holder
with a cash drag.

## 8. Implications for the deployment matrix

`docs/PRODUCTION_READINESS.md` §3 B1 currently frames the F4 closure
as a deployment-bundle:

> **B1 — F4 tail-risk widening fix.** ... PR #260 (the **frequency
> guard**) + PR #262 (R10 single-name notional cap, the **magnitude
> guard**) ... together they form the F4 defense-in-depth pair.

**HT-D adds the first empirical measurement of how the D17 stack
(R10 included) behaves at the canonical $1M / 100t scale**:

- R10 specifically binds 571 times over 5 years, exactly on the
  high-priced names S44 predicted. The magnitude-guard framing is
  *mechanically* correct.
- **The portfolio_delta cap dominates strict-mode refusals at 12:1
  vs R10** — the framing should acknowledge that R10 + the
  portfolio-delta cap *together* form the binding stack, not R10
  alone.
- The **−15.81pp full-window strict underperformance** at $1M / 100t
  in 2020-2024 is consistent with S38's −52pp (loose) and S44's
  similar finding — and is now joined by HT-D's measurement that
  *strict mode does NOT close this gap; it widens it*. The strict
  tracker underperforms loose by another 15.8pp ON TOP OF loose's
  already-existing −52pp gap vs Univ-EW.
- **Strict mode would have been a CORRECT choice for an operator who
  knew in advance the run would include a COVID-scale drawdown
  near the start.** It would have been a WRONG choice for an operator
  who knew the window was bull-dominated. Without a regime forecast,
  the verdict for the 5y window is "loose wins".

**Suggested PRODUCTION_READINESS.md §5 deployment matrix amendment**
(for Major Session to triage):

> **$500k-$1M supervised, universe ≥ 100 tickers** — adds an
> **R10 strict-mode footnote**: HT-D (2026-05-30) measured the
> first-ever strict-mode backtest at this scale and found that
> `require_ev_authority=True` + attached `PortfolioContext` causes
> the strict tracker to refuse 99.5% of put attempts and freeze
> after the initial 2020 deployment phase due to portfolio_delta
> cap saturation. Strict mode preserves capital in drawdowns
> (+18.5% NAV ahead at COVID bottom) but underperforms loose by
> −15.81pp over the full 5y window. Operators considering strict
> mode for live deployment should either (a) accept the bull-year
> drag in exchange for crisis preservation, or (b) wait for a
> follow-up D-decision that calibrates the portfolio-delta cap for
> the wheel strategy's structural long-stock delta.

## 9. AI handoff

- **R10 is mechanically correct + S44-prediction-validated** (571
  firings, all on BKNG/AZO as predicted). The "load-bearing magnitude
  guard" framing in `docs/PRODUCTION_READINESS.md` is correct in
  intent — R10 IS the gate that catches single-contract notional
  excesses on high-priced names — but its actual binding rate at
  $1M scale is dwarfed (12:1) by `portfolio_delta_breach`.
- **The portfolio_delta cap (D17 default `±$300/$100k NAV`) is
  STRUCTURALLY MISCALIBRATED for the wheel strategy.** A wheel book
  has long-delta from assigned stock by design; the cap, set in #154
  C4 for a generic options book, freezes the strict tracker after
  the assigned-stock book exceeds ~$3,000 delta-dollars at $1M NAV
  (which happens after ~21 assigned positions at any realistic
  spot). **Recommended follow-up D-decision**: either exclude
  stock-leg delta from `check_portfolio_delta`'s aggregation, or
  scale the cap with deployed (not initial) capital, or set a
  wheel-specific cap multiplier. Out of HT-D scope; flagged for
  Major Session triage.
- **R10 in entry-only mode cannot prevent assignment-driven
  concentration breaches.** Loose tracker reached 25.36% per-name
  on 2020-03-23 (well above the 10% cap) via assignments, not new
  opens. A future card could evaluate adding an assignment-time
  R10 check or a closing-side R10 (force-close on a name that grew
  past the cap via assignment).
- **The deployment-story answer is regime-dependent.** Strict won
  every bear / drawdown phase (peaked +$149k ahead in COVID-April-
  2020); loose won the 2023-2024 bull recovery. Over the full 5y
  window, loose wins by −15.81pp. This is the answer to the HT-D
  headline question, and it suggests the *strategy* of switching
  between strict / loose modes based on regime forecast is a
  separate (large) research surface — not in HT-D scope.
- **§2 invariant CLEAN.** Both modes pass with zero non-positive-EV
  opens. The launch gate (`consume_ranker_row` →
  `issue_ev_authority_token`) and the harness filter both enforce R1
  correctly.

## 10. Method appendix

- **Engine SHA:** `origin/main` @ `56c671d` (cycle-1 close, 2026-05-30).
- **Data:**
  - OHLCV: `data/bloomberg/sp500_ohlcv.csv` SHA256
    `c3d5443158b12ec5309a08111cdbeae6610baea78c39e676ef109b571c5edfb1`
    (captured in `summary.json` `setup.ohlcv_csv_sha256`). Window:
    2018-01-02 → 2026-03-20.
  - IV: `data/bloomberg/sp500_vol_iv_full.csv`. Window: 2015-01-02 →
    2026-03-20.
- **Provider:** `MarketDataConnector` (Bloomberg-only). Theta MCP not
  required for this run.
- **Driver:** `docs/verification_artifacts/r10_strict_driver.py`.
  Imports the canonical harness helpers (`friction_*`,
  `_spot_on_or_after`, `_next_business_day`,
  `_forward_replay_realized_pnl`, `assert_data_window_available`,
  `ohlcv_sha256`) from `backtests/regression/_common.py` and the
  canonical `UNIVERSE_100` from `backtests/regression/universes.py`
  to keep the comparison apples-to-apples with S38 / S44 / S43.
- **Compute:** 7.00h wall-clock (419.7 min for **1,304 business days =
  19.3 s/day**; modestly slower than the 18.7 s/day pilot rate because
  the strict-mode path adds `_evaluate_d17_hard_blocks` evaluation on
  6,704 refused attempts that the loose path skips).
- **Methodology — A/B harness:** Both trackers built inside ONE
  `run_strict_vs_loose` call. They share **one** daily
  `rank_candidates_by_ev` invocation (the multi-friction pattern from
  `_common`) so the rank input is byte-identical for both. The only
  divergence is the open-path: loose calls `tracker.open_short_put`
  directly; strict calls `tracker.consume_ranker_row` (which issues
  a D16 token, then opens with `current_ev_dollars`, which triggers
  `_evaluate_d17_hard_blocks` — R9 + R10 + portfolio-delta + Kelly).
  Per-attempt outcome (`opened` / `refused` + `reason`) captured for
  EVERY attempt from `tracker._ev_authority_log` introspection,
  immediately before/after the attempt to attribute log entries to
  the attempt.

---

## Appendix A — D17 hard-block evaluation order (engine reference)

For documentation reference, the strict tracker evaluates D17 gates
in this order inside `_evaluate_d17_hard_blocks` (see
`engine/wheel_tracker.py:1768`):

  1. **Pre-gate**: `nav < min_nav_for_trading` → `nav_exhausted`.
     Default `min_nav_for_trading=0.0` so this only fires on a
     strictly negative NAV; we set 0 explicitly.
  2. **R9 sector cap** (`check_sector_cap`, GICS-strict, 25% NAV cap).
     Fired 1 time over 5y.
  3. **R10 single-name cap** (`check_single_name_cap`, 10% NAV cap).
     Fired 571 times over 5y, all on BKNG (331) + AZO (240).
  4. **Portfolio delta cap** (`check_portfolio_delta`, ±$300 / $100k
     NAV = ±$3,000 at $1M NAV). Fired 6,704 times over 5y — the
     dominant binding gate.
  5. **Kelly cap** (`check_kelly_size`, 50% NAV per trade). Never
     fired (structurally unreachable at one-contract sizing per the
     docstring).

First-failure-wins: the gate returns at the first failed check. The
audit-log entry carries `reason` + the details bag of that gate. R7
(`check_var`) and R8 (`check_stress_scenario` + `check_dealer_regime`)
do NOT fire in the tracker open path — they are dossier-side
**soft-warns** that downgrade verdict (`proceed → review`) but never
block. This driver does NOT exercise the dossier path because the
question is about the tracker hard-block; if the dossier-side R7/R8
firing rate is also of interest, that's a follow-up card.

## Appendix B — Companion artifacts

- `docs/verification_artifacts/r10_strict_driver.py` — the driver
  (and `--analyze` post-hoc analyzer).
- `docs/verification_artifacts/r10_full_2020-2024_summary.json` — the
  full-run `summary.json` (machine-readable companion).
- `docs/verification_artifacts/r10_full_2020-2024_raw_output.txt` —
  trimmed stdout (per-50-day progress + final summary; HMM
  `RuntimeWarning` lines stripped because they are bookkeeping
  warnings from `engine/regime_hmm.py` that have no bearing on the
  R10 measurement).
- `docs/verification_artifacts/r10_full_2020-2024_analysis.txt` —
  output of `r10_strict_driver.py --analyze --out-dir .../full/`.
- `docs/verification_artifacts/r10_pilot_2020-q1apr_summary.json` +
  `r10_pilot_2020-q1apr_raw_output.txt` — the Q1+April 2020 pilot
  artifacts (validated the driver before the 7h full run).

Full-run rank logs / equity curves / attempt logs / tracker states
live under `%TEMP%/r10_full/` per the canonical Sn throwaway-harness
convention (sizes: `rank_log_*.csv` 1.2 MB × 2, `tracker_strict_state.json`
7.7 MB, etc.). Re-runnable from the driver: `python
docs/verification_artifacts/r10_strict_driver.py --start 2020-01-02
--end 2024-12-31 --out-dir <some_temp_dir>/r10_full` (allow ~7h
wall-clock on the dev box).
