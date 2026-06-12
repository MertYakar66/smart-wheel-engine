# SIM-200K — Eight One-Year Wheel Campaigns at $200,000 (Reliability Study)

**Run date:** 2026-06-11/12 (overnight autonomous campaign, operator-commissioned).
**Engine:** `main` @ `83eacdd` (post brain-audit fix wave: #407 D16 token binding,
#408 as_of-frontier staleness gate, #405 S47 defensive rolls — all merged).
**Driver:** `backtests/regression/sim200k_reliability.py` (thin wrapper over the
canonical `run_backtest_multi_friction`; S38/S43 lineage).
**Question under test:** if a $200k retail account had switched the engine on at
eight different past dates and traded its output for one year, what is the
*distribution* of outcomes — and is the engine's edge stable across entry
regimes, or an artifact of one lucky period?

---

## 1. Design

Eight one-year windows, each pretending a different past date was "day one" of
live trading. Chosen for regime diversity *before* seeing any results
(pre-registered in the driver; no window was added or dropped after results):

| ID | Start | End | Entry regime |
|---|---|---|---|
| w1_2020_crash_entry | 2020-02-03 | 2021-01-29 | Worst-case timing: first month runs into the COVID crash |
| w2_2020_recovery | 2020-06-01 | 2021-05-28 | Post-crash recovery, elevated IV |
| w3_2021_calm_bull | 2021-01-04 | 2021-12-31 | Calm grind-up bull, premium-thin |
| w4_2022_bear | 2022-01-03 | 2022-12-30 | Rate-shock bear year (adverse for short puts) |
| w5_2022_bottom_entry | 2022-10-03 | 2023-09-29 | Entry near the bear low, high IV |
| w6_2023_chop | 2023-07-03 | 2024-06-28 | Chop into renewed bull |
| w7_2024_late_cycle | 2024-06-03 | 2025-05-30 | Late-cycle bull |
| w8_2025_recent | 2025-03-03 | 2026-02-27 | Most recent full year inside the data frontier |

**Capital:** $200,000 per window (independent books).
**Strategy:** the engine's wheel — 35-DTE / 25-delta short cash-secured puts,
ranked by `EVEngine.evaluate` through `WheelRunner.rank_candidates_by_ev`
(the §2 authoritative path, no bypass), wheeled into covered calls on
assignment via `rank_covered_calls_by_ev`. Max 3 new positions/day, top-15
ranking depth, 1 contract per position, EV > 0 required to execute.
**Friction:** three levels per window — `none`, `bid_ask` (half-spread
`max($0.05, 8% of premium)`), `full` (spread + $0.65/contract commission +
assignment slippage `0.1% of notional + $0.65`). **All headline numbers in
this document are `full` friction** — the most adverse cost model.
**Universe:** `UNIVERSE_100` (canonical 100-ticker S&P subset).

## 2. No-lookahead protocol (the "do not cheat" contract)

1. Every ranking call passes `as_of = <simulated day>` — the engine's PIT
   machinery (as-of OHLCV/IV slicing, event-lockout calendar, staleness gate
   per #408) sees only data dated on or before the simulated day.
   (`_common.py` rank call sites; verified by independent adversarial review,
   §8.)
2. Expiration settlement consults the spot price of the expiry date only once
   the day-stepping loop has *reached* that date; the forward-replay of ranked
   rows is pure post-hoc evaluation and cannot influence trade selection.
3. Window ends are capped ≥ 35 calendar days inside the 2026-06-04 OHLCV
   frontier, so **every opened position settles on real data** — no
   mark-to-model tail in the realized P&L columns of the rank log (positions
   still open at window end are marked in NAV and counted in `open_at_end`).
4. Data inputs are fingerprinted: each window's `metrics.json` embeds SHA-256
   of every connector CSV, pinning exactly which data produced the result.
5. No parameter was tuned on any window's outcome. Knobs are the canonical
   S38/S43 set; the single change vs S43 is `capital=200_000` (vs $1M), fixed
   before the first run.
6. Windows were not re-run, swapped, or dropped after seeing results.

## 3. Honest-limitations register (read before the results)

- **Universe survivorship.** `UNIVERSE_100` is a fixed snapshot of *current*
  S&P members. A 2020 window therefore trades names known (to the universe
  constructor, not to the engine) to have survived to 2026. This inflates
  results vs a true PIT membership universe by an unquantified but real
  margin. Same property as every prior campaign (S38/S43); flagged as the
  price of canonical comparability.
- **Window overlap.** The eight windows overlap in places (e.g. w1/w2/w3 share
  parts of 2020-21). They are regime samples, not independent draws; treat the
  cross-window dispersion as indicative, not as eight i.i.d. observations.
- **Synthetic expiries.** Simulated contracts expire exactly 35 calendar days
  after entry (business-day adjusted), not on listed-chain Fridays. Consistent
  with how the EV was computed (`dte_target=35`), but real fills would land on
  exchange expiration dates.
- **No early assignment.** American-style early exercise (deep-ITM puts,
  ex-dividend call assignment) is not simulated on the put leg; positions run
  to expiry. Covered-call ex-div early-assignment IS modeled in the CC EV
  (D-series fix), but the put-side omission flatters path realism slightly.
- **Holiday calendar.** The day loop steps `pd.bdate_range` (weekday) dates;
  on exchange holidays the marks roll to the next session's close
  (`max_lookahead_days=1` marking convention — settlement of already-fixed
  obligations only, not decision input).
- **Caps-off canonical config.** `require_ev_authority=False`, R9 sector /
  R10 single-name caps not armed — the canonical S38/S43 configuration, kept
  for comparability. A post-hoc would-fire audit (§7) reports what the armed
  caps *would* have refused, using running-peak (not end-state) exposure.
- **Single seed, deterministic.** `seed=42`; the ranker is deterministic given
  data + as_of, so re-runs reproduce exactly. There is no Monte-Carlo
  dispersion *within* a window — dispersion comes from the eight windows.
- **Zero-skew IV.** The Bloomberg IV source carries `put_iv == call_iv`
  exactly (no smile). Per the D9 annotation (PR #410): short-put EV is
  *conservative* (25Δ put premium understated 13-41% vs real smile), CC EV
  optimistic ~6-12%. Net effect on the wheel: realized premiums in live
  trading would likely be *richer* than simulated on the put side.
- **$200k scale.** Cash-secured collateral (strike × 100) saturates a $200k
  book quickly; names with strike × 100 > available BP are naturally
  unaffordable. Trade counts are far below the $1M campaigns by construction.
- **Forward-distribution tier degradation on early windows.** At w1's start
  (2020-02-03) only ~525 OHLCV bars exist; the non-overlapping sampler needs
  more, so w1/early-w2 rank on the lower-quality `empirical_overlapping` tier
  (autocorrelated samples; Wilson CI on prob_profit suppressed there by
  design). Verified by direct probe; per-window tier histogram in §8. Windows
  from 2021 onward run on the full-quality tier.
- **External overlays inert (verified by probe).** Credit-regime (FRED) has
  no API key in this environment → deterministic `regime=unknown`,
  `credit_mult=1.0`; dealer positioning is dormant on the Bloomberg connector
  (no chain access); news sentiment store is empty → neutral. The results
  measure the core OHLCV/IV EV engine, not these overlays — and are exactly
  reproducible because no live network data entered the run.
- **Dateless-fundamentals leak (known, small, disclosed).** `get_fundamentals`
  is an undated 2026 snapshot; the ranker's BSM dividend-yield input and GICS
  sector come from it on every historical day (issue #354 lineage; the dated
  PIT fundamentals pull is Block A item A3). Direction of bias: mixed and
  small (dividend yields drift slowly); named here rather than buried.
- **Synthetic BSM premiums.** Premiums are Black-Scholes mids
  (`premium_source="synthetic_bsm"`), not real chain quotes; the friction
  overlay models spread/commission on top, but real fills would differ
  beyond the modeled half-spread.
- **D21 horizon over-statement.** The forward sampler indexes calendar DTE as
  trading bars (~46% over-long horizon → over-dispersion; deliberately not
  yet corrected — it rides the Block B coordinated re-baseline). Applies
  uniformly to all eight windows.

## 4. Headline results (full friction)

| Window | Period | Final NAV | Return | Max DD | Trades | Assign. | Hit rate | ρ (EV→realized) | EW B&H | Engine − B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| w1 crash entry | 2020-02-03 → 2021-01-29 | $204,259 | **+2.1%** | −34.4% | 95 | 17 | 87.4% | 0.52 | +13.2% | −11.1pp |
| w2 recovery | 2020-06-01 → 2021-05-28 | $228,297 | **+14.2%** | −5.9% | 96 | 9 | 94.5% | 0.50 | +45.0% | −30.9pp |
| w3 calm bull | 2021-01-04 → 2021-12-31 | $241,791 | **+20.9%** | −8.0% | 94 | 8 | 87.2% | 0.24 | +31.3% | −10.4pp |
| w4 bear | 2022-01-03 → 2022-12-30 | $193,597 | **−3.2%** | −15.6% | 57 | 20 | 73.9% | 0.35 | −12.1% | **+8.9pp** |
| w5 bottom entry | 2022-10-03 → 2023-09-29 | $227,741 | **+13.9%** | −8.1% | 103 | 17 | 79.4% | 0.34 | +18.4% | −4.5pp |
| w6 chop | 2023-07-03 → 2024-06-28 | $223,792 | **+11.9%** | −7.9% | 71 | 17 | 77.7% | 0.23 | +15.9% | −4.0pp |
| w7 late cycle | 2024-06-03 → 2025-05-30 | $220,314 | **+10.2%** | −15.5% | 88 | 17 | 80.6% | 0.36 | +13.1% | −2.9pp |
| w8 recent | 2025-03-03 → 2026-02-27 | $289,050 | **+44.5%** | −9.2% | 76 | 8 | 83.5% | 0.49 | +17.2% | **+27.4pp** |

EW B&H = equal-weight buy-and-hold of the same universe, computed on the
connector's TRUE close (the raw CSV's column labels are rotated one slot;
the first benchmark cut accidentally used the mislabeled raw column — the
adversarial verifier caught it (§8), and this table carries the corrected
basis; the verifier's independent w8 recompute agrees within 0.23pp).
All Spearman ρ values have p ≈ 0 (n ≈ 3,600 settled rank rows per window).
Every opened position in every window settled on real data (0 unsettled rows).
Friction sensitivity is small and non-monotone (±$1–7k across the three
levels — friction perturbs the cash path, which changes which trades fit
buying power, so books diverge; w4/w5/w8 full-friction NAV slightly exceeds
frictionless for this reason, not because costs help).

## 5. Outcome distribution & benchmark comparison

**Distribution of the eight one-year outcomes (full friction):**

- Positive: **7 of 8** windows. Mean **+14.3%**, median **+12.9%**,
  σ ≈ 14.3pp, range **−3.2% … +44.5%**.
- The single loss is the 2022 rate-shock bear (−3.2%) — the regime short
  puts are structurally worst in — and there the engine still **beat the
  equal-weight benchmark by +8.9pp** (EW B&H: −12.1%).
- Max drawdown: median ≈ −8.6%; worst −34.4% (w1: peak 2020-02-19 →
  trough 2020-03-23, the COVID crash itself — the simulated trough lands on
  the real market-bottom date. First month −15.4%; the book recovered to
  +2.1% by window end, but an operator would have sat through a one-third
  drawdown in month two of live trading).
- w8's path: the April-2025 vol event cost −9.2% peak-to-trough
  (2025-03-25 → 04-04), after which return accrued steadily (~3.4%/month
  average, only one negative month) — its +44.5% was premium-richness
  spread across the year, not a single lucky month.

**Against the same-universe equal-weight buy-and-hold:**

- Engine mean +14.3% vs EW mean +17.7% → the engine **lags passive by
  ~3.4pp per year on average across these windows**, beating it in 2 of 8
  (the bear year and w8).
- What the engine buys for that lag is downside truncation: worst engine
  year −3.2% vs worst EW year −12.1%; engine max-DD beats the underlying
  basket's in the two adverse windows.
- This is the documented wheel character (bear-alpha / bull-lag): short
  25Δ puts cap upside participation and monetize fear. The 2020 recovery
  window shows it most starkly (engine +14.2% vs EW +45.0% — premium
  collection cannot keep pace with a 45% melt-up).
- **w8's +27.4pp excess decomposes as path-dependent equity beta, not
  premium edge (§8 forensics):** of the +$89,050 gain, ~$73,448 (76%) is
  unrealized appreciation in three names the wheel was assigned near the
  April-2025 bottom (CAT $309.5→$742.8 = +$43.3k; AMAT +$23.1k; CEG
  +$12.1k) and then held as stock through the rally. Closed-cycle net
  P&L after costs was +$16,989; w8's IV mean (0.32) is mid-pack. The
  wheel "bought the dip" mechanically and got paid for it — real, but
  not a repeatable harvest claim.

**Reliability claim supported by this campaign:** at $200k, full friction,
across eight regime-diverse entry points, the engine never produced a
catastrophic year, was positive in seven, and its single negative year was
shallower than the market's. The claim it does NOT support: beating
buy-and-hold in sustained bull markets.

## 6. Calibration — did the engine's numbers mean what they said?

Computed over **all settled ranked rows** (≈3,600/window — including rows
never executed), not just the book, so this measures the engine's scoring,
not execution luck. Full per-window tables in
`%TEMP%\sim200k_backtest\campaign_table.md`.

**Ranking signal (Spearman ρ, predicted EV vs realized P&L):** positive in
all eight windows, 0.23–0.52, p ≈ 0 throughout. The EV ordering carries
real information in every regime tested.

**Top prob_profit bin (0.9–1.0, midpoint 95%) — realized win rate:**

| | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 |
|---|---|---|---|---|---|---|---|---|
| realized | 88.0% | 93.7% | 88.1% | 77.6% | 84.5% | 76.2% | 78.0% | 85.7% |
| gap | −7.0pp | −1.3pp | −6.9pp | −17.4pp | −10.5pp | −18.8pp | −17.0pp | −9.3pp |

The top-bin **over-confidence is confirmed in all eight windows**
(−1.3 to −18.8pp, worst in bear/chop regimes) — exactly the known
calibration defect (IBKR Phase-3, heavy-verify I1-I5) whose recalibration
rides Block B. Mid bins (0.7–0.9) are much closer to honest.

**EV levels by regime:** in calm/recovery windows the EV-decile table is
broadly monotone (w2 textbook: every decile's realized P&L rises with
predicted EV). In the 2022 bear the engine's *ranking* still worked
(ρ=0.35) but its *levels* were over-optimistic — every positive-EV decile
realized negative on average. The forward distribution lags regime
deterioration; documented limitation, not new.

**Two notable distortions, disclosed:**
- *Bottom-decile pessimism:* in 6 of 8 windows the most-negative-EV decile
  realized **positive** mean P&L — the engine is too pessimistic at the
  bottom. Harmless to the book (those rows are never executed) but it says
  EV levels, not just the top bin, need the Block-B recalibration.
- *w8 top-decile "inversion" is a data artifact, not a calibration
  finding:* w8's highest-EV decile shows mean realized **−$2,505** against
  predicted +$858 — the verifier traced this to the **2026-03-23
  BKNG/CVNA unadjusted-split seam** in the OHLCV CSV (BKNG 4286.81→175.87
  ×0.041, CVNA ×0.212, overnight on ~40× volume — the only two such seams
  since 2020). Four late-February BKNG rank rows expiring after the seam
  settle against the post-split price vs ~$3,900 strikes → −$1.46M of
  *fake hypothetical* rank-log P&L. **Neither name was ever executed into
  any window's book** (collateral alone excluded BKNG at $200k), so zero
  NAV impact; cleaned of the seam, w8's all-rows mean realized is +$124
  (reported: −$277) and ρ improves to 0.497. Logged as a data-layer fix
  item for the Block-A session (split-adjustment continuity at the
  reconstitution seam).

## 7. Concentration audit (R9/R10 would-fire, post-hoc)

Running-peak exposure (not end-state) reconstructed from the daily open-
position timeline vs the daily portfolio value:

| Window | Max single-name | R10 (10%) would-fire days | Max sector | R9 (25%) would-fire days |
|---|---|---|---|---|
| w1 | 30.2% | 249 | 41.0% | 136 |
| w2 | **53.8%** | 201 | 58.6% | 144 |
| w3 | 23.6% | 169 | 53.5% | 72 |
| w4 | 25.3% | 126 | 62.4% | 102 |
| w5 | 26.8% | 203 | 49.1% | 84 |
| w6 | 23.2% | 156 | 53.8% | 87 |
| w7 | 26.1% | 239 | 65.0% | 152 |
| w8 | 26.4% | 207 | 55.9% | 169 |

**Read this table before trusting §4.** At $200k with `contracts=1`, a
single ~$50-strike position is 2.5% of NAV but a ~$400 strike is 20% —
the caps-off canonical book ran single names to 23–54% of NAV and sectors
to 41–65%, and the armed R10 (10%) would have refused opens on 126–249
days per window. A production-armed book (`make_live_book_tracker`) would
have been **materially smaller and differently composed** — almost
certainly lower returns AND lower idiosyncratic risk. These results
characterize the engine's ranking + the canonical harness, not the armed
production configuration. (Sector-map gaps: BK in four windows and
"CBOE UF" in two fell to the Unknown bucket — the R9 numbers slightly
understate those windows' true sector concentration.)

## 8. Independent no-lookahead verification

Two independent adversarial passes (Opus-class reviewer, instructed to
refute), one before accepting any results and one on the finished
artifacts.

**Pre-run design review — APPROVE_WITH_CHANGES.** Traced the harness day
loop and the engine internals end to end with file:line evidence: every
rank call passes `as_of=today`; OHLCV, IV history, HMM regime fit, and all
four forward-distribution samplers slice `≤ as_of`; `EVEngine.evaluate` is
a pure function; the forward-replay runs after the loop and cannot feed
back into selection; mark-to-market's next-day fallback touches only the
equity curve, never a decision input. Verdict: *no lookahead influences
trade selection*. Its three required changes (tier disclosure, overlay
neutralization proof, dateless-fundamentals caveat) are §3 items.

**Post-run artifact verification — CONFIRMED_WITH_NOTES.**

| Check | Result |
|---|---|
| Trade re-derivation (5/window × 8, deterministic sampling incl. largest \|P&L\|) | **40/40 exact** — independently fetched closes match `spot_at_expiry` to the cent; P&L recomputes from the formula; all DTE=35; no fabricated prices |
| NAV reconciliation | **8/8 exact** — equity-curve final = metrics `final_nav`; tracker cash = `final_cash`; NAV−cash explained by open marks |
| Determinism | **30/30 exact** — re-running `rank_candidates_by_ev` at w4 2022-06-15 and w8 2025-09-15 reproduces the logged rows to the cent |
| §2 invariants | 0 non-finite EVs in 24 rank logs; **0/680** executed trades without a same-day positive-EV ranked row; no buying-power overdraw (min cash $5,039, w1 March 2020) |
| Data frontier | max expiration 2026-04-03 ≤ 2026-06-04; 0 unsettled rows |
| Forward-distribution tier probe | w1 ranked on degraded `empirical_overlapping` at start AND mid-window; w2 at start only; w3–w8 effectively all `empirical_non_overlapping` |

**Notes that survived verification (both folded into this doc):** the
2026-03-23 BKNG/CVNA split seam corrupting w8's rank-log tail (§6 — zero
book impact), and the benchmark price-column basis error (§4 table now
carries the corrected true-close benchmark). The verifier also confirmed
the harness's settlement column is the *true* close after the connector's
column-rotation repair — i.e., the engine settled on the right prices all
along; only the first benchmark cut used the wrong raw column.

## 9. Realism check

| Aspect | Engine (this campaign) | Reality | Verdict |
|---|---|---|---|
| Premiums | Synthetic BSM mid at zero-skew IV | Real 25Δ put quotes carry skew — live premium 13–41% richer (D9 annotation) | Conservative on the put leg |
| Fills | Mid − `max($0.05, 8%)` half-spread + $0.65/ct | Retail wheel fills near mid on liquid names | Comparable or conservative |
| Expiries | Synthetic 35-calendar-day dates | Listed Fridays/monthlies | Minor structural difference |
| Assignment | European, at expiry close | American; early assignment on deep-ITM/ex-div | Flatters path slightly |
| Book concentration | Caps off (canonical) — single names to 54% NAV | Armed R10 caps at 10% | **Unrealistic vs production config** (§7) |
| Affordable universe | BP check only — book skews to low-priced names | Same in reality at $200k cash-secured | Realistic |
| Capital | $200k, no margin relief (cash-secured) | Reg-T short puts need less margin | Conservative |
| Position P&L | Held to expiry, no management | S47 finding: management surface is operator's job | Matches engine scope |
| Universe | 2026-member snapshot | PIT membership would include delisted losers | **Flatters results** (disclosed) |
| Data | Split-adjusted Bloomberg EOD, SHA-pinned | — | Sound |

Net read: cost/premium assumptions lean conservative; survivorship and
caps-off concentration lean generous. The two biases pull in opposite
directions and neither is quantified here — treat absolute return levels
with a band of uncertainty, and the cross-regime *shape* (positive skew,
bear-alpha, bull-lag, no catastrophic year) as the robust finding.

## 10. Verdict

**The engine is reliable in the sense the operator asked about: switched
on at eight different past dates with $200k, it produced a positive year
seven times out of eight, never a catastrophic one, and its single loss
(−3.2%, 2022 bear) was ~9pp shallower than holding the same universe.**
Results are verified honest: PIT-clean selection (two independent
adversarial passes), exact trade-level re-derivation from raw data,
reproducible to the cent, zero §2 violations.

What this campaign establishes:

1. **Positive-outcome reliability at retail scale** — mean +14.3%/yr,
   median +12.9%, range −3.2% … +44.5% (full friction).
2. **The edge shape, not generic alpha** — bear-alpha (+8.9pp in 2022),
   bull-lag (−3 to −31pp in up markets), downside truncation (worst year
   −3.2% vs market's −12.1%). Anyone wanting to beat buy-and-hold in a
   melt-up should not run this strategy.
3. **The ranking signal generalizes** — ρ(EV→realized) positive in all
   eight regimes (0.23–0.52, p≈0, ~3,600 rows each).
4. **The known calibration defect generalizes too** — the 0.9–1.0
   prob_profit bin over-states by 1–19pp in ALL eight windows (worst in
   adverse regimes). This is now an 8-window mandate for the Block-B
   recalibration, with the same R11-VIX-gate guardrail as before.
5. **Production caveats that matter at $200k** — caps-off concentration
   (single names to 54% NAV; armed R10 would have intervened on 126–249
   days/window) and survivorship universe both flatter the absolute
   levels; zero-skew premiums and cash-secured sizing lean conservative.
   The cross-regime *shape* is the robust claim; the absolute levels
   carry a band.

Follow-ups routed to existing queues: BKNG/CVNA split-seam repair →
Block A (data session); recalibration evidence → Block B (already
scoped); optional armed-caps re-run of this campaign → operator's call
(cheap: ~2h compute).

---

*Artifacts: `%TEMP%\sim200k_backtest\<window_id>\{none,bid_ask,full}\
{rank_log.csv, metrics.json, tracker_state.json}` + `summary.json` +
`analysis.json` per window; `campaign_table.md` at the root. Analysis tool:
`scripts/sim200k_analysis.py`.*
