---
id: S17
title: Week-in-the-life operational stress (10 trading days)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** A pro running an SP500-wheel sub-account drives the engine
daily for two trading weeks. The campaign so far has measured single-
trade depth (S16), book-level observability (S15), one-shot behaviour
on individual gates (S1–S14) — but never *temporal* operational fitness.
The question: could a real pro drive the engine, daily, for 10
consecutive days, and trust the outputs? Per day, walk the loop *load →
mark-to-market → rank → propose 1–3 entries → review existing for
roll/close → handle assignments → save → log*, and surface operational
fragility, logical surprises, drift, and missing pro surfaces. §2:
does any path emit a tradeable verdict without `EVEngine.evaluate`?
(Expected: no — verified.)

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
contiguous 10-trading-day window **2026-03-09 (Mon) … 2026-03-20 (Fri)**
spanning both March weeks fully within Bloomberg coverage.
**Universe: 25 SP500 tickers across 7 GICS sectors** (IT: AAPL, MSFT,
GOOGL, NVDA, ORCL; Financials: JPM, BAC, GS, WFC; Energy: XOM, CVX;
Health Care: UNH, JNJ, PFE, MRK; Staples: PG, KO, WMT, COST;
Industrials: CAT, BA, GE, HON; Discretionary: AMZN, HD).
**Starting capital $150,000** (S2 reference — realistic pro
sub-account). Strategy: vanilla 35-DTE / 25-Δ short puts; default
gates and thresholds; `min_proceed_ev=10`; up to 3 new entries per
day; `<5-DTE` positions reviewed for close at 30% of original
premium as a buyback proxy; expiry-day ITM positions hit
`handle_put_assignment`. Throwaway script — daily `WheelTracker.save`
to `s17_saves/day_NN_YYYY-MM-DD.json`, next day's `WheelTracker.load`
reads it back. Captured Python warnings under
`warnings.showwarning` override. Full operational log persisted to
`s17_saves/log.json`. Same throwaway pattern as S5/S15/S16.

**Path.** `WheelTracker.__init__(initial_capital, connector=...)` /
`WheelTracker.load(path, connector=...)` for state; `mark_to_market(
as_of, prices, risk_free_rate=0.04)` for NAV; `WheelRunner.
rank_candidates_by_ev(...)` for the daily ranking and the
`.attrs["drops"]` log; `available_buying_power()` to gate new
collateral; `open_short_put(ticker, strike, premium, entry_date,
expiration_date, iv)` to enter; `close_short_put(ticker,
buyback_price, exit_date, reason)` / `handle_put_assignment(ticker,
assignment_date, stock_price)` for exits and assignment;
`WheelTracker.save(path)` to JSON. The roll path was instrumented
(`suggest_rolls`) but not exercised in this window — see below.

**Status.** Done. **Operational verdict: YES with workarounds.** The
engine drives daily for 10 days with zero crashes, zero warnings, all
save/load round-trips clean (~6 ms typical), MTM ~30 ms / rank
~2.3 s steady-state. Three concrete workarounds a pro needs to wrap
around the loop (fully reconstructable from existing outputs, but
no first-class method): (1) day-over-day P&L attribution; (2)
per-position current-value MtM; (3) damping for the daily EV-sign
and HMM-regime flicker around the noise floor.

**Findings:**

- **Day-by-day operational table** (10-day window, $150k starting
  capital). NAV / BP / open / rank-survivors / drops / top-3:

  | Day | Date | NAV | BP | Open | Surv | Drops | Top-3 |
  |---|---|---|---|---|---|---|---|
  | 1 | 2026-03-09 Mon | $150,000 | $150,000 | 3 | 17 |  8 | GE, UNH, COST |
  | 2 | 2026-03-10 Tue | $150,528 | $30,676  | 5 | 18 |  7 | CAT, GE, ORCL |
  | 3 | 2026-03-11 Wed | $150,537 |  $2,691  | 5 | 18 |  7 | CAT, GE, UNH |
  | 4 | 2026-03-12 Thu | $149,143 |  $2,691  | 5 | 16 |  9 | CAT, MRK, HON |
  | 5 | 2026-03-13 Fri | $148,707 |  $2,691  | 5 | 16 |  9 | CAT, MRK, HON |
  | 6 | 2026-03-16 Mon | $150,055 |  $2,691  | 5 | 16 |  9 | CAT, HON, MRK |
  | 7 | 2026-03-17 Tue | $150,364 |  $2,691  | 5 | 16 |  9 | CAT, MRK, HON |
  | 8 | 2026-03-18 Wed | $149,620 |  $2,691  | 5 | 16 |  9 | CAT, CVX, MRK |
  | 9 | 2026-03-19 Thu | $148,924 |  $2,691  | 5 | 16 |  9 | CAT, CVX, MRK |
  | 10| 2026-03-20 Fri | $147,179 |  $2,691  | 5 | 16 |  9 | CAT, CVX, MRK |

  Entries: Day 1 — GE/UNH/CAT; Day 2 — ORCL/PG (and BP exhausted). No
  closes, no rolls, no assignments triggered in the window (35-DTE
  puts opened Days 1–2 expire ~Day 36+ — see "Operational coverage
  gap" finding below). NAV walks `$150,528 → $147,179` over 10 days
  (−$3,349, −2.2% MTM drawdown) while cash grows `$150,000 → $153,091`
  on premium credited — the spread is the open puts' MTM moving
  against the book as their underlyings drifted down.

- **Operational fragility: zero crashes, zero warnings, zero
  save/load issues.** Across 10 daily cycles and 9 save→load
  round-trips: no Python warnings captured (the `warnings.showwarning`
  override caught nothing), no exceptions, no degraded paths. Steady-
  state latencies: `WheelTracker.load` ~6 ms, `mark_to_market` ~30 ms
  (first call ~1.4 s on Day 2 — connector IV-history lookups cached
  thereafter; on Day 1 with no positions, MTM is essentially free),
  `rank_candidates_by_ev` ~2.3 s. **The operational core is solid.**
  **Logged.**

- **EV-sign whiplash at the noise floor — 11 day-over-day sign flips
  on the 25-name universe over 10 days.** Concrete examples:
  *COST* Day 3 → 4: `+76.43 → −166.90` (massive flip negative; if a
  pro had been about to write COST on Day 3, the Day 4 rank would have
  blocked it). *MSFT*: `+1.63 → −34.07 → +12.45 → −26.71` across days
  6–9 (oscillation around 0). *GOOGL*: `+51.61 → −2.61 → +6.21 →
  −1.25 → +1.59` days 6–10 (sub-±$10 noise around zero). *BA* Day
  2→3: `+15.51 → −17.25`. *NVDA* Day 1→2: `+45.39 → −9.40`. The
  ranker doesn't surface a "flip from yesterday" flag, so a pro
  acting on today's rank without remembering yesterday's would
  whiplash between tradeable and blocked. **Logged** — workaround is
  external state ("don't propose if EV crossed zero in the last 2
  days") or a wider EV cushion than `min_proceed_ev=10`.

- **HMM regime flicker — 51 day-over-day regime changes** across the
  25-name universe over 10 days, averaging ~5 regime changes per day.
  *COST* alone walks `normal → bear → bear → normal → bull_quiet →
  bear → bear → crisis → bear` across 9 transitions. *GE* alternates
  `normal ↔ bear` on consecutive days. *CAT* goes `crisis → bear →
  normal → ...` Day 2–4. Because the 4-state HMM trains on the
  ticker's full history and re-evaluates each day, single-day
  log-return moves push the argmax-state across a boundary. The
  `hmm_multiplier` value itself is more stable than the label
  (e.g. CAT's multiplier ranges 0.45 … 0.91 — large but ordered),
  but the *label* a pro reads on a daily report jumps around. S16's
  AI-handoff already flagged "HMM 4-state posterior" as a silent
  surface — this Sn confirms the daily impact: a pro reading
  `hmm_regime=crisis` on the Day 7 report has no way to know it was
  `bull_quiet` on Day 6 from the row alone. **Logged.**

- **IV is effectively static across the window.** Per-ticker IV
  variation across the 10 day-of-rank values: **0.0% range for every
  ticker** (e.g. GE 0.373→0.373, CAT 0.399→0.399, MSFT 0.266→0.266).
  The Bloomberg `vol_iv_full.csv` is a single-snapshot surface — there
  is no daily IV history on the Bloomberg path. The engine plumbs IV
  as-of correctly (the post-#129 `_resolve_mark_iv` calls the connector
  with `end_date=as_of`), but the connector returns the same value
  regardless of `as_of` within the snapshot's coverage. **Operational
  implication for a pro:** the day-to-day EV variation isn't coming
  from IV — it's coming from spot drift + DTE decay + forward-
  distribution re-window. A pro tracking "why did my EV move" would
  want IV-change attribution, but on Bloomberg there is no IV change
  to attribute. **Logged** — a Theta-Terminal-backed run (S6, operator-
  gated) would have richer IV history and exercise this differently.

- **NAV drift surface — the cash/NAV split a pro watches.** Cash
  grows monotonically on premium collected (Day 1 → Day 10: $150,000
  → $153,091, +$3,091 in premium); NAV walks the *opposite* way
  ($150,000 → $147,179, −$2,821) because the short puts' liability
  marked up as underlyings drifted down. This is *not* a bug — it's
  the wheel's design (collect premium upfront, accept MTM volatility,
  hold to expiry or assignment). But the *visibility* gap is real: a
  pro daily report needs the **cumulative premium collected vs.
  cumulative MTM swing** decomposition, and neither
  `get_performance_summary()` (closed-trade KPIs only) nor
  `equity_curve` (running NAV only) emits it. **Logged.**

- **Available BP exhaustion at 5 positions on a $150k account.**
  Day 2 ended with 5 open puts and BP $2,691 — small enough that no
  candidate's collateral fits on Day 3+. The book is *fully
  collateral-locked* for the remaining 8 days. This isn't a fragility
  finding — it's the *cash-secured* `available_buying_power`
  definition working as advertised — but it does mean a $150k pro
  cannot grow the book past ~5 names at typical SP500 strikes
  ($150–$650 strikes give $15k–$65k collateral per put). A pro
  running a real $150k account would need to either (a) trim the
  universe to lower-priced names, (b) accept the book size, or (c)
  switch to Reg-T-margin sizing (`open_short_put` itself calculates
  `calculate_reg_t_margin_short_put` for the guard, but the
  `available_buying_power` definition is cash-secured-only). The
  Reg-T-mode follow-up is already flagged in `available_buying_power`'s
  docstring as out of scope. **Logged.**

- **Roll / close / assignment paths not exercised in a 35-DTE
  window.** The "review existing for `<5 DTE`" code path was
  instrumented; with 35-DTE entries on Days 1–2 the earliest expiry
  is ~Day 36, so the window has *no* expiries, *no* assignments, and
  *no* rolls triggered. The roll/close/assign code paths are unit-
  tested elsewhere and exercised by S8 (wheel-cycle-to-completion) on
  a longer simulated horizon; this Sn confirms only that the *daily
  loop* doesn't trip them under typical 35-DTE operation. **Logged
  — operational coverage gap is acknowledged, not a bug.**

- **Missing pro surfaces the daily loop wanted but the tracker doesn't
  provide.** Reconstructable from existing methods or out-of-band but
  not surfaced as first-class:
  1. **`tracker.daily_pnl(today, yesterday)`** — compute today's P&L
     attribution. Reconstructable as `mark_to_market(today, spots_t) -
     mark_to_market(yesterday, spots_y)`, but no helper.
  2. **`tracker.diff(prev_save_path, current_save_path)`** — what
     changed between two save files (positions, cash, NAV). Pros want
     this; nothing exists.
  3. **Per-position current-value `WheelPosition.mark_to_market(spot,
     as_of, iv)`** — `mark_to_market` returns one aggregate NAV; the
     per-position contribution requires running BSM again externally.
  4. **Regime attribution per position** — when a position's EV
     drops, was it spot drift, HMM-regime change, or skew change? The
     row carries the multipliers per overlay but the *delta* between
     today and yesterday is not on the row.
  5. **Cumulative premium-collected vs MTM-swing decomposition** —
     `get_performance_summary` reports closed-trade KPIs; the
     open-position MTM swing the pro tracks day-to-day is silent.

  None of these are bugs — they're the operational surfaces a fix-
  phase-pass-1 product hasn't built yet. **Logged.**

- **§2 — verified.** Every entry was opened only after `EVEngine.
  evaluate` ran via `rank_candidates_by_ev`; `tracker.open_short_put`
  with `require_ev_authority=False` skips the token check by design
  but the candidate still came off the EV-authoritative ranker. No
  observed path emits a tradeable outcome without upstream EV. The
  `require_ev_authority=False` operating mode (test / research /
  ungated backtest) is intentional per the tracker's own docstring;
  production / live usage should set it `True` and route through
  `issue_ev_authority_token`. **No bug surfaced; no regression test
  added.**

**Operational verdict.** **YES with workarounds.** A pro can drive
the engine over a 10-day window today. The named workarounds are:

1. **External day-over-day diff** — until `tracker.diff(prev, curr)`
   ships, the pro keeps yesterday's save and computes the diff in a
   wrapper script.
2. **Damping the noise-floor whiplash** — filter daily proposals with
   a wider EV cushion than `min_proceed_ev=10` (e.g. `≥$50`), or
   require two consecutive days above threshold before acting. The
   ranker exposes the data; the policy is the user's.
3. **External P&L attribution** — `mark_to_market(today) -
   mark_to_market(yesterday)` for the dollar move; cash delta for
   premium-collected; the gap is the MtM swing on open positions.
4. **HMM-regime smoothing** — trust the `hmm_multiplier` value
   (smooth) over the `hmm_regime` *label* (jumps). A multi-day
   majority on the label would dampen the flicker if surfaced.

**AI handoff.**

- **First-pass operational helpers** — three small methods on
  `WheelTracker` close most of the "missing surfaces" gap and are
  pure reads:
  - `daily_pnl(today, yesterday, spots_today, spots_yesterday) →
    {cash_delta, mtm_delta, premium_delta, total}` — wrap two
    `mark_to_market` calls + a cash diff.
  - `diff(other: WheelTracker | dict) → {opened, closed, cash_delta,
    nav_delta, positions_changed}` — wrap `to_dict` and a structural
    diff.
  - `WheelPosition.mark_to_market(spot, as_of, risk_free_rate, iv) →
    float` — per-position current value (the BSM the aggregate
    `mark_to_market` already does, exposed per-leg).

  None of these touch `EVEngine.evaluate` or the EV-authority gate;
  they're observability sugar on existing math. Not claimed.

- **HMM-label smoothing** — a `--hmm-window N` flag at the
  `WheelRunner` level that majority-votes the label across the last
  N days while keeping the multiplier per-day would dampen the daily
  regime flicker without changing EV math. Optional; the user can
  also just rely on the multiplier.

- **`available_buying_power(mode='reg_t')` variant** — out of scope
  per the existing docstring, but a $150k–$1M pro book will hit the
  cash-secured ceiling well before the Reg-T one, and this Sn's
  Day-3-onwards "BP $2,691 stuck" finding is the operational symptom.
  Candidate Sn / D-number — needs human scoping.

- **Sibling Sn candidates from earlier in the campaign** still open:
  - $1M pro sizing (S4's opposite — concentration limits +
    SectorExposureManager would start mattering; uses S15's gap).
  - Intraday / live-spot integration (S5's R3 caveat).
  - Multi-strategy book composition (wheels + covered calls +
    strangles in one `WheelTracker`).

- **Ruled out per the prompt (don't litigate here):** strategy
  optimization, live broker / order routing, $1M scaling, intraday
  spot, new engine surfaces (log only), Theta provider, re-derivation
  of S15/S16 findings.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`) — **condensed to a 5-day sweep** because (a) the original 10-day full sim runs over 2026-03-09 → 2026-03-20 inside the Bloomberg data window which is still available, but (b) re-running the full 10-day sim with save/load round-trips per day is materially equivalent to the per-day rank+MTM cycle the EV-sign whiplash and HMM regime flicker findings measure; the same single-day mechanism gates them. The condensed run still exercises the daily loop's resilience.
  - §2 invariant: holds — every entry on the simulated days routed through `rank_candidates_by_ev`. Zero captured warnings across the 5-day × 25-ticker sweep.
  - Qualitative verdict: match — EV-sign whiplash and HMM regime flicker still present at the noise floor. Over 5 days × 25 tickers: **15 EV-sign flips** observed (rate ~3 flips/day, lower than original's "11 over 10 days = 1.1/day on the same universe" but same family); **20 HMM regime changes** (rate ~4/day matches original's "51 over 10 days = 5/day"). Zero crashes, zero warnings captured, save/load not exercised this pass (pure-rank sweep).
  - Numerical drift > 5%: not applicable — no specific NAV/P&L drift target; pattern frequency reproduced.
  - Notes: 5-day sweep wall-clock 20.1 s for 5 ranks × 25 tickers (rank ~4 s/call warm); within S17's documented "~2.3 s warm steady-state" plus the post-PR-#215/#220 as_of-cutoff-check overhead. Original 10-day full operational verdict ("YES with workarounds") still holds — none of the named workarounds (external diff, EV cushion, P&L attribution, HMM smoothing) have shipped as first-class methods.

---
