---
id: S4
title: Account-size-constrained book selection
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Force a return-on-capital lens by setting a realistic
small account ($50k retail) as a hard constraint, then try to build a
book from the ranking. Exercises the no-account-size / no-ROC gap S1
logged.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, offline charts, 36-name
diversified watchlist (11 GICS sectors, mega-caps through mid-caps),
`as_of=2026-03-20`, 35-DTE / 25-delta puts, `contracts=1`,
`include_diagnostic_fields=True`. $50k hard cap; each cash-secured put
reserves `strike × 100` collateral. The book is walked top-down off
`WheelRunner.rank_candidates_by_ev` output.

**Status.** Done. 30 of 36 names ranked, 20 of them positive-EV; the
ranker's top three by `ev_per_day` (FIX, MCK, BKNG) each demand more
collateral than the entire $50k account ($122.2k / $84.1k / $401.4k).
No bug — every figure computes correctly; S4 surfaces missing
*capability*, not a defect. All findings logged; no code change (a
usage test's deliverable is the writeup). The top-10 names by ROC —
CF, FIX, FDS, AJG, EXE, JBHT, EG, MCK, HUM, BR — are the sample S7
fed to the advisor committee.

**Findings:**

- **No account-size input — the ranking is capital-blind.**
  `rank_candidates_by_ev` (`wheel_runner.py:448`) takes `contracts`,
  `top_n`, `min_ev_dollars` — nothing for account size or a collateral
  budget. It returns an identical ranking for a $50k account and a $5M
  one. Optimizing `ev_per_day` with no capital term, it front-loads the
  most expensive names: a strict top-down walk (stop at the first
  unaffordable pick) opens **zero** positions and strands all $50k; the
  first candidate that fits at all is the ranker's #4. Nothing in the
  output flags that the top names are unbuyable. **Logged.** (the core
  S1 gap, now exercised.)

- **No return-on-capital column.** Output carries `ev_dollars`,
  `ev_per_day`, `prob_profit`, CVaR, Omega, `edge_vs_fair` and the
  dealer/skew/HMM diagnostics — but nothing dividing EV by collateral.
  ROC (`ev_dollars / (strike × 100)`) had to be hand-computed.
  Re-ranking by ROC reorders the book sharply: CF #8 → #1, EXE #12 →
  #5, against BKNG #3 → #18, MCK #2 → #8, CAT #7 → #12. The ranker's
  order and the capital-efficiency order disagree most on exactly the
  names a $50k trader must decide between. **Logged.**

- **ROC ordering beats ranker ordering on the same $50k.** A greedy
  fill down `ev_per_day` order fits 2 names (EG + FDS — $48.75k
  collateral, $666 EV). The same fill down ROC order fits 4 (CF + FDS +
  EXE + KO — $47.35k, $722 EV): +8.4 % EV, twice the positions (genuine
  diversification vs a two-name book), and less capital committed. The
  absolute-EV lens leaves both money and diversification on the table
  for a constrained account. **Logged.**

- **No buying-power / book-builder helper; `contracts` is global.**
  The collateral math, the budget walk and the ROC re-sort were all
  done by hand. S2 logged the workaround `available = cash −
  Σ(strike × 100)`; S4 confirms there is still no
  `available_buying_power()` and no "fit a book under budget X" helper
  for what is really a knapsack problem. `contracts` is one
  ranker-wide argument, not per-candidate — "as many contracts of the
  cheap names as the budget allows" cannot be expressed. **Logged.**

- **No concentration guard; $50k is structurally forced to
  concentrate.** The ranker-order book put 97.5 % of the account into
  2 names (EG alone = 60 %). S2 logged "max 25 % per name" as a
  trader-imposed rule the engine does not enforce — and under $50k that
  rule caps collateral at $12.5k/name, i.e. `strike ≤ $125`. Only 3 of
  the 20 positive-EV names (CF, EXE, KO) clear it at one contract. A
  properly diversified $50k wheel book is barely buildable from S&P
  names at current share prices, and the engine surfaces none of it.
  **Logged.**

- **`ev_per_day` is EV over the *effective* hold, not the 35-DTE
  nominal.** `ev_per_day = ev_dollars / expected_days_held`
  (`ev_engine.py:506`); `expected_days_held` is the
  probability-weighted blend profit→`dte/2`, stop→`dte/3`, hold→`dte`
  (`ev_engine.py:444`). For these high-prob-profit names the effective
  hold is ~17–19.5 days, not 35 — so `ev_per_day ≈ ev_dollars / ~18`,
  and `ev_per_day` order is *close to but not identical* to
  `ev_dollars` order (AJG outranks FDS on EV yet trails it on
  `ev_per_day`, its effective hold being longer). Neither metric
  carries a capital term; both front-load expensive names. Recorded so
  the ROC contrast is read against the metric the ranker actually
  sorts by — and so a future S4 reader does not mistake `ev_per_day`
  for `ev_dollars / 35`. **Logged.**

- **Silent drops recur, and bite harder under the constraint.** 6 of
  36 names returned nothing (NFLX, JPM, UNH, JNJ, GE, XOM); re-running
  with `use_event_gate=False` returns all 6 with `days_to_earnings`
  18–32 — earnings inside the option's life, correctly event-gated, 5
  of them otherwise positive-EV. Correct behavior, but invisible: the
  trader sees a 30-name list with no signal that 6 candidates were
  removed or why. Same as S1 / S2. **Logged.**

**Follow-up.** Shipped in #109 — `collateral` / `roc` columns on the
`rank_candidates_by_ev` output and `WheelRunner.select_book(...)`, an
account-aware skip-and-fill book selector. Both consume and subset
post-`EVEngine.evaluate` ranker output — neither rescues a candidate
nor bypasses the EV authority (§2-safe, pinned by a zero-evaluate-call
regression test). Deferred: per-name multi-contract sizing.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `select_book` invokes `rank_candidates_by_ev` internally and filters by collateral/ROC; no candidate bypasses `EVEngine.evaluate`.
  - Qualitative verdict: match — `collateral` + `roc` columns present on ranker output; `WheelRunner.select_book(account_size, tickers=..., as_of=..., max_weight_per_name=0.25, min_roc=0.0)` exists with the published signature. On a 36-name watchlist at `as_of=2026-03-20` it returns 27 rows / 21 positive-EV; with $50k account + 25% per-name cap it selects 3 names (CF $11,400 / EXE $10,000 / KO $7,150) — closer to S4's documented "ROC-ordered greedy fill" outcome than the absolute-EV book.
  - Numerical drift > 5%: original S4 quoted "30 of 36 names ranked, 20 positive-EV" — current run gives 27 / 21. Row count delta −10% (30→27); positive-EV delta +5% (20→21). Likely attributable to the post-S4 universe data refresh (some 2026-Q1-earnings names now dropped by event gate) + post-IV-PIT-fix EV recalibration (PR #179). Not attributable to one single PR — composite of #119/#121/#179/#208/#210/#220.
  - Notes: `select_book` parameter name is `max_weight_per_name` (not `max_per_ticker_pct` as my first probe used) — pure-API note; behaves as documented.
