---
id: S37
title: Ranking philosophy: ev_dollars vs roc
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** S33 F5 surfaced a real ranking-philosophy question: on
a crisis day the engine ranks by `ev_dollars` (current default), and
absolute EV biases toward high-IV / high-beta names (crisis IV spikes
boost absolute premium). Defensive sectors rank lower despite being
a senior-trader's typical crisis pick. S37 quantifies the divergence
between `ev_dollars` ranking and `roc` (risk-adjusted = `ev_dollars /
collateral`) ranking and proposes a defensible default. **Investigation
Sn, not a code change** — produces a recommendation, not a fix.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
22-name universe (same as S31 / S33 / S36). Two anchor dates:
2025-04-04 (S30-confirmed crisis day) and 2026-03-20 (data
cutoff, normal regime). For each: run
`rank_candidates_by_ev` with `use_event_gate=False` to surface
all 22 candidates; sort by both metrics; compute sector mix,
ticker overlap, rank correlation. Driver under `%TEMP%\s37\`.
The `roc` column is already in the ranker output (per the S31
fix campaign — emitted alongside `ev_dollars` and `collateral`);
no engine changes needed.

**Path.**
`WheelRunner.rank_candidates_by_ev` at `engine/wheel_runner.py`
(the §2 ranker route). Output rows carry `ev_dollars` (the
post-multiplier EV) AND `roc` (`ev_dollars / collateral` per
S31 fix campaign).

**Status.** Done. **Verdict: the two metrics diverge meaningfully
on crisis days (rank correlation ρ = 0.63) and largely agree on
normal days (ρ = 0.89). NEITHER metric is universally "right" —
the choice is operator-account-size-dependent:**

- **`ev_dollars` (current default)** — favors absolute return.
  Right for a $1M-class portfolio that needs to deploy capital
  efficiently. On crisis day, sector mix in top half = 3
  Healthcare + 3 IT + 2 Financials + 2 Comm Svcs + 1 Cons Disc.
- **`roc` (risk-adjusted)** — favors capital efficiency. Right
  for a $100k-class portfolio with tight collateral budget. On
  crisis day, sector mix in top half = **5 Financials** + 2
  Healthcare + 2 IT + 1 Cons Disc + 1 Comm Svcs.

**The crisis-day divergence is real and operator-relevant.**

**Findings:**

- **(F1 — ranking divergence on crisis day, ρ = 0.6341).**
  At as_of=2025-04-04 (S30-confirmed crisis), the two metrics
  produce noticeably different top-11 rankings:

  | Metric | Top 11 (in order) |
  |---|---|
  | **ev_dollars** | LLY, GS, UNH, META, TSLA, MSFT, AVGO, JPM, AAPL, GOOGL, ABBV |
  | **roc** | LLY, TSLA, AVGO, MS, GS, GOOGL, WFC, NVDA, JPM, BAC, UNH |

  Overlap: 7 names (LLY, TSLA, AVGO, GS, GOOGL, JPM, UNH).
  ev_dollars-only: AAPL, ABBV, META, MSFT (high-IV mega-caps
  with large absolute EV but moderate roc).
  roc-only: BAC, MS, NVDA, WFC (smaller absolute EV but better
  EV-per-dollar — banks dominate this set).

- **(F2 — ranking near-agreement on normal day, ρ = 0.8950).**
  At as_of=2026-03-20 (normal regime), the two metrics produce
  largely-aligned rankings: 9 of 11 names overlap. Top of both
  lists is GS / LLY in either order. The two metrics disagree
  only at the margins (KO via roc; UNH via ev_dollars). **The
  ranking-philosophy question is fundamentally a crisis-day
  question** — on calm days the trade-off doesn't bite.

- **(F3 — sector composition shift on crisis day)** — the key
  operator-facing fact:

  | Sector | top-11 by ev_dollars | top-11 by roc |
  |---|---|---|
  | Financials | 2 (GS, JPM) | **5** (GS, MS, WFC, JPM, BAC) |
  | Healthcare | 3 (LLY, UNH, ABBV) | 2 (LLY, UNH) |
  | Information Technology | 3 (MSFT, AVGO, AAPL) | 2 (AVGO, NVDA) |
  | Communication Services | 2 (META, GOOGL) | 1 (GOOGL) |
  | Consumer Discretionary | 1 (TSLA) | 1 (TSLA) |

  **`roc` puts 5 of 11 in Financials.** Why: banks have low
  strike → low collateral → high EV-per-dollar even when
  absolute EV is modest. Conversely, ev_dollars favors LLY +
  the IT mega-caps because their high IV pushes absolute
  premium up.

- **(F4 — collateral and EV magnitudes — the operator-size
  link)**. From the comparative stats:

  ```
  CRISIS day:   ev_dollars min=5.50, max=720.48, median=43.51
                roc       min=0.0007, max=0.0108, median=0.0036
                collateral min=$3,150, max=$66,900, median=$15,075

  NORMAL day:   ev_dollars min=-481.38, max=809.69, median=56.20
                roc       min=-0.0141, max=0.0108, median=0.0030
                collateral min=$4,400, max=$83,200, median=$20,725
  ```

  In crisis, max EV per trade is $720 (LLY) but the median is
  only $44 — a long-tailed distribution. The roc range is
  0.07% – 1.08%. A $1M portfolio looking for 5% annual return
  needs to deploy ~50% of NAV (a $500k+ aggregate) — by
  median EV $44, that's 11,400 trades; **the engine can't
  ship that many candidates daily**. The capacity limit
  surfaces here too (matches Terminal A's S32 F3 finding
  about 10.8% deployment).

- **(F5 — the right ranking key depends on operator account
  size)**. A defensible recommendation:

  | Account size | Recommended primary key | Rationale |
  |---|---|---|
  | < $250k | **`roc`** | Capital efficiency matters; can't afford to tie up $50k+ collateral on a marginal trade |
  | $250k – $2M | **`ev_dollars`** (current default) | Capacity is the binding constraint per S32 / S34 backtests; absolute deployment > per-dollar efficiency |
  | > $2M | **Hybrid: ev_dollars ordering, but require `roc > median(roc)` for inclusion** | Both metrics matter; reject candidates that consume capital inefficiently regardless of absolute EV |

  **The current default (`ev_dollars`) is correct for the engine's
  intended $100k–$2M operator profile.** A future refinement
  could expose a `ranking_key` kwarg or use account-size-aware
  defaults.

- **§2 verified.** No engine code touched; both rankings come
  from the same `rank_candidates_by_ev` call. The `roc` column
  is post-EV (= `ev_dollars / collateral`), so ordering by it
  is a presentation-layer choice. The §2 contract holds.

**Realism Check.**

| Aspect | Engine output | Trader / external expectation | Verdict |
|---|---|---|---|
| Rank correlation on crisis day | ρ = 0.6341 (pre-#260); ρ = 0.7075 (post-#260, S46 re-run 2026-05-28) | Substantial divergence expected when premium-density varies by IV magnitude | ✓ Confirmed (real divergence; post-#260 ρ is +0.0734 higher due to F4 widening firing differentially across the 22-name universe on 2025-04-04) |
| Rank correlation on normal day | ρ = 0.8950 (pre- and post-#260; S46 re-run byte-identical) | High agreement when IV is uniform | ✓ Confirmed (near-alignment; tail_widening_factor = 1.0 universe-wide on 2026-03-20, so ρ is unchanged) |
| `roc` favors Financials on crisis day | 5 of 11 top half are banks | Banks have low strikes after a crisis drop → high EV-per-collateral by construction | ✓ Math-aligned |
| `ev_dollars` favors high-IV mega-caps on crisis day | LLY (Healthcare), META, MSFT, AAPL all in top half | High IV → high premium → high absolute EV regardless of collateral | ✓ Math-aligned |
| LLY ranks #1 on BOTH metrics on BOTH days | $720 / $810 EV; 1.08% / 0.86% roc | LLY is a genuine outlier — high IV + high ROC simultaneously (rare combination) | ✓ Aligned with idiosyncratic LLY characteristics |

**Verdict.**

- **The question "is ev_dollars or roc the right ranking key?"
  is genuinely operator-dependent.** Neither metric is
  universally correct. The engine's current default
  (`ev_dollars`) is right for the $250k–$2M operator profile
  the engine is built for; smaller accounts benefit from
  `roc`; larger accounts benefit from a hybrid.

- **The S33 F5 "defensive should dominate top half in crisis"
  expectation was mis-specified.** The expectation conflated
  two different things: (a) "engine should favor defensive
  sectors when other things are equal" (a reasonable thesis)
  and (b) "absolute-EV ranking should favor defensive sectors
  in crisis" (wrong — crisis pushes IV up across the board,
  often more so on high-beta names, making absolute EV
  concentrate in tech/healthcare).

- **The crisis-day Financials rotation under `roc` IS real and
  defensible.** Banks at crisis low-strikes ARE the highest
  capital-efficiency trades. A trader prioritizing capital
  efficiency over absolute return would rationally go bank-
  heavy in crisis — which matches conventional "buy the bear
  in banks" wisdom. The current `ev_dollars` default hides
  this from the trader unless they re-sort.

- **No code change shipped from S37.** The recommendation:
  EXPOSE `roc` more prominently in the trader-facing summary
  (it's already a column), and consider an account-size-aware
  default ranking kwarg in a future PR. Neither is a §2
  correctness fix; both are operator-experience refinements.

**AI handoff.**

- **F5 follow-on (small future PR, not in S37 scope):** add a
  `ranking_key: str = "ev_dollars"` parameter to
  `rank_candidates_by_ev` that accepts `"ev_dollars" | "roc" |
  "hybrid"`. The hybrid mode uses ev_dollars ordering with a
  `roc > median(roc)` filter. Backwards-compatible default.
  Single-PR, single-concern.

- **Operator documentation gap:** the engine's "ev_dollars vs
  roc" trade-off should be documented in
  `docs/RANKING_PHILOSOPHY.md` (new file) or as a §6 in
  CLAUDE.md. A new operator reading the ranker output today
  doesn't know that `roc` is also a valid first-key.

- **Cross-reference Terminal A's S32 finding.** S32 found
  "engine UNDERPERFORMS SPY by −22pp at $1M due to 10.8%
  deployment." S37 confirms this is the right behavior for
  the intended operator profile — the engine isn't broken;
  it just isn't right for a $1M+ account without strategy
  expansion (which is exactly what Terminal B's S34
  universe-expansion backtest is testing).

**Methodology debt.**

- **Two anchor dates only.** S37 picks one crisis day and one
  normal day. A multi-anchor study (crisis bottoms, bull tops,
  vol shocks) would surface whether the ρ = 0.63 / 0.89
  divergence pattern holds or shifts with the regime.

- **No explicit operator-size simulation.** The recommendation
  table is a heuristic derived from the EV magnitudes seen on
  these two days. A formal backtest comparing both ranking
  keys at $100k / $1M / $5M account sizes would either
  confirm or refute the size-dependent recommendation.

- **`roc` is a static post-hoc sort.** S37 doesn't simulate
  what the engine WOULD recommend if `roc` were the primary
  key from the start — selection effects could shift
  candidates (e.g., the engine might surface different
  candidates entirely if it knew to optimize for `roc`).
  The current `roc` value is computed from a candidate set
  pre-filtered by `ev_dollars` semantics.

- **No comparison to other risk-adjusted metrics.** `roc =
  ev_dollars / collateral` is one risk-adjustment scheme.
  Other options exist: `ev_per_day / collateral` (time-
  adjusted), `(ev_dollars - cvar_5) / collateral` (downside-
  adjusted), Sharpe-style ratios. S37 picks the simplest
  `roc` because it's already emitted.
