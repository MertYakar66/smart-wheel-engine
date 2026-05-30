---
id: S25
title: Vol-shock recovery (MU 2026-03-18 earnings, beat-but-tank)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** First explicit **realism check** Sn (per the new
campaign framing): compare engine output to the realized market
behavior around a real, high-vol earnings event. Specifically —
when a 35-DTE put or covered call's holding window crosses a known
earnings, does the engine's empirical forward distribution
adequately bound the realized post-event move, or does it
systematically understate post-event tail risk?

MU 2026-03-18 is the concrete event: actual EPS beat
(12.2 vs 9.0 estimate) followed by a -8.83% 2-day sell-off — a
classic "sell the news" outcome with measurable IV crush in the
file.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
ticker MU. Three observation dates around the event:

- **2026-03-17** (Tue, pre-event, IV elevated at 69.39%)
- **2026-03-18** (Wed, event day, close essentially flat)
- **2026-03-19** (Thu, post-event, -3.77% from pre-event close,
  IV crushed to 65.15%)
- **2026-03-20** (Fri, T+2, -8.83% from pre-event close, IV
  re-elevated to 69.82%)

35-DTE / 25-delta covered call. Driver under `%TEMP%\s25\`, not
committed; `sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]].

**Path.** `engine.forward_distribution.best_available_forward_distribution`
at `engine/forward_distribution.py` builds the 35-DTE log-return
distribution from OHLCV up to as_of (block-bootstrap / HAR-RV
cascade). `WheelRunner.rank_covered_calls_by_ev` at
`engine/wheel_runner.py:1790` builds a synthetic CC at the target
delta, prices it via BSM at the IV from
`conn.get_fundamentals(ticker)['implied_vol_atm']` (S23 F3 — a
snapshot, no date column), and runs each through
`EVEngine.evaluate`.

**Status.** Done. **Verdict: the engine's empirical forward
distribution is wide enough to comfortably contain MU's realized
post-event move — tail risk is NOT understated on this scenario.
But the IV input bug (S23 F3) re-surfaces on MU exactly as on
AVGO. The engine also correctly classifies a 25-delta MU CC as
net-negative EV both pre- and post-event — sound conservative
behavior on a high-tail-risk name.**

**Findings:**

- **Headline cross-section.** MU 35-DTE / 0.25-delta CC with the
  event gate forcibly disabled (so the post-event re-evaluation
  is observable; the gate would otherwise block the 03-17 cell
  with a 1-day-out earnings event):

  ```
  as_of=2026-03-17   spot=461.69  strike=541.0  premium=$12.597  iv=0.6485  ev_dollars=-$1058.28
  as_of=2026-03-19   spot=444.27  strike=521.0  premium=$12.044  iv=0.6485  ev_dollars=-$  803.41
  ```

  Both cells engine-recommend AGAINST the trade
  (`ev_dollars < 0`). Spot drop is fully observable; IV is NOT
  (same value at both as_ofs because of S23 F3). **Logged.**

- **(F1) Engine forward distribution is wide enough to bound the
  realized move.** At `as_of=2026-03-17`, 35-DTE non-overlapping
  block bootstrap on 2062 rows of MU OHLCV (2018-01-02 →
  2026-03-17) gave:

  ```
  method:               empirical_non_overlapping
  samples:              35
  log-return std:       0.1886   (= 18.86%, 35-day; scaled to ~3.19% daily)
  p1  log-return:       -0.2660  (= -23.4% price move)
  p5  log-return:       -0.2060  (= -18.6% price move)
  ```

  Realized 2-day post-event move: log-return = -0.0925 = -8.83%
  price drop. That is **-0.49 sigmas** of the 35-DTE distribution
  — well inside the body, far from the 5th percentile (-0.2060).
  The engine's tail is *consistent with* the realization; no
  systematic underestimation visible at this single scenario.
  **Logged as a positive — F1 confirms the empirical block
  bootstrap is doing its job on a real high-vol earnings event.**

- **(F2) Engine correctly flags 25-delta MU CC as net-negative
  EV.** Both pre and post the event, the engine returns
  `ev_dollars < 0` for the synthetic CC. The combination of MU's
  wide forward distribution (F1) and elevated absolute spot
  (~$460) means the engine sees enough tail risk that even the
  fat $12.60 premium doesn't compensate. A "sell vol around
  earnings" trader would override; the engine is structurally
  conservative on high-tail-risk names. **Logged as a positive —
  the engine's decision aligns with the "MU is too volatile for
  credit-selling" prior a pro options trader would hold.**

- **(F3) IV-snapshot bug re-confirmed on MU (validates S23 F3
  generality).** The engine used `iv=0.6485` (64.85%) at BOTH
  `as_of=2026-03-17` and `as_of=2026-03-19` — the
  `implied_vol_atm` snapshot from `sp500_fundamentals.csv`. The
  IV file's actual PIT values for the same dates:

  ```
  date         hist_put_imp_vol  hist_call_imp_vol  iv_avg     vs snapshot 64.85%
  2026-03-17           69.39             69.39    69.39%       +4.54 pp ( +7.0% rel)
  2026-03-18           70.42             70.42    70.42%       +5.57 pp ( +8.6% rel)
  2026-03-19           65.15             65.15    65.15%       +0.30 pp ( +0.5% rel)
  2026-03-20           69.82             69.82    69.82%       +4.97 pp ( +7.7% rel)
  ```

  At 2026-03-17 (pre-event), the engine used 64.85% when the
  actual IV was 69.39%. **The engine pre-event was pricing as if
  the market expected a calmer day than it did.** Same direction
  as S23's AVGO finding, different ticker, larger gap. After
  Fix #1 lands (`claude/fix-ranker-iv-pit-aware` @ `d26a8d6`),
  this gap closes mechanically. **Logged — confirms the bug is
  not AVGO-specific and motivates Fix #1.**

- **(F4) IV crush is observable in the data but invisible
  through the engine.** The IV file shows a clean -5.27 pp
  crush from 03-18 (70.42%) to 03-19 (65.15%) on MU. The
  ranker's `iv` column is the snapshot value, so a trader
  inspecting the ranker output sees no evidence of the crush.
  Same root cause as F3. Post Fix #1 this becomes a usable
  signal in the ranker output. **Logged.**

- **(F5) `volatility_30d` (realized) shows the dispersion in the
  underlying.** Realized 30-day vol jumped from 71.14% on 03-17
  to a momentary 63.50% on 03-18 (lookback windowing artifact)
  then 64.92% on 03-19 — the post-event realized vol normalized
  quickly. Not directly used by the engine for synthetic
  pricing (BSM uses IV), but useful diagnostic. **Logged.**

- **§2 verified.** The forward distribution call sits inside
  `rank_covered_calls_by_ev`, which still routes each candidate
  through `EVEngine.evaluate`. No bypass. The two negative-EV
  surfaces in the headline came through `evaluate` honestly —
  the conservative recommendation is the engine's product, not
  a side-channel veto. **Logged as a positive.**

**Realism Check.**

| Aspect | Engine | Reality (file / market) | Verdict |
|---|---|---|---|
| 35-DTE log-return std (block bootstrap) | 0.1886 | Realized 2-day move = -0.0925 (= -0.49σ) | **Consistent** — engine tail bounds reality |
| IV input to BSM strike-solve at 2026-03-17 | 0.6485 (snapshot) | 0.6939 (PIT) | **Mismatch** — engine under by 4.54 pp (-6.5% relative). S23 F3 / Fix #1 |
| IV crush observable in ranker output | No (snapshot frozen) | Yes (-5.27 pp on 03-18 → 03-19) | **Mismatch** — invisible until Fix #1 |
| 25-delta CC EV verdict | Negative both runs | Pro trader would hold the conservative view on MU at high-IV | **Aligned** — engine's bearish-on-credit-selling-MU stance is sound |
| Spot move tracked through re-evaluation | Yes (461.69 → 444.27) | -3.77% / -8.83% cumulative | **Aligned** — engine reads spot from OHLCV correctly |

**Verdict.**

- **The quant layer is doing better than the data layer on this
  scenario.** Forward distribution (F1) cleanly bounds the
  realized move, with the realized 2-day drop sitting at -0.49σ
  of the 35-DTE distribution — well within the body. The block
  bootstrap on 8 years of MU history captures enough idiosyncratic
  vol that an 8.83% drop is unsurprising.
- **The IV-snapshot bug is the binding constraint on realism for
  this trade**. Fix #1 (`claude/fix-ranker-iv-pit-aware`) closes
  F3 and F4 mechanically. The engine's evaluation pre- and post-
  event would still both be negative-EV after Fix #1 (the IV move
  is in the right direction to slightly improve the synthetic
  premium and reduce the EV magnitude), but the trader-visible
  numbers would be **correct** as of each date.
- **Decision quality is sound**. The engine refusing to credit-
  sell MU at a 25-delta CC even with $12.60 of premium reflects
  the wide empirical forward distribution. A trader who disagrees
  ("sell into the IV pop, take the premium, manage if it goes
  bad") would have to override the engine's verdict — but the
  engine is being honest about the tail.

**No new bug surfaced beyond the F3/F4 re-confirmation of the
S23 finding.** This is the intended realism-check outcome: a
single, real, high-vol event that lets us *quantify* whether the
engine's distributional output is realistic. Answer: yes for the
forward distribution, no for the IV input until Fix #1 lands.

**AI handoff.**

- The realism-check verdict on the forward distribution is from a
  single scenario. To make a stronger claim ("the engine's
  empirical bootstrap is *systematically* well-calibrated for
  earnings tails"), the same machinery should be run across a
  basket of historical earnings events with various IV regimes
  and sector mixes. A natural follow-up Sn: 20-event basket of
  S&P 500 names with earnings in the 2024-2026 window, each
  evaluated at `as_of=earnings_date - 1d` for 35-DTE horizon, and
  the realized move converted to engine-sigma units. Histogram
  the result. If the distribution of "realized in engine-sigma"
  is well-centered with no fat left tail beyond the engine's
  predicted tail, that's confirmation. If it has a fat left
  tail, that's evidence the empirical bootstrap is missing the
  pure-earnings-jump regime that's underrepresented in 8 years
  of post-2018 data.

- The reason the engine returns NEGATIVE EV on a 25-delta MU CC
  even with $12.60 premium deserves a separate audit. Hypothesis:
  the synthetic forward distribution's left tail
  (`p1=-0.2660 logret` = -23.4% price drop) generates large
  ITM assignment losses that dominate the expected premium
  collection. The engine is pricing a fat-tailed name correctly,
  but the "wheel premium harvester" trader's mental model
  ("just sell the vol, manage on assignment") doesn't line up
  with the engine's "honest expected dollar P&L over a single
  hold-to-expiry sample" framing. Worth a documentation pass on
  what `ev_dollars` means semantically — it's not a "premium
  harvested if all goes well" number, it's "expected $ P&L
  including the bad scenarios weighted by their probability."

- Re-running this entry after Fix #1 lands will move the
  `iv=0.6485` rows to `iv=0.6939` (03-17) and `iv=0.6515` (03-19).
  Premium and ev_dollars will shift; sign almost certainly stays
  negative (the IV move is modest in absolute terms compared to
  the wide forward distribution). The realism table's "Mismatch"
  rows for IV will become "Aligned".

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`). **Fix #1 has landed (PR #179) — the predicted IV values now match exactly.**
  - §2 invariant: holds — CC ranker still routes through `EVEngine.evaluate`; sign of 25-delta MU CC stays negative on both dates.
  - Qualitative verdict: **match — the predicted post-Fix-#1 outcome is the observed outcome**:
    - MU CC IV @ 2026-03-17: **`iv=0.6939`** (was `0.6485` pre-fix) — matches `hist_put_imp_vol=hist_call_imp_vol=0.6939` in the IV file exactly.
    - MU CC IV @ 2026-03-19: **`iv=0.6515`** (was `0.6485` pre-fix) — matches the IV file (0.6515) exactly.
    - 25-delta CC `ev_dollars` stays negative both dates (forward-distribution wide enough to keep the engine bearish on selling vol around earnings).
  - Numerical drift > 5% (with attribution):
    - metric `MU_CC_iv[2026-03-17]`: orig `0.6485` → new `0.6939` (`+7.0%`); attributable to **PR #179** (per the original entry's explicit prediction).
    - metric `MU_CC_iv[2026-03-19]`: orig `0.6485` → new `0.6515` (`+0.46%`) — minimal change at the post-IV-crush date when snapshot and PIT are coincidentally close.
    - metric `MU_CC_ev_dollars[2026-03-17]` (25-delta proxy): orig `-1058.28` → new `-147.93` (`-86% magnitude`); the strike selection shifted under higher IV (557.5 vs 541) and the premium shape changed accordingly. The engine's "negative EV verdict on selling MU CC at high-vol earnings" — i.e. the headline F2 result — holds in direction.
  - Notes: F1 (forward distribution bounds realized move) and F2 (engine refuses 25-delta MU CC) verdicts re-confirmed — the engine still classifies a 25-delta MU CC as net-negative EV. The realism table's "Mismatch" rows for IV are now **Aligned** (per the original entry's exit prediction).
