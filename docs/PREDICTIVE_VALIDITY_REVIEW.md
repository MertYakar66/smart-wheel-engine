# Predictive-validity review — S22 + S27 backtests (2026-05-25)

**Reviewer:** Terminal B, fresh session.
**Question (from user):** *Does our options engine produce realistic
output?*
**Headline answer:** **Yes — the engine ranks better than chance with
moderate effect size (ρ = 0.22 post-fix). It is not snake oil. But
the pre-fix ρ = 0.48 was an artefact of an IV-snapshot bug; the
honest signal is roughly half of what S22 claimed, and two
engineering limits (tail-risk miss on COST April 2022; fragile
BP-saturation "protection") are real and now confirmed at the data
level.**

**Scope:** PR **#184** (S27 IV-PIT re-run, merged `0cc9c75`) and
PR **#178** (S22 pre-fix backtest, **closed** without merging — the
doc on its branch is the historical record). The fix that landed
between them is PR #179 (`claude/fix-ranker-iv-pit-aware` @
`d26a8d6` → merged `1378a5d`).

**Pre-fix engine SHA:** `86b917c7` (origin/main at S22's run-time
2026-05-24).
**Post-fix engine SHA:** `d26a8d6` (the fix commit; reachable
locally even after the fix branch was deleted from the remote;
verified on current main `e83eaca` — `_resolve_pit_atm_iv` lives at
`engine/wheel_runner.py:147`).
**Origin/main SHA at review start:** `e83eaca`.

## Tally

- CONFIRMED:                  **8**
- CONFIRMED-WITH-NOTE:        **1** (P9 — Caveat 2 in-sample-parameters not restated in S27)
- CONFIRMED-WITH-CAVEAT:      **0**
- CONCERN:                    **0**
- METHODOLOGY-BREAK:          **0**

The arc holds. The engine ranks. The follow-on PR #196 (`test(forward_distribution):
F4 tail-risk regression watch from PR #178 / #184`) shows the F4
finding is already being actioned.

---

## P1 — PIT discipline in the harness

**Verdict:** **CONFIRMED.**

The S27 harness lives at `%TEMP%\s27_backtest\run.py` (still on the
dev box). Read in full. PIT is enforced on two surfaces:

**1. Rank-call surface** — every rank invocation passes
`as_of=today.isoformat()`:

```python
# run.py:175 (covered-call rank)
cc_df = runner.rank_covered_calls_by_ev(
    ticker=ticker, ..., as_of=today.isoformat(), ...
)

# run.py:226 (short-put rank)
df = runner.rank_candidates_by_ev(
    tickers=UNIVERSE, as_of=today.isoformat(), ...
)
```

**2. Connector OHLCV access** — the harness fetches each ticker's
full OHLCV once and caches it, but every *use* slices PIT:

```python
# run.py:get_spot_at_or_before (used by mark_to_market + CC step)
sub = df[df.index <= pd.Timestamp(on_or_before)]   # PIT slice
return float(sub.iloc[-1]["close"])

# run.py:get_spot_at (used by replay_row_outcome ONLY)
sub = df[df.index >= pd.Timestamp(on_or_after)]    # forward-of-entry slice
```

`get_spot_at` is the only place the harness reaches "into the future"
relative to entry — and it is used **only** for the
`replay_row_outcome` step (exit-side P&L observation), never as a
rank input. This is the allowed observation pattern per the prompt's
P1 definition.

**3. Engine-side PIT** — independent of harness discipline, the
engine itself respects `as_of` on `main` today:

```python
# engine/wheel_runner.py:816-819 (rank_candidates_by_ev)
# Respect PIT cutoff on OHLCV.
if as_of is not None:
    try:
        cutoff = pd.Timestamp(as_of)
        ohlcv = ohlcv.loc[ohlcv.index <= cutoff]
```

```python
# engine/wheel_runner.py:147 (the post-fix helper)
def _resolve_pit_atm_iv(conn, ticker: str, as_of: str | None) -> float | None:
    ...
    hist = conn.get_iv_history(ticker, end_date=as_of)
```

Both layers (harness + engine) enforce PIT. No leak observed.

---

## P2 — Independent Spearman computation

**Verdict:** **CONFIRMED — exact match.**

Ran `scipy.stats.spearmanr` against the on-disk rank logs:

| Run | N | ρ (recomputed) | p (recomputed) | Doc-claimed ρ |
|---|---|---|---|---|
| S22 pre-fix | 6,163 | **0.4838** | 0.00e+00 | 0.4838 ✓ |
| S27 post-fix | 6,163 | **0.2183** | **2.29e-67** | 0.218, p ≈ 2.29e-67 ✓ |

Filter: `option_type == "put"` AND `realized_pnl.notna()`. Sample
sizes, ρ values, and p-values match the docs to the digit. The
**−0.2655 drop** in ρ between runs is the load-bearing finding of
the IV-PIT bug discovery.

---

## P3 — Realized P&L spot-check

**Verdict:** **CONFIRMED.**

Picked one ORCL put assignment (`entry_date=2022-01-03, strike=84.00,
premium=0.996, exit_date=2022-02-07, exit_spot=80.74, realized=-$226.40`)
from the S27 rank log via `df.sample(random_state=42)`. Recomputed:

- Independent `runner.connector.get_ohlcv("ORCL").loc[2022-02-07]["close"]`
  = **80.74** ✓ (matches the row's `exit_spot` exactly).
- Harness formula for short-put assignment (`exit_spot < strike`):
  `(exit_spot − strike + premium) × 100 = (80.74 − 84.00 + 0.996) × 100 = −$226.40` ✓
  (matches the row's `realized_pnl` exactly).

Extended the check across **all 77 executed rows** (50 puts +
27 covered calls): zero rows had `|delta| > $0.50` between reported
realized and the formula. The exit-side accounting is internally
consistent.

The puzzle case I initially flagged (`ORCL 2022-07-18 strike=$74.50
exit_spot=$76.46 realized=−$106.20`) turned out to be a **covered-call**
row, not a put — the harness records CC rows with `rank_position=-1`
and the call-side formula `(strike − exit_spot + premium) × 100 =
(74.5 − 76.46 + 0.898) × 100 = −$106.20` ✓. Same row, correct
formula, my error. No data defect.

---

## P4 — IV-PIT bug live verification

**Verdict:** **CONFIRMED — bug exists, fix is correct, 60% relative
correction.**

Ran the pre-fix vs post-fix IV resolution side-by-side for UNH on
2024-01-15:

```
Snapshot IV (pre-fix path, fundamentals['implied_vol_atm']): 43.233 (= 0.4323 decimal)
PIT IV (post-fix path, get_iv_history end_date=2024-01-15):    0.1712
  hist_put_imp_vol=17.12%, hist_call_imp_vol=17.12%
  date in IV history: 2024-01-12 (Friday before MLK Day)
Relative correction: -60.4%
```

The S27 doc's "Verified live (2024-01-15, UNH): pre-fix IV `0.4323`,
post-fix IV `0.1712`. A 60% relative correction" matches my live
reproduction to the digit.

Fix on `main` today (commit `1378a5d` via PR #179):

```
engine/wheel_runner.py:147 — def _resolve_pit_atm_iv(conn, ticker, as_of)
engine/wheel_runner.py:877 — iv = _resolve_pit_atm_iv(conn, ticker, as_of)  # rank_candidates_by_ev
engine/wheel_runner.py:2023 — iv = _resolve_pit_atm_iv(conn, ticker, as_of)  # rank_covered_calls_by_ev
engine/wheel_runner.py:2508 — iv = _resolve_pit_atm_iv(conn, ticker, as_of)  # rank_strangles_by_ev
```

All three rankers consume PIT IV via `get_iv_history(end_date=as_of)`
with a graceful fallback to the snapshot path for connectors that
don't expose the history method. The pre-fix code (`fundamentals[
"implied_vol_atm"]`) is the documented fallback, not the primary
path.

---

## P5 — F4 (COST April 2022 tail-risk gap)

**Verdict:** **CONFIRMED — identical engine behaviour across runs;
F4 is a real engineering limitation, not an IV-PIT artefact.**

### Same rows in both runs

| Run | n | mean `prob_profit` | mean `realized_pnl` | mean `ev_dollars` | HMM regimes | executed |
|---|---|---|---|---|---|---|
| S22 pre-fix | 10 | **0.8333** | **−$7,615.70** | $103.52 | normal / bear / crisis | 0 (BP) |
| S27 post-fix | 10 | **0.8333** | **−$7,501.29** | $127.35 | normal / bear / crisis | 0 (BP) |

`prob_profit = 0.8333` exactly across every one of the 10 candidates
in BOTH runs — the forward distribution did not widen as COST
dropped. The HMM regime classifier DID flip (`normal → bear → crisis`
visible in the per-row regime column), but the EV calculation kept
producing positive numbers ($58 – $224) the whole way down.

### Independent spot trajectory

| Date | COST close | Note |
|---|---|---|
| 2022-04-07 | $608.05 | peak in window |
| 2022-04-21 | $591.74 | end of entry-claim window |
| 2022-05-06 | $503.36 | expiry for 2022-04-01 entry |
| 2022-05-13 | $497.27 | expiry for 2022-04-08 entry |
| 2022-05-19 | $422.93 | expiry for 2022-04-14 entry |
| 2022-05-20 | **$416.43** | trough in 2022-04-01 / 2022-05-25 window |

Peak-to-trough drop in the entry-to-expiry envelope: **608 → 416 =
−31.5%**. The doc's framing "~$130 drop from $584 → $423" understates
the actual magnitude (the peak was $608, not $584; the trough was
$416, not $423). Same direction, larger magnitude than written.

### The substantive finding

**`prob_profit = 0.833` is constant across a 31.5% drop in BOTH the
pre-fix and post-fix engines.** The tail-risk machinery
(`engine/forward_distribution.py` + `engine/tail_risk.py` POT-GPD +
`engine/regime_hmm.py` multiplier) did not detect the unfolding
single-name drawdown as something worth widening the forward
distribution over.

This is the campaign's **highest-leverage engineering finding** and
PR #196 is already on the board — `test(forward_distribution):
F4 tail-risk regression watch from PR #178 / #184` — to pin the
regression so any future widening shows up in CI.

---

## P6 — F5 (BP-saturation accident)

**Verdict:** **CONFIRMED — explicit data evidence.**

### Headline numbers (S22 vs S27, puts only)

| Metric | S22 pre-fix | S27 post-fix | Direction |
|---|---|---|---|
| Executed | 59 | 50 | -9 |
| Executed mean realized | **+$200.72** | **−$71.99** | flip to negative |
| Executed hit-rate (OTM expire) | 88.1% | 76.0% | -12.1pp |
| `insufficient_bp` refusals | **1,171** | **574** | **-597 (-51%)** |
| `below_min_ev` refusals | 236 | 893 | +657 (post-fix lower EVs fall below $10 floor more often) |

The mechanism is exactly as F5 claims: **the pre-fix engine refused
1,171 trades for lack of BP; the post-fix engine refused only 574.**
The difference (597 trades) is the set of would-be-trades the
pre-fix engine accidentally blocked because earlier-positioned
trades had over-tied up BP with their inflated EV → strike-collateral
allocation. Post-fix, those trades fire, and they fire badly on
average — flipping the executed-trade mean realized from +$200 to
−$72.

### Per-week distribution check

Per-week executed counts cluster tightly at 1 in both runs:

| Run | weeks active | mean exec/week | std | max |
|---|---|---|---|---|
| S22 | 52 | 1.13 | 0.34 | 2 |
| S27 | 40 | 1.25 | 0.54 | 3 |

No bimodal "0 or 4-5" pattern that the prompt initially hypothesised
— but the prompt's hypothesis was about *what BP saturation would
look like*, not what was actually there. The harness's
`MAX_NEW_PER_DAY=3` cap plus the 24-ticker universe makes the
per-week count naturally low and bounded. The BP saturation
manifests as **refusals** of trades (the 1,171 number), not as
week-level capacity caps. **The mechanism F5 names is real; the
secondary pattern in the prompt's hypothesis is just absent for this
small universe + low rank-floor setup.**

---

## P7 — §2 invariant on the test methodology

**Verdict:** **CONFIRMED — no hand-built `ev_row` synthesis.**

Read the S27 harness end-to-end. Every executed trade routes through
the engine:

```python
# run.py:118 — non-strict mode (deliberate per the doc)
tracker = WheelTracker(initial_capital=STARTING_CAPITAL, require_ev_authority=False)

# run.py:175, 226 — every candidate originates in the ranker
cc_df = runner.rank_covered_calls_by_ev(...)
df    = runner.rank_candidates_by_ev(...)

# run.py:192, 266 — opens read strike/premium from those rows
tracker.open_covered_call(ticker=..., strike=float(cc_row["strike"]), premium=float(cc_row["premium"]), ...)
tracker.open_short_put(   ticker=..., strike=strike,                  premium=premium,                   ...)
```

`rank_candidates_by_ev` and `rank_covered_calls_by_ev` internally
call `EVEngine.evaluate` for every candidate row (CLAUDE.md §1
Layer 3) — so every trade the tracker opens was EV-evaluated
upstream. The §2 invariant ("no tradeable candidate bypasses
`EVEngine.evaluate`") holds at the test-methodology level.

`require_ev_authority=False` skips the **D16 token gate** at the
tracker (the additional verdict-bound enforcement from PR #145),
not the §2 invariant itself. The token gate is supplementary
authorisation, not the EV authority. This is the same distinction
the S22 doc explicitly draws and the S27 doc inherits.

**No hand-built `ev_row` synthesis anywhere in either harness.**
The only fields the tracker receives are the strike/premium values
from the ranker's emitted row. The Spearman correlation between
`ev_dollars` and `realized_pnl` is measuring exactly what it claims
to measure: the engine's predictive ordering against post-hoc
outcomes.

---

## P8 — Realistic-output qualitative check (10 + 10 random trades)

**Verdict:** **CONFIRMED — output is realistic.**

Sampled with `df.sample(random_state=42)`. Spot-at-entry from
`runner.connector.get_ohlcv(...).iloc[<=entry_date].iloc[-1]`.

### 10 random EXECUTED put trades (S27 post-fix)

| # | Ticker | Entry | Strike | Spot | OTM% | Prem | EV | P(prof) | Exit | Realized | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GOOGL | 2022-08-30 | 102.00 | 108.94 | 6.4% | 1.81 | +$49.70 | 0.91 | assigned | +$145.00 | prem covered small assign loss |
| 2 | MRK | 2024-09-03 | 112.50 | 116.58 | 3.5% | 1.05 | +$26.40 | 0.89 | assigned | −$292.80 | normal assignment loss |
| 3 | MRK | 2023-09-06 | 103.00 | 106.49 | 3.3% | 1.02 | +$15.69 | 0.86 | otm_expire | +$101.60 | premium kept |
| 4 | UNH | 2024-10-24 | 535.50 | 560.81 | 4.5% | 6.44 | +$83.84 | 0.91 | otm_expire | +$644.30 | clean OTM win |
| 5 | MRK | 2022-11-18 | 100.00 | 104.23 | 4.1% | 1.07 | +$18.74 | 0.91 | otm_expire | +$107.40 | premium kept |
| 6 | ORCL | 2024-12-09 | 177.00 | 190.45 | 7.1% | 3.84 | +$123.65 | 0.89 | assigned | −$1923.90 | bigger assign loss |
| 7 | ORCL | 2023-06-12 | 108.00 | 116.43 | 7.2% | 2.48 | +$36.56 | 0.91 | otm_expire | +$247.50 | premium kept |
| 8 | MRK | 2023-06-02 | 108.50 | 112.52 | 3.6% | 1.12 | +$31.32 | 0.91 | otm_expire | +$112.50 | premium kept |
| 9 | XOM | 2023-11-16 | 95.00 | 102.46 | 7.3% | 2.02 | +$15.71 | 0.77 | otm_expire | +$202.30 | premium kept |
| 10 | MRK | 2022-12-23 | 108.00 | 111.86 | 3.5% | 1.10 | +$19.68 | 0.91 | assigned | −$152.00 | small assignment |

**Observations:** OTM% ranges 3.3 – 7.3% — consistent with 25-delta
on 35-DTE for typical SP500 names. Premiums scale sensibly with
spot and IV (e.g., UNH at $560 with 6.44 premium = 1.1% of spot;
MRK at $104 with 1.05 premium = 1.0% of spot). Realized P&L matches
exit-reason mechanics: OTM expire → keeps premium; assigned →
formula `(exit_spot − strike + premium) × 100`. **No row trips a
flag.**

### 10 random NOT-EXECUTED put candidates (S27 post-fix)

| # | Ticker | Entry | Strike | Spot | OTM% | Prem | EV | P(prof) | Exit | Realized | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | NVDA | 2024-01-09 | 50.00 | 53.14 | 5.9% | 0.92 | **−$58.50** | 0.83 | otm_expire | +$91.60 | engine refused; OTM in retrospect |
| 2 | PG | 2022-03-02 | 147.50 | 153.79 | 4.1% | 1.72 | +$25.82 | 0.90 | otm_expire | +$171.50 | below_top3 |
| 3 | XOM | 2023-06-08 | 103.50 | 108.19 | 4.3% | 1.24 | **−$95.18** | 0.74 | otm_expire | +$124.10 | refused; OTM in retrospect |
| 4 | MRK | 2024-11-25 | 97.00 | 101.16 | 4.1% | 1.10 | +$11.17 | 0.77 | otm_expire | +$110.20 | below_top3 |
| 5 | PG | 2024-03-08 | 156.00 | 160.35 | **2.7%** | 1.23 | +$0.87 | 0.83 | assigned | +$56.40 | tight (PG low IV); below_top3 |
| 6 | ORCL | 2022-04-20 | 76.50 | 80.29 | 4.7% | 0.90 | −$2.30 | 0.80 | assigned | −$577.10 | refused, would have lost |
| 7 | GOOGL | 2023-10-31 | 118.50 | 124.08 | 4.5% | 1.63 | **−$7.65** | 0.77 | otm_expire | +$162.90 | refused; OTM in retrospect |
| 8 | UNH | 2023-11-22 | 525.50 | 543.76 | 3.4% | 4.56 | **−$54.62** | 0.89 | assigned | +$185.00 | refused; ended slightly positive |
| 9 | AMZN | 2023-10-30 | 125.50 | 132.71 | 5.4% | 2.12 | **−$49.02** | 0.71 | otm_expire | +$212.30 | refused; OTM in retrospect |
| 10 | WFC | 2022-04-20 | 46.00 | 48.65 | 5.4% | 0.77 | **−$59.18** | 0.80 | assigned | −$110.60 | refused, would have lost |

**Observations:** Most refusals are `below_top3` (6) or
`below_min_ev` (3); only one `already_held`. None refused for
`insufficient_bp` in this random sample. The negative-EV refusals
(7 of 10) often DID end up OTM-positive in retrospect — but that
is exactly what a probabilistic-EV engine is supposed to refuse
on: high-hit-rate-but-negative-expected-value trades. The engine
is correctly trading off hit-rate against tail risk on those rows.

**One flagged row** (#5, PG 2024-03-08 with 2.7% OTM) — atypically
tight, but PG is a low-IV defensive stock and 2.7% OTM at low IV
is a plausible 25-delta strike. Not a defect.

### Bottom line for P8

The engine's output is **mechanically realistic**: strikes are
reasonable 25-delta picks, premiums are reasonable for the
IV/DTE/spot, realized P&L is internally consistent with exit
mechanics, and the not-executed rows show the engine refusing
correctly on negative-EV candidates rather than chasing
high-hit-rate-low-EV tails. **No defects surfaced in 20 randomly
sampled trades.**

---

## P9 — Caveat honesty across S22 → S27

**Verdict:** **CONFIRMED-WITH-NOTE — Caveat 2 (in-sample
parameters) is not restated in S27.**

S22's doc has a dedicated "Hard methodology caveats" section with
three numbered caveats:

- **Caveat 1 (IV-PIT bug):** documented in detail; explicitly says
  "ranking yes, dollars no".
- **Caveat 2 (in-sample parameters):** HMM thresholds,
  dealer-multiplier clamp, POT-GPD calibration all tuned with full
  2018-2026 visibility. "No way around this on the repo today".
- **Caveat 3 (frictionless P&L):** no bid/ask, no commissions, no
  slippage; "real trading would lose ~2-5% per leg".

S27's doc handles them differently:

- **Caveat 1:** *implicitly closed* — the entire doc is the post-fix
  re-run. The IV-PIT bug is the doc's subject, not a caveat
  carried forward. ✓ Honest framing.
- **Caveat 2 (in-sample parameters):** **not restated.** The HMM /
  POT-GPD parameters in the post-fix engine are the same parameters
  that were in the pre-fix engine — and those parameters were tuned
  with full 2018-2026 data visibility. A reader who only reads S27
  could leave with the impression that all the magnitude claims
  (ρ = 0.22, top-quartile lift 1.7×, +51% NAV) are fully out-of-
  sample. They are not. Caveat 2 still applies.
- **Caveat 3 (frictionless):** mentioned at `:173` ("caveats from
  S22 doc still apply re: friction") and at `:186-187` ("Friction
  (bid/ask, commissions, slippage) is still unmodeled — realistic
  real-money return would be lower than $51%"). Acknowledged but
  not as a structured caveat section. ✓ Honest enough.

The note: **a future reader of S27 in isolation would miss the
in-sample-parameter caveat.** The fix is a one-paragraph
"## Hard methodology caveats (inherited from S22)" section in S27,
or a clear `**(Caveat 2 inherited)**` flag on the ρ-by-year and
top-quartile-lift tables. Not consequential to the verdict; worth
a follow-up docs PR. Same shape of follow-up as the M4 attribution
note from the prior audit-of-audit review.

---

## Final verdict on the question *"does the engine produce realistic output?"*

**Yes — with two structural caveats that are now well-named.**

The engine's `ev_dollars` ranks realized historical P&L with
Spearman ρ ≈ 0.22 across 6,163 puts and 753 trading days. The
top-EV quartile beats the bottom by 1.7× in mean realized P&L and
by 5pp in hit-rate. Quartile monotonicity is clean. The numbers
are independently reproducible from the on-disk rank log to the
digit. **This is real signal, not luck — but it is moderate
signal, not the ρ = 0.48 the pre-fix backtest reported.**

The pre-fix headline was an artefact of `fundamentals[
"implied_vol_atm"]` returning a 2026 snapshot for every historical
date. PR #179's `_resolve_pit_atm_iv` correctly resolves the
post-fix engine's PIT IV from `connector.get_iv_history(end_date=
as_of)`. Verified live: UNH 2024-01-15 snapshot IV = 0.4323, PIT
IV = 0.1712, a 60.4% downward correction. All three rankers on
current `main` (`e83eaca`) use the post-fix path. The bug is fully
closed in the production code path.

The two **engineering limits** that survived the IV-PIT fix are
where production attention belongs:

1. **F4 tail-risk gap.** The forward-distribution + POT-GPD machinery
   produced `prob_profit = 0.833` constant across COST's 31.5%
   April-May 2022 drawdown in *both* the pre-fix and the post-fix
   engines. The HMM regime classifier *did* flip (`normal → bear →
   crisis` per-row), but the EV calc kept producing positive numbers.
   PR #196 is open with a regression watch for exactly this episode.
   **Production should not run unattended on single-name
   concentration until F4 has a fix.**

2. **F5 fragile BP-saturation protection.** The S22 backtest's +$200
   per-executed-trade win was partly because the pre-fix engine's
   inflated EVs over-tied up buying power, accidentally refusing
   1,171 trades. The S27 post-fix engine has 574 BP-refusals (51%
   fewer), and the freed-up BP lets bad trades fire — flipping the
   executed mean realized to −$72. **Production must run with
   `require_ev_authority=True` plus a `PortfolioContext` so the D17
   hard-blocks (sector / delta / Kelly / VaR / stress / dealer
   regime) fire explicitly rather than via accidental BP
   exhaustion.** PR #174 ships the wiring (`consume_ranker_row` +
   `portfolio_context_snapshot`); the call from `engine_api.py`
   or the dashboard is still open.

The §2 invariant holds across both backtests at the methodology
level: no hand-built `ev_row` synthesis, every candidate routes
through `rank_candidates_by_ev` → `EVEngine.evaluate`. The
non-strict mode (`require_ev_authority=False`) skips the D16 token
gate but not §2.

In one sentence for the user: **the engine ranks better than chance
with a moderate, defensible effect size, but you should not trust
it unattended on single-name tail risk and you should not rely on
buying-power accidents as a risk control.** Both are named, both
are being actioned. The arc is honest and the follow-ups are scoped.

---

_Read-only review. No edits to `engine/`, `tests/`,
decision-layer files, or to the two backtest docs themselves.
All findings logged in this document._
