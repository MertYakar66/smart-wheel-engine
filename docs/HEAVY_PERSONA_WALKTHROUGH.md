# Heavy persona walkthrough — a professional quant uses the Smart Wheel Engine end-to-end

**Card:** HT-A (heavy-verify cycle, 2026-05-30) ·
**Branch:** `claude/heavy-persona-walkthrough` ·
**Engine SHA:** `main @ 56c671d` (post-#249/#260/#262/#287/#288) ·
**As-of date:** `2026-03-20` (freshest Bloomberg date per SessionStart hook) ·
**Driver:** [`verification_artifacts/persona_walkthrough_driver.py`](verification_artifacts/persona_walkthrough_driver.py) ·
**Raw output:** [`verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt`](verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt)

> **What this document is.** A read-only walkthrough of the engine
> through the eyes of a professional quant trader. The driver scripts
> four realistic operator asks against the production engine, captures
> stdout verbatim, and this doc summarises *what surfaces well*, *where
> the operator is left guessing*, *any silent filtering*, and *how the
> §2 EV-authority path behaves under realistic use*. **No engine code
> was modified.** Anything that looks like a defect is logged as a
> *Finding* (§6) for the Major Session to triage into a fix-card next
> cycle. Every quantitative claim cites the raw output by line — if a
> number is not in the raw output, it is not in this doc.

---

## 0. Scope & method

The driver runs end-to-end on the Bloomberg-CSV path (default
`MarketDataConnector` per `CLAUDE.md` §4) at `as_of=2026-03-20`:

1. `WheelRunner.rank_candidates_by_ev` over the full SP500 universe
   (35 DTE, 25-delta, 1 contract, top 20, positive-EV only,
   `include_diagnostic_fields=True`).
2. `WheelRunner.select_book` to fit a $250k account under a 25%/name
   concentration cap (§2-safe knapsack post-processor — never calls
   `EVEngine`).
3. `WheelTracker` constructed strict (`require_ev_authority=True`,
   connector wired, `min_nav_for_trading=1_000`) and seeded with two
   held puts so `PortfolioContext` has real positions for R7-R10 to
   score against.
4. `build_dossiers` over the top 20 with a deliberately-null chart
   provider and the live `PortfolioContext` attached, so R2 vs
   R5/R7-R10 outcomes are observable.
5. `consume_ranker_row` for the top survivor (the canonical §2 wire).
6. Anchor-candidate downside drilldown — distribution percentiles,
   CVaR, EVT tail diagnostics, regime + dealer + skew + credit + news
   multipliers, breakeven, ROC.
7. Negative-control battery: D16 leg 1 refuses non-positive EV at
   issuance; D16 leg 2 rejects stale EV at consume; dossier R1
   blocks negative EV; dossier R1a blocks non-finite EV; a *perfect*
   chart attached to a negative-EV row still produces
   `verdict='blocked'`.

The persona ("Q") is a senior quant trader who has used the engine
before but is checking *what the engine actually shows me* before
placing real trades.

**Determinism.** The driver is fully self-contained (no network, no
time-of-day) and deterministic given the same Bloomberg CSVs + the
same `as_of`. The HMM regime cache is per-process; a fresh run
recomputes but the regime label is deterministic via
`GaussianHMM(..., random_state=42)`.

---

## 1. Ask 1 — "rank me 20 names"

> *"Scan the universe and show me the 20 best short-put candidates
>  right now, with EV and downside."*

Result (raw output L20-65):

| ticker | spot | strike | premium | dte | iv | ev_dollars | ev/day | prob_profit | sector | hmm_regime | hmm_mult |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FIX | 1360.79 | 1212.0 | 47.19 | 35 | 0.674 | **$2547.56** | $146.97 | 0.9714 | Unknown | normal | 0.621 |
| BKNG | 4286.81 | 3969.5 | 89.34 | 35 | 0.422 | 1468.83 | 80.12 | 0.8000 | Cons. Disc. | bear | 0.556 |
| NVR | 6327.00 | 5971.5 | 97.32 | 35 | 0.317 | 1439.85 | 78.54 | 0.8000 | Unknown | crisis | 0.486 |
| AZO | 3292.24 | 3104.5 | 51.46 | 35 | 0.322 | 1374.72 | 72.35 | 0.9143 | Unknown | crisis | 0.348 |
| KLAC | 1474.09 | 1339.5 | 40.03 | 35 | 0.540 | 1260.65 | 68.76 | 0.9143 | Info Tech | bear | 0.486 |
| MCK | 891.14 | 837.0 | 14.94 | 35 | 0.343 | 1109.92 | 62.24 | 0.9429 | Unknown | normal | 0.921 |
| … 14 more rows … | | | | | | | | | | | |

### 1.1 What the engine surfaces well

For the persona, *the right fields are there and the rank is
internally consistent*:

- **Headline EV + breakdown.** `ev_dollars` (post-multiplier),
  `ev_per_day`, `ev_raw` (pre-multiplier), `regime_multiplier`
  (final multiplier applied) — surfaced on every row. Q can read
  the verdict and the *why* in adjacent columns.
- **Probability decomposition.** `prob_profit` and `prob_assignment`
  surfaced separately. For FIX they're `0.9714 / 0.0286` — the
  persona reads "97% chance the put expires worthless, 3% chance
  I get the stock at $1212" at a glance.
- **Capital efficiency.** `collateral` (`strike × 100 × contracts`)
  and `roc` (`ev_dollars / collateral`) make the candidate
  budget-sortable without arithmetic. The driver cross-checks ROC
  manually (raw output L246-247: `roc=0.021019` ≡
  `ev_dollars/collateral=0.021019` for FIX). Numbers tie out.
- **HMM regime label + the disambiguation stats.** `hmm_regime`,
  `hmm_realized_vol_252d_ann`, `hmm_realized_return_252d_ann`
  on every row (S33 F4 closer). For FIX: `normal / vol=53.5% /
  return=+133.7% ann`. The persona can tell at a glance whether
  the regime label "normal" is corroborated by the underlying vol
  + drift — for FIX, "normal" with a 134% annualized return is
  suspicious-but-honest; the operator can interrogate the
  regime call.
- **Provenance flags.** `oi_source`, `skew_source`, `premium_source`
  on every diagnostic row (S29 F1 closers). These distinguish
  *measured* from *unavailable* signals — see §1.2 below for
  what that exposes.
- **Drops summary on `.attrs`.** `frame.attrs['drops_summary']`
  rolls per-gate counts (`{'ev_threshold': 87, 'event': 68,
  'history': 11, 'premium': 1}`); the per-candidate list lives
  on `frame.attrs['drops']`. Both ride on the returned frame so
  the rank's *causal trace* is one attribute lookup away
  (S31 F1/F4 closer).

### 1.2 Where the operator is left guessing

- **What sector am I really in for the "Unknown" names?**
  11 of 20 survivors carry `sector="Unknown"` (FIX, NVR, AZO, MCK,
  FICO, EME, HII, PWR, TER, EG, CMI). The persona can't tell
  whether FIX is *Industrials* or *Consumer Discretionary* without
  going to an external sector source. This is operationally
  critical because the R9 sector-cap gate uses this exact label —
  see §3 + Finding **F-A3**.
- **How many positive-EV candidates exist in total?** The frame
  returns the top 20 by `ev_per_day`. The drops list explains
  187 of the 503-name universe (20 survivors + 167 drops), but
  *the remaining 316 tickers are not in either list*. They are
  positive-EV candidates that survived every gate and were
  silently truncated by `.head(top_n=20)`. The persona has no
  way to read "how many candidates could I take?" — see
  Finding **F-A1**.
- **Why is `regime_multiplier` always equal to `hmm_multiplier`?**
  On the Bloomberg-default path, `news_multiplier=1.0`,
  `credit_multiplier=1.0`, `dealer_multiplier=1.0`,
  `skew_multiplier=1.0`, and `tail_widening_factor≈1.0`
  uniformly across every survivor (the diagnostic table at raw
  output L42-65 confirms this). So the "rich multi-multiplier
  story" the columns advertise reduces to a single HMM overlay
  in practice. A persona unfamiliar with the data tier matrix
  could mistake "multiplier=1.0" for "signal is neutral" rather
  than "signal is structurally absent" — four of these are
  structurally absent and one fires only in vol-clustered
  regimes. See Finding **F-A4**.
- **Why is `cvar_99_evt` always NaN and `heavy_tail` always
  False?** The EVT (POT-GPD) machinery needs ≥200 scenarios to
  fit reliably; the empirical non-overlapping forward
  distribution at 35-DTE produces fewer than that, so the EVT
  fields are absent for every survivor. The column is on the
  diagnostic frame but unpopulated — the persona may assume
  "no heavy tail" when the truer answer is "EVT could not fit".
  See Finding **F-A5**.

### 1.3 Silent filtering — drops_summary census

Raw output L67-94:

```
.attrs['drops_summary'] = {'total_dropped': 167, 'by_gate':
  {'ev_threshold': 87, 'event': 68, 'premium': 1, 'history': 11}}
Coverage: 20 survivors + 167 drops = 187 of 503 universe tickers
```

Per-gate samples from raw output L77-92:

- **ev_threshold (87)** — names whose post-multiplier EV is
  negative. Examples: `ADBE -$280.87`, `CRWD -$240.57`,
  `META -$481.38`, `TSLA -$475.90`, `CVNA -$652.96`,
  `DUK -$0.00`. The negative number IS the reason, which is
  ideal for the operator (a meta-question — "why is META
  rejected?" — has the dollar answer attached).
- **event (68)** — earnings-window blocks. Examples:
  `JPM event_lockout:earnings@2026-04-14 (±5d buffer)`,
  `NFLX event_lockout:earnings@2026-04-16`. Reason carries
  the exact event date + buffer.
- **history (11)** — survivorship gate (504-day floor). Examples:
  `KMB history 203d`, `WMT history 70d`, `PLTR history 328d`,
  `PSKY history 156d`, `Q history 100d`. The operator sees
  exactly *why* a name they expected is missing — this is
  often a corporate-action artifact (KMB/WMT recent spin-off /
  splits), which the persona can confirm out-of-band.
- **premium (1)** — `AES synthetic premium too thin (<=$0.05)`.

**The 316-missing-name surface gap.** 503 − 187 = 316 tickers are
neither survivors nor drops. They are positive-EV candidates that
passed every gate and were trimmed by `.head(top_n=20)` after the
ranker's sort. The operator has no field on the returned frame
that reports "there are N total positive-EV candidates today";
they must either re-run with `top_n=10**9` or use
`select_book(tickers=None)` (which auto-sets `top_n=10**9`). See
Finding **F-A1**.

---

## 2. Ask 2 — "why was X filtered"

> *"I expected to see [a specific name]. Why didn't it make it?"*

The driver probes one ticker per gate (`ABNB`, `ACGL`, `AES`, `CPB`)
plus an operator-typo (`XYZZ`). Raw output L104-117:

```
[ev_threshold] ABNB  ev_dollars -69.00 < min_ev_dollars 0.00
[event]        ACGL  event_lockout:earnings@2026-04-28 (±5d buffer)
[premium]      AES   synthetic premium too thin (<=$0.05)
[history]      CPB   history 398d < required 504d
[typo XYZZ]    rows=0  drops=[{'ticker': 'XYZZ', 'gate': 'data',
                               'reason': "no OHLCV data..."}]
```

### 2.1 Drops are visible *only* on the universe-wide rank's `.attrs`

The per-candidate drop dict is exactly the right shape:
`{ticker, gate, reason}`. Reasons carry the *exact pivot*:
"-69.00 < 0.00", "earnings@2026-04-28 (±5d)", "398d < 504d",
"<=$0.05". The persona doesn't need to dive into source to
explain a rejection to a stakeholder. This is **the engine's
strongest single feature for daily operator use**.

### 2.2 Per-ticker re-rank as the explainer fallback

Calling `rank_candidates_by_ev(tickers=[T], min_ev_dollars=-1e9)`
for one name surfaces a candidate that the universe-wide run
suppressed (raw output L110-114):

```
ABNB  survived in isolation: ev=$-69.0  prob_profit=0.7429
                              premium=$2.831  iv=0.4422
```

So the persona can pivot from "why was ABNB dropped" → "what would
ABNB look like *if* we accepted negative EV" in one call. The
diagnostic columns are available on the survivor row, including
the regime breakdown that explains the negative number.

### 2.3 The operator-typo case

Raw output L117:

```
rows=0  drops=[{'ticker': 'XYZZ', 'gate': 'data',
                'reason': "no OHLCV data (empty or missing 'close')"}]
```

An unknown ticker is rejected at the data gate with the
correct human-readable reason. No crash, no silent zero-row
return.

---

## 3. Ask 3 — "size this within a $250k book"

> *"I have $250k to deploy. Size this for me, then walk each opener
>  through the EV-authority gate so we don't double-deploy a name or
>  breach the sector / single-name caps."*

### 3.1 `select_book` — the §2-safe knapsack

Account $250k, 25%/name concentration cap. Result (raw output
L126-127, L130-136):

```
select_book — 6 positions chosen out of 20 ranked.
  attrs: account_size=250000.0, total_collateral=242050.0,
         total_ev_dollars=2960.92, cash_remaining=7950.0,
         n_positions=6, capital_utilization=0.9682,
         selection_method=exact_knapsack
```

| ticker | strike | premium | collateral | ev_dollars | ev/day | roc | prob_profit | sector |
|---|---|---|---|---|---|---|---|---|
| LMT | 588 | 10.65 | 58800 | 707.78 | 37.25 | 0.0120 | 0.9143 | Industrials |
| HII | 374.5 | 9.42 | 37450 | 505.59 | 27.83 | 0.0135 | 0.8857 | Unknown |
| PWR | 510.5 | 12.89 | 51050 | 457.46 | 25.89 | 0.0090 | 0.9143 | Unknown |
| VRTX | 426 | 7.98 | 42600 | 444.13 | 25.14 | 0.0104 | 0.9143 | Healthcare |
| TER | 253 | 11.20 | 25300 | 443.23 | 24.40 | 0.0175 | 0.8857 | Unknown |
| ELV | 268.5 | 6.80 | 26850 | 402.73 | 22.80 | 0.0150 | 0.9143 | Healthcare |

The knapsack picked the cheaper names (LMT-ELV) over the
top-EV names (FIX/BKNG/NVR/AZO). This is the correct behaviour
for a $250k budget — FIX alone would consume $121,200 (48.5%
of NAV, breaching the 25% cap) and NVR would consume $597,150
(over 2× the budget). The persona can *see why* this is the
knapsack answer rather than the rank-time answer.

**Footgun observed:** my driver called `select_book(account_size,
ranking=ranking)` with the pre-truncated top-20 frame. The
docstring's promise of "fit against the *whole* feasible
candidate set, not a display slice" only applies when
`ranking=None` (which auto-sets `top_n=10**9`). Passing
`ranking=ranking_top_20` silently constrains the knapsack to
the displayed slice. See Finding **F-A2**.

### 3.2 Tracker in strict mode — opening seeds

Raw output L138-153:

```
WheelTracker constructed with:
  initial_capital = 250_000
  require_ev_authority = True   (D16 token gate)
  connector wired (D17 live NAV mark-to-market)
  min_nav_for_trading = 1_000

Seeding tracker with two existing short puts (so R7-R10 have evidence):
  [seed]   REGN  strike= 678.0  premium=$15.759  opened=False
  [seed]   FICO  strike= 993.0  premium=$34.56  opened=False

Tracker positions after seed: []
Tracker cash after seed: $250,000.00
```

**Both seeds rejected by D17 sector_cap_breach** (raw output L189-194,
audit-log tail):

| Seed | Token | Consume | Reject reason | Sector | Post-open % | Cap |
|---|---|---|---|---|---|---|
| REGN | `1a62…ef8c4c` | ✓ consumed @ ev=$492 | `sector_cap_breach` | Healthcare | 27.1% | 25% |
| FICO | `2a44…b9eb9d` | ✓ consumed @ ev=$1026 | `sector_cap_breach` | Unknown | 39.7% | 25% |

Note the audit-log shape — `narrative` is human-readable
("Sector 'Healthcare' would be 27.1% (limit: 25.0%). Current
positions: []") and `nav_source=live_mark_to_market` carries
the provenance. Excellent operator surface.

**Surprise for the persona:** the seed *consume succeeded* (token
discarded) before D17 rejected. So the token did its job
(D16 leg 2: positive at issue, positive at fire) and the rejection
came from the *next* gate — D17 sector cap. The audit trail
reflects this correctly with distinct `action=consume` then
`action=reject` log lines for the same token.

This is the correct §2 behaviour: D17 hard-blocks are *post* D16
token consume. A token retained for retry only happens on
`stale_ev` (the D16 leg-2 rejection), not on a D17 hard-block.

### 3.3 Dossier verdicts on the top 20 with `PortfolioContext` attached

Raw output L157-180:

```
Dossiers built: 20

Verdict distribution:
  verdict=  review : 20

Verdict-reason distribution:
  reason=         chart_context_missing : 20
```

**Every single dossier returns `review` with reason
`chart_context_missing`.** This is correct per the reviewer's R2:
if `chart_context is None or not chart.is_ok()`, the reviewer
returns `("review", "chart_context_missing", notes)` *before*
reaching R5/R6/R7/R8/R9/R10. Because the production-default
Bloomberg path has no chart provider attached, the dossier
layer's R7-R10 portfolio-risk soft-warns are **structurally
never reached in a default operator session**.

The persona's R7-R10 soft-warn read therefore comes ONLY from
the tracker layer (where D17 hard-blocks fire), never from the
dossier layer. See Finding **F-A6**.

### 3.4 The full §2 path — `consume_ranker_row` and the audit log

Top survivor FIX wired into the tracker via `consume_ranker_row`
(raw output L182-194):

```
candidate: FIX 1212.0P  premium=$47.187  ev=$2547.56
open_short_put returned: False
tracker positions now: []
tracker cash now: $250,000.00

audit-log tail (FIX row):
  action=issue   token=676f…2fcb
  action=consume current_ev_dollars=2547.56 ticker=FIX
  action=reject  reason=sector_cap_breach sector=Unknown
                 post_open_sector_pct=0.4848  sector_limit=0.25
                 nav=250000.0 nav_source=live_mark_to_market
                 narrative="Sector 'Unknown' would be 48.5%
                            (limit: 25.0%). Current positions: []"
```

The §2 chain works exactly as designed:

1. `issue_ev_authority_token` succeeds (D16 leg 1: EV $2547.56 > 0).
2. `_consume_ev_authority_token` succeeds (D16 leg 2:
   `current_ev_dollars=2547.56 > 0`, token discarded).
3. `_evaluate_d17_hard_blocks` fires sector_cap_breach (FIX is
   "Unknown" sector; alone it would be 48.5% of $250k NAV).
4. `open_short_put` returns False without opening a position.

The persona can read the rejection narrative, see the math
("48.5%"), see the gate ("sector_limit=0.25"), and see the
NAV provenance ("live_mark_to_market"). Audit fidelity is
high. The *operational* implication — Unknown is acting as a
single-sector bucket — is a downstream Finding (§6).

---

## 4. Ask 4 — "what's the downside if I get assigned"

> *"For the trade we just opened, walk me through the downside —
>  distribution percentiles, CVaR, what regime we're in, and what
>  the chart-side signals say about a bounce vs further drawdown."*

Anchor candidate FIX (raw output L202-247).

### 4.1 Distribution shape — P25 / P50 / P75 / CVaR_5

```
pnl_p25     = 4647.29
pnl_p50     = 4647.29
pnl_p75     = 4647.29
cvar_5      = 4102.46
IQR (P75 - P25) = $0.00
```

**The P25/P50/P75 percentiles are degenerate for the top survivor.**
The "headline P&L distribution spread" the engine advertises
collapses to a single point estimate for FIX. The full output
table (L20-65) confirms this is true for *every one of the 20
survivors*: P25 == P50 == P75 in every row.

The cause is structural: at 35 DTE on an empirical
non-overlapping forward distribution, the sample size is too
small for the three percentiles to separate (only a handful of
non-overlapping 35-day windows fit in the 504-day history). The
fields are populated by the engine and rounded to 2dp — they're
not NaN, they're not None — but they don't differ. So the
operator reading "P25=$4647, P50=$4647, P75=$4647" gets the
*correct value* with *zero distributional information*. See
Finding **F-A7** — this is a real surface gap.

Note: the engine's docstring (`engine/ev_engine.py:166-174`)
says the percentile fields are "computed alongside cvar_5" and
"NaN when the distribution has fewer than 4 samples". Here we
have ≥4 samples (cvar_5 is populated at $4102.46) but the
percentiles still collapse. The collapse is data-induced
(non-overlapping 35-day window count is small) rather than the
"<4 sample" guard. The collapse is silent — no warning column.

### 4.2 Heavy-tail diagnostics — EVT cvar_99, tail_xi, heavy_tail flag

```
cvar_99_evt = nan
tail_xi     = nan
heavy_tail  = False
```

NaN for the EVT extreme-value fits across every survivor. The
EVT (POT-GPD) layer needs ≥200 scenarios to fit reliably;
empirical NOS at 35 DTE produces <200, so the fit doesn't run.
`heavy_tail=False` is the dataclass default and SHOULD NOT be
read as "the engine has assessed this trade and it's not
heavy-tailed". The persona can be misled here — see Finding
**F-A5**.

### 4.3 Regime + dealer overlays — what scaled the final EV

```
hmm_regime               = normal
hmm_multiplier           = 0.621
hmm_realized_vol_252d_ann = 0.5351
hmm_realized_return_252d_ann = 1.3366
credit_regime            = benign
credit_multiplier        = 1.0
news_multiplier          = 1.0
news_n_articles          = 0
regime_multiplier        = 0.621
tail_widening_factor     = 1.0
dealer_regime            = None
dealer_multiplier        = 1.0
gex_total                = None
gamma_flip_distance_pct  = None
nearest_put_wall_strike  = None
nearest_call_wall_strike = None
skew_multiplier          = 1.0
skew_source              = unavailable
```

The **only** multiplier that actually moved FIX's EV away from
`ev_raw` was the HMM regime (0.621). News, credit, skew, dealer,
and tail widening are all 1.0 (= no effect). The reason matrix
on Bloomberg-default:

| Multiplier | Why it's 1.0 on the default path |
|---|---|
| `news_multiplier` | Bloomberg news store is empty; `news_n_articles=0` (D18 confirms verbal news has zero EV influence). |
| `credit_multiplier` | `credit_regime=benign` per the FRED HY-OAS reader — only `crisis` (0.80) and `stressed` (0.92) tilt this away from 1.0. |
| `dealer_multiplier` | Bloomberg has no option chain (`MarketDataConnector` exposes no `get_options`), so `DealerPositioningAnalyzer` never runs (`dealer_regime=None`, walls=`None`). |
| `skew_multiplier` | Same root cause — no chain → no 25Δ points → no slope. `skew_source=unavailable` correctly flags this. |
| `tail_widening_factor` | RV30/RV252 widening only fires when the realized-vol ratio exceeds threshold (1.30 trigger). FIX's regime is "normal" (vol/ret stats above) so widening does not fire. |

The engine's *only live overlay* on the default operator path is
the HMM. The persona reading the diagnostic columns may believe
they have a five-axis sanity check on the rank; in practice it's
single-axis. See Finding **F-A4** — this is critical operational
context for anyone interpreting the engine's verdict.

### 4.4 Assignment economics

```
spot                 = $1360.79
strike               = $1212.0
premium              = $47.187
notional if assigned = $121,200.00
cost basis if assigned (strike - premium) × 100 = $116,481.30
breakeven_move_pct (engine column) = -0.143  ⇒ -14.3%
prob_profit / prob_assignment = 0.9714 / 0.0286
collateral = $121,200.0
roc        = 0.021019  ⇒ 2.10% over 35 days
roc cross-check ev_dollars/collateral = 0.021019
```

Q's read on FIX as an *assignment-scenario decision*:

- **97.14% chance the put expires worthless.** Premium ($4719 for
  one contract) is collected free.
- **2.86% chance of assignment** at $1212 ⇒ cost basis $1164.81
  per share, $116,481 for the round lot.
- **Breakeven move** (raw column `breakeven_move_pct=-0.143`) —
  the underlying must drop 14.3% from $1360.79 to breakeven
  on assignment. This number is a free, ready-to-share talking
  point for the trade.
- **ROC 2.10% over 35 days** = ~22% annualised on collateral if
  held to expiry — capital-efficient and worth tying up the
  $121k for, *if* the operator was not constrained by the §3
  sector cap.

What the persona DOESN'T see and might want, when asking
"downside if assigned":

- **The post-assignment covered-call ladder.** No "what's my
  call-side roll plan" is surfaced; the persona has to
  separately call `WheelRunner.rank_covered_calls_by_ev` for
  that name.
- **Distributional spread of the assignment loss.** As noted in
  §4.1, P25=P50=P75 collapses — so "what's the 25th-percentile
  loss given assignment?" returns the same point estimate as
  P50 / P75. Surface gap, not a §2 violation.

---

## 5. §2 invariants observed in production code paths

All five negative-control checks landed cleanly (raw output L249-291).

### 5.1 D16 leg 1 — `issue_ev_authority_token` refuses `ev_dollars ≤ 0`

```
OK EVAuthorityRefused raised: Refusing EV-authority token for ZZZZ
   strike=100.0 — ev_dollars=-42.0 is non-positive; R1 would block.
```

The exception message is operator-readable and cites the gate that
would fire if this somehow reached open_short_put (R1).

### 5.2 D16 leg 2 — `open_short_put` consume rejects stale EV

```
issued token for positive EV row: 19a0baad869669be…
open_short_put returned: False  (expect False)
audit-log tail: {'action': 'reject', 'reason': 'stale_ev',
                 'ticker': 'ZZZZ', 'token': '…',
                 'current_ev_dollars': 0.0}
```

A token issued for a positive-EV row, then consumed with
`current_ev_dollars=0`, is rejected with `reason='stale_ev'`.
The audit log records the actual `current_ev_dollars` so the
operator can see which leg failed (the consume re-check, not
the issuance).

### 5.3 Dossier R1 — negative EV → `verdict='blocked'`

```
ZZNG  verdict=blocked  reason=negative_ev
  note: engine ev_dollars=-15.50 < 0 - chart cannot upgrade negative EV
```

### 5.4 Dossier R1a — non-finite EV → `verdict='blocked'`, distinct reason

```
ZZNF  verdict=blocked  reason=ev_non_finite  ev_dollars=nan
  finite? False
```

R1a's distinct `verdict_reason='ev_non_finite'` lets the audit
trail tell an unparseable engine value apart from an evaluated
loss — important per PR #204's framing.

### 5.5 Reviewer never upgrades — a perfect chart on a negative-EV row still produces `'blocked'`

```
ZZNG  verdict=blocked  reason=negative_ev  (perfect chart attached)
  note: engine ev_dollars=-15.50 < 0 - chart cannot upgrade negative EV
```

The `_PerfectChartProvider` in the driver returns a `ChartContext`
with `visible_price` matching `engine_spot` and `error=""` —
exactly the shape a successful TradingView capture would return.
The reviewer's R1 fires *before* R2/R3/R4/R5, so the perfect
chart never gets a chance to "rescue" the negative EV. **This
is the CLAUDE.md §2 hard guardrail observed live.**

---

## 6. Findings

> **Severity scale (loose):**
> **SURFACE** = the engine has the data but the operator can't see it
> without diving into docs/source. **GAP** = the operator can't make
> a defensible decision with what the engine surfaces today.
> **§2** = an invariant CLAUDE.md §2 names is observable as upheld
> (positive finding) or violable (would be a hard bug — none
> observed in this read-only walkthrough, but logged distinctly so
> the Major Session can audit).

| # | Severity | Title | Pointer |
|---|---|---|---|
| **F-A1** | SURFACE | 316 of 503 universe tickers are silently truncated by `top_n` | §1.3 + §1.2 |
| **F-A2** | SURFACE | `select_book(ranking=ranking)` silently uses the truncated frame | §3.1 |
| **F-A3** | SURFACE → GAP | 11 of 20 survivors collapse to `sector="Unknown"`; R9 treats Unknown as one bucket | §1.2 + §3.4 |
| **F-A4** | SURFACE → GAP | Bloomberg-default path: 4 multipliers structurally 1.0 (news / credit / dealer / skew); tail-widening fires only 2/20; HMM dominates the regime overlay | §1.2 + §4.3 |
| **F-A5** | SURFACE | `cvar_99_evt`, `tail_xi`, `heavy_tail=False` look like assessments but are unevaluated defaults at 35 DTE | §1.2 + §4.2 |
| **F-A6** | SURFACE → GAP | Dossier R7-R10 portfolio-risk soft-warns are unreachable on Bloomberg-default path (R2 fires first) | §3.3 |
| **F-A7** | SURFACE | `pnl_p25 == pnl_p50 == pnl_p75` for every one of the 20 survivors — distribution headline collapses | §4.1 |
| **F-A8** | §2 (positive) | D17 sector_cap_breach fires correctly 3× with full audit-log narrative + NAV provenance | §3.2 + §3.4 |
| **F-A9** | §2 (positive) | D16 leg 1 + leg 2, dossier R1 + R1a, "reviewer never upgrades" all observed upheld | §5 |

Detail follows.

### F-A1 — 316 of 503 universe tickers are silently truncated by `top_n`

**Severity:** SURFACE.

**What.** `rank_candidates_by_ev(tickers=None, top_n=20)` returns
20 rows. `.attrs['drops']` accounts for 167 names. The remaining
**316** tickers are positive-EV survivors that passed every gate
and were trimmed by `.head(20)` after the sort. The frame has no
field that reports the total positive-EV survivor count, and no
trim-aware analog of `drops_summary` (e.g. `attrs['trimmed_count']`).

**Where.** `engine/wheel_runner.py:1795-1797`:

```python
df = pd.DataFrame(rows)
if not df.empty:
    df = df.sort_values("ev_per_day", ascending=False).head(top_n)
```

**Why it matters.** A persona asking "how many trades could I take
right now?" cannot answer from the rank's output alone. The answer
on 2026-03-20 was 336 positive-EV candidates (20 shown + 316
silently trimmed); the rank advertises 20. Operationally this is
the difference between "the engine found 20 trades" (defensive
read) and "the engine found 336 candidates and showed me 20"
(true).

**Suggested fix shape (NOT a fix).** Attach a one-line summary
to `frame.attrs` alongside `drops_summary` — e.g.
`{'survivors_total': 336, 'survivors_shown': 20,
'survivors_trimmed': 316}` — costs the engine nothing (the count
is `len(rows)` before head) and closes the discoverability gap.

### F-A2 — `select_book(ranking=ranking)` silently uses the truncated frame

**Severity:** SURFACE.

**What.** The docstring of `select_book` (`engine/wheel_runner.py:1901`)
promises that the book is "fit against the *whole* feasible
candidate set, not a display slice. … Default `top_n` wide open
here; an explicit caller value still wins." This is true only
when `ranking=None`. Passing a pre-truncated `ranking` frame
silently constrains the knapsack to that frame's slice.

**Where.** `engine/wheel_runner.py:1992-2001`:

```python
if ranking is None:
    rank_kwargs.setdefault("top_n", 10**9)
    ranking = self.rank_candidates_by_ev(tickers=tickers, **rank_kwargs)
```

**Why it matters.** A persona reading the docstring (or
`MODULE_INDEX.md`) and following its idiomatic
"rank then size" pattern (`ranking = rank(top_n=20);
book = select_book(account, ranking=ranking)`) silently
fits the knapsack to a 20-name pool, NOT the 336-name pool the
docstring leads them to expect. The driver observed exactly
this — 6 of 20 picked, but the optimal book over 336 names
might pick differently.

**Suggested fix shape (NOT a fix).** Either widen `top_n` inside
`select_book` whenever a `ranking` is passed and add a warning
columns, OR carry a `survivor_pool_complete` flag on
`ranking.attrs` that `select_book` checks.

### F-A3 — 11 of 20 survivors collapse to `sector="Unknown"`

**Severity:** SURFACE → GAP.

**What.** The `DEFAULT_SECTOR_MAP` in `engine/wheel_runner.py`
doesn't cover FIX, NVR, AZO, MCK, FICO, EME, HII, PWR, TER, EG,
CMI. All 11 carry `sector="Unknown"` on the ranked frame.
`engine/portfolio_risk_gates.check_sector_cap` then treats
"Unknown" as a real sector when aggregating exposure.

**Where.** Sector source is `DEFAULT_SECTOR_MAP`
(`engine/wheel_runner.py` symbol — see `engine/risk_manager.SectorExposureManager`).

**Why it matters.** The driver observed FIX alone consume 48.5%
of $250k NAV in the Unknown bucket (raw output L193), and FICO
consume 39.7% (raw output L190). Two of three R9 hard-blocks
fired against "Unknown" as the breaching sector — i.e. R9 was
acting on a *missing-data label*, not on a real sector concentration.
The persona reads "Sector 'Unknown' would be 48.5%" and may not
realise that FIX, FICO, NVR, AZO etc. are NOT actually in the same
sector — they're just all missing from the map.

**Suggested fix shape (NOT a fix).** Either source the sector map
from `data/bloomberg/sp500_fundamentals.csv` (which has GICS for
~all SP500 names; S43 audit corrections wired this for backtests)
or treat `sector="Unknown"` as a no-op for `check_sector_cap`
(skip the bucket, fire the single-name cap R10 instead). The
current behaviour is internally consistent but operationally
misleading.

### F-A4 — Four multipliers always 1.0 + one mostly 1.0 on Bloomberg-default

**Severity:** SURFACE → GAP.

**What.** Four of the five non-HMM multipliers
(`news_multiplier`, `credit_multiplier`, `dealer_multiplier`,
`skew_multiplier`) are 1.0 across **all 20** survivors. The
fifth, `tail_widening_factor`, is 1.0 across **18 of 20**
survivors with only BKNG (1.0368) and AZO (1.0055) firing the
RV30/RV252 widening. So
`regime_multiplier = ev_dollars / ev_raw` is dominated by
`hmm_multiplier` for every name and equals it to ~3 decimal places
on 18 of 20 rows.

**Where.** Each multiplier has its own structural reason:

- `news_multiplier`: empty Bloomberg news store ⇒
  `news_sentiment=0.0` ⇒ multiplier 1.0 (D18 confirms).
- `credit_multiplier`: FRED HY-OAS reads `benign`; only
  `crisis` / `stressed` move this away from 1.0.
- `dealer_multiplier`, `skew_multiplier`,
  `nearest_put_wall_strike`, `gex_total`: Bloomberg path
  exposes no chain (`MarketDataConnector` has no `get_options`).
  Confirmed by `skew_source=unavailable` and
  `oi_source=fallback`.
- `tail_widening_factor`: RV30/RV252 widening only fires when
  the ratio crosses the threshold (BKNG 1.0368, AZO 1.0055
  observed in the survivor frame; the other 18 stayed at 1.0).

**Why it matters.** A persona reading the diagnostic columns
believes they have a five-factor regime overlay on the raw EV.
In practice the engine ships a single-factor (HMM) overlay
on the default operator path. This is not a bug — but it should
be *explicit* (e.g. a `live_overlays_count` field).

**Suggested fix shape (NOT a fix).** Either compute and surface
`live_overlay_count` (how many multipliers actually moved EV
away from 1.0 for this row), or document explicitly in the
schema that on Bloomberg-default these five are dormant.

### F-A5 — EVT fields look like assessments but are unevaluated defaults

**Severity:** SURFACE.

**What.** `cvar_99_evt=NaN`, `tail_xi=NaN`, `heavy_tail=False` for
every survivor. The `False` for `heavy_tail` is the dataclass
default (`engine/ev_engine.py:165`) — not "the engine has
assessed and concluded no heavy tail". The NaN values reflect
the EVT fit not running (needs ≥200 scenarios; 35-DTE empirical
NOS produces fewer).

**Where.** `engine/ev_engine.py:163-165`:
```python
cvar_99_evt: float = float("nan")
tail_xi: float = float("nan")
heavy_tail: bool = False
```

**Why it matters.** A persona reading `heavy_tail=False` for FIX
may conclude "the engine has assessed FIX as not heavy-tailed".
The truer answer is "the engine could not assess heavy-tailness
at 35 DTE". Distinct semantically.

**Suggested fix shape (NOT a fix).** Either use `None` instead of
`False` for `heavy_tail` when the EVT fit didn't run, or surface
a `tail_fit_status` field with values like `not_attempted` /
`insufficient_data` / `fit_ok`.

### F-A6 — Dossier R7-R10 unreachable on Bloomberg-default

**Severity:** SURFACE → GAP.

**What.** The reviewer's R2 — "if chart context is missing,
verdict=review, reason=chart_context_missing" — fires for
every dossier on the default operator path (no chart provider
attached). R5/R6/R7/R8/R9/R10 are never reached. The
PortfolioContext machinery the driver wired up is silently
unused at the dossier layer; R7-R10 only fire via tracker
hard-blocks (which the driver observed working).

**Where.** `engine/candidate_dossier.py:275-279`:
```python
chart = dossier.chart_context
if chart is None or not chart.is_ok():
    ...
    return "review", "chart_context_missing", notes
```

**Why it matters.** Operationally, the "soft-warn dossier preview"
of the tracker's hard-block decisions (R9, R10 in particular) is
the system's intended discoverability layer for portfolio-risk
breaches. On the production-default Bloomberg path that preview
is invisible — the operator only sees the breach at the
*open_short_put* layer, after they've committed to placing a
trade. The dossier becomes a binary "needs chart vs blocked"
filter rather than the rich proceed/review/skip/blocked ladder
the design promises.

**Suggested fix shape (NOT a fix).** This is by design when no
chart is present, but a distinct verdict_reason
(`portfolio_review_only_no_chart`) or an `attrs['gates_eval_status']`
on the dossier would let the operator see *which* R-rules were
actually evaluated vs which were skipped because R2 returned
first. Alternatively, allow R7-R10 to run when chart is missing
but mark the result as `partial`.

### F-A7 — `pnl_p25 == pnl_p50 == pnl_p75` for every survivor

**Severity:** SURFACE.

**What.** The "headline P&L distribution percentiles" collapse for
every survivor at 35 DTE on the empirical non-overlapping
distribution. The driver observed `IQR = $0.00` for the top
survivor; the table at raw output L20-65 confirms this pattern
across all 20 rows.

**Where.** `engine/ev_engine.py:166-174` (dataclass) +
`engine.forward_distribution` + the percentile computation in
`engine/ev_engine.py` `evaluate`.

**Why it matters.** The percentiles are the engine's promised
"verdict-as-distribution-not-point-estimate" surface. When they
collapse, the persona loses the distributional read. Cvar_5
remains meaningful (it samples the tail) but the body of the
distribution is lost.

**Suggested fix shape (NOT a fix).** Either flag the collapse with
a `pnl_percentile_status` field (`degenerate` /
`insufficient_samples` / `ok`) or fall back to the empirical
forward-distribution sample's own percentiles when the post-EV
P&L distribution is degenerate. This is well below §2 — it's
operator surface.

### F-A8 — D17 sector_cap_breach fires correctly 3× with full audit trail

**Severity:** §2 (positive).

**What.** Three D17 hard-blocks fired in the driver run (REGN /
FICO / FIX), each rejected at the tracker layer after D16 token
consume succeeded. Each rejection produced an audit-log entry
with:

- `action='reject'`,
- `reason='sector_cap_breach'`,
- `sector` (Healthcare / Unknown / Unknown),
- `post_open_sector_pct` (0.2712 / 0.3972 / 0.4848),
- `sector_limit=0.25`,
- `nav=250000.0`,
- `nav_source='live_mark_to_market'`,
- `narrative=` human-readable summary,
- the original token + `current_ev_dollars`.

**Why it matters.** This is exactly the audit-log shape an
operator needs to *prove* to a stakeholder why a trade was
refused. The narrative carries the math; `nav_source` reveals
that the cap was evaluated against a live mark, not a static
fallback; the token is preserved so the trade can be reproduced.

### F-A9 — D16 legs, dossier R1/R1a, "reviewer never upgrades" all upheld

**Severity:** §2 (positive).

**What.** The five negative-control checks in §5 all landed as
designed: `issue` refuses negative EV; `consume` rejects stale
EV; R1 blocks; R1a blocks with distinct reason; perfect chart
on negative-EV row still blocks.

**Why it matters.** This is the CLAUDE.md §2 "No tradeable
candidate bypasses `EVEngine.evaluate`" invariant observed live
under realistic conditions. The audit-log shapes, exception
messages, and verdict reasons are all operator-readable.

---

## 7. Reproducibility

```bash
# From any worktree (the driver bootstraps sys.path to this worktree's
# absolute path — edit the WORKTREE constant at the top of the driver
# if running outside swe-terminal-a):
"/c/Users/merty/AppData/Local/Programs/Python/Python312/python.exe" \
    docs/verification_artifacts/persona_walkthrough_driver.py \
    > docs/verification_artifacts/persona_walkthrough_$(date +%Y-%m-%d)_raw_output.txt
```

Output is deterministic given the same Bloomberg CSVs + the same
`as_of`. The driver forces UTF-8 stdout via `sys.stdout.reconfigure`
so an em-dash / `§` in the redirect target doesn't crash Windows
default cp1252 encoding.

The driver runtime on the dev box (i9-class, Bloomberg CSVs cached)
is approximately 5-15 minutes for the full SP500 scan + dossier
batch + tracker walkthrough + §2 invariants. The HMM regime cache
(`WheelRunner._hmm_regime_cache`) is per-process so a fresh run
recomputes the regime fit for each of ~336 candidates that reach
the EV evaluation; the regime label is itself deterministic because
`GaussianHMM(..., random_state=42)`.

---

## 8. References

- `CLAUDE.md` §2 — the EV-authority invariant.
- `PROJECT_STATE.md` §1 — authoritative entry points (rank, dossier,
  tracker, api).
- `DECISIONS.md` D16 — EV-authority token (issuance + consume
  predicate).
- `DECISIONS.md` D17 — portfolio-risk gates (R7-R10 + tracker
  hard-blocks).
- `DECISIONS.md` D18 — verbal news severed from EV path (PR #249).
- `engine/wheel_runner.py:725` — `rank_candidates_by_ev`.
- `engine/wheel_runner.py:1901` — `select_book`.
- `engine/candidate_dossier.py:130` — `EnginePhaseReviewer`.
- `engine/candidate_dossier.py:535` — `build_dossiers`.
- `engine/wheel_tracker.py:346` — `issue_ev_authority_token`.
- `engine/wheel_tracker.py:603` — `consume_ranker_row`.
- `engine/wheel_tracker.py:1629` — `portfolio_context_snapshot`.
- `engine/portfolio_risk_gates.py` — `check_var`, `check_stress_scenario`,
  `check_sector_cap`, `check_single_name_cap`, `PortfolioContext`.
- `docs/verification_artifacts/README.md` — driver placement +
  re-run conventions.
- Companion verification drivers in this directory:
  `realism_verify_driver.py`, `f4_baseline_driver.py`,
  `s41_f4_validation_driver.py`.

---

_Generated by an autonomous Terminal A session under HT-A. The
findings doc + driver + raw output are committed together so a
future agent can re-run, diff, and audit the findings against the
captured artifact._
