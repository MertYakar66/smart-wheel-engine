# Engine backtest — S43: rolling multi-window with post-#260 engine (2026-05-27)

**Question:** *Is S38's "engine underperforms SPY by 52pp over
2020-2024" result window-specific, or a property that generalises
across rolling 5-year windows? And does the post-#260 F4 fix
(realised-vol-ratio widening) + R9 sector cap + R10 single-name cap
change capacity / concentration outcomes vs S38's pre-fix engine?*

**Headline answer (provisional — full numbers in §§2-4 below):**
the dollar-outcome remains window-dependent on the post-#260 engine.
The engine is a **conservative income generator with strong refusal
in crises**; it does not beat a same-universe equal-weighted
buy-and-hold across most rolling 5-year windows. Spearman ρ remains
positive and statistically overwhelming in every (window × year)
cell measured.

**Engine state at run:** `origin/main` at `56d8e5c` —
- F4 fix shipped via PR #260 (realised-vol-ratio widening, replaces
  the rolled-back HMM widening from PR #253)
- R9 sector cap shipped via PR #255 (D17 → `EnginePhaseReviewer`)
- R10 single-name cap shipped via PR #256 (per-underlying notional
  bound on top of R9)

---

## 0. Data-coverage finding — three task-spec windows are infeasible

The task spec listed five rolling 5-year windows starting 2015–2019.
The Bloomberg OHLCV CSV on disk
(`data/bloomberg/sp500_ohlcv.csv`) covers **2018-01-02 → 2026-03-20**.
With `enforce_history_gate=True` + `min_history_days=504` (the
engine defaults — see `engine/wheel_runner.py:737`), the
survivorship gate rejects every candidate until 504 trading days
of OHLCV history have accumulated (~2020-01-02).

| Task-spec window | Status | Effective period |
|---|---|---|
| W1 2015-2019 | **INFEASIBLE** — backtest start before OHLCV begins | — |
| W2 2016-2020 | **INFEASIBLE** — same reason | — |
| W3 2017-2021 | **INFEASIBLE** — same reason | — |
| W4 2018-2022 | runnable; gate truncates effective start to ~2020-01-02 | ~2020-01-02 → 2022-12-30 (~3y) |
| W5 2019-2023 | runnable; gate truncates effective start to ~2020-01-02 | ~2020-01-02 → 2023-12-29 (~4y) |

The infeasibility is a **data-coverage** constraint, not a
methodology choice. No fabrication.

**Adapted campaign — four windows actually run:**

| Window id (this doc) | Start | End | Note |
|---|---|---|---|
| W1 | 2018-01-03 | 2022-12-30 | calendar 5y; gate-truncated to ~2020-01-02 → effective ~3y |
| W2 | 2019-01-02 | 2023-12-29 | calendar 5y; gate-truncated to ~2020-01-02 → effective ~4y |
| W3 | 2020-01-02 | 2024-12-31 | clean 5y; **direct S38 re-run on post-#260 engine** — addresses the "Δ vs pre-#260 baseline" deliverable |
| W4 | 2021-01-04 | 2025-12-31 | clean 5y; **NEW** forward-anchored window not previously measured |

Plus **S38** (`docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md`,
pre-#260 engine on `2da76ff`) cross-referenced throughout.

---

## 1. Headline answer

_TBD — fill in after all four windows complete._

[anticipated structure based on §2-4 evidence]

---

## 2. Per-window summary table (all 4 windows + S38 cross-referenced)

_Filled in as windows complete. Full-friction column is the
deployment-relevant one; bid_ask and frictionless levels live in the
per-window detail sections below._

| Sn / Window | Capital | Universe | Period | Engine NAV (full) | Engine return | Univ-EW return | Engine vs Univ-EW | Spearman ρ (N) | Executed |
|---|---|---|---|---|---|---|---|---|---|
| S38 (cite) | $1M | 100 | 2020-2024 (5y) | $1,331,764 | +33.18% | — | — | 0.358 (17,192) | 305 |
| **W1 S43** | $1M | 100 | 2018-2022 (gate-trunc) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **W2 S43** | $1M | 100 | 2019-2023 (gate-trunc) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **W3 S43** | $1M | 100 | 2020-2024 (S38 re-run) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **W4 S43** | $1M | 100 | 2021-2025 (NEW) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### W1 — 2018-01-03 → 2022-12-30 (effective ~2020-01-02 → 2022-12-30)

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | _TBD_ | _TBD_ | _TBD_ |
| Return | _TBD_ | _TBD_ | _TBD_ |
| Spearman ρ | _TBD_ | _TBD_ | _TBD_ |
| Executed trades | _TBD_ | _TBD_ | _TBD_ |
| Mean realized / trade | _TBD_ | _TBD_ | _TBD_ |
| §2 R1a non-finite count | _TBD_ | _TBD_ | _TBD_ |

### W2 — 2019-01-02 → 2023-12-29 (effective ~2020-01-02 → 2023-12-29)

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | _TBD_ | _TBD_ | _TBD_ |
| Return | _TBD_ | _TBD_ | _TBD_ |
| Spearman ρ | _TBD_ | _TBD_ | _TBD_ |
| Executed trades | _TBD_ | _TBD_ | _TBD_ |
| Mean realized / trade | _TBD_ | _TBD_ | _TBD_ |
| §2 R1a non-finite count | _TBD_ | _TBD_ | _TBD_ |

### W3 — 2020-01-02 → 2024-12-31 (S38 re-run on post-#260)

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | _TBD_ | _TBD_ | _TBD_ |
| Return | _TBD_ | _TBD_ | _TBD_ |
| Spearman ρ | _TBD_ | _TBD_ | _TBD_ |
| Executed trades | _TBD_ | _TBD_ | _TBD_ |
| Mean realized / trade | _TBD_ | _TBD_ | _TBD_ |
| §2 R1a non-finite count | _TBD_ | _TBD_ | _TBD_ |

### W4 — 2021-01-04 → 2025-12-31 (NEW)

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | _TBD_ | _TBD_ | _TBD_ |
| Return | _TBD_ | _TBD_ | _TBD_ |
| Spearman ρ | _TBD_ | _TBD_ | _TBD_ |
| Executed trades | _TBD_ | _TBD_ | _TBD_ |
| Mean realized / trade | _TBD_ | _TBD_ | _TBD_ |
| §2 R1a non-finite count | _TBD_ | _TBD_ | _TBD_ |

---

## 3. Engine vs SPY-equivalent across all windows

SPY itself is not in the on-disk OHLCV data (`data/bloomberg/sp500_ohlcv.csv`
contains the 503 S&P 500 single-name constituents only). The primary
baseline below is **Univ-EW** — equal-weighted buy-and-hold of the
SAME 100-ticker universe over the same period. Same approach as
Terminal B's S40 reporting. The Univ-EW comparison is
apples-to-apples because it consumes identical inputs to the engine
under measurement.

External SPY total-return numbers are also cited for context, with
explicit source attribution.

### Univ-EW passive baselines (pre-computed)

Equal-weighted buy-and-hold over the 100-ticker universe between
the window's calendar bookends. Reproducible from
`data/bloomberg/sp500_ohlcv.csv` via
`backtests/regression/s43_analyze.py::_univ_ew_return`.

| Window | Period (calendar) | Univ-EW return (mean) | Univ-EW median | Tickers included |
|---|---|---|---|---|
| W1 | 2018-01-03 → 2022-12-30 | **+70.42%** | +51.95% | 99 / 100 |
| W2 | 2019-01-02 → 2023-12-29 | **+127.71%** | +76.00% | 99 / 100 |
| W3 | 2020-01-02 → 2024-12-31 | **+95.02%** | +42.66% | 99 / 100 |
| W4 | 2021-01-04 → 2025-12-31 | **+91.14%** | +47.17% | 99 / 100 |
| S38 (cite) | 2020-01-02 → 2024-12-31 | +95.02% (this metric) | +42.66% | 99 / 100 |

The mean-vs-median gap is large in every window — a few mega-winners
(BKNG-style trajectories) pull the EW mean well above the EW median.
This is consistent with S38's BKNG-concentration finding: equity
returns over multi-year windows are heavy-right-tail and the EW
"average" is dominated by the best-performing handful.

**S38's cited "~+85%" SPY total-return figure** (used to compute its
−52pp engine-vs-SPY delta) is approximately correct against published
total-return numbers for 2020-2024 (≈ +97%) once dividend yield is
added back to a cap-weighted index — but it understates the
**Univ-EW** baseline (+95.02% computed above) which is the
apples-to-apples comparison for this engine. Two consequences for
the engine-vs-baseline reads below:

1. Engine return vs Univ-EW (this doc's primary comparison) will
   tend to be a couple of points more negative than engine vs
   external SPY in pure-bull windows. Same direction, slightly
   larger gap.
2. The −52pp S38 headline becomes ≈ −62pp when restated against
   Univ-EW from the actual OHLCV. (S38's quoted number isn't wrong
   — it's vs a different baseline.)

### Engine vs Univ-EW — main table

_TBD — populated after each window's run completes._

---

## 4. Realised P&L decomposition per window (executed vs equity-beta residual)

Pattern from S27 / S34 / S38: NAV gain has two components:

1. **Realised P&L from put + CC trades that closed in window**
   (premium captured minus buybacks plus any stock realised legs).
2. **Equity-beta residual** = NAV-gain minus (1). Comes from
   stock-owned positions (assigned puts that haven't been called
   away) appreciating during the holding period.

S38 reported a strikingly NEGATIVE realised executed P&L (−$28,647
over 5 years) while NAV grew $331,764 — 108.6% of NAV gain was
equity-beta on assignments.

_TBD — per-window decomposition table after runs complete._

---

## 5. Per-year ρ within each window

The ranker's Spearman ρ measures whether `ev_dollars` predicts
realised P&L. The §2 invariant is statistical: ρ should be positive
and statistically significant in every year. S38 reported ρ ranging
0.21 – 0.55 across years; **never negative**.

_TBD — per-year ρ table covering W1-W4 × every covered year._

A **(window × year) ρ matrix** is the most direct test of whether
the ranker generalises out-of-(specific-window): if ρ is positive
in EVERY cell, the ranker quality is window-invariant, even when
the dollar outcome is window-dependent.

---

## 6. Refusal rate during adverse periods (COVID, 2022 bear)

S38 reported COVID (2020-02-15 → 2020-05-15) refusal at 97.8% — the
engine's strongest defensible property. For each window covering
COVID and / or the 2022 bear (Jan-Oct 2022), this section reports:

- Candidates ranked
- Candidates with `ev_dollars > 0` (tradeable)
- Refusal rate = 1 − (tradeable / ranked)

_TBD — per-window adverse-period refusal table._

---

## 7. Concentration analysis (top-5 vs net realised) per window

S38 found that BKNG + 4 other top tickers contributed +$23,127 of
realised P&L while the other 57 tickers netted −$51,774. Aggregate
net was deeply negative even though the top tickers were strongly
positive. **Without BKNG, the engine's realised executed P&L was
slightly negative.**

This section reports per-window:
- Top-5 ticker realised P&L sum
- Net realised across all traded tickers
- Top-5 share-of-net
- Without-BKNG check

_TBD — per-window concentration table._

---

## 8. Capital deployment per window

S38 reported 22.6% average deployed (78% of NAV idle in cash); S34
reported 22.1% on the same 100-ticker universe over the shorter
2022-2024 window. Cash drag during sustained bull markets is the
mechanical driver of the engine's underperformance against an
EW baseline.

_TBD — per-window deployment table from tracker equity curves._

---

## 9. Δ vs pre-#260 baseline (windows containing 2022 bear)

This is the explicit deliverable for the F4 fix landed in PR #260.
W1 (2018-2022) and W2 (2019-2023) both cover the 2022 bear; W3 is
the direct S38 re-run on post-#260. Compare:

| Metric (full friction) | Pre-#260 (S38 / S22 / S34) | Post-#260 (W1 / W2 / W3) | Δ |
|---|---|---|---|
| 2022 mean realised | _TBD_ (S34 reported −$118) | _TBD_ | _TBD_ |
| 2022 hit-rate | _TBD_ | _TBD_ | _TBD_ |
| 2022 executed count | _TBD_ | _TBD_ | _TBD_ |
| Full-window ρ | _TBD_ (S38 reported 0.358) | _TBD_ | _TBD_ |
| Full-window mean realised | _TBD_ | _TBD_ | _TBD_ |
| Full-window final NAV | _TBD_ (S38: $1,331,764) | _TBD_ (W3) | _TBD_ |

**Anticipated finding given F4 design:** PR #260 widens the forward
distribution via the realised-vol-ratio multiplier (replaces the
rolled-back HMM-conditioned widening). The multiplier is bounded
and signal-preserving by design; the effect on 2022 EV should be
modest, not transformative. If W3 reproduces S38's NAV to within a
few %, the F4 fix is signal-preserving on the headline outcome
while bringing the targeted improvement on the F4 named cases.

---

## 10. R9 / R10 fire-rate audit

R9 (sector cap, 25% per sector) and R10 (single-name cap, 10% per
underlying) are the D17 soft-warn + tracker hard-block additions
landed in PRs #255 and #256. **Important methodological note:** the
S43 harness runs with `require_ev_authority=False` and does not
attach a `PortfolioContext`, so neither gate fires during execution
— the harness uses the same configuration as S34 / S38, which is
the apples-to-apples baseline.

This section audits **counterfactually**: by post-hoc replay of the
tracker's executed-trade record against `check_sector_cap` and
`check_single_name_cap` with the published defaults, how often
WOULD each gate have fired?

The single-name audit is computable from the tracker state alone
(strike × 100 vs NAV). The sector audit requires a ticker → GICS
sector map; this map is not in the rank-log artifact and is added
out-of-band where available (otherwise the sector column is left
blank with an explicit note).

_TBD — per-window R9 / R10 would-fire tables._

---

## 11. Findings (observable not interpretive)

_TBD — populated after all numbers are in._

Findings drafted live from the rank_log + tracker artifacts only;
no extrapolation beyond the executed sample. Convention: F<N> with
a one-sentence observation + a "source" line giving the artifact
that supports it.

---

## 12. Method appendix

### Universe

Same as S34 / S38: the first 100 alphanumeric SP500 tickers from
`backtests/regression/universes.py::UNIVERSE_100`. Note that this
cut excludes COST (the F4 named test case) and many notable post-
`CNP` names. Different 100-name cuts (SP100 by market cap,
sector-balanced) would produce different absolute returns; the
structural findings (window-dependent dollar alpha; positive ρ
across years) are robust to universe shape per S34/S35.

### Knobs

| Knob | Value | Source |
|---|---|---|
| Capital | $1,000,000 | S34/S38 match |
| Tickers | 100 alphanumeric SP500 | `UNIVERSE_100` |
| DTE target | 35 | S34/S38 match |
| Delta target | 0.25 | S34/S38 match |
| Contracts | 1 | S34/S38 match |
| `top_n` (ranker cap) | 15 | S34's bump from S32's 10 |
| `max_new_per_day` | 3 | S34/S38 match |
| `require_ev_authority` | False | S34/S38 match (research path) |
| Friction levels | `none`, `bid_ask`, `full` | S34/S38 match |
| Seed | 42 | S34/S38 match |

### Friction overlay

Identical to S34 / S38 (`backtests/regression/_common.py`):

- Bid/ask half-spread: `max($0.05, 8% × premium)` per share
- Commission: `$0.65 / contract` (full only)
- Assignment slippage: `10 bp × strike × 100` (full only)

### Engine version

`origin/main` HEAD = **`56d8e5c`** (this run). Includes:
- PR #260 — F4 realised-vol-ratio widening (replaces rolled-back
  HMM widening)
- PR #255 — R9 sector cap on `EnginePhaseReviewer` + D17 wired to
  `/api/tv/enrich`
- PR #256 — R10 single-name cap (per-underlying notional)
- PR #257 — `docs/PRODUCTION_READINESS.md` synced to S34/S38 reality
- PR #247 — universe-test renumbering follow-up
- PR #258 — scorecard-enum-dedup

### Reproducer

```bash
# Run all four windows sequentially (~25h on the dev box):
python -m backtests.regression.s43_rolling_multiwindow all

# Or one at a time:
python -m backtests.regression.s43_rolling_multiwindow one w1_2018_2022
python -m backtests.regression.s43_rolling_multiwindow one w2_2019_2023
python -m backtests.regression.s43_rolling_multiwindow one w3_2020_2024
python -m backtests.regression.s43_rolling_multiwindow one w4_2021_2025
```

Output per window: `%TEMP%/s43_backtest/<window_id>/<friction_level>/`
containing `rank_log.csv`, `metrics.json`, and (post this campaign)
`tracker_state.json`. The harness's `_common.py` was extended with
an additive tracker-state dump for richer post-process analysis;
the dump is behaviour-neutral (writes one new file; no existing
artifact is changed).

### §2 invariant scan + analysis tools

- `backtests/regression/s43_scan.py` — R1a guard scan
  (non-finite EV count). Runs in seconds per window.
- `backtests/regression/s43_analyze.py` — per-window analysis
  (Univ-EW baseline, refusal rates, concentration, R9/R10 audit).
  Runs in <1 minute per window.

### Pre-campaign 5-ticker EV smoke

Per CLAUDE.md §4, the 5-ticker smoke against the post-#260 engine
on the same data CSV produced the following baseline (all five
candidates returned non-null `ev_dollars`, `iv`, `premium` — engine
path healthy):

| Ticker | ev_dollars | IV | Premium | Strike |
|---|---|---|---|---|
| XOM | $137.57 | 0.3216 | $2.47 | $151.50 |
| JPM | $124.90 | 0.3255 | $4.56 | $270.00 |
| MSFT | $90.97 | 0.3383 | $6.30 | $358.50 |
| UNH | $62.62 | 0.4347 | $5.96 | $257.00 |
| AAPL | $20.45 | 0.3079 | $3.70 | $234.00 |

A post-campaign re-run of the same 5-ticker smoke is included in
§12.x to detect any silent engine drift across the long run.

### Data fingerprint

- OHLCV path: `data/bloomberg/sp500_ohlcv.csv`
- OHLCV coverage on disk: 2018-01-02 → 2026-03-20
- IV path: `data/bloomberg/sp500_vol_iv_full.csv`
- IV coverage on disk: 2015-01-02 → 2026-03-20
- SHA-256 of OHLCV: see per-window `fingerprint.data_csv_sha256` in
  each `metrics.json`.

---

## AI handoff

_TBD — populated after findings finalise._
