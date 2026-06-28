# Real-premium rail — data-quality + wiring audit (2026-06-27)

**Terminal:** Windows main. **Scope:** the gitignored real-premium rail
(`data_processed/option_premium`, 154 parquets, PR #435) that only this machine
has — complements the Mac terminal's rail-independent campaign (#436). **Method:**
`audit_rail.py` (light: rank calls + parquet scans), run from the worktree with
`SWE_OPTION_PREMIUM_DIR` → the produced rail. Raw: `out/summary.json`.

**Verdict: the rail is wired accurately. No defects found.** One minor
diagnostic-frame unit inconsistency noted (C3); the engine's event lockout around
both corporate-actions and earnings is verified correct (a bonus risk-control
confirmation, C2).

## C1 — Coverage: how much does the rail actually engage?

Fraction of ranked candidates served a real `market_mid` (vs synthetic-BSM
fallback), across regimes:

| as_of | market_mid % | (mm / candidates, tickers) |
|---|---|---|
| 2020-03-16 | 24.6% | 17/69, 69t |
| 2021-06-15 | 26.3% | 20/76, 76t |
| 2022-10-14 | 37.5% | 3/8, 8t* |
| 2023-06-15 | 27.6% | 21/76, 76t |
| 2024-06-14 | 24.3% | 17/70, 70t |

**~25% of the ranked book uses real mids** at the chosen ~25-delta strike; the
rest fall back to synthetic. So the real-premium channel (and R11b, which gates
only `market_mid` rows) touches roughly a quarter of candidates — useful context
for sizing its impact. *2022-10-14 ranked only 8 tickers (a thin candidate day —
broad event/earnings lockout cluster); not a rail issue.*

## C2 — Split-adjustment: correct

The `_split_adjust_option_premium` hazard (raw rail strikes vs split-ADJUSTED
engine spot) is **cleared**. At dates clear of the event lockout, the served
strike scale matches the engine's adjusted spot:

| name | split | as_of | adj spot | strike | strike/spot | real mid | verdict |
|---|---|---|---|---|---|---|---|
| NVDA | 10:1 | 2024-04-08 | 87.13 | 81.00 | 0.93 | 1.69 | ✅ |
| AMZN | 20:1 | 2022-02-16 | 158.10 | 149.50 | 0.95 | 2.93 | ✅ |
| GOOGL | 20:1 | 2021-12-06 | 143.16 | 135.50 | 0.95 | 2.26 | ✅ |

NVDA was the headline hazard (raw ~$870 → ÷10 = adjusted ~$87): **correctly
divided.** The three highest-ratio cases (10:1, 20:1, 20:1) all pass, so the
cumulative-split-factor logic is sound.

**Bonus (verified-good behavior):** ranking NVDA/AAPL/TSLA *near* their splits
returns zero candidates because the engine applies a **corporate-action event
lockout** (`event_lockout:corp_action@<split> ±3d`) — and an **earnings lockout**
(`±5d`) — refusing to open positions whose lifetime spans a split or earnings.
That is correct risk control, confirmed live.

## C3 — Units: `edge_vs_fair` is per-contract (minor inconsistency)

Confirmed on 17 `market_mid` rows: `edge_vs_fair == (premium − fair_value) × 100`
(max abs diff $0.05). So **`edge_vs_fair` is reported per-contract**, while
`premium` and `fair_value` are per-share, in the same ranker frame. Not a
correctness bug (EV is sound; R11b's `edge > 0` gate is sign-based so unaffected),
but a **mixed-units inconsistency** in the diagnostic frame that can mislead a
reader (it briefly misled this audit). Candidate follow-up: normalize the
diagnostic frame's units or document them in `docs/GREEKS_UNIT_CONTRACT.md`.
*(Filed for the Mac/owner as a non-trio doc/clarity item — not fixed here.)*

## C4 — Skew edge is real and pervasive (validates R11b)

`edge_vs_fair` (per-contract) on `market_mid` rows, by VIX bucket:

| VIX bucket | n | mean | median | % positive |
|---|---|---|---|---|
| vix<=15 | 38 | +$16.17 | +$12.34 | 68.4% |
| 15–25 | 20 | +$63.18 | +$48.16 | 100% |
| 25–35 | 3 | +$34.35 | +$14.71 | 100% |
| vix>35 | 17 | +$45.55 | +$42.09 | 70.6% |

The real mid is **richer than ATM-IV BSM fair across every regime** (positive
edge), i.e. a persistent VRP/skew premium — which is exactly the calibration win
the with/without sweep measured, and confirms **R11b gates on a genuine signal,
not noise.**

## C5 — Quote quality: clean

8 sampled parquets (AAPL…ADP, 1.0–2.6M rows each): **zero** crossed quotes
(`bid>ask`), zero non-positive mids, zero `mid≠(bid+ask)/2`, zero DTE-out-of-belt,
zero bad `right`. The producer's two-sided-uncrossed validity gate holds.

## C6 — PIT / no-lookahead: clean (critical)

If the rail accessor leaked future quotes it would invalidate **every** backtest
that uses it (the R11b A/B and the earlier with/without sweep). Live check on real
data — for each `(ticker, as_of)`, the served chain's max snapshot date must be
≤ `as_of`, AND the raw parquet must contain later-dated rows that were correctly
excluded (proving the gate has teeth, not just that no future data existed):

| ticker | as_of | served snapshot | later rows in raw (excluded) |
|---|---|---|---|
| AAPL | 2023-06-15 | 2023-06-15 ✅ | 689,522 |
| MSFT | 2022-03-10 | 2022-03-10 ✅ | 1,248,105 |
| JPM | 2021-09-20 | 2021-09-20 ✅ | 874,784 |
| XOM | 2024-02-15 | 2024-02-15 ✅ | 387,075 |

**4/4 checks, 0 lookahead violations.** The rail is point-in-time correct on real
data → the backtests built on it are not future-contaminated.

## Net

The real-premium rail is **well-formed and correctly wired** into the engine
(coverage ~25%, split-adjust correct, quotes clean, skew edge real). The only
actionable item is the cosmetic per-contract/per-share unit inconsistency in the
diagnostic frame (C3). This audit is complementary to — and disjoint from — the
Mac terminal's committed-data wiring campaign (#436).
