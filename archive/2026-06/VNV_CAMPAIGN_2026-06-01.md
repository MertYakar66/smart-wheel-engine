# Engine V&V campaign — efficiency / realism / reliability (2026-06-01)

Read-only verification sweep of the production engine on `origin/main`
(`1c69062`), provider `MarketDataConnector` (Bloomberg). Goal: stress the
engine's outputs for **realism** and **reliability**, profile its
**efficiency**, and record concrete findings + reproducible drivers. No engine
code was modified for this campaign; every candidate was produced by the
authoritative `WheelRunner.rank_candidates_by_ev` (§2 intact).

Drivers (re-runnable): `scripts/vnv_funnel_tier_report.py`,
`scripts/vnv_prob_profit_calibration.py`. Captured outputs:
`docs/verification_artifacts/vnv_2026-06-01/`.

> **Theta note.** The real-time/historical Theta MCP was unavailable during
> this run (Terminal busy pulling data), so the realism checks are against the
> committed Bloomberg reference files, not a third-party option-price oracle.

---

## 1. Reliability — candidate funnel is transparent (PASS)

Full-universe scan at `as_of=2026-03-20` (`vnv_funnel_tier_report.py`):

| stage | count |
|---|---|
| tickers entering | 503 |
| **survivors (a ranked candidate)** | **423 (84%)** |
| dropped | 80 |

Drops, by gate (all carry an auditable reason via `df.attrs["drops"]` — the
HT-A transparency surface, so this is **not** silent filtering):

| gate | n | example |
|---|---|---|
| `event` (earnings lockout) | 68 | `ACGL: earnings@2026-04-28 (±5d buffer)` |
| `history` (<504d) | 11 | `CPB: history 398d < required 504d` |
| `premium` (too thin) | 1 | `AES: synthetic premium <= $0.05` |

The dominant filter is the legitimate earnings event-lockout; the rest are
data-availability gates. **Conclusion: the funnel is healthy and fully
auditable.** (See §5 for the PIT subtlety in the earnings gate.)

## 2. Reliability — Wilson-CI coverage validates the tier-gate (PASS)

The prob_profit Wilson sampling CI (PR #317) is emitted only on the IID
`empirical_non_overlapping` forward tier. Coverage across the 423 put survivors:

| forward tier | n | share | CI |
|---|---|---|---|
| `empirical_non_overlapping` | 417 | **98.6%** | shown |
| `empirical_overlapping` | 6 | 1.4% | suppressed |

So on the put ranker the CI is shown on **98.6%** of candidates and suppressed
on exactly the **1.4%** where the count is autocorrelated and the interval
would be false precision. The tier-gate is a **narrowly-targeted real fix**,
not over-suppression. (The CC/strangle rankers, with longer DTE, fall to the
overlapping tier more often — which is precisely why the gate matters more
there; see PR #317 / RA-4.)

## 3. Realism — prob_profit calibration (top-bin over-confidence, regime-dependent)

Method: for a regime-spanning bi-monthly `as_of` grid (2020-06 → 2026-02, all
with expiry ≤ 2026-03-20 data end), rank the full universe and compare each
candidate's predicted `prob_profit` to the **realized** hold-to-expiry outcome
(`S_expiry > strike − premium`, the cash-secured-put breakeven — the
engine-EXACT rule, not "expired OTM", to avoid the ~12pp methodology artifact
HT-B flagged). Caveats: candidates are not independent (Wilson interval shown
is optimistically tight); costs omitted (realized marginally optimistic → any
over-confidence is a LOWER bound); current-member survivorship.

Full universe, **n = 9,612** candidate-outcomes across 35 as_of dates:

| prob_profit bin | n | predicted | realized | gap | Wilson95 |
|---|---|---|---|---|---|
| [0.60,0.70) | 931 | 0.665 | 0.745 | +0.080 | [0.716,0.772] |
| [0.70,0.80) | 3452 | 0.749 | 0.758 | +0.009 | [0.744,0.772] |
| [0.80,0.90) | 4378 | 0.836 | 0.789 | −0.047 | [0.777,0.801] |
| [0.90,0.95) | 742 | 0.915 | 0.802 | **−0.113** | [0.772,0.829] |
| [0.95,1.00) | 60 | 0.963 | 0.800 | **−0.163** | [0.682,0.882] |

Mid bins (0.60–0.80) are well-calibrated to slightly *under*-confident;
over-confidence appears only in the top two bins and is **mild unconditionally**
(−0.11 / −0.16). But it is sharply **regime-dependent**:

| regime | n | predicted | realized | gap |
|---|---|---|---|---|
| 2020 covid-recovery | 147 | 0.924 | 0.932 | +0.008 |
| 2021 calm-bull | 126 | 0.918 | 0.802 | −0.116 |
| 2022 rate-bear | 201 | 0.922 | **0.577** | **−0.345** |
| 2023 recovery | 48 | 0.924 | 0.958 | +0.034 |
| 2024 bull | 34 | 0.926 | 0.706 | −0.220 |
| 2025 bull | 80 | 0.927 | 0.875 | −0.052 |
| 2026 early | 19 | 0.926 | 0.684 | −0.242 |

**The cockpit's crisis-realized ghost (~0.57) is exactly reproduced:** the 2022
rate-bear top bin realizes **0.577**. Unconditionally the top bin realizes ~0.80;
only in the rate-bear regime does it collapse to ~0.58. This (a) validates the
cockpit's VIX>25-conditioned red caution — it fires precisely when the
miscalibration is large — and (b) is direct evidence for *why* a static
recalibration band fails leave-one-crisis-out: applying 2022's −0.345 everywhere
would badly mis-correct the calibrated calm/recovery regimes (2020 +0.008, 2023
+0.034). **Confirms the gated-band decision.**

This independently reproduces `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` (top
bin MISCAL −5pp to −18pp across 10 backtests) — my unconditional −0.11/−0.16 is
in that range — and adds the regime split that explains *when* the spread blows
out (the 2022 rate-bear crisis), plus a forward-realized rather than
in-backtest measurement.

## 4. Realism — does `ev_dollars` predict realized $? (CONFIRMS the cockpit)

Full universe, **n = 9,612**:

- Pearson `corr(ev_dollars, realized_pnl)` = **−0.018 (≈ 0)** — confirms the
  cockpit/I11 statement that `ev_dollars` has **~0 linear correlation** to
  realized dollars (the heavy-tailed realized-P&L variance washes out a linear
  fit). *(Honesty note: the limit-25 smoke showed a misleading +0.24 — a
  small-sample / alphabetical-subset artifact. The full universe is the
  authoritative ~0; I am reporting the full number, not the smoke.)*
- **But as a RANKING signal it works** — monotonic EV-quintile lift:

  | EV quintile | mean ev_dollars | mean realized $ | n |
  |---|---|---|---|
  | Q1 | −233 | −42.86 | 1923 |
  | Q2 | −61 | −0.14 | 1923 |
  | Q3 | −23 | +9.48 | 1922 |
  | Q4 | −0.06 | +19.93 | 1922 |
  | Q5 | +97 | **+106.55** | 1922 |

- Sign gate: `ev_dollars > 0` → mean realized **+$85.22** (n=2846);
  `ev_dollars ≤ 0` → **−$9.45** (n=6766).

Both halves of the cockpit's framing hold **simultaneously**: `ev_dollars` is a
**ranking score** (monotonic quintiles; the sign separates winners from losers)
with **~0 linear correlation** to the realized dollar amount (heavy tails).
**No tension with the cockpit — full agreement**, and a positive result for the
§2 negative-EV gate: it routes the +$85 winners through and blocks the rest.
(Realized $ skew positive overall — a largely up-market 2020–2025 sample +
current-member survivorship; the EV *differential* is the signal, not the
absolute level.)

## 5. Realism — input limitations (confirmed)

- **IV has zero skew.** `hist_put_imp_vol == hist_call_imp_vol` in
  **100.0000%** of 1,353,901 non-null rows (`sp500_vol_iv_full.csv`); mean abs
  diff = 0. The engine prices short puts off symmetric IV, missing the
  real-market put-skew premium → downside is structurally under-priced. This is
  a plausible *contributor* to the §3 top-bin over-confidence and means the
  Nelson-Siegel skew tooling is dormant on Bloomberg (consistent with
  `DECISIONS.md` D9 / known `bloomberg-iv-no-skew`).
- **Earnings gate PIT subtlety (RA-2).** `get_next_earnings` reads a static
  `sp500_earnings.csv` with no as-of vintage, so a historical `as_of` filters on
  the *realized* announcement date (e.g. ACGL@2026-04-28, 39 days past the
  2026-03-20 as_of). Mild + conservative (the gate only removes candidates), but
  not strictly PIT. Off-trio fix = stamp a calendar vintage / diagnostic column.

## 6. Efficiency — connector ticker handling dominates a universe scan

cProfile of a 150-ticker scan (`sort_stats('tottime')`; absolute times inflated
by a concurrent run, but call-counts exact):

| cost | tottime | calls | note |
|---|---|---|---|
| pandas `comp_method_OBJECT_ARRAY` | 10.1s | 742 | full object-column compares in `_filter_ticker` (`df[df["ticker"]==t]`) |
| `regime_hmm._forward_backward` | 7.9s | 3,087 | HMM E-step — inherent |
| `data_connector.normalize_ticker` | 1.7s (+~0.9s helpers) | **2.4M** | `_load` line 115 `.apply()` over every OHLCV+IV row (~500 unique tickers) |

Two **non-trio** data-layer wins (see follow-up efficiency PR):
1. `_load`: `df["ticker"].apply(normalize_ticker)` → normalize the ~500 uniques
   and `.map()` — output-identical, removes ~2.3M function calls.
2. `_filter_ticker`: replace the repeated full object-column scan with a cached
   per-ticker groupby (or categorical dtype) — the single biggest cost in a
   universe scan. Must ship with an output-equivalence test.

## 7. Reliability — HMM numerical warnings (minor)

`regime_hmm._forward_backward` emits `RuntimeWarning: invalid value encountered
in reduce` (logaddexp over all-`-inf` rows) on some series. The engine handles
it and continues; outputs are unaffected in the smoke. Low-priority hardening:
guard the `logaddexp.reduce` against all-`-inf` rows. (Pre-existing on `main`.)

---

## Verdict

- **Reliability:** funnel transparent (§1), Wilson-CI coverage 98.6% (§2), full
  suite CI-green on #317/#318/#319. PASS.
- **Realism:** top-bin prob_profit over-confidence is real, mild
  unconditionally but −0.345 in the 2022 rate-bear regime — reproducing the
  cockpit's 0.57 crisis ghost (§3); gated-band decision confirmed. `ev_dollars`
  has ~0 linear correlation to realized $ (confirms the cockpit) yet is a valid
  ranking/sign signal (§4). Symmetric-IV and earnings-PIT are known input
  limits (§5). All consistent with the shipped engine's own caution layer.
- **Efficiency:** clear non-trio connector wins identified (§6).

No §2 issues found. Follow-ups (operator's call): the §6 connector optimization
(separate PR), the RA-2 earnings-vintage diagnostic, and the §4 cockpit-copy
reconciliation. None require a trio edit.
