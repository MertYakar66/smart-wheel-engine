# Data fix — OHLCV 2026-03-23 split-scale splice (#439 / D-W1-1)

**Lane:** Mac terminal (data). **Scope:** data-layer only — no decision trio, no
connector behaviour, no §2 surface. **Status:** Mac-lane fix complete; regression
snapshot re-baseline handed to the #402/Windows lane (see §6).

---

## 1. The defect

`data/bloomberg/sp500_ohlcv.csv` is a splice of an older split-adjusted history
pull and a recent slice (≥ 2026-03-23). Names that split **between** the two
pulls carry the split adjustment only on the recent slice, so the committed
series has a one-day scale break at the **2026-03-23 pull boundary** (the vendor
applied the split ~2 weeks before its official effective date):

| Ticker | 2026-03-20 close | 2026-03-23 close | implied | true split (corp actions) | factor |
|---|---:|---:|---:|---|---:|
| BKNG UW Equity | 4293.38 | 176.31 | ÷24.35 | 25:1, eff 2026-04-06 | **÷25** |
| CVNA UN Equity | 291.36 | 58.40 | ÷4.99 | 5:1, eff 2026-05-08 | **÷5** |

A fake −96% (BKNG) / −80% (CVNA) single-day return on 2026-03-23 poisons any
realized-vol / return / forward-distribution / tail computation crossing that
date. The default/live ranker is unaffected (it ranks at the 2026-06-04 frontier,
already split-adjusted); only backtests with a historical `as_of` crossing
2026-03-23 see the break.

## 2. Completeness — exactly two split-splice names

Two independent methods agree the **split-scale** affected set is `{BKNG, CVNA}`:

1. **Boundary sweep** (all 511 tickers): flag every name with a >12 % single-day
   move at 2026-03-23 (a sub-split threshold that catches even 3:2 / 5:4 / 4:3
   small-ratio splits a >2× filter would miss). Four flagged: BKNG, CVNA (splits)
   + FDX, APTV (see §5).
2. **Move-size-independent enumeration**: every stock split with an effective
   date in the splice-risk window `[2026-03-23, frontier 2026-06-04]`. Only
   BKNG (25:1, 04-06) and CVNA (5:1, 05-08). AMCR (eff 2026-01-15) is *before*
   the boundary (consistently adjusted); KLAC/DD/CRWD are effective *after* the
   frontier (not in the recent pull). None straddle the boundary.

Split signature confirmed per name: all OHLC columns rescale by the *same* clean
factor (BKNG per-column boundary ratios 0.0411/0.0410/0.0410/0.0411 = 1/25), the
series trades normally before and after, and the only anomaly is the boundary.

## 3. The fix — full-history back-adjustment to the split (frontier) scale

The engine already treats the whole OHLCV series as living in the split-adjusted
frame (`engine/data_connector.py:909` "OHLCV is Bloomberg SPLIT-adjusted"; the
option-strike back-adjust assumes it; the frontier is at split scale). So the
correct convention is to lift the **pre-splice history** onto that scale, not to
re-apply the split at its effective date (which would manufacture a new seam at
04-06/05-08 since the file's scale actually flips at 03-23):

- **OHLCV** (`data/bloomberg/sp500_ohlcv.csv`): rows with `date < 2026-03-23` for
  BKNG/CVNA → `open/high/low/close /= factor`, `volume *= factor`.
- **Dividends** (`data/bloomberg/sp500_dividends.csv`): BKNG `dividend_amount /= 25`
  (CVNA pays none). **Lockstep, required**: leaving per-share dividends at the
  un-split scale (8.75) on a back-adjusted ~$150 share is a ~6 % quarterly yield
  → the ex-div early-assignment path would always early-assign. ÷25 preserves the
  yield (10.5 → 0.42, the real post-split per-share level; 2024-02-22 BKNG
  dividend is inside the s34 window).

Factors are the **true corporate-action ratios** (25 / 5), not the raw boundary
ratios (which fold in a genuine weekend move: 4293.38/25 = 171.74 → +2.66 % →
176.31; 291.36/5 = 58.27 → +0.22 % → 58.40). `÷5` and `÷25` always terminate, so
the back-adjustment is `Decimal`-exact (no rounding); dividing a price block by a
constant leaves its internal day-over-day returns unchanged. Reproduced by
`scripts/fix_ohlcv_split_scale_439.py` (streaming, CRLF-preserving,
idempotency-guarded — every untouched row byte-identical).

## 4. Verification (connector-served)

- Boundary continuity restored: BKNG seam ratio 1.026, CVNA 1.058 (both in
  [0.5, 2.0]); **0** remaining >2× adjacent-day breaks across the full
  2018-01-02…2026-06-04 series for either name.
- Internal returns preserved (e.g. BKNG 2022-06 +5.98 %, CVNA +13.55 % unchanged).
- Frontier unchanged: BKNG 2026-06-04 close still 167.49 (live ranker untouched).
- Dividends: BKNG `dividend_amount` now 0.42 / 0.384 / 0.35 (= ÷25).
- Tests flipped to passing pins (default per-PR lane): `tests/test_w1_data_wiring.py
  ::test_ohlcv_has_no_split_scale_discontinuity[BKNG,CVNA]`,
  `tests/test_data_integrity_bloomberg.py::test_ohlcv_no_split_adjustment_scale_breaks`
  (strict-xfails removed), and `::test_ohlcv_scale_breaks_are_the_known_two`
  (`KNOWN_SCALE_BREAKS` emptied). 55 passed, 0 XPASS-strict failures.

## 5. New finding — FDX/APTV are SPINOFF-splice artifacts (separate defect)

FDX and APTV show a boundary discontinuity from the **same pull-splice mechanism**
but driven by **spinoffs**, not splits: FDX (Spinoff eff 2026-06-01, boundary
−21.6 %, factor ~1.276) and APTV (Spinoff eff 2026-04-01, boundary −13.6 %,
factor ~1.157). Both are smooth across their spinoff effective dates (so the step
lands at the 03-23 pull boundary, confirming back-adjusted-only-on-recent-slice),
and neither factor is a clean share-count ratio. **Excluded from this split-ratio
fix** — they need a separate spinoff value back-adjustment. Filed as its own
issue (see #439 thread). The other prior >2× names (GL/OXY/TRGP/PCG) are genuine
historical crashes years from the boundary — not artifacts.

## 6. Cross-terminal handoff — regression snapshot re-baseline (#402/Windows lane)

Editing the two connector CSVs changes their SHA, so the default-lane drift guard
`tests/test_backtest_regression.py::test_snapshot_data_fingerprint_matches_current`
goes red for all four snapshots (drift keys `['dividends', 'ohlcv']`) — **by
design** ("EXPECTED to fail on every legitimate data change until the snapshots
are regenerated"). These snapshots are READ-ONLY on this lane:

- **s34_universe_100t_1m** — UNIVERSE_100 includes BKNG (a dominant driver),
  window 2022–2024 entirely pre-split → **value re-baseline** required.
- **s27 / s32 / s35** — UNIVERSE_24, no BKNG/CVNA, values unchanged → **hash-only
  re-pin**.

The Windows/#402 lane must re-baseline all four against this corrected data per
TESTING.md "Backtest regression — re-baseline workflow". This is the only red on
the PR.

**Stale (read-only, non-blocking) evidence** that will no longer match the fixed
data but is not test-asserted: `docs/verification_artifacts/data_wiring_2026-06-27/
{w1_ohlcv,w1_summary}.json` (discontinuity_count 6 → 4) and
`docs/verification_artifacts/efix_ab_2026-06-10/{main,integ}_s34_*.json`.
