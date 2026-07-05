# T0-3 / #354 — dated PIT dividend-yield panel + carry-coverage investigation

Pull 2026-06-18 (xbbg 1.3.0). Deliverable: `sp500_dividend_yield_pit.csv` — a **dated monthly**
dividend-yield panel replacing the dateless 2026 snapshot `get_fundamentals` used (the #354
lookahead: BSM `q` can now be selected point-in-time). Staging only; engine wiring out of scope.

## Panel
- `sp500_dividend_yield_pit.csv`: **72,461 rows, 421 names, 2010-01-29 → 2026-05-29 · monthly**.
- Fields: `dvd_yld_12m` (EQY_DVD_YLD_12M, median 2.08%), `dvd_yld_ind` (EQY_DVD_YLD_IND, median 2.13%),
  `dvd_sh_12m` (DVD_SH_12M, median 1.48). FLDS-verified entitled; `EQY_DVD_YLD_EST` / `DVD_SH_LAST`
  all-NaN (not entitled).
- One band outlier: `dvd_yld_12m` max 68.57% — a special-dividend / depressed-price month; flag for
  the review's winsorization, not a pull error.

## #354 question — is the gap "no dividend" or "missing data"? → **"no dividend"** (confirmed)
Bloomberg returns **NaN, never 0**, for non-payers (snapshot zeros=0). Of 511 universe tickers, 421
(82.4%) have ≥1 non-null 12m yield in the panel; **90 are never-yield**. Resolution of the 90:
- **81** have no >0 dividend record anywhere → genuine non-payers.
- **8** confirmed genuine non-payers in the 2010–2026 window by direct probe + DVD_HIST:
  ADBE & ADSK (regular dividends **discontinued 2005**), NVR (discontinued 1990), TYL (1998),
  AMD & TTWO (only a 1990s/2008 "Rights Redemption", no cash dividend), UAL (one 2008 "Return of
  Capital"), **IT/Gartner** (DVD_HIST = exactly one **Special Cash** dividend in 1999, nothing since).
  → NaN is correct (q=0). (My first cross-ref over-counted via a `startswith` bug — "IT" matched ITW,
  "BK" matched BKNG/BKR; direct DVD_HIST probes corrected it.)
- **1 — BK (Bank of New York Mellon)**: a genuine payer, but a **ticker-line/field anomaly**. The
  universe line `BK UN Equity` returns OHLCV yet **no BDP price, no dividend-yield fields, and
  DVD_HIST = 0 rows**; the composite `BK US Equity` also returns NaN for EQY_DVD_YLD_12M. BNY's
  dividend yield is **unretrievable via the standard fields at this tier** → documented gap (1/503),
  not fillable here. Fallback for BK's `q` would be its actual cash dividends, which this session's
  BK line also cannot retrieve — a review/wiring item.

**Bottom line:** the carry gap is **"no dividend," not "missing"** — 89/90 never-yield names are
correct NaN (non-payers); the lone exception (BK) is a per-name data anomaly, flagged. The dated
panel fixes the lookahead for the 421 payers.
