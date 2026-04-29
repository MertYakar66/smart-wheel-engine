# Bloomberg BQL pulls — copy/paste into Excel

One section per missing dataset. For each:
1. Paste the BQL into cell A1 of a new sheet.
2. Wait for the formula to resolve (usually < 60s for S&P-wide queries).
3. Save the sheet as `<filename>.csv` under `data/bloomberg/`.
4. Re-run `python scripts/feature_smoke_test.py` — the new data will show up in its own section.

All queries use the BQL functional form `=BQL.QUERY("…")`. Column aliases
in the queries match what the loader expects (see `data/consolidated_loader.py`
column maps).

> **Note on date range.** Queries are written against `range(2015-01-01, 2026-04-23)`.
> Adjust the end date to *today* before running if you want the freshest data.

---

## 1. Macro / economic-release calendar  → `sp500_macro_calendar.csv`

Activates `engine/event_gate.py` macro-buffer path. Today there's no macro
calendar at all; FOMC, CPI, NFP events are invisible to the gate.

```bql
=BQL.QUERY("get(eco_release_dt, eco_release_event, eco_importance, eco_country) for(['FDTR Index', 'CPI YOY Index', 'NFP TCH Index', 'GDP CQOQ Index', 'NAPMPMI Index', 'INJCJC Index', 'FOMC Dec Index', 'RSTAMOM Index', 'CONCCONF Index', 'INDPRO Index']) with(dates=range(2015-01-01, 2026-12-31))")
```

Expected columns:
`release_date, event, importance, country, ticker`

---

## 2. Corporate actions (splits, specials, M&A)  → `sp500_corporate_actions.csv`

Currently the file on disk is 2 bytes. Required by the event gate to avoid
early-assignment on short calls around specials / splits.

```bql
=BQL.QUERY("get(cac_announcement_date, cac_effective_date, cac_type, cac_ratio, cac_amount) for(members('SPX Index')) with(dates=range(2015-01-01, 2026-12-31))")
```

Expected columns:
`announcement_date, effective_date, action_type, ratio, amount, ticker`

---

## 3. Short interest & borrow fee  → `sp500_short_interest.csv`

`engine/risk_manager.py` sizes positions without knowing which names are
heavily shorted. Crowded-short names blow up put sellers.

```bql
=BQL.QUERY("get(eqy_short_interest, eqy_short_interest_pct_of_float, equity_short_borrow_rate_net, eqy_sh_out, eqy_float_pct) for(members('SPX Index')) with(dates=range(2020-01-01, 2026-12-31), fill=prev)")
```

Expected columns:
`date, ticker, short_interest, short_interest_pct_float, borrow_rate_net, shares_out, float_pct`

---

## 4. VIX futures term structure UX1-UX7  → `vix_futures_curve.csv`

You already have VIX3M/VIX6M. Full UX1-UX7 lets the regime detector see
contango vs backwardation precisely.

```bql
=BQL.QUERY("get(px_last) for(['UX1 Index', 'UX2 Index', 'UX3 Index', 'UX4 Index', 'UX5 Index', 'UX6 Index', 'UX7 Index']) with(dates=range(2015-01-01, 2026-12-31), fill=prev)")
```

Expected columns:
`date, ticker (UX1..UX7), close`

---

## 5. Skew / vol-of-vol indices  → `vol_indices.csv`

SKEW captures tail-risk regimes VIX misses. VVIX captures vol-of-vol
(important for dealer-positioning module). Single-pull, one tidy file.

```bql
=BQL.QUERY("get(px_last) for(['SKEW Index', 'VVIX Index', 'VIX9D Index', 'VXN Index', 'RVX Index', 'VXEEM Index', 'OVX Index', 'GVZ Index']) with(dates=range(2015-01-01, 2026-12-31), fill=prev)")
```

Expected columns:
`date, ticker, close`

---

## 6. Analyst revisions stream (not snapshot)  → `sp500_analyst_revisions.csv`

`sp500_analyst.csv` is one row per ticker today. Revision momentum is the
clean fundamentals signal — monotonic on short-put win rate.

```bql
=BQL.QUERY("get(best_eps, best_target_price, best_analyst_rating, tot_analyst_rec) for(members('SPX Index')) with(dates=range(2015-01-01, 2026-12-31), fill=prev)")
```

Expected columns:
`date, ticker, best_eps, best_target_price, best_analyst_rating, tot_analyst_rec`

---

## 7. Realised-correlation index  → `spx_correlation.csv`

Correlation regime is first-order for portfolio CVaR. Cheap pull.

```bql
=BQL.QUERY("get(px_last) for(['COR3M Index', 'COR1M Index', 'COR6M Index']) with(dates=range(2015-01-01, 2026-12-31), fill=prev)")
```

---

## 8. Issuer credit spreads (CDS 5y)  → `sp500_cds.csv`

Complements the `altman_z_score` that's already in `sp500_credit_risk.csv`.
Live CDS reprices faster than the Altman Z.

```bql
=BQL.QUERY("get(cds_spread_5y) for(members('SPX Index')) with(dates=range(2018-01-01, 2026-12-31), fill=prev)")
```

---

## 9. Intraday SPX option market-maker hedge proxy  → `spx_mm_hedge.csv`

Dealer-positioning module today estimates GEX from single-ticker OI.
SPX-level MM hedge proxy is what institutional desks use.

```bql
=BQL.QUERY("get(dealer_gex_total, dealer_dex_total, dealer_gex_profile) for(['SPX Index', 'SPY US Equity', 'QQQ US Equity']) with(dates=range(2020-01-01, 2026-12-31), fill=prev)")
```

Note: this one requires a BQL dealer-positioning subscription. If your
tier doesn't expose it, skip and rely on `engine/dealer_positioning.py`'s
approximation from option chains.

---

## 10. ETF sector constituent weights (rebalancing flow signal)  → `sector_etf_constituents.csv`

```bql
=BQL.QUERY("get(weight) for(members('XLF US Equity') + members('XLK US Equity') + members('XLV US Equity') + members('XLY US Equity') + members('XLP US Equity') + members('XLE US Equity') + members('XLI US Equity') + members('XLU US Equity') + members('XLB US Equity') + members('XLC US Equity') + members('XLRE US Equity')) with(dates=range(2018-01-01, 2026-12-31, frq=M), fill=prev)")
```

---

## 11. Foreign-index vol (for global regime)  → `global_vol_indices.csv`

```bql
=BQL.QUERY("get(px_last) for(['V2X Index', 'VHSI Index', 'VNKY Index', 'VKOSPI Index']) with(dates=range(2015-01-01, 2026-12-31), fill=prev)")
```

---

## 12. Rates vol & FX vol (macro regime)  → `rates_fx_vol.csv`

```bql
=BQL.QUERY("get(px_last) for(['MOVE Index', 'CVIX Index', 'JPMVXYG7 Index']) with(dates=range(2015-01-01, 2026-12-31), fill=prev)")
```

---

# After you've pulled everything

1. Save each CSV to `data/bloomberg/`.
2. Re-run the feature-pipeline backfill to let the new fields propagate:
   ```
   python scripts/backfill_features.py --force --workers 6
   ```
3. Run the smoke test to validate:
   ```
   python scripts/feature_smoke_test.py
   ```
4. Wire the new files into `data/consolidated_loader.py` (one `load_*`
   method per file, following the existing pattern of `load_analyst()`).
