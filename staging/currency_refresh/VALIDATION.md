# Bucket A — currency refresh validation (2026-06-05 → 2026-06-18)

Pull date 2026-06-18 (xbbg 1.3.0 / blpapi 3.26.5.1). Method: pulled from the 06-04 overlap day,
validated each fragment against the committed monolith **to the cent**, then kept 06-05+. Fragments
replicate each monolith's exact column set + ticker form (append-compatible). **Monoliths byte-untouched.**

| fragment | rows | names | dates | result |
|---|---|---|---|---|
| `sp500_ohlcv__2026-06-05_2026-06-18.csv` | 5075 | 508 | 06-05..06-18 | rotation gate `open==max`&`low==min`=**1.0000** (0 violations) |
| `sp500_liquidity__2026-06-05_2026-06-18.csv` | 5080 | 508 | 06-05..06-18 | cols `date,avg_vol_30d,turnover,shares_out,ticker` (BID_ASK_SPREAD all-NaN → not in schema, matches monolith) |
| `vix_term_structure__2026-06-05_2026-06-18.csv` | 10 | — | 06-05..06-18 | overlap 06-04 vix/vix_3m/vix_6m **Δ=0 exact** |

## Overlap-to-the-cent findings (06-04 vs monolith)
- **VIX term: exact match** (Δ=0 on all three series). Clean.
- **OHLCV/liquidity price: 1 seam — `KLAC UW Equity` (KLA Corp) 10:1 forward split.**
  `mono_high 2131.10 → pull_high 213.11` (÷10), volume ×10 (mono/pull ratio 0.10). Confirmed split,
  not a puller bug. The monolith holds KLAC's **pre-split** history; this fragment holds **post-split**
  06-05+ prices. Appending for KLAC creates a 10× seam at 06-04/06-05 → **flag for the review's
  monolith integration** (re-baseline KLAC split-adjusted). Monolith left untouched here.
  - IV/skew surface is split-invariant (IV is a %), so the banked T0-2 surface is unaffected.
- **OHLCV/liquidity volume: 82 benign finalization revisions** (ex-KLAC, ratios 0.98–0.997, <2%).
  The monolith captured preliminary 06-04 volume; now final. Expected; not a seam.
- avg_vol_30d (11) / turnover (82) / shares_out (2): rolling/derived revisions, same benign cause.

## Side-effect catch for bucket E
The KLAC split is a **post-2026-06-05 corporate action**, but `sp500_corporate_actions.csv` ends
2026-06-05 → corp-actions is **not merely verify-only**; it needs a 06-06→06-18 tail (the split is
missing). Tracked in the manifest's bucket E.

## Sanity bands
VIX 06-05..06-18: 16.2–22.2 (vix), term structure in contango (vix < vix_3m < vix_6m every day) ✓.
OHLCV rotation held across all 508 names. No negative prices/volumes; no NaN in kept rows.
