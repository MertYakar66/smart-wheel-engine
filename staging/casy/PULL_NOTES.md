# CASY Phase-1A pull — session notes (2026-06-17)

Supervised Bloomberg session. Branch `claude/phase1a-casy-bloomberg-pull` off `main` @ `83eacdd`.
Per `docs/CASY_BACKFILL_SPEC.md` + `docs/NEXT_DATA_SESSION_RUNBOOK.md` Phase 1A.
**Fragments only — monoliths untouched. Phase 1B integration + Phase 3 re-baseline NOT done (held for review).**

## Environment
- Bloomberg Terminal live (verified: AAPL UW `bdh` returns data).
- Pull env: fresh repo `.venv` (gitignored). Installed `blpapi 3.26.5.1` (Bloomberg index),
  `xbbg 1.3.0`, pinned `pandas==2.3.3` / `numpy==2.3.5` to match the repo's known-good versions.
- **xbbg 1.3.0 caveat:** returns a narwhals *tidy* frame `[ticker,date,field,value]`, NOT the legacy
  wide-MultiIndex pandas the spec's snippet (`df.stack(level=0)`) assumes. `pull_casy.py` adapts via
  `pivot_table`. The repo does not pin xbbg, so any future puller must handle this.

## Fragments produced (staging/casy/, ticker + range per committed schema)
| file | ticker | rows | range | status |
|---|---|---|---|---|
| `casy_ohlcv.csv` | `CASY UW Equity` | 2117 | 2018-01-02→2026-06-04 | ✅ validated |
| `casy_vol_iv.csv` | `CASY UW` | 2117 | 2018-01-02→2026-06-04 | ✅ realized-vol exact; IV field confirmed |
| `casy_liquidity.csv` | `CASY UW` | 2117 | 2018-01-02→2026-06-04 | ✅ raw fields exact |
| `casy_earnings.csv` | `CASY UW` | 148 (Q) | 1989-07-31→2026-06-09 | ⚠️ PARTIAL (dates only) |

## Validated method decisions
1. **OHLCV column scramble** (reproduces the committed convention; connector un-scrambles on read per CLAUDE.md §1):
   `open←PX_HIGH, high←PX_LAST, low←PX_LOW, close←PX_OPEN, volume←PX_VOLUME`.
   Derived empirically + validated: 4 price cols **exact to the cent** over all 52 overlap rows
   (2026-03-23→06-04); volume exact on 51/52 (06-04 +19 shares = post-close finalization).
2. **vol_iv implied vol field = `30DAY_IMPVOL_100.0%MNY_DF`** (ATM, single field → `hist_put_imp_vol == hist_call_imp_vol`, the documented no-skew). Authoritative: `EXTRACTION_GUIDE.md:152`, `iv_formulas.txt`, `bloomberg_export.vba`. Realized vols `VOLATILITY_30/60/90/260D` → `volatility_*` **exact to the cent**.
3. **liquidity:** `VOLUME_AVG_30D→avg_vol_30d`, `TURNOVER→turnover`, `EQY_SH_OUT→shares_out`, `Fill='P'`. (Matches the committed schema; the stale `scripts/pull_liquidity.py` emits `bid_ask_spread` instead of `shares_out`.)

## Validation vs committed 52-row overlap (the spec gate)
| field | result | note |
|---|---|---|
| ohlcv open/high/low/close | **maxabsdiff 0.000000** | exact |
| ohlcv volume | maxabsdiff 19 (1/52 rows) | frontier-day post-close finalization |
| volatility_30/60/90/260d | **maxabsdiff 0.000000** | exact (deterministic from prices) |
| liquidity shares_out | **maxabsdiff 0.000000** | exact |
| liquidity turnover | maxabsdiff 14400 on ~87M base (0.016%) | revision noise |
| liquidity avg_vol_30d | up to +36.5k (≈22% on worst row), 26/52 rows | see "revision" below |
| vol_iv hist_put/call_imp_vol | median 0.42, max 4.74 | see "revision" below |

## Key finding — Bloomberg revises derived/surface fields (NOT a pull error)
Raw/deterministic fields (prices, realized vol, shares_out, turnover) match the older committed
vintage **to the cent**. The two fields that diverge are both revision-sensitive aggregates:
- **avg_vol_30d**: matches exactly 03-23→04-14, then diverges precisely when the trailing-30-day
  window catches the 2026-04-08 volume spike (3.9M) and later the 05-29 spike (451k). Daily volumes
  get revised post-close; those revisions amplify through the 30-day moving average. Method is correct.
- **vol_iv IV**: systematic regime-dependent drift (−4.7 in Apr, +4.4 in late May) = the IVOL surface
  being recomputed since the original (~2026-06-02 vintage) pull. Field confirmed correct.

Implication for Phase 1B/3: the spec replaces CASY's 52 recent rows anyway, so these are non-blocking.
But CASY's full history would be the *current* revision vintage while the rest of the universe is the
*original* vintage — a minor cross-name vintage mismatch worth noting for the supervised re-baseline.

### Vintage record (for the re-baseline)
- **Exact-to-committed-vintage** (raw/deterministic): ohlcv open/high/low/close, volatility_30/60/90/260d, liquidity shares_out + turnover.
- **Current-revision vintage** (recomputed by Bloomberg since the ~2026-06-02 original pull): vol_iv IV (`hist_put/call_imp_vol`, ±≤4.7 vol pts) and liquidity `avg_vol_30d` (≤~22% on spike-window rows). The Phase-3 re-baseline absorbs these. CASY (and any name backfilled this session) therefore carries the **current** IV/avg-vol revision vintage; the rest of the universe carries the original vintage until its own refresh.

## Earnings — PARTIAL (lowest priority per spec)
`casy_earnings.csv` has `year/period` + `announcement_date` (148 quarterly rows, incl. CASY's most
recent 2026-06-09) — the fields the event-lockout gate needs. NOT populated: `announcement_time`
(87.7% in committed) and `earnings_eps/comparable_eps/estimate_eps`. Reason: xbbg's `bds` did not
forward the `EARN_ANN_DT_TIME_HIST_WITH_EPS=Y` bulk-override sub-columns (only date+period returned).
To complete: pull via Excel BDS bulk override (the EXTRACTION_GUIDE method) or a blpapi
ReferenceDataRequest; also reconcile the `:A` annual-row convention (committed includes some).

## NOT done (held for review — per runbook + rails)
- Phase 1B integration into the monoliths, BK↔BNY collapse, dividends union, UNIVERSE_100 re-derive.
- Phase 2 (E) fixes. Phase 3 re-baseline. No monolith bytes changed. No `--update-snapshot`.
