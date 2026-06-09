# BLOOMBERG PULL LIST — what data we need (manual-pull checklist)

_Compiled 2026-06-09. This is the **what-to-fetch** checklist for the Bloomberg terminal — no
scripts. For the full rationale, engine consumer, benefit tier, and §2 role of each item, see the
companion [`DATA_ACQUISITION_ROADMAP.md`](DATA_ACQUISITION_ROADMAP.md). For what we already hold (so
you don't re-pull), see [`DATA_INVENTORY.md`](DATA_INVENTORY.md)._

## How to read this

- **Every field mnemonic is FLDS-verify** — confirm the exact code in the terminal (`FLDS <field>`)
  before pulling; Bloomberg renames fields. Items the analysis flagged as uncertain are marked ⚠.
- **Universe A** = the ~503 current S&P 500 constituents (see `sp500_index_membership.csv`). Index /
  ETF / macro tickers are spelled out per row.
- **Freq:** D = daily · M = monthly · Q = quarterly · E = event-driven.
- **History:** pull as deep as Bloomberg allows. Where a deep archive already exists on disk
  (`data/bloomberg/deep/`), only the **recent tail** is needed — marked *(extend)*.
- **Target** = destination file under `data/bloomberg/`. Bloomberg prices are **split-adjusted**;
  never mix with Theta raw.
- **§2:** none of this rescues a negative-EV trade — every input is a correctness input to
  `EVEngine.evaluate`, a remove-only event gate, or a downgrade-only/advisory signal.

---

## Tier 0 — pull these first (correctness blockers)

| ☐ | Dataset | Universe | Fields (FLDS-verify) | History · freq | Target file |
|---|---|---|---|---|---|
| ☐ | **Moneyness-IV grid (skew)** | Universe A | `30/60/90/180/365DAY_IMPVOL_{80,90,95,100,105,110,120}%MNY_DF` — **put & call legs** | 2005→now · D *(extend; deep 5×5 surface already on disk 2005-2026 — wire it, this adds breadth + recent)* | `sp500_iv_surface.csv` |
| ☐ | **Dividend yield (dated/PIT)** | Universe A | `EQY_DVD_YLD_12M`, `EQY_DVD_YLD_IND` | 2015→now · D | `sp500_dividend_yield.csv` |
| ☐ | **Macro-event calendar** | `FDTR`, `CPI YOY`, `NFP TCH`, `PCE`, `GDP CQOQ`, `NAPMPMI` Index | `ECO_RELEASE_DT`, `_EVENT`, `_ACTUAL`, `_SURVEY`, `_IMPORTANCE`, `_COUNTRY` | 2015→now + forward · E | `sp500_macro_calendar.csv` |
| ☐ | **UST yield curve** | `USGG3M/6M/2YR/5YR/10YR/30YR` Index | `PX_LAST` | 1994→now · D *(file starts 2021-05 — backfill)* | `treasury_yields.csv` *(extend)* |
| ☐ | **Corporate actions** | members(`SPX Index`) | `CAC_ANNOUNCEMENT_DATE`, `CAC_EFFECTIVE_DATE`, `CAC_TYPE`, `CAC_RATIO`, `CAC_AMOUNT` | 2015→now · E *(populated copy exists off-main — restore first, then top up)* | `sp500_corporate_actions.csv` *(empty)* |

---

## A · Volatility & options-derived

| ☐ | Dataset | Universe | Fields (FLDS-verify) | History · freq | Target file |
|---|---|---|---|---|---|
| ☐ | ATM IV term structure | Universe A | `7/14/30/60/90/180/365/730DAY_IMPVOL_100.0%MNY_DF` | 2010→now · D | `sp500_iv_term.csv` |
| ☐ | Risk-reversals & butterflies (25Δ/10Δ) | Universe A | put vs call legs of the moneyness grid (Tier 0) | 2010→now · D | `sp500_iv_surface.csv` |
| ☐ | Realized-vol cones | Universe A | `10/20/30/60/90/120/180/260DAY_HV` (or compute from OHLCV) | 2010→now · D | `sp500_realized_vol.csv` |
| ☐ | VIX complex | `VIX VVIX SKEW VXN RVX VXEEM OVX GVZ MOVE` Index | `PX_LAST` | 2011→now · D *(also in `vol_indices.parquet`; commit a PIT copy)* | `sp500_vol_indices.csv` |
| ☐ | VIX futures term structure | `UX1…UX7` Index | `PX_LAST` | 2007→now · D *(file lacks futures)* | `vix_term_structure.csv` *(extend)* |
| ☐ | IV rank / percentile | Universe A | `IMPVOL_RANK` ⚠ or derive 252d pct of ATM IV | 2010→now · D | `sp500_iv_rank.csv` |
| ☐ | Single-name IV indices (P3) | mega-caps | `AAPLV/NVDAV/TSLAV…` Index ⚠ | as avail · D | `sp500_singlename_ivx.csv` |

> Per-strike OI / greeks / smile is **not on Bloomberg** (OMON manual only) — source from **Theta** (we already hold ~185M EOD-chain rows, 2016-2026).

## B · Equity pricing & microstructure

| ☐ | Dataset | Universe | Fields (FLDS-verify) | History · freq | Target file |
|---|---|---|---|---|---|
| ☐ | Total-return series (div-adjusted) | Universe A | `TOT_RETURN_INDEX_NET_DVDS` ⚠ | 1994→now · D | `sp500_total_return.csv` |
| ☐ | EOD bid/ask spread | Universe A | `PX_BID`, `PX_ASK` (or `LAST_BID/LAST_ASK` ⚠) | 2015→now · D | extend `sp500_liquidity.csv` |
| ☐ | Average daily volume | Universe A | `EQY_AVG_VOL_{20,60,90}D` ⚠ (or rolling `PX_VOLUME`) | 2015→now · D | extend `sp500_liquidity.csv` |
| ☐ | Turnover | Universe A | `EQY_TURNOVER` ⚠ | 2015→now · D | extend `sp500_liquidity.csv` |
| ☐ | Shares outstanding (PIT) | Universe A | `EQY_SH_OUT` | 2010→now · M | `sp500_shares_out.csv` |
| ☐ | Beta (raw & adjusted, 1Y/2Y/5Y) | Universe A | `BETA_RAW_OVERRIDABLE`, `BETA_ADJUSTED_OVERRIDABLE` | 2015→now · M | `sp500_beta.csv` |
| ☐ | Ex-div & split adjustment factors | Universe A | `EXD_ADJ_FACTOR`, `SPLIT_ADJ_FACTOR` ⚠ | full · E | `sp500_adj_factors.csv` |
| ☐ | VWAP (P2) | Universe A | `VWAP` | 2018→now · D | extend `sp500_liquidity.csv` |
| ☐ | Float % (P3) | Universe A | `EQY_FLOAT_PCT` | snapshot | extend `sp500_institutional.csv` |
| ☐ | Intraday bars (P2, tier-dependent) ⚠ | Universe A | 1m/5m/15m/1h OHLCV if tier allows; else Theta | recent · intraday | `data_processed/intraday/` |

> Pre-2018 daily OHLCV is **already on disk** (`deep/sp500_ohlcv__1994_2018.csv.gz`) — wire it, don't re-pull (split-adj vs Theta raw; re-baseline required).

## C · Fundamentals & estimates (PIT — pull period-end/filing dates alongside)

| ☐ | Dataset | Universe | Fields (FLDS-verify) | History · freq | Target file |
|---|---|---|---|---|---|
| ☐ | Income / Balance / Cash-flow line items | Universe A | `IS_REVENUE/OPERATING_INCOME/NET_INCOME/EBITDA/EPS`, `BS_TOT_ASSET/CASH/LONG_TERM_DEBT/TOT_EQUITY/GOODWILL`, `CF_OPERATING/CAPITAL_EXPENDITURE/FREE_CASH_FLOW` + `BEST_PERIOD_END_DT`, `BEST_FISPD_SHEET_DT` | 2010→now · Q | `sp500_financials_pit.csv` |
| ☐ | Forward consensus estimates | Universe A | `BEST_EPS`, `BEST_REVENUE`, `BEST_EBITDA`, `BEST_TARGET_PRICE`, `BEST_EPS_GROWTH` | 2010→now · M | `sp500_estimates.csv` |
| ☐ | Estimate revisions & dispersion | Universe A | `BEST_EPS_REVISIONS_UD_3M`, `BEST_EPS_20D_REVISION`, `BEST_EPS_STD_DEV`, `EARN_REVISIONS_UP/DOWN_1YR_PERCENT` | 2010→now · M | `sp500_estimates.csv` |
| ☐ | Earnings surprise history | Universe A | `IS_EPS`, `BEST_EPS_MEDIAN`, `EARN_EST_EPS_SURPRISE_PCT` | 2010→now · E | `sp500_earnings_surprise.csv` |
| ☐ | Valuation multiples (history) | Universe A | `BEST_PE_RATIO`, `PX_TO_BOOK_RATIO`, `EV_EBITDA`, `PX_TO_SALES_RATIO`, `PEG_RATIO` | 2010→now · M | `sp500_valuation.csv` |
| ☐ | Profitability & returns | Universe A | `RETURN_COM_EQY`, `ROIC`, `RETURN_ON_ASSET`, `GROSS/OPERATING/NET/EBITDA_MARGIN` | 2010→now · Q | extend `sp500_historical_fundamentals.csv` |
| ☐ | Leverage & coverage | Universe A | `INTEREST_COVERAGE`, `NET_DEBT_TO_EBITDA`, `NET_DEBT_TO_EQUITY` | 2010→now · Q | extend `sp500_historical_fundamentals.csv` |
| ☐ | FCF metrics | Universe A | `FCF_YIELD`, `FCF_CONVERSION_RATIO`, `CAPEX_AS_PCT_SALES`, `CF_FREE_CASH_FLOW_MARGIN` | 2010→now · Q | extend `sp500_historical_fundamentals.csv` |
| ☐ | GICS classification (full) | Universe A | `GICS_SECTOR_NAME`, `GICS_INDUSTRY_GROUP/INDUSTRY/SUB_INDUSTRY_NAME` | PIT · static | extend `sp500_fundamentals.csv` |
| ☐ | Dividend payout & coverage | Universe A | `DVD_PAYOUT_RATIO`, `BEST_DPS`, `FCF_PAYOUT_RATIO` | 2010→now · Q | `sp500_dividend_yield.csv` |
| ☐ | Growth (rev/EPS YoY + CAGR) | Universe A | `REVENUE_GROWTH_1Y/5Y`, `BEST_EPS_GROWTH`, `SALES_GROWTH_FORECAST` | 2010→now · Q | extend `sp500_historical_fundamentals.csv` |
| ☐ | Quality scores (P2) | Universe A | `ALTMAN_ZSCORE`, `PIOTROSKI_SCORE` ⚠, `BENEISH_MSCORE` ⚠ (else compute) | 2010→now · Q | `sp500_quality_scores.csv` |
| ☐ | R&D / book value / buybacks (P3) | Universe A | `RESEARCH_AND_DEVELOPMENT_EXPENSE`, `BOOK_VALUE_PER_SHARE`, `BUYBACK_AUTHORIZATION` ⚠ | 2010→now · Q | extend `sp500_financials_pit.csv` |

## D · Credit, ownership & positioning

| ☐ | Dataset | Universe | Fields (FLDS-verify) | History · freq | Target file |
|---|---|---|---|---|---|
| ☐ | CDS spreads | Universe A | `CDS_SPREAD_5Y` (also 1Y/10Y) | 2015→now · D | `sp500_cds.csv` |
| ☐ | Credit ratings + watch + outlook + changes | Universe A | `RTG_SP_LT_LC_ISSUER_CREDIT`, `RATING_WATCH`, `RATING_OUTLOOK`, `RATING_CHANGE_DT` | history · E | extend `sp500_credit_risk.csv` |
| ☐ | Short interest & days-to-cover | Universe A | `EQY_SHORT_INTEREST`, `EQY_SHORT_INTEREST_PCT_OF_FLOAT` | 2015→now · biweekly | `sp500_short_interest.csv` |
| ☐ | Borrow rate / availability | Universe A | `EQUITY_SHORT_BORROW_RATE_NET` | 2015→now · M | `sp500_short_interest.csv` |
| ☐ | Institutional / insider / float | Universe A | `EQY_INST_PCT_SH_OUT`, `EQY_INSIDER_HOLDINGS_PCT`, `EQY_FREE_FLOAT_PCT` | 2015→now · M | extend `sp500_institutional.csv` |
| ☐ | DRSK default probability (P2) | Universe A | `DRSK_DERIVED_PROB_DEFAULT` | 2015→now · M | `sp500_drsk.csv` |
| ☐ | Index membership history + PIT weights | members(`SPX Index`) | `INDEX_ADDITION_DATE`, `idx_weight`, `idx_weight_date` | full · M *(history already on disk — verify/extend)* | extend `sp500_index_membership.csv` |
| ☐ | Insider transactions / 13F (P3, manual) ⚠ | Universe A | Form-4 / 13F via EDGAR or DES portal | quarterly | `data_processed/ownership/` |

## E · Macro, rates, FX & cross-asset

| ☐ | Dataset | Universe | Fields (FLDS-verify) | History · freq | Target file |
|---|---|---|---|---|---|
| ☐ | UST full curve (1m–30y) | `USGG1M…USGG30YR` Index | `PX_LAST` | 1994→now · D | `treasury_yields.csv` *(extend)* |
| ☐ | OIS / SOFR term structure | `USSO{tenor}` Index ⚠ | `PX_LAST` | 2018→now · D | `sofr_curve.csv` |
| ☐ | Real yields / TIPS / inflation swaps (P2) | `USGGT{tenor}`, `USSWIT{tenor}` Index ⚠ | `PX_LAST` | 2015→now · D | `real_yields.csv` |
| ☐ | Fed funds path | `FDTR` Index, `ZQ{contract}` ⚠ | `PX_LAST` | 2015→now · D | `fed_funds.csv` |
| ☐ | Financial conditions & rates-vol | `NFCI`, `MOVE`, `CVIX` Index | `PX_LAST` | 2015→now · D | `sp500_vol_indices.csv` |
| ☐ | Credit spreads (IG/HY OAS) | `LUACOAS`, `LF98OAS` Index | `PX_LAST` | 2015→now · D | `credit_spreads.csv` |
| ☐ | Realized correlation | `COR1M/3M/6M` Index | `PX_LAST` | 2010→now · D | `spx_correlation.csv` |
| ☐ | Sector & factor ETFs | `XLK/XLF/XLE/XLV/XLI/XLP/XLY/XLU/XLB/XLRE/XLC`, `SPY/QQQ/IWM/DIA` US Equity | OHLCV | 2010→now · D | extend `sp500_sector_etfs.csv` |
| ☐ | Macro surprise indices (P2) | `CESIUSD`, `BESIUSD` Index | `PX_LAST` | 2015→now · D | `macro_surprise.csv` |
| ☐ | FX majors (P3) | `DXY Curncy`, `EURUSD/USDJPY/GBPUSD Curncy` | `PX_LAST` | 2015→now · D | `fx.csv` |
| ☐ | Commodities (P3) | `CL1/GC1/HG1/NG1 Comdty` | `PX_LAST` | 2015→now · D | `commodities.csv` |
| ☐ | Global vol & CDX (P3) | `V2X/VHSI/VNKY/VKOSPI` Index, `CDXIG/CDXHY US5Y` Index | `PX_LAST` | 2015→now · D | `global_vol.csv` |

## F · Sentiment, ESG, news & events

| ☐ | Dataset | Universe | Fields (FLDS-verify) | History · freq | Target file |
|---|---|---|---|---|---|
| ☐ | Special-dividend calendar | Universe A | `DVD_ANNOUNCE_DT`, `DVD_EX_DT`, `DVD_RECORD_DT`, `DVD_AMOUNT`, `DVD_TYPE` | full · E | extend `sp500_dividends.csv` |
| ☐ | Earnings dates + BMO/AMC + confirmed | Universe A | `EARNING_ANNOUNCEMENT_TIMING`, `_CONFIRMED` | history + forward · E | extend `sp500_earnings.csv` |
| ☐ | Analyst recs / targets / changes (history) | Universe A | `BEST_ANALYST_RATING`, `TOT_ANALYST_REC`, `BEST_TARGET_PRICE`, `REC_BUY/HOLD/SELL_CNT`, `REC_RATING_CHG_DT` | 2015→now · M | `sp500_analyst_history.csv` |
| ☐ | ESG scores & controversies (P2) | Universe A | `ESG_SCORE`, `ENVIRONMENTAL/SOCIAL/GOVERNANCE_SCORE`, `ESG_CONTROVERSIES` ⚠ | 2018→now · Q | `sp500_esg.csv` |
| ☐ | Options market structure (P2) | Universe A | `OPT_PUT_CALL_OI_RATIO`, `OPT_PUT_CALL_VOLUME_RATIO`, `OPT_SKEW` ⚠ | 2015→now · D | `sp500_options_structure.csv` |
| ☐ | News sentiment & volume (P3 · D18 transparency) | Universe A | `NEWS_SENTIMENT`, `NEWS_VOLUME_PCT_CHANGE_252D` ⚠ | 2018→now · D | `sp500_news_sentiment.csv` |
| ☐ | Social sentiment (P3) ⚠ | Universe A | `SOCIAL_SENTIMENT`, `SOCIAL_VOLUME` ⚠ | recent · D | `sp500_social_sentiment.csv` |

---

## Already on disk — do NOT re-pull (wire/restore instead)

- **Deep OHLCV / vol-IV / liquidity / 5×5 IV surface, 1994/2005→2026** — in `data/bloomberg/deep/` (gitignored); wire into the connector.
- **VVIX / SKEW / MOVE / GVZ / OVX / VXN, 2011→2026** — in `data_processed/vol_indices.parquet`.
- **Per-strike option OI / greeks / IV, 2016→2026** — in Theta (`data_processed/theta/`, ~185M rows); wire into the dealer path.
- **Corporate actions (873 KB populated)** — on branch `deep-history/bloomberg-raw`; restore rather than re-pull.

## No-pull implementation work (code, not data)

See `DATA_ACQUISITION_ROADMAP.md` §9: wire the deep IV-surface archive → skew; add `as_of` to
`get_fundamentals` (PIT dividend fix); wire Theta per-strike OI → `DealerPositioningAnalyzer`; land
D19 (exit cost) + D21 (DTE→trading-bars) + recalibration together; restore corporate actions;
backfill pre-2018 OHLCV from the deep archive.
