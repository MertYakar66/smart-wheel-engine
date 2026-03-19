# Smart Wheel Engine - Data Specification

## Overview

This document defines the complete data architecture for the Smart Wheel Trading System.
All data is stored in **Parquet format** with partitioning by ticker and date for scalability.

---

## Data Storage Architecture

```
/data/
├── raw/                          # Original data from Bloomberg
│   ├── ohlcv/
│   ├── fundamentals/
│   └── ...
├── processed/                    # Cleaned, normalized data
│   ├── ohlcv/
│   │   └── ticker={TICKER}/
│   │       └── year={YYYY}/
│   │           └── data.parquet
│   ├── options_flow/
│   │   └── ticker={TICKER}/
│   │       └── date={YYYY-MM-DD}/
│   │           └── data.parquet
│   └── ...
├── features/                     # Computed features
│   ├── technical/
│   ├── fundamental/
│   ├── sentiment/
│   └── composite/
└── cache/                        # Temporary computation cache
```

---

## Phase 1: Core Data (MUST HAVE)

### 1.1 Price Data (OHLCV)

**File:** `sp500_ohlcv.parquet`
**Frequency:** Daily
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| ticker | string | Bloomberg ticker |
| open | float64 | Opening price (adjusted) |
| high | float64 | High price (adjusted) |
| low | float64 | Low price (adjusted) |
| close | float64 | Closing price (adjusted) |
| volume | int64 | Trading volume |
| vwap | float64 | Volume-weighted average price |
| adj_factor | float64 | Adjustment factor for splits/dividends |

**Bloomberg Query:**
```
=BQL.QUERY("get(px_open, px_high, px_low, px_last, px_volume, eqy_weighted_avg_px) for(members('SPX Index')) with(dates=range(2015-01-01, 2026-03-17), fill=prev)")
```

---

### 1.2 Options Flow (CRITICAL - HIGH ALPHA)

**File:** `sp500_options_flow.parquet`
**Frequency:** Daily
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| ticker | string | Bloomberg ticker |
| call_volume | int64 | Total call volume |
| put_volume | int64 | Total put volume |
| call_oi | int64 | Call open interest |
| put_oi | int64 | Put open interest |
| call_oi_change | int64 | Daily change in call OI |
| put_oi_change | int64 | Daily change in put OI |
| put_call_volume_ratio | float64 | Put/Call volume ratio |
| put_call_oi_ratio | float64 | Put/Call OI ratio |
| atm_iv | float64 | At-the-money implied volatility |
| iv_rank | float64 | IV percentile (52-week) |
| iv_percentile | float64 | IV percentile vs history |

**Bloomberg Query:**
```
=BQL.QUERY("get(put_opt_vol, call_opt_vol, put_opt_oi, call_opt_oi, opt_iv_30d_atmf) for(members('SPX Index')) with(dates=range(2015-01-01, 2026-03-17), fill=prev)")
```

**Derived Fields (compute in pipeline):**
- `call_oi_change = call_oi - call_oi.shift(1)`
- `put_call_volume_ratio = put_volume / call_volume`
- `iv_rank = rolling_percentile(atm_iv, 252)`

---

### 1.3 Realized Volatility (Multi-Horizon)

**File:** `sp500_realized_vol.parquet`
**Frequency:** Daily
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| ticker | string | Bloomberg ticker |
| rv_5d | float64 | 5-day realized volatility (annualized) |
| rv_10d | float64 | 10-day realized volatility |
| rv_21d | float64 | 21-day realized volatility |
| rv_63d | float64 | 63-day realized volatility |
| rv_parkinson | float64 | Parkinson estimator (21d) |
| rv_garman_klass | float64 | Garman-Klass estimator (21d) |
| rv_yang_zhang | float64 | Yang-Zhang estimator (21d) |
| iv_rv_spread | float64 | IV - RV (volatility risk premium) |

**Calculation Methods:**

```python
# Close-to-close realized volatility
rv_cc = np.sqrt(252) * returns.rolling(window).std()

# Parkinson (uses high-low range)
rv_parkinson = np.sqrt(252 / (4 * np.log(2))) * np.sqrt(
    (np.log(high/low)**2).rolling(window).mean()
)

# Garman-Klass (uses OHLC)
rv_gk = np.sqrt(252) * np.sqrt(
    0.5 * (np.log(high/low)**2) -
    (2*np.log(2) - 1) * (np.log(close/open)**2)
).rolling(window).mean()
```

---

### 1.4 Earnings Enriched

**File:** `sp500_earnings_enriched.parquet`
**Frequency:** Quarterly
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| ticker | string | Bloomberg ticker |
| earnings_date | datetime | Announcement date |
| fiscal_quarter | string | e.g., "2024Q1" |
| eps_actual | float64 | Actual EPS |
| eps_estimate | float64 | Consensus estimate |
| eps_surprise | float64 | Actual - Estimate |
| eps_surprise_pct | float64 | Surprise as % of estimate |
| revenue_actual | float64 | Actual revenue |
| revenue_estimate | float64 | Consensus estimate |
| revenue_surprise_pct | float64 | Revenue surprise % |
| guidance_direction | int8 | -1 (lowered), 0 (maintained), +1 (raised) |
| pre_earnings_iv | float64 | IV 1 day before earnings |
| post_earnings_iv | float64 | IV 1 day after earnings |
| earnings_move | float64 | 1-day return post-earnings |
| implied_move | float64 | Straddle-implied move |
| move_vs_implied | float64 | Actual / Implied move |

**Bloomberg Query:**
```
=BQL.QUERY("get(is_eps, best_eps, earn_dt, sales_rev_turn, best_sales) for(members('SPX Index')) with(dates=range(2015-01-01, 2026-03-17), frq=Q)")
```

---

### 1.5 Fundamentals (Historical Quarterly)

**File:** `sp500_fundamentals.parquet`
**Frequency:** Quarterly
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Report date |
| ticker | string | Bloomberg ticker |
| market_cap | float64 | Market capitalization |
| pe_ratio | float64 | Price-to-earnings |
| pe_forward | float64 | Forward P/E |
| pb_ratio | float64 | Price-to-book |
| ps_ratio | float64 | Price-to-sales |
| ev_ebitda | float64 | EV/EBITDA |
| roe | float64 | Return on equity |
| roa | float64 | Return on assets |
| gross_margin | float64 | Gross margin % |
| operating_margin | float64 | Operating margin % |
| net_margin | float64 | Net margin % |
| debt_equity | float64 | Debt/Equity ratio |
| current_ratio | float64 | Current ratio |
| free_cash_flow | float64 | Free cash flow |
| fcf_yield | float64 | FCF / Market Cap |
| dividend_yield | float64 | Annual dividend yield |
| payout_ratio | float64 | Dividend payout ratio |

---

## Phase 2: Enhanced Data

### 2.1 News Sentiment

**File:** `sp500_news_sentiment.parquet`
**Frequency:** Daily (aggregated)
**History:** 2020-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| ticker | string | Bloomberg ticker |
| news_count | int32 | Number of news articles |
| sentiment_score | float64 | Aggregate sentiment (-1 to +1) |
| sentiment_std | float64 | Sentiment dispersion |
| positive_count | int32 | Positive articles |
| negative_count | int32 | Negative articles |
| event_tags | list[string] | Event types (M&A, upgrade, etc.) |

**Bloomberg Query:**
```
=BQL.QUERY("get(news_sentiment, news_heat_score) for(members('SPX Index')) with(dates=range(2020-01-01, 2026-03-17), fill=prev)")
```

---

### 2.2 Factor Exposures

**File:** `sp500_factor_exposure.parquet`
**Frequency:** Monthly
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Month end |
| ticker | string | Bloomberg ticker |
| factor_value | float64 | Value factor exposure |
| factor_growth | float64 | Growth factor exposure |
| factor_momentum | float64 | Momentum factor exposure |
| factor_quality | float64 | Quality factor exposure |
| factor_low_vol | float64 | Low volatility exposure |
| factor_size | float64 | Size factor exposure |
| factor_yield | float64 | Yield factor exposure |

**Calculation Methods:**
```python
# Value: composite of P/E, P/B, P/S (z-scored)
# Momentum: 12-1 month return (skip last month)
# Quality: ROE + margin stability + low leverage
# Low Vol: inverse of 252-day volatility
```

---

### 2.3 Dynamic Correlations

**File:** `sp500_correlations.parquet`
**Frequency:** Daily
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| ticker | string | Bloomberg ticker |
| corr_spx_21d | float64 | 21-day correlation to SPX |
| corr_spx_63d | float64 | 63-day correlation to SPX |
| corr_sector_21d | float64 | Correlation to sector ETF |
| beta_21d | float64 | 21-day rolling beta |
| beta_63d | float64 | 63-day rolling beta |
| idio_vol | float64 | Idiosyncratic volatility |

---

## Phase 3: Advanced Data

### 3.1 Borrow Rates / Short Data

**File:** `sp500_borrow_rates.parquet`
**Frequency:** Daily
**History:** 2018-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| ticker | string | Bloomberg ticker |
| short_interest | int64 | Shares sold short |
| short_interest_ratio | float64 | Days to cover |
| utilization | float64 | % of lendable shares borrowed |
| borrow_rate | float64 | Annualized cost to borrow |
| available_shares | int64 | Shares available to borrow |

---

### 3.2 Macro Events Calendar

**File:** `macro_events.parquet`
**Frequency:** Event-based
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Event date |
| event_type | string | CPI, FOMC, NFP, etc. |
| event_time | string | Time of release |
| actual | float64 | Actual value |
| consensus | float64 | Consensus estimate |
| prior | float64 | Prior reading |
| surprise | float64 | Actual - Consensus |
| market_impact | float64 | SPX move in 30 min |

---

### 3.3 ETF Flows

**File:** `etf_flows.parquet`
**Frequency:** Daily
**History:** 2015-01-01 to present

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| etf_ticker | string | ETF ticker (SPY, XLF, etc.) |
| flow_usd | float64 | Daily flow in USD |
| flow_shares | int64 | Daily flow in shares |
| aum | float64 | Assets under management |
| flow_pct | float64 | Flow as % of AUM |
| cumulative_flow_30d | float64 | 30-day cumulative flow |

---

## Index & Membership Data

### Historical Index Membership

**File:** `sp500_membership_history.parquet`

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Effective date |
| ticker | string | Bloomberg ticker |
| action | string | "ADD" or "REMOVE" |
| replacing | string | Ticker being replaced (if ADD) |

**Critical for survivorship-bias-free backtesting.**

---

## Intraday Data (Selective)

### 5-Minute Bars (Last 2 Years)

**File:** `sp500_intraday_5min.parquet`
**Partitioning:** `ticker={TICKER}/date={YYYY-MM-DD}/`

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Bar timestamp (ET) |
| ticker | string | Bloomberg ticker |
| open | float64 | Open |
| high | float64 | High |
| low | float64 | Low |
| close | float64 | Close |
| volume | int64 | Volume |
| vwap | float64 | VWAP |

**Storage estimate:** ~200GB for 2 years, all SPX members

---

## Data Quality Requirements

### Validation Rules

1. **No future data leakage** - All point-in-time data must use `as_of_date`
2. **Survivorship bias handling** - Use historical index membership
3. **Adjustment consistency** - All prices must be split/dividend adjusted
4. **Missing data handling:**
   - Forward-fill prices (max 5 days)
   - Flag stale fundamentals
   - Never interpolate earnings dates

### Quality Checks (Automated)

```python
def validate_ohlcv(df):
    assert (df['high'] >= df['low']).all(), "High < Low violation"
    assert (df['high'] >= df['open']).all(), "High < Open violation"
    assert (df['high'] >= df['close']).all(), "High < Close violation"
    assert (df['low'] <= df['open']).all(), "Low > Open violation"
    assert (df['low'] <= df['close']).all(), "Low > Close violation"
    assert (df['volume'] >= 0).all(), "Negative volume"
    assert df['close'].notna().mean() > 0.99, "Too many missing prices"
```

---

## Data Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Bloomberg     │────▶│   Raw Parquet   │────▶│   Validation    │
│   Terminal      │     │   (raw/)        │     │   Pipeline      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │   Processed     │◀─────────────┘
                        │   (processed/)  │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Technical   │    │  Fundamental  │    │   Sentiment   │
│   Features    │    │   Features    │    │   Features    │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                   ┌─────────────────┐
                   │   Feature Store │
                   │   (features/)   │
                   └─────────────────┘
```

---

## Refresh Schedule

| Data Type | Refresh Frequency | Lag |
|-----------|-------------------|-----|
| OHLCV | Daily 6:00 PM ET | T+0 |
| Options Flow | Daily 6:00 PM ET | T+0 |
| Fundamentals | After earnings | T+1 |
| Earnings | As announced | T+0 |
| News Sentiment | Daily 6:00 PM ET | T+0 |
| Factor Exposure | Monthly | T+5 |
| Borrow Rates | Daily | T+1 |
| Macro Events | As released | T+0 |

---

## Storage Estimates

| Dataset | Rows (10Y) | Size (Parquet) |
|---------|------------|----------------|
| OHLCV | ~1.3M | ~500 MB |
| Options Flow | ~1.3M | ~400 MB |
| Fundamentals | ~200K | ~100 MB |
| Realized Vol | ~1.3M | ~300 MB |
| Earnings | ~20K | ~10 MB |
| News Sentiment | ~650K | ~200 MB |
| Correlations | ~1.3M | ~200 MB |
| Intraday 5min | ~500M | ~200 GB |
| **Total (ex-intraday)** | | **~2 GB** |
| **Total (with intraday)** | | **~200 GB** |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-19 | Initial specification |
