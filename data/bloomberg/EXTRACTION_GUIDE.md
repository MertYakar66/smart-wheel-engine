# Bloomberg Data Extraction Guide

## Overview

This guide tells you exactly what to type in Bloomberg Excel Add-In to extract
all data needed for the Smart Wheel Engine. There are **7 Excel workbooks** to
create and save as CSV.

---

## Setup

1. Open Excel on the Bloomberg terminal
2. Make sure Bloomberg Add-In is loaded (check for Bloomberg ribbon tab)
3. Create one workbook per category below
4. Paste formulas, let them populate, then **Save As → CSV (UTF-8)**
5. Put the saved CSVs in the directories specified below

---

## Tickers to Extract

Primary universe (paste into formulas where you see `{TICKER}`):

```
MAG7:    AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
Finance: JPM, BAC, WFC, GS
Health:  UNH, JNJ, LLY, ABBV
Energy:  XOM, CVX
Consumer: PG, KO, HD, MCD
Industrial: CAT, HON, GE
```

You can add more SP500 tickers later. Start with these 24.

---

## Workbook 1: OHLCV Price History

**Save to:** `data/bloomberg/ohlcv/{TICKER}.csv` (one file per ticker)

**Create one sheet per ticker.** In cell A1:

```
=BDH("{TICKER} US Equity","PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME","20190101","20260217","Dir=V")
```

**Examples:**
```
=BDH("AAPL US Equity","PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME","20190101","20260217","Dir=V")
=BDH("MSFT US Equity","PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME","20190101","20260217","Dir=V")
```

**Output will look like:**
| Date | PX_OPEN | PX_HIGH | PX_LOW | PX_LAST | PX_VOLUME |
|------|---------|---------|--------|---------|-----------|
| 01/02/2019 | 154.89 | 158.85 | 154.23 | 157.92 | 37039737 |

**Save each sheet as:** `AAPL.csv`, `MSFT.csv`, etc.

---

## Workbook 2: Option Chains

**Save to:** `data/bloomberg/options/{TICKER}.csv`

### Step 1: Get the option chain tickers

In cell A1:
```
=BDS("{TICKER} US Equity","OPT_CHAIN")
```

This returns a list of option ticker strings like `AAPL 3/21/25 C200`.

### Step 2: For each option ticker, pull data

Once you have the option tickers in column A, use BDP in adjacent columns:

| Column B (header: strike) | Column C (header: option_type) | Column D (header: expiration) | Column E (header: bid) | Column F (header: ask) | Column G (header: implied_vol) | Column H (header: open_interest) | Column I (header: volume) | Column J (header: delta) | Column K (header: underlying_price) |
|---|---|---|---|---|---|---|---|---|---|
| `=BDP(A2,"OPT_STRIKE_PX")` | `=BDP(A2,"OPT_PUT_CALL")` | `=BDP(A2,"OPT_EXPIRE_DT")` | `=BDP(A2,"BID")` | `=BDP(A2,"ASK")` | `=BDP(A2,"OPT_IMPLIED_VOLATILITY_MID")` | `=BDP(A2,"OPEN_INT")` | `=BDP(A2,"VOLUME")` | `=BDP(A2,"OPT_DELTA")` | `=BDP(A2,"OPT_UNDL_PX")` |

### Alternative: Simpler approach with OMON

1. Type `OMON` in Bloomberg terminal for the ticker
2. Set your filters (30-60 DTE, strikes near ATM)
3. Use `Export to Excel` button in OMON
4. Save as CSV with the column headers above

**Tip:** Focus on options with 20-60 DTE and strikes within ±15% of spot.
You do NOT need every strike — just the liquid ones near ATM.

---

## Workbook 3: Earnings History

**Save to:** `data/bloomberg/earnings/{TICKER}.csv`

In cell A1:
```
=BDS("{TICKER} US Equity","ERN_ANN_DT_AND_PER","EARN_ANN_DT_TIME_HIST_WITH_EPS=Y","START_DT=20190101")
```

If the above doesn't give enough fields, use BDH with quarterly periodicity:
```
=BDH("{TICKER} US Equity","IS_EPS,BEST_EPS_MEDIAN,EARN_EST_EPS_SURPRISE_PCT","20190101","20260217","Dir=V","Per=Q")
```

**Expected output:**
| Date | IS_EPS | BEST_EPS_MEDIAN | EARN_EST_EPS_SURPRISE_PCT |
|------|--------|-----------------|--------------------------|
| 01/28/2025 | 2.40 | 2.35 | 2.13 |

**Also get earnings timing (BMO/AMC):**
```
=BDP("{TICKER} US Equity","EARNING_ANNOUNCEMENT_TIMING")
```

**Save each as:** `AAPL.csv`, `MSFT.csv`, etc.

---

## Workbook 4: Dividend History

**Save to:** `data/bloomberg/dividends/{TICKER}.csv`

In cell A1:
```
=BDS("{TICKER} US Equity","DVD_HIST_ALL","DVD_START_DT=20190101","DVD_END_DT=20260217")
```

**Expected output:**
| Ex-Date | Record Date | Payable Date | Dividend Amount | Dividend Frequency | Dividend Type |
|---------|-------------|-------------|-----------------|-------------------|---------------|
| 02/07/2025 | 02/10/2025 | 02/13/2025 | 0.25 | Quarterly | Regular Cash |

**Save each as:** `AAPL.csv`, `MSFT.csv`, etc.

---

## Workbook 5: IV History (Daily)

**Save to:** `data/bloomberg/iv_history/{TICKER}.csv`

In cell A1:
```
=BDH("{TICKER} US Equity","30DAY_IMPVOL_100.0%MNY_DF,60DAY_IMPVOL_100.0%MNY_DF,30DAY_IMPVOL_90.0%MNY_DF,30DAY_IMPVOL_110.0%MNY_DF,20DAY_HV,60DAY_HV","20190101","20260217","Dir=V")
```

**Breakdown of fields:**
- `30DAY_IMPVOL_100.0%MNY_DF` → 30-day ATM implied vol
- `60DAY_IMPVOL_100.0%MNY_DF` → 60-day ATM implied vol
- `30DAY_IMPVOL_90.0%MNY_DF` → 30-day 25-delta put IV (skew)
- `30DAY_IMPVOL_110.0%MNY_DF` → 30-day 25-delta call IV (skew)
- `20DAY_HV` → 20-day historical (realized) volatility
- `60DAY_HV` → 60-day historical (realized) volatility

**Expected output:**
| Date | 30DAY_IMPVOL_100.0%MNY_DF | 60DAY_IMPVOL_100.0%MNY_DF | 30DAY_IMPVOL_90.0%MNY_DF | 30DAY_IMPVOL_110.0%MNY_DF | 20DAY_HV | 60DAY_HV |
|------|---------------------------|---------------------------|--------------------------|--------------------------|----------|----------|
| 01/02/2019 | 35.42 | 33.18 | 38.91 | 32.45 | 42.12 | 28.76 |

**Note:** Bloomberg reports IV as percentage (35.42 = 35.42%). The loader
handles normalization to decimal (0.3542) automatically.

**Save each as:** `AAPL.csv`, `MSFT.csv`, etc.

---

## Workbook 6: Treasury Yields

**Save to:** `data/bloomberg/rates/treasury_yields.csv` (single file)

In cell A1:
```
=BDH("USGG3M Index","PX_LAST","20190101","20260217","Dir=V")
```

For multiple tenors (put each in a separate column starting from B1):
```
Cell A1: =BDH("USGG3M Index","PX_LAST","20190101","20260217","Dir=V")
Cell C1: =BDH("USGG6M Index","PX_LAST","20190101","20260217","Dir=V")
Cell E1: =BDH("USGG2YR Index","PX_LAST","20190101","20260217","Dir=V")
Cell G1: =BDH("USGG10YR Index","PX_LAST","20190101","20260217","Dir=V")
```

**Simplest option:** Just the 3-month T-bill rate is sufficient. The engine
defaults to 5% if no rates data is available.

**Save as:** `treasury_yields.csv`

---

## Workbook 7: Fundamentals

**Save to:** `data/bloomberg/fundamentals/sp500_fundamentals.csv` (single file)

Create a column of tickers in A2:A25 (e.g., `AAPL US Equity`, `MSFT US Equity`, ...).
Then use BDP:

| Col A (ticker) | Col B | Col C | Col D | Col E | Col F |
|---|---|---|---|---|---|
| Header: Security | Market_Cap | GICS_Sector | GICS_Industry | Div_Yield | PE |
| AAPL US Equity | `=BDP(A2,"CUR_MKT_CAP")` | `=BDP(A2,"GICS_SECTOR_NAME")` | `=BDP(A2,"GICS_INDUSTRY_GROUP_NAME")` | `=BDP(A2,"EQY_DVD_YLD_IND")` | `=BDP(A2,"PE_RATIO")` |

**Save as:** `sp500_fundamentals.csv`

---

## File Structure After Extraction

```
data/bloomberg/
├── ohlcv/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── GOOGL.csv
│   └── ... (24 files)
├── options/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ... (24 files)
├── earnings/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ... (24 files)
├── dividends/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ... (24 files)
├── iv_history/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ... (24 files)
├── rates/
│   └── treasury_yields.csv
└── fundamentals/
    └── sp500_fundamentals.csv
```

Total files: ~122 CSVs (24 tickers × 5 per-ticker categories + 2 shared)

---

## Priority Order (If Short on Time)

If you can't get everything in one session, extract in this order:

1. **OHLCV** (essential — everything depends on price history)
2. **Option chains** (essential — needed for trade universe)
3. **IV history** (high priority — IV rank drives entry signals)
4. **Earnings** (high priority — earnings ML model needs this)
5. **Dividends** (medium — needed for BSM pricing accuracy and LSM)
6. **Rates** (low — engine has 5% default, small impact)
7. **Fundamentals** (low — engine has hardcoded sector map already)

---

## Troubleshooting

**"#N/A" in cells:**
- The security might need a different suffix (e.g., `GOOG` vs `GOOGL`)
- BDH date range might be too long — try shorter range
- Field might not exist for that security

**Slow loading:**
- BDH with many fields + long date range takes time
- Let it fully populate before saving (watch the status bar)
- Consider splitting into 2-year chunks if very slow

**Excel Add-In not showing:**
- File → Options → Add-ins → Manage COM Add-ins → Check Bloomberg

**Data looks wrong:**
- Check if values are in percentage or decimal (IV especially)
- The loader handles both formats, but verify a few values manually
