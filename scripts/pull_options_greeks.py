"""
#13 - Options Greeks & IV Term Structure
BQL: get(opt_iv_30d_atmf, opt_iv_60d_atmf, opt_iv_90d_atmf, opt_skew_30d)
     for(members('SPX Index')) with(dates=range(2015-01-01, 2026-03-17), fill=prev)
"""
from xbbg import blp
import pandas as pd
import os

print("Getting S&P 500 members...")
members = blp.bds("SPX Index", "INDX_MWEIGHT")
tickers = [t + " Equity" for t in members['member_ticker_and_exchange_code'].tolist()]
print(f"Found {len(tickers)} tickers")

# Try multiple field name variants - Bloomberg field names vary by terminal
FIELD_SETS = [
    {
        'names': ['30D_IMPVOL_100.0%MNY_DF', '60D_IMPVOL_100.0%MNY_DF', '90D_IMPVOL_100.0%MNY_DF', 'PUT_CALL_OPEN_INTEREST_RATIO'],
        'columns': ['iv_30d_atm', 'iv_60d_atm', 'iv_90d_atm', 'skew_30d']
    },
    {
        'names': ['IVOL_30D', 'IVOL_60D', 'IVOL_90D', 'PUT_CALL_OPEN_INTEREST_RATIO'],
        'columns': ['iv_30d_atm', 'iv_60d_atm', 'iv_90d_atm', 'skew_30d']
    },
    {
        'names': ['HIST_PUT_IMP_VOL', 'HIST_CALL_IMP_VOL', '30DAY_IMPVOL', '90DAY_IMPVOL'],
        'columns': ['iv_put', 'iv_call', 'iv_30d', 'iv_90d']
    }
]

CHUNK_SIZE = 20
all_chunks = []
working_fields = None

# Test with first ticker to find which fields work
print("Testing field availability...")
for fs in FIELD_SETS:
    try:
        test = blp.bdh(
            tickers=[tickers[0]],
            flds=fs['names'],
            start_date='2025-01-01',
            end_date='2026-03-20'
        )
        if len(test) > 0:
            working_fields = fs
            print(f"  Using fields: {fs['names']}")
            break
    except Exception as e:
        print(f"  Fields {fs['names'][:2]}... not available: {e}")

if working_fields is None:
    print("WARNING: No IV field set worked. Trying individual BDP approach...")
    # Fallback: pull current IV snapshot per ticker
    records = []
    for i, t in enumerate(tickers):
        try:
            data = blp.bdp(t, ['IVOL_30D', 'IVOL_60D', 'IVOL_90D', 'HIST_PUT_IMP_VOL', 'HIST_CALL_IMP_VOL'])
            if len(data) > 0:
                row = data.iloc[0].to_dict()
                row['ticker'] = t.replace(' Equity', '')
                records.append(row)
        except:
            pass
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)}")

    result = pd.DataFrame(records)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bloomberg', 'sp500_options_greeks.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Done (snapshot only)! Saved {len(result)} rows")
else:
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i+CHUNK_SIZE]
        print(f"Pulling {i+1} to {min(i+CHUNK_SIZE, len(tickers))} of {len(tickers)}...")
        try:
            df = blp.bdh(
                tickers=chunk,
                flds=working_fields['names'],
                start_date='2015-01-01',
                end_date='2026-03-20',
                Fill='P'
            )
            df.columns.names = ['ticker', 'field']
            long = df.stack(level=0).reset_index()
            cols = ['date', 'ticker'] + working_fields['columns']
            long.columns = cols
            long['ticker'] = long['ticker'].str.replace(' Equity', '')
            all_chunks.append(long)
            print(f"  Got {len(long)} rows")
        except Exception as e:
            print(f"  Error on chunk {i}: {e}")

    print("Combining all chunks...")
    result = pd.concat(all_chunks, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bloomberg', 'sp500_options_greeks.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\nDone! Saved {len(result):,} rows")
    print(f"Tickers: {result['ticker'].nunique()}")
