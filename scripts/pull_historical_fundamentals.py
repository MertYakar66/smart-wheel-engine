"""
#12 - Historical Fundamentals (Quarterly)
BQL: get(pe_ratio, eps, revenue, ebitda, book_value_per_share) for(members('SPX Index'))
     with(dates=range(2015-01-01, 2026-03-17), frq=Q, fill=prev)
"""
from xbbg import blp
import pandas as pd
import os

print("Getting S&P 500 members...")
members = blp.bds("SPX Index", "INDX_MWEIGHT")
tickers = [t + " Equity" for t in members['member_ticker_and_exchange_code'].tolist()]
print(f"Found {len(tickers)} tickers")

FIELDS = ['PE_RATIO', 'IS_EPS', 'SALES_REV_TURN', 'EBITDA', 'BOOK_VAL_PER_SH']
CHUNK_SIZE = 25
all_chunks = []

for i in range(0, len(tickers), CHUNK_SIZE):
    chunk = tickers[i:i+CHUNK_SIZE]
    print(f"Pulling {i+1} to {min(i+CHUNK_SIZE, len(tickers))} of {len(tickers)}...")
    try:
        df = blp.bdh(
            tickers=chunk,
            flds=FIELDS,
            start_date='2015-01-01',
            end_date='2026-03-20',
            Per='Q',
            Fill='P'
        )
        df.columns.names = ['ticker', 'field']
        long = df.stack(level=0).reset_index()
        long.columns = ['date', 'ticker', 'pe_ratio', 'eps', 'revenue', 'ebitda', 'book_value_per_share']
        long['ticker'] = long['ticker'].str.replace(' Equity', '')
        all_chunks.append(long)
        print(f"  Got {len(long)} rows")
    except Exception as e:
        print(f"  Error on chunk {i}: {e}")

print("Combining all chunks...")
result = pd.concat(all_chunks, ignore_index=True)
result['date'] = pd.to_datetime(result['date'])
result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bloomberg', 'sp500_historical_fundamentals.csv')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result.to_csv(out_path, index=False)

print(f"\nDone! Saved {len(result):,} rows")
print(f"Tickers: {result['ticker'].nunique()}")
print(f"Date range: {result['date'].min()} to {result['date'].max()}")
