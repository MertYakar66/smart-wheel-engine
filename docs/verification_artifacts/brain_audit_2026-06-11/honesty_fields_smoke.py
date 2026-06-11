import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

from engine.wheel_runner import WheelRunner
wr = WheelRunner()
print('provider:', type(wr.connector).__name__)
df = wr.rank_candidates_by_ev(
    tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
    top_n=10, min_ev_dollars=-1e9,
    include_diagnostic_fields=True,
)
cols = [c for c in df.columns if 'prob_profit' in c or c in ('n_scenarios', 'distribution_source', 'ev_dollars', 'iv', 'premium')]
print(df[['ticker'] + cols].to_string())
