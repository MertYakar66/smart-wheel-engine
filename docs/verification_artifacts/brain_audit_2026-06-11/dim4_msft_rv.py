import os
import sys
import warnings

WT = r"C:\Users\merty\Desktop\swe-main"
os.chdir(WT)
sys.path.insert(0, WT)
warnings.filterwarnings("ignore")
import engine  # noqa: E402

assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

from engine.data_connector import MarketDataConnector  # noqa: E402
from engine.forward_distribution import realized_vol_ratio, realized_vol_widening_factor  # noqa: E402

conn = MarketDataConnector()
for t in ["AAPL", "MSFT", "JPM", "XOM", "UNH"]:
    o = conn.get_ohlcv(t)
    if "date" in o.columns:
        o = o.set_index("date")
    r = realized_vol_ratio(o, as_of="2026-03-20")
    f = realized_vol_widening_factor(o, as_of="2026-03-20")
    print(f"{t}: rv30/rv252={r:.4f} factor={f:.4f} frontier={o.index.max()}")
