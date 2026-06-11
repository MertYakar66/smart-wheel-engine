import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

from engine.wheel_runner import WheelRunner
conn = WheelRunner().connector
for t in ['NFLX', 'BKNG', 'CVNA']:
    df = conn.get_ohlcv(t)
    if df is None or len(df) == 0:
        print(t, 'NO DATA'); continue
    c = df['close']
    last = c.iloc[-1]
    print(f"{t}: rows={len(df)} close_min={c.min():.2f} close_max={c.max():.2f} last={last:.2f} "
          f"last20_mean={c.iloc[-20:].mean():.2f} pre-seam(2026-03-20 area) vs post check")
    # ratio across the documented 2026-03-23 seam
    d = df.copy()
    import pandas as pd
    d['date'] = pd.to_datetime(d['date'])
    pre = d[d['date'] <= '2026-03-20']['close']
    post = d[d['date'] > '2026-03-20']['close']
    if len(pre) and len(post):
        print(f"   pre-seam last={pre.iloc[-1]:.2f}  post-seam first={post.iloc[0]:.2f}  ratio={pre.iloc[-1]/post.iloc[0]:.2f}")
    else:
        print(f"   pre n={len(pre)}, post n={len(post)} (frontier 2026-03-20 -> no post-seam rows)")
