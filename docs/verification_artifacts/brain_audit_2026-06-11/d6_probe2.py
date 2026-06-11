import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

from datetime import date
from engine.wheel_tracker import WheelTracker

print("=== A: token-gate isolation — consume for a different ticker ===")
t = WheelTracker(initial_capital=1_000_000.0, require_ev_authority=True)
row = {"ticker": "AAPL", "strike": 180.0, "premium": 2.50, "dte": 32,
       "ev_dollars": 25.0, "prob_profit": 0.85, "distribution_source": "empirical"}
tok = t.issue_ev_authority_token(row)
ok = t._consume_ev_authority_token(tok, "MSFT", current_ev_dollars=5.0)
print(f"_consume_ev_authority_token(AAPL-token, ticker='MSFT') -> {ok}")
print(f"audit: {t._ev_authority_log[-1]['action']} ticker={t._ev_authority_log[-1].get('ticker')}")

print()
print("=== B: full open path, cross-ticker token, sized to clear all D17 gates ===")
t2 = WheelTracker(initial_capital=1_000_000.0, require_ev_authority=True)
tok2 = t2.issue_ev_authority_token(row)  # issued for AAPL strike 180
ok2 = t2.open_short_put(
    ticker="ZZZQ",            # never EV-ranked
    strike=50.0,              # small => delta-dollars under the $3,000 cap
    premium=0.50,
    entry_date=date(2026, 3, 20), expiration_date=date(2026, 4, 20),
    iv=0.25,
    ev_authority_token=tok2,
    current_ev_dollars=5.0,   # caller-supplied "fresh EV"
    prob_profit=0.90,
)
print(f"open ZZZQ (never ranked) with AAPL-issued token -> {ok2}")
print(f"positions: {list(t2.positions.keys())}")
for e in t2._ev_authority_log:
    print(f"  log: action={e.get('action')} reason={e.get('reason','')} ticker={e.get('ticker','')}")

print()
print("=== C: same but with strike/premium also mismatched vs token row (already shown) — and replay check ===")
ok3 = t2.open_short_put(
    ticker="AAPL", strike=180.0, premium=2.50,
    entry_date=date(2026, 3, 20), expiration_date=date(2026, 4, 21),
    iv=0.25, ev_authority_token=tok2, current_ev_dollars=25.0, prob_profit=0.85,
)
print(f"replay consumed token for the ORIGINAL AAPL row -> {ok3} (expected False: single-use held)")
