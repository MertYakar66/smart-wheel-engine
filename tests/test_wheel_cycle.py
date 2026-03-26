from datetime import date
from engine.wheel_tracker import WheelTracker

tracker = WheelTracker(100000.0)

tracker.open_short_put(
    ticker='TEST',
    strike=150.0,
    premium=2.50,
    entry_date=date(2024, 1, 1),
    expiration_date=date(2024, 2, 1),
    iv=0.25
)
print(f"After put entry: State={tracker.positions['TEST'].state.value}, Cash=${tracker.cash:.2f}")

tracker.handle_put_assignment('TEST', date(2024, 1, 15), 145.0)
print(f"After assignment: State={tracker.positions['TEST'].state.value}, Cash=${tracker.cash:.2f}")

tracker.open_covered_call(
    ticker='TEST',
    strike=155.0,
    premium=1.50,
    entry_date=date(2024, 1, 16),
    expiration_date=date(2024, 2, 16),
    iv=0.23
)
print(f"After call entry: State={tracker.positions['TEST'].state.value}, Cash=${tracker.cash:.2f}")

result = tracker.close_covered_call('TEST', 0.75, date(2024, 1, 25), "profit_target")
print(f"After buyback: State={tracker.positions['TEST'].state.value}, Cash=${tracker.cash:.2f}")
print(f"Call leg P&L: ${result['call_leg_pnl']:.2f}")

tracker.open_covered_call(
    ticker='TEST',
    strike=156.0,
    premium=1.60,
    entry_date=date(2024, 1, 26),
    expiration_date=date(2024, 2, 26),
    iv=0.24
)
print(f"After 2nd call: State={tracker.positions['TEST'].state.value}, Cash=${tracker.cash:.2f}")

tracker.handle_call_expiration('TEST', date(2024, 2, 26), 154.0)
print(f"After expiry: State={tracker.positions['TEST'].state.value}, Cash=${tracker.cash:.2f}")
print(f"Total realized P&L: ${tracker.positions['TEST'].realized_pnl:.2f}")

print("\n✓ Wheel cycle complete - position cycled through multiple calls")
