from datetime import date
from engine.wheel_tracker import WheelTracker
from engine.transaction_costs import (
    calculate_commission,
    calculate_slippage,
    calculate_total_entry_cost,
    calculate_total_exit_cost,
    calculate_assignment_fee
)

print("=== Testing Transaction Cost Module ===\n")

print("1. Testing individual cost functions:")
comm = calculate_commission("option")
print(f"   Commission per contract: ${comm:.2f}")

slippage = calculate_slippage(mid_price=2.50, bid_ask_spread=0.25, trade_direction="sell")
print(f"   Slippage (15% of $0.25 spread): ${slippage:.4f}")

assignment_fee = calculate_assignment_fee()
print(f"   Assignment fee: ${assignment_fee:.2f}")

print("\n2. Testing entry cost calculation:")
entry_costs = calculate_total_entry_cost(premium_per_share=2.50, bid_ask_spread=0.25)
print(f"   Gross premium: ${entry_costs['gross_premium']:.2f}")
print(f"   Slippage: ${entry_costs['slippage']:.2f}")
print(f"   Commission: ${entry_costs['commission']:.2f}")
print(f"   Total cost: ${entry_costs['total_cost']:.2f}")
print(f"   Net premium collected: ${entry_costs['net_premium_collected']:.2f}")

print("\n3. Testing exit cost calculation:")
exit_costs = calculate_total_exit_cost(buyback_price_per_share=1.00, bid_ask_spread=0.10)
print(f"   Gross buyback: ${exit_costs['gross_buyback_cost']:.2f}")
print(f"   Slippage: ${exit_costs['slippage']:.2f}")
print(f"   Commission: ${exit_costs['commission']:.2f}")
print(f"   Total cost: ${exit_costs['total_cost']:.2f}")
print(f"   Total buyback cost: ${exit_costs['total_buyback_cost']:.2f}")

print("\n4. Testing full Wheel cycle with cost tracking:")
tracker = WheelTracker(100000.0)

tracker.open_short_put(
    ticker='COST_TEST',
    strike=150.0,
    premium=2.50,
    entry_date=date(2024, 1, 1),
    expiration_date=date(2024, 2, 1),
    iv=0.25
)
pos = tracker.positions['COST_TEST']
print(f"   After put entry:")
print(f"     Cash: ${tracker.cash:.2f}")
print(f"     Realized P&L: ${pos.realized_pnl:.2f}")
print(f"     Transaction costs: ${pos.transaction_costs:.2f}")

tracker.handle_put_assignment('COST_TEST', date(2024, 1, 15), 145.0)
print(f"   After assignment:")
print(f"     Cash: ${tracker.cash:.2f}")
print(f"     Transaction costs: ${pos.transaction_costs:.2f}")

tracker.open_covered_call(
    ticker='COST_TEST',
    strike=155.0,
    premium=1.50,
    entry_date=date(2024, 1, 16),
    expiration_date=date(2024, 2, 16),
    iv=0.23
)
print(f"   After call entry:")
print(f"     Cash: ${tracker.cash:.2f}")
print(f"     Realized P&L: ${pos.realized_pnl:.2f}")
print(f"     Transaction costs: ${pos.transaction_costs:.2f}")

result = tracker.close_covered_call('COST_TEST', 0.75, date(2024, 1, 25), "profit_target")
print(f"   After call buyback:")
print(f"     Cash: ${tracker.cash:.2f}")
print(f"     Realized P&L: ${pos.realized_pnl:.2f}")
print(f"     Transaction costs: ${pos.transaction_costs:.2f}")
print(f"     Net P&L: ${pos.realized_pnl - pos.transaction_costs:.2f}")

expected_min_costs = 4 * 0.65 + 5.0
print(f"\n5. Cost validation:")
print(f"   Total transaction costs: ${pos.transaction_costs:.2f}")
print(f"   Expected minimum (4 commissions + 1 assignment): ${expected_min_costs:.2f}")
print(f"   Additional costs from slippage: ${pos.transaction_costs - expected_min_costs:.2f}")

if pos.transaction_costs > expected_min_costs:
    print(f"\n✓ Slippage is being modeled correctly (costs exceed commission-only baseline)")
else:
    print(f"\n✗ Warning: Costs do not exceed baseline, slippage may not be applied")

print(f"\n✓ Transaction cost module validation complete")
