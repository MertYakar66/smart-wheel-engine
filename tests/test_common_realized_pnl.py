"""Ground-truth dollar value-asserts for the canonical realized-P&L + friction
helpers in ``backtests/regression/_common.py`` (#456 item C).

These helpers underpin every backtest scenario (S22/S27/S32/S34) and the W6/W7
heavy-verify drivers, but were only exercised *indirectly* (via wheel-level
outcomes) — no test pinned a hand-computed dollar value. The W7 double-count
(#451) is exactly the bug class that slips when accounting is asserted by shape
(finiteness / sign / inequality) rather than by value. Every number below is
computed by hand in the assertion comment and is NOT copied from the code under
test — break the formula and the literal fails.
"""

from __future__ import annotations

import pytest

from backtests.regression._common import (
    _forward_replay_realized_pnl as realized_pnl,
)
from backtests.regression._common import (
    friction_adjusted_premium,
    friction_assignment_cost,
    friction_open_cost,
)

# ---------------------------------------------------------------------------
# _forward_replay_realized_pnl(strike, premium, spot_at_expiry)
#   = (premium - max(0, strike - spot)) * 100   [dollars per 1-contract]
# ---------------------------------------------------------------------------


def test_realized_pnl_otm_keeps_full_premium():
    # spot 105 >= strike 100 -> intrinsic 0 -> (3 - 0)*100 = +300.00
    assert realized_pnl(100.0, 3.0, 105.0) == pytest.approx(300.0)


def test_realized_pnl_atm_exact_keeps_full_premium():
    # spot == strike -> intrinsic max(0, 0) = 0 -> +300.00 (boundary is OTM-side)
    assert realized_pnl(100.0, 3.0, 100.0) == pytest.approx(300.0)


def test_realized_pnl_breakeven_is_exactly_zero():
    # spot 97 = strike 100 - premium 3 -> intrinsic 3 -> (3 - 3)*100 = 0.00
    assert realized_pnl(100.0, 3.0, 97.0) == pytest.approx(0.0)


def test_realized_pnl_itm_loses_intrinsic_minus_premium():
    # spot 90 -> intrinsic 10 -> (3 - 10)*100 = -700.00
    assert realized_pnl(100.0, 3.0, 90.0) == pytest.approx(-700.0)


def test_realized_pnl_worst_case_spot_zero():
    # spot 0 -> intrinsic 100 -> (3 - 100)*100 = -9700.00
    assert realized_pnl(100.0, 3.0, 0.0) == pytest.approx(-9700.0)


def test_realized_pnl_second_independent_example():
    # strike 50, prem 1.25, spot 48 -> intrinsic 2 -> (1.25 - 2)*100 = -75.00
    assert realized_pnl(50.0, 1.25, 48.0) == pytest.approx(-75.0)


# ---------------------------------------------------------------------------
# friction_adjusted_premium(premium, level)
#   none      -> premium
#   bid_ask / full -> max(0, premium - max(0.05, 0.08*premium))
# ---------------------------------------------------------------------------


def test_friction_premium_none_is_passthrough():
    assert friction_adjusted_premium(3.0, "none") == pytest.approx(3.0)


def test_friction_premium_percent_branch():
    # 0.08*3.0 = 0.24 > 0.05 floor -> 3.0 - 0.24 = 2.76
    assert friction_adjusted_premium(3.0, "full") == pytest.approx(2.76)


def test_friction_premium_floor_branch():
    # 0.08*0.40 = 0.032 < 0.05 floor -> 0.40 - 0.05 = 0.35
    assert friction_adjusted_premium(0.40, "full") == pytest.approx(0.35)


def test_friction_premium_floor_boundary():
    # 0.08*0.625 = 0.05 exactly -> half_spread = 0.05 -> 0.625 - 0.05 = 0.575
    assert friction_adjusted_premium(0.625, "full") == pytest.approx(0.575)


def test_friction_premium_cannot_go_negative():
    # 0.03 - max(0.05, 0.0024) = 0.03 - 0.05 = -0.02 -> clamped to 0.0
    assert friction_adjusted_premium(0.03, "full") == pytest.approx(0.0)


def test_friction_premium_bid_ask_equals_full_on_open_spread():
    assert friction_adjusted_premium(3.0, "bid_ask") == pytest.approx(
        friction_adjusted_premium(3.0, "full")
    )


def test_friction_premium_invalid_level_raises():
    with pytest.raises(ValueError):
        friction_adjusted_premium(3.0, "bogus")


# ---------------------------------------------------------------------------
# friction_open_cost(contracts, level): full -> 0.65*contracts, else 0.0
# ---------------------------------------------------------------------------


def test_open_cost_full_one_contract():
    assert friction_open_cost(1, "full") == pytest.approx(0.65)


def test_open_cost_full_scales_per_contract():
    assert friction_open_cost(5, "full") == pytest.approx(3.25)  # 0.65 * 5


def test_open_cost_zero_below_full():
    assert friction_open_cost(1, "none") == 0.0
    assert friction_open_cost(3, "bid_ask") == 0.0


# ---------------------------------------------------------------------------
# friction_assignment_cost(strike, contracts, level)
#   full -> 0.0010*strike*100*contracts + 0.65*contracts ; else 0.0
# ---------------------------------------------------------------------------


def test_assignment_cost_full_single():
    # 0.0010*100*100*1 + 0.65*1 = 10.00 + 0.65 = 10.65
    assert friction_assignment_cost(100.0, 1, "full") == pytest.approx(10.65)


def test_assignment_cost_full_scales_strike_and_contracts():
    # 0.0010*250*100*2 + 0.65*2 = 50.00 + 1.30 = 51.30
    assert friction_assignment_cost(250.0, 2, "full") == pytest.approx(51.30)


def test_assignment_cost_zero_below_full():
    assert friction_assignment_cost(100.0, 1, "none") == 0.0
    assert friction_assignment_cost(100.0, 1, "bid_ask") == 0.0


# ---------------------------------------------------------------------------
# Composition — a held-to-expiry ITM short put at FULL friction, the way the
# backtest accounts it. Pins the COMPOSED dollar value (the W7 lesson: assert
# the whole leg, not just that each piece is finite).
# ---------------------------------------------------------------------------


def test_full_friction_itm_put_leg_composed_value():
    # strike 100, gross premium 3.0, spot@expiry 90, 1 contract, full friction:
    #   net premium   = max(0, 3.0 - max(0.05, 0.24))      = 2.76
    #   replay P&L    = (2.76 - max(0, 100-90)) * 100       = (2.76 - 10)*100 = -724.00
    #   open cost     = 0.65
    #   assign cost   = 0.0010*100*100 + 0.65               = 10.65
    #   total         = -724.00 - 0.65 - 10.65              = -735.30
    net_prem = friction_adjusted_premium(3.0, "full")
    leg = realized_pnl(100.0, net_prem, 90.0)
    total = leg - friction_open_cost(1, "full") - friction_assignment_cost(100.0, 1, "full")
    assert leg == pytest.approx(-724.0)
    assert total == pytest.approx(-735.30)


def test_full_friction_otm_put_leg_keeps_net_premium_minus_open():
    # OTM (spot 105 > strike 100): no assignment cost; net premium kept minus open
    #   net premium = 2.76 -> replay (2.76)*100 = 276.00 ; minus open 0.65 = 275.35
    net_prem = friction_adjusted_premium(3.0, "full")
    total = realized_pnl(100.0, net_prem, 105.0) - friction_open_cost(1, "full")
    assert total == pytest.approx(275.35)
