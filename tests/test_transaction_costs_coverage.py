"""Coverage tests for engine/transaction_costs.py.

Targets the missing-line set identified by the 2026-05-08 audit (F7):
lines 66, 73-74, 133, 135, 138->146 branch, 143, 283, 354-372.

Tests are hermetic — pure-Python arithmetic, no fixtures from disk.
"""

from __future__ import annotations

import math

import pytest

from engine import transaction_costs as tc


# ---------------------------------------------------------------------------
# calculate_actual_spread — covers lines 66 (normal bid/ask), 73-74 (basis fallback)
# ---------------------------------------------------------------------------


class TestCalculateActualSpread:
    def test_normal_bid_ask_returns_difference(self):
        # Line 66: ask >= bid >= 0 → return ask - bid
        assert tc.calculate_actual_spread(bid=1.00, ask=1.20) == pytest.approx(0.20)

    def test_zero_bid_zero_ask_returns_zero(self):
        # Edge: bid=ask=0 still satisfies ask >= bid >= 0
        assert tc.calculate_actual_spread(bid=0.0, ask=0.0) == 0.0

    def test_inverted_quote_falls_through_to_mid(self):
        # ask < bid violates the ask >= bid guard — should fall through.
        # With mid_price provided, returns mid_price * fallback_pct.
        spread = tc.calculate_actual_spread(bid=1.5, ask=1.0, mid_price=1.25)
        assert spread == pytest.approx(0.125)  # 1.25 * 0.10

    def test_no_bid_or_ask_uses_mid_fallback(self):
        # First fallback: mid_price * fallback_pct
        spread = tc.calculate_actual_spread(bid=None, ask=None, mid_price=2.0, fallback_pct=0.10)
        assert spread == pytest.approx(0.20)

    def test_no_mid_uses_ask_basis(self):
        # Lines 73-74: mid is None → basis = ask, returns ask * fallback_pct
        spread = tc.calculate_actual_spread(bid=None, ask=2.0, mid_price=None, fallback_pct=0.10)
        assert spread == pytest.approx(0.20)

    def test_no_mid_no_ask_uses_bid_basis(self):
        # Lines 73-74: mid + ask both None → basis = bid
        spread = tc.calculate_actual_spread(bid=1.5, ask=None, mid_price=None, fallback_pct=0.10)
        assert spread == pytest.approx(0.15)

    def test_no_inputs_returns_zero(self):
        # Lines 73-74 final branch: everything None → basis=0 → 0 * pct = 0
        assert tc.calculate_actual_spread(bid=None, ask=None, mid_price=None) == 0.0

    def test_zero_mid_falls_through_to_basis(self):
        # mid_price=0 fails `mid_price > 0`, so we fall through to basis.
        spread = tc.calculate_actual_spread(bid=None, ask=3.0, mid_price=0.0, fallback_pct=0.10)
        assert spread == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# calculate_slippage — covers lines 133, 135, 143, 138->146 branch
# ---------------------------------------------------------------------------


class TestCalculateSlippageLiquidityBuckets:
    """Open-interest tier multipliers (lines 130-135)."""

    def test_open_interest_below_50_doubles_and_a_half(self):
        # OI < 50 → base_factor *= 2.5 (line 131)
        slippage = tc.calculate_slippage(
            mid_price=10.0,
            bid_ask_spread=0.10,
            trade_direction="sell",
            open_interest=10,
        )
        # base = 0.15 * 2.5 = 0.375; clamped to 0.50 cap; spread_slippage = 0.10 * 0.375
        assert slippage == pytest.approx(0.10 * 0.375)

    def test_open_interest_below_100_doubles(self):
        # Line 133: OI in [50, 100) → base_factor *= 2.0
        slippage = tc.calculate_slippage(
            mid_price=10.0,
            bid_ask_spread=0.10,
            trade_direction="sell",
            open_interest=80,
        )
        # base = 0.15 * 2.0 = 0.30; spread_slippage = 0.10 * 0.30
        assert slippage == pytest.approx(0.10 * 0.30)

    def test_open_interest_below_500_one_and_a_half(self):
        # Line 135: OI in [100, 500) → base_factor *= 1.5
        slippage = tc.calculate_slippage(
            mid_price=10.0,
            bid_ask_spread=0.10,
            trade_direction="sell",
            open_interest=300,
        )
        # base = 0.15 * 1.5 = 0.225; spread_slippage = 0.10 * 0.225
        assert slippage == pytest.approx(0.10 * 0.225)

    def test_open_interest_at_or_above_500_unchanged(self):
        # OI >= 500 → no liquidity penalty
        slippage = tc.calculate_slippage(
            mid_price=10.0,
            bid_ask_spread=0.10,
            trade_direction="sell",
            open_interest=1000,
        )
        # base = 0.15; spread_slippage = 0.10 * 0.15
        assert slippage == pytest.approx(0.10 * 0.15)


class TestCalculateSlippageSpreadPenalties:
    """Spread-pct penalty branches (lines 138-143)."""

    def test_zero_mid_skips_spread_pct_branch(self):
        # Branch 138->146: mid_price <= 0 skips the spread_pct adjustment.
        # Sqrt-impact also disabled (no adv_contracts), so size_slippage = 0.
        slippage = tc.calculate_slippage(
            mid_price=0.0,
            bid_ask_spread=0.05,
            trade_direction="sell",
        )
        # base_factor unchanged at 0.15; spread_slippage = 0.05 * 0.15
        assert slippage == pytest.approx(0.05 * 0.15)

    def test_wide_spread_above_30pct_increases_penalty(self):
        # Line 141: spread_pct > 0.30 → base_factor *= 1.5
        # mid_price=1.0, spread=0.40 → spread_pct = 0.40 > 0.30
        slippage = tc.calculate_slippage(
            mid_price=1.0,
            bid_ask_spread=0.40,
            trade_direction="buy",
        )
        # base = 0.15 * 1.5 = 0.225; capped under 0.50; spread_slippage = 0.40 * 0.225
        assert slippage == pytest.approx(0.40 * 0.225)


class TestCalculateSlippageSqrtImpact:
    """Almgren-Chriss sqrt impact (lines 149-153)."""

    def test_sqrt_impact_zero_when_adv_unknown(self):
        # adv_contracts=None → size_slippage = 0
        slippage = tc.calculate_slippage(
            mid_price=10.0,
            bid_ask_spread=0.10,
            trade_direction="sell",
            num_contracts=10,
            adv_contracts=None,
        )
        assert slippage == pytest.approx(0.10 * 0.15)  # spread-only

    def test_sqrt_impact_disabled_returns_spread_only(self):
        slippage = tc.calculate_slippage(
            mid_price=10.0,
            bid_ask_spread=0.10,
            trade_direction="sell",
            num_contracts=10,
            adv_contracts=100,
            use_sqrt_impact=False,
        )
        assert slippage == pytest.approx(0.10 * 0.15)

    def test_sqrt_impact_scales_with_participation(self):
        # 10 contracts on 100 ADV → participation = 0.10 → sqrt = 0.3162...
        # size_slippage = 0.10 * 10.0 * 0.3162... = ~0.3162
        slippage = tc.calculate_slippage(
            mid_price=10.0,
            bid_ask_spread=0.10,
            trade_direction="sell",
            num_contracts=10,
            adv_contracts=100,
            use_sqrt_impact=True,
            impact_coefficient=0.10,
        )
        expected_size = 0.10 * 10.0 * math.sqrt(0.10)
        expected_spread = 0.10 * 0.15
        assert slippage == pytest.approx(expected_spread + expected_size)


# ---------------------------------------------------------------------------
# calculate_total_exit_cost — covers line 283 (auto-spread fallback)
# ---------------------------------------------------------------------------


class TestExitCostSpreadFallback:
    def test_exit_cost_without_spread_computes_one(self):
        # Line 283: bid_ask_spread is None → calculate_actual_spread is called
        result = tc.calculate_total_exit_cost(
            buyback_price_per_share=2.0,
            bid_ask_spread=None,
            bid=1.95,
            ask=2.05,
        )
        # Spread should be 2.05 - 1.95 = 0.10; slippage = 0.10 * 0.15 = 0.015 per share
        # slippage_per_contract = 0.015 * 100 = 1.50
        assert result["slippage"] == pytest.approx(1.50)
        assert result["commission"] == pytest.approx(tc.DEFAULT_COMMISSION_PER_CONTRACT)
        assert result["gross_buyback_cost"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# estimate_round_trip_cost — covers lines 354-372
# ---------------------------------------------------------------------------


class TestEstimateRoundTripCost:
    def test_full_round_trip_with_explicit_spreads(self):
        result = tc.estimate_round_trip_cost(
            entry_premium=2.50,
            expected_exit_premium=1.00,
            entry_spread=0.10,
            exit_spread=0.08,
            open_interest=200,
        )
        assert result["entry_costs"] > 0
        assert result["exit_costs"] > 0
        assert result["total_costs"] == pytest.approx(
            result["entry_costs"] + result["exit_costs"]
        )
        # Premium is non-zero → cost-as-pct must be the ratio.
        assert result["cost_as_pct_of_premium"] == pytest.approx(
            result["total_costs"] / (2.50 * 100)
        )
        assert result["breakeven_decay_needed"] == pytest.approx(result["total_costs"] / 100)

    def test_round_trip_defaults_entry_spread_when_missing(self):
        # Line 354-355: entry_spread is None → use entry_premium * 0.10
        result_no_spread = tc.estimate_round_trip_cost(
            entry_premium=2.50,
            expected_exit_premium=1.00,
            entry_spread=None,
            exit_spread=None,
        )
        result_with_spread = tc.estimate_round_trip_cost(
            entry_premium=2.50,
            expected_exit_premium=1.00,
            entry_spread=0.25,  # = 2.50 * 0.10
            exit_spread=0.25,
        )
        assert result_no_spread["total_costs"] == pytest.approx(result_with_spread["total_costs"])

    def test_round_trip_defaults_exit_spread_to_entry(self):
        # Line 357-358: exit_spread is None → mirrors entry_spread
        explicit_entry = tc.estimate_round_trip_cost(
            entry_premium=3.0,
            expected_exit_premium=1.5,
            entry_spread=0.20,
            exit_spread=None,
        )
        both_explicit = tc.estimate_round_trip_cost(
            entry_premium=3.0,
            expected_exit_premium=1.5,
            entry_spread=0.20,
            exit_spread=0.20,
        )
        assert explicit_entry["total_costs"] == pytest.approx(both_explicit["total_costs"])

    def test_round_trip_zero_entry_premium_returns_zero_pct(self):
        # Line 376: entry_premium <= 0 short-circuits cost_as_pct_of_premium to 0.
        result = tc.estimate_round_trip_cost(
            entry_premium=0.0,
            expected_exit_premium=0.0,
            entry_spread=0.05,
            exit_spread=0.05,
        )
        assert result["cost_as_pct_of_premium"] == 0


# ---------------------------------------------------------------------------
# Bug pinned by audit F7 — line 143 unreachable
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "engine/transaction_costs.py:140-143 has an unreachable elif. "
        "`if spread_pct > 0.30: base_factor *= 1.5` always catches inputs "
        "where `> 0.50`, so the elif `*= 2.0` 'severe penalty' never fires. "
        "Pinned per audit F7. Fix is to swap the order (test > 0.50 first) "
        "in a separate behavior-change PR."
    ),
)
def test_severe_spread_penalty_doubles_base_factor():
    """Pins the intent that spread_pct > 0.50 triggers a 2.0× penalty.

    Currently fires the 1.5× branch instead because of the elif ordering bug
    documented in the chore commit on this branch.
    """
    # mid_price=1.0, spread=0.60 → spread_pct = 0.60 (> 0.50, also > 0.30)
    # Intended: base_factor = 0.15 * 2.0 = 0.30; spread_slippage = 0.60 * 0.30 = 0.180
    # Actual: base_factor = 0.15 * 1.5 = 0.225; spread_slippage = 0.60 * 0.225 = 0.135
    slippage = tc.calculate_slippage(
        mid_price=1.0,
        bid_ask_spread=0.60,
        trade_direction="sell",
    )
    intended = 0.60 * (tc.DEFAULT_SLIPPAGE_PCT * 2.0)
    assert slippage == pytest.approx(intended)
