"""Tests for payoff_engine: payoff diagrams, expected move, strike recommendations."""

from engine.payoff_engine import (
    StrikeRecommendation,
    compute_expected_move,
    compute_payoff,
    recommend_strikes,
)


class TestComputePayoff:
    """Test payoff diagram generation for each strategy."""

    def test_csp_above_strike(self):
        data = compute_payoff(spot=100, strike=95, premium=2.0, strategy="csp", n_points=10)
        # Grid has n_points linspace samples plus up to 3 anchors (spot, strike, breakeven)
        assert 10 <= len(data) <= 13
        # Above strike = max profit = premium * 100
        above = [d for d in data if d["price"] >= 95]
        for d in above:
            assert d["pnl"] == 200.0  # 2.0 * 100

    def test_csp_below_strike(self):
        data = compute_payoff(spot=100, strike=95, premium=2.0, strategy="csp", n_points=50)
        below = [d for d in data if d["price"] < 90]
        for d in below:
            assert d["pnl"] < 0  # Loss when deep below strike

    def test_cc_payoff(self):
        data = compute_payoff(spot=100, strike=105, premium=3.0, strategy="cc", n_points=20)
        assert 20 <= len(data) <= 23
        # Above strike = capped profit
        above = [d for d in data if d["price"] > 110]
        for d in above:
            # Profit is capped at (strike - spot + premium) * 100
            assert d["pnl"] == (105 - 100 + 3.0) * 100

    def test_short_strangle(self):
        data = compute_payoff(
            spot=100, strike=95, premium=4.0, strategy="short_strangle", n_points=20
        )
        assert 20 <= len(data) <= 22
        # Near the money = max profit
        near_money = [d for d in data if 95 <= d["price"] <= 100]
        for d in near_money:
            assert d["pnl"] > 0

    def test_long_call(self):
        data = compute_payoff(spot=100, strike=100, premium=5.0, strategy="long_call", n_points=20)
        below = [d for d in data if d["price"] <= 100]
        for d in below:
            assert d["pnl"] == -500  # Lost premium
        above = [d for d in data if d["price"] > 110]
        for d in above:
            assert d["pnl"] > 0  # Profit above breakeven

    def test_long_put(self):
        data = compute_payoff(spot=100, strike=100, premium=5.0, strategy="long_put", n_points=20)
        above = [d for d in data if d["price"] >= 100]
        for d in above:
            assert d["pnl"] == -500  # Lost premium

    def test_unknown_strategy(self):
        data = compute_payoff(spot=100, strike=100, premium=5.0, strategy="unknown", n_points=5)
        assert len(data) == 5
        for d in data:
            assert d["pnl"] == 0

    def test_pnl_pct_calculated(self):
        data = compute_payoff(spot=100, strike=95, premium=2.0, strategy="csp", n_points=5)
        for d in data:
            assert "pnlPct" in d

    def test_multiple_contracts(self):
        single = compute_payoff(
            spot=100, strike=95, premium=2.0, strategy="csp", contracts=1, n_points=5
        )
        double = compute_payoff(
            spot=100, strike=95, premium=2.0, strategy="csp", contracts=2, n_points=5
        )
        # 2 contracts = 2x PnL
        for s, d in zip(single, double, strict=True):
            assert abs(d["pnl"] - s["pnl"] * 2) < 0.01


class TestExpectedMove:
    """Test expected move band calculations."""

    def test_basic_bands(self):
        result = compute_expected_move(spot=100, iv=25, dte=45)
        assert result["spot"] == 100
        assert result["dte"] == 45
        assert len(result["bands"]) == 3
        labels = [b["label"] for b in result["bands"]]
        assert labels == ["1σ", "1.5σ", "2σ"]

    def test_band_ordering(self):
        result = compute_expected_move(spot=100, iv=25, dte=45)
        for b in result["bands"]:
            assert b["upper"] > 100
            assert b["lower"] < 100
            assert b["probability_within"] > 0
            assert b["move_pct"] > 0

    def test_wider_bands_at_2sigma(self):
        result = compute_expected_move(spot=100, iv=25, dte=45)
        b1 = result["bands"][0]  # 1σ
        b2 = result["bands"][2]  # 2σ
        assert b2["upper"] > b1["upper"]
        assert b2["lower"] < b1["lower"]

    def test_zero_iv(self):
        result = compute_expected_move(spot=100, iv=0, dte=45)
        assert result["bands"] == []

    def test_zero_dte(self):
        result = compute_expected_move(spot=100, iv=25, dte=0)
        assert result["bands"] == []

    def test_period_vol_calculated(self):
        result = compute_expected_move(spot=100, iv=25, dte=45)
        assert "period_vol" in result
        assert result["period_vol"] > 0

    def test_probability_1sigma_about_68(self):
        result = compute_expected_move(spot=100, iv=25, dte=45)
        assert 68 <= result["bands"][0]["probability_within"] <= 69


class TestRecommendStrikes:
    """Test strike recommendation engine."""

    def test_csp_recommendations(self):
        recs = recommend_strikes("AAPL", spot=200, iv=25, dte=45, strategy="csp")
        assert len(recs) == 5
        for r in recs:
            assert r["strike"] < 200  # OTM puts below spot
            assert r["premium"] > 0
            assert r["delta"] < 0  # Put delta is negative
            assert 0 < r["probabilityOtm"] < 100
            assert r["score"] >= 0

    def test_cc_recommendations(self):
        recs = recommend_strikes("AAPL", spot=200, iv=25, dte=45, strategy="cc")
        assert len(recs) == 5
        for r in recs:
            assert r["premium"] > 0
            assert r["delta"] > 0  # Call delta is positive
        # Most strikes should be above spot (low-delta ones)
        above_spot = [r for r in recs if r["strike"] >= 200]
        assert len(above_spot) >= 3

    def test_sorted_by_score(self):
        recs = recommend_strikes("AAPL", spot=200, iv=25, dte=45, strategy="csp")
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_strategy_empty(self):
        recs = recommend_strikes("AAPL", spot=200, iv=25, dte=45, strategy="butterfly")
        assert recs == []

    def test_zero_spot_empty(self):
        recs = recommend_strikes("AAPL", spot=0, iv=25, dte=45)
        assert recs == []

    def test_zero_iv_empty(self):
        recs = recommend_strikes("AAPL", spot=200, iv=0, dte=45)
        assert recs == []

    def test_strike_recommendation_dataclass(self):
        sr = StrikeRecommendation(
            ticker="AAPL",
            strategy="csp",
            strike=190,
            premium_estimate=2.5,
            delta=-0.25,
            probability_otm=0.75,
            annualized_return=12.0,
            breakeven=187.5,
            distance_from_spot_pct=5.0,
            max_loss=18750,
            expected_value=1.5,
            dte=45,
            score=72,
        )
        assert sr.ticker == "AAPL"
        assert sr.strike == 190

    def test_n_candidates_limit(self):
        recs = recommend_strikes("AAPL", spot=200, iv=25, dte=45, n_candidates=3)
        assert len(recs) == 3
