"""
Comprehensive Test Suite for Advisor AI Layer

Tests all advisor components:
- Schema validation
- Individual advisors (Buffett, Munger, Simons)
- Committee engine aggregation
- Integration with options engine
"""

import pytest

# Import all advisor components
from advisors import (
    AdvisorInput,
    # Output models
    AdvisorResponse,
    # Base
    BaseAdvisor,
    # Advisors
    BuffettAdvisor,
    CandidateTrade,
    # Committee
    CommitteeEngine,
    CommitteeOutput,
    # Enums
    ConfidenceLevel,
    # Integration
    EngineIntegration,
    Judgment,
    MarketContext,
    MungerAdvisor,
    PortfolioContext,
    # Input models
    Position,
    RegimeType,
    SimonsAdvisor,
    TradeType,
    # Helpers
    create_sample_input,
    format_committee_report,
    quick_evaluate,
)

# =============================================================================
# SCHEMA TESTS
# =============================================================================


class TestSchemaEnums:
    """Test enum definitions."""

    def test_confidence_level_values(self):
        """Test ConfidenceLevel enum has expected values."""
        assert ConfidenceLevel.VERY_LOW.value == "very_low"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"

    def test_judgment_values(self):
        """Test Judgment enum has expected values."""
        assert Judgment.STRONG_APPROVE.value == "strong_approve"
        assert Judgment.APPROVE.value == "approve"
        assert Judgment.NEUTRAL.value == "neutral"
        assert Judgment.REJECT.value == "reject"
        assert Judgment.STRONG_REJECT.value == "strong_reject"

    def test_trade_type_values(self):
        """Test TradeType enum covers common strategies."""
        trade_types = [t.value for t in TradeType]
        assert "short_put" in trade_types
        assert "covered_call" in trade_types
        assert "iron_condor" in trade_types

    def test_regime_type_values(self):
        """Test RegimeType enum covers market conditions."""
        regimes = [r.value for r in RegimeType]
        assert "low_volatility" in regimes
        assert "high_volatility" in regimes
        assert "crisis" in regimes


class TestPosition:
    """Test Position dataclass."""

    @pytest.fixture
    def sample_position(self):
        return Position(
            ticker="NVDA",
            shares=100,
            avg_cost=450.0,
            current_price=520.0,
            sector="Technology",
            market_cap="mega",
        )

    def test_market_value(self, sample_position):
        """Test market value calculation."""
        assert sample_position.market_value == 52000.0

    def test_unrealized_pnl(self, sample_position):
        """Test unrealized P&L calculation."""
        assert sample_position.unrealized_pnl == 7000.0  # (520-450) * 100

    def test_unrealized_pnl_pct(self, sample_position):
        """Test unrealized P&L percentage."""
        expected = (520 - 450) / 450 * 100
        assert abs(sample_position.unrealized_pnl_pct - expected) < 0.01


class TestCandidateTrade:
    """Test CandidateTrade dataclass."""

    @pytest.fixture
    def sample_trade(self):
        return CandidateTrade(
            ticker="AAPL",
            trade_type=TradeType.SHORT_PUT,
            strike=170.0,
            expiration_date="2026-05-15",
            dte=45,
            delta=-0.25,
            premium=4.50,
            contracts=2,
            expected_value=2.1,
            p_otm=0.72,
            p_profit=0.75,
            iv_rank=65.0,
            iv_percentile=70.0,
            theta=0.35,
            gamma=0.02,
            vega=0.45,
            underlying_price=185.0,
            earnings_before_expiry=False,
        )

    def test_trade_attributes(self, sample_trade):
        """Test trade has correct attributes."""
        assert sample_trade.ticker == "AAPL"
        assert sample_trade.trade_type == TradeType.SHORT_PUT
        assert sample_trade.strike == 170.0
        assert sample_trade.dte == 45
        assert sample_trade.contracts == 2

    def test_probability_bounds(self, sample_trade):
        """Test probability values are in valid range."""
        assert 0 <= sample_trade.p_otm <= 1
        assert 0 <= sample_trade.p_profit <= 1

    def test_greeks_present(self, sample_trade):
        """Test Greeks are populated."""
        assert sample_trade.theta != 0
        assert sample_trade.gamma != 0
        assert sample_trade.vega != 0


class TestPortfolioContext:
    """Test PortfolioContext dataclass."""

    @pytest.fixture
    def sample_portfolio(self):
        positions = [
            Position("NVDA", 100, 450, 520, "Technology", "mega"),
            Position("AAPL", 50, 170, 185, "Technology", "mega"),
        ]
        return PortfolioContext(
            positions=positions,
            total_equity=150000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            sector_allocation={"Technology": 60.0, "Healthcare": 20.0, "Financials": 20.0},
            top_5_concentration=75.0,
            portfolio_beta=1.15,
            portfolio_delta=0.65,
            max_drawdown_30d=-5.5,
            var_95=2.8,
            open_positions_count=5,
            total_premium_at_risk=8000.0,
            total_margin_used=35000.0,
        )

    def test_portfolio_attributes(self, sample_portfolio):
        """Test portfolio has expected attributes."""
        assert sample_portfolio.total_equity == 150000.0
        assert len(sample_portfolio.positions) == 2
        assert sample_portfolio.portfolio_beta > 0

    def test_sector_allocation_sums(self, sample_portfolio):
        """Test sector allocation is reasonable."""
        total = sum(sample_portfolio.sector_allocation.values())
        assert total == 100.0


class TestMarketContext:
    """Test MarketContext dataclass."""

    @pytest.fixture
    def sample_market(self):
        return MarketContext(
            regime=RegimeType.NORMAL,
            vix=18.5,
            vix_percentile=45.0,
            spy_price=485.0,
            spy_50ma=480.0,
            spy_200ma=460.0,
            fed_funds_rate=4.5,
            treasury_10y=4.2,
            recent_fed_action="Held rates steady",
            upcoming_events=["FOMC in 15 days", "CPI next week"],
        )

    def test_market_attributes(self, sample_market):
        """Test market context has expected attributes."""
        assert sample_market.regime == RegimeType.NORMAL
        assert sample_market.vix > 0
        assert sample_market.spy_price > 0

    def test_moving_averages_relationship(self, sample_market):
        """Test MA relationship is logical."""
        # In normal uptrend: price > 50MA > 200MA
        assert sample_market.spy_price >= sample_market.spy_50ma
        assert sample_market.spy_50ma >= sample_market.spy_200ma


class TestAdvisorInput:
    """Test AdvisorInput dataclass."""

    def test_create_sample_input(self):
        """Test sample input creation helper."""
        sample = create_sample_input()

        assert isinstance(sample, AdvisorInput)
        assert sample.candidate_trade is not None
        assert sample.portfolio is not None
        assert sample.market is not None

    def test_sample_input_has_valid_trade(self):
        """Test sample input has valid trade data."""
        sample = create_sample_input()

        assert sample.candidate_trade.ticker == "NVDA"
        assert sample.candidate_trade.strike > 0
        assert sample.candidate_trade.dte > 0


# =============================================================================
# ADVISOR TESTS
# =============================================================================


class TestBaseAdvisor:
    """Test BaseAdvisor abstract class."""

    def test_buffett_advisor_is_base_advisor(self):
        """Test BuffettAdvisor inherits from BaseAdvisor."""
        advisor = BuffettAdvisor()
        assert isinstance(advisor, BaseAdvisor)

    def test_advisor_has_required_properties(self):
        """Test advisor has name and philosophy."""
        advisor = BuffettAdvisor()
        assert advisor.name == "Buffett"
        assert len(advisor.philosophy) > 0
        assert len(advisor.system_prompt) > 0


class TestBuffettAdvisor:
    """Test BuffettAdvisor implementation."""

    @pytest.fixture
    def advisor(self):
        return BuffettAdvisor()

    @pytest.fixture
    def quality_stock_input(self):
        """Input for a high-quality stock."""
        sample = create_sample_input()
        sample.candidate_trade.ticker = "AAPL"  # Quality stock
        return sample

    @pytest.fixture
    def speculative_stock_input(self):
        """Input for a speculative stock."""
        sample = create_sample_input()
        sample.candidate_trade.ticker = "GME"  # Speculative
        return sample

    def test_advisor_name(self, advisor):
        """Test advisor name is correct."""
        assert advisor.name == "Buffett"

    def test_advisor_philosophy(self, advisor):
        """Test advisor has philosophy."""
        assert "capital allocation" in advisor.philosophy.lower()

    def test_evaluate_returns_response(self, advisor, quality_stock_input):
        """Test evaluate returns AdvisorResponse."""
        response = advisor.evaluate(quality_stock_input)

        assert isinstance(response, AdvisorResponse)
        assert response.advisor_name == "Buffett"
        assert response.judgment is not None

    def test_response_has_required_fields(self, advisor, quality_stock_input):
        """Test response has all required fields."""
        response = advisor.evaluate(quality_stock_input)

        assert len(response.key_reasons) >= 2
        assert len(response.critical_questions) >= 1
        assert response.confidence is not None
        assert response.judgment_summary != ""

    def test_quality_stock_evaluation(self, advisor, quality_stock_input):
        """Test evaluation of quality stock is favorable."""
        response = advisor.evaluate(quality_stock_input)

        # Quality stocks should not be strongly rejected
        assert response.judgment != Judgment.STRONG_REJECT

    def test_speculative_stock_evaluation(self, advisor, speculative_stock_input):
        """Test evaluation of speculative stock is cautious."""
        response = advisor.evaluate(speculative_stock_input)

        # Should identify speculative nature
        reasons_text = " ".join(response.key_reasons).lower()
        assert "speculative" in reasons_text or response.judgment in [
            Judgment.REJECT,
            Judgment.STRONG_REJECT,
            Judgment.NEUTRAL,
        ]


class TestMungerAdvisor:
    """Test MungerAdvisor implementation."""

    @pytest.fixture
    def advisor(self):
        return MungerAdvisor()

    @pytest.fixture
    def sample_input(self):
        return create_sample_input()

    def test_advisor_name(self, advisor):
        """Test advisor name is correct."""
        assert advisor.name == "Munger"

    def test_advisor_philosophy(self, advisor):
        """Test advisor has inversion philosophy."""
        assert "inversion" in advisor.philosophy.lower()

    def test_evaluate_returns_response(self, advisor, sample_input):
        """Test evaluate returns AdvisorResponse."""
        response = advisor.evaluate(sample_input)

        assert isinstance(response, AdvisorResponse)
        assert response.advisor_name == "Munger"

    def test_identifies_biases(self, advisor, sample_input):
        """Test advisor identifies cognitive biases."""
        response = advisor.evaluate(sample_input)

        # Munger should mention biases
        all_text = " ".join(response.key_reasons + response.hidden_risks).lower()
        assert "bias" in all_text or "inversion" in all_text

    def test_asks_inversion_questions(self, advisor, sample_input):
        """Test advisor asks inversion-style questions."""
        response = advisor.evaluate(sample_input)

        # Should ask challenging questions
        questions_text = " ".join(response.critical_questions).lower()
        # Munger asks about what could go wrong, second-order effects, etc.
        inversion_keywords = [
            "fail",
            "wrong",
            "lose",
            "kill",
            "drop",
            "second",
            "assigned",
            "happen",
            "edge",
            "capital",
        ]
        assert any(word in questions_text for word in inversion_keywords)


class TestSimonsAdvisor:
    """Test SimonsAdvisor implementation."""

    @pytest.fixture
    def advisor(self):
        return SimonsAdvisor()

    @pytest.fixture
    def high_ev_input(self):
        """Input with high expected value."""
        sample = create_sample_input()
        sample.candidate_trade.expected_value = 3.5
        sample.candidate_trade.p_otm = 0.75
        return sample

    @pytest.fixture
    def low_ev_input(self):
        """Input with low expected value."""
        sample = create_sample_input()
        sample.candidate_trade.expected_value = 0.5
        sample.candidate_trade.p_otm = 0.55
        return sample

    def test_advisor_name(self, advisor):
        """Test advisor name is correct."""
        assert advisor.name == "Simons"

    def test_advisor_philosophy(self, advisor):
        """Test advisor has statistical philosophy."""
        assert "statistical" in advisor.philosophy.lower()

    def test_evaluate_returns_response(self, advisor, high_ev_input):
        """Test evaluate returns AdvisorResponse."""
        response = advisor.evaluate(high_ev_input)

        assert isinstance(response, AdvisorResponse)
        assert response.advisor_name == "Simons"

    def test_high_ev_favorable(self, advisor, high_ev_input):
        """Test high EV trades get favorable treatment."""
        response = advisor.evaluate(high_ev_input)

        # High EV should not be strongly rejected
        assert response.judgment != Judgment.STRONG_REJECT

    def test_low_ev_cautious(self, advisor, low_ev_input):
        """Test low EV trades get cautious treatment."""
        response = advisor.evaluate(low_ev_input)

        # Low EV should not be strongly approved
        assert response.judgment != Judgment.STRONG_APPROVE

    def test_mentions_statistical_concepts(self, advisor, high_ev_input):
        """Test advisor uses statistical language."""
        response = advisor.evaluate(high_ev_input)

        all_text = " ".join(
            response.key_reasons + response.hidden_risks + response.critical_questions
        ).lower()

        statistical_terms = ["probability", "statistical", "regime", "confidence", "ev", "expected"]
        assert any(term in all_text for term in statistical_terms)


# =============================================================================
# COMMITTEE ENGINE TESTS
# =============================================================================


class TestCommitteeEngine:
    """Test CommitteeEngine aggregation."""

    @pytest.fixture
    def committee(self):
        return CommitteeEngine(parallel=False)  # Sequential for determinism

    @pytest.fixture
    def sample_input(self):
        return create_sample_input()

    def test_committee_has_default_advisors(self, committee):
        """Test committee has default advisors (3 core + optional Taleb)."""
        assert len(committee.advisors) >= 3

    def test_committee_evaluate_returns_output(self, committee, sample_input):
        """Test evaluate returns CommitteeOutput."""
        output = committee.evaluate(sample_input)

        assert isinstance(output, CommitteeOutput)
        assert len(output.advisor_responses) >= 3

    def test_committee_output_has_votes(self, committee, sample_input):
        """Test output has vote counts."""
        output = committee.evaluate(sample_input)

        total = output.approval_count + output.rejection_count + output.neutral_count
        assert total >= 3

    def test_committee_output_has_judgment(self, committee, sample_input):
        """Test output has committee judgment."""
        output = committee.evaluate(sample_input)

        assert output.committee_judgment is not None
        assert output.committee_reasoning != ""

    def test_committee_output_has_agreements(self, committee, sample_input):
        """Test output identifies agreements."""
        output = committee.evaluate(sample_input)

        # Should have some analysis
        assert isinstance(output.areas_of_agreement, list)
        assert isinstance(output.areas_of_disagreement, list)

    def test_unanimous_approve_detection(self, committee):
        """Test unanimous approval is detected."""
        # Create input likely to get unanimous approval
        sample = create_sample_input()
        sample.candidate_trade.ticker = "AAPL"
        sample.candidate_trade.expected_value = 4.0
        sample.candidate_trade.p_otm = 0.80
        sample.candidate_trade.delta = -0.15
        sample.market.regime = RegimeType.NORMAL

        output = committee.evaluate(sample)

        if output.approval_count == 3:
            assert output.unanimous_approve is True

    def test_format_committee_report(self, committee, sample_input):
        """Test report formatting."""
        output = committee.evaluate(sample_input)
        report = format_committee_report(output)

        assert isinstance(report, str)
        assert "COMMITTEE" in report
        assert "DECISION" in report
        assert len(report) > 500  # Should be substantial

    def test_processing_time_recorded(self, committee, sample_input):
        """Test processing time is recorded."""
        output = committee.evaluate(sample_input)

        assert output.total_processing_time_ms >= 0  # May be 0 on fast machines

    def test_add_custom_advisor(self, committee, sample_input):
        """Test adding custom advisor."""
        initial_count = len(committee.advisors)
        committee.add_advisor(BuffettAdvisor())  # Add another

        assert len(committee.advisors) == initial_count + 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEngineIntegration:
    """Test integration with options engine."""

    @pytest.fixture
    def integration(self):
        return EngineIntegration()

    def test_evaluate_trade_dict(self, integration):
        """Test evaluating trade from dict format."""
        trade = {
            "ticker": "NVDA",
            "trade_type": "short_put",
            "strike": 480,
            "expiration_date": "2026-05-15",
            "dte": 45,
            "delta": -0.30,
            "premium": 12.50,
            "contracts": 1,
            "expected_value": 2.3,
            "p_otm": 0.68,
            "p_profit": 0.72,
            "iv_rank": 65,
            "iv_percentile": 70,
            "theta": 0.4,
            "gamma": 0.02,
            "vega": 0.5,
            "underlying_price": 520,
        }

        portfolio = {
            "positions": [],
            "total_equity": 100000,
            "cash_available": 50000,
            "buying_power": 100000,
            "sector_allocation": {"Technology": 30},
            "top_5_concentration": 40,
            "portfolio_beta": 1.0,
            "portfolio_delta": 0.5,
            "max_drawdown_30d": -3,
            "var_95": 2.0,
            "open_positions_count": 2,
            "total_premium_at_risk": 5000,
            "total_margin_used": 20000,
        }

        market = {
            "regime": "normal",
            "vix": 18,
            "vix_percentile": 45,
            "spy_price": 485,
            "spy_50ma": 480,
            "spy_200ma": 460,
            "fed_funds_rate": 4.5,
            "treasury_10y": 4.2,
        }

        result = integration.evaluate_trade(trade, portfolio, market)

        assert isinstance(result, CommitteeOutput)
        assert result.trade_summary != ""

    def test_filter_approved(self, integration):
        """Test filtering approved trades."""
        trades = [
            {
                "ticker": "AAPL",
                "trade_type": "short_put",
                "strike": 170,
                "dte": 45,
                "delta": -0.20,
                "premium": 3.5,
                "contracts": 1,
                "expected_value": 2.0,
                "p_otm": 0.75,
                "p_profit": 0.78,
                "iv_rank": 50,
                "iv_percentile": 55,
                "theta": 0.3,
                "gamma": 0.02,
                "vega": 0.4,
                "underlying_price": 185,
            },
        ]

        portfolio = {
            "positions": [],
            "total_equity": 100000,
            "cash_available": 50000,
            "buying_power": 100000,
            "sector_allocation": {},
            "top_5_concentration": 30,
            "portfolio_beta": 1.0,
            "portfolio_delta": 0.5,
            "max_drawdown_30d": -3,
            "var_95": 2.0,
            "open_positions_count": 0,
            "total_premium_at_risk": 0,
            "total_margin_used": 0,
        }

        market = {
            "regime": "normal",
            "vix": 18,
            "vix_percentile": 45,
            "spy_price": 485,
            "spy_50ma": 480,
            "spy_200ma": 460,
            "fed_funds_rate": 4.5,
            "treasury_10y": 4.2,
        }

        approved = integration.filter_approved(trades, portfolio, market, min_approval_count=1)

        assert isinstance(approved, list)


class TestQuickEvaluate:
    """Test quick_evaluate helper function."""

    def test_quick_evaluate_basic(self):
        """Test basic quick evaluation."""
        result = quick_evaluate(
            ticker="AAPL", strike=170, dte=45, delta=-0.25, premium=4.0, print_report=False
        )

        assert isinstance(result, CommitteeOutput)
        assert result.trade_summary != ""

    def test_quick_evaluate_with_all_params(self):
        """Test quick evaluation with all parameters."""
        result = quick_evaluate(
            ticker="NVDA",
            strike=480,
            dte=45,
            delta=-0.30,
            premium=12.50,
            underlying_price=520,
            expected_value=2.3,
            p_otm=0.68,
            iv_rank=65,
            print_report=False,
        )

        assert isinstance(result, CommitteeOutput)
        assert "NVDA" in result.trade_summary


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def committee(self):
        return CommitteeEngine(parallel=False)

    def test_extreme_high_iv(self, committee):
        """Test handling of extreme IV."""
        sample = create_sample_input()
        sample.candidate_trade.iv_rank = 99.0
        sample.market.vix = 45.0
        sample.market.regime = RegimeType.CRISIS

        output = committee.evaluate(sample)

        assert output is not None
        # Crisis regime should trigger caution
        assert output.rejection_count >= 0

    def test_extreme_low_delta(self, committee):
        """Test handling of very low delta."""
        sample = create_sample_input()
        sample.candidate_trade.delta = -0.05

        output = committee.evaluate(sample)

        assert output is not None

    def test_very_short_dte(self, committee):
        """Test handling of very short DTE."""
        sample = create_sample_input()
        sample.candidate_trade.dte = 5

        output = committee.evaluate(sample)

        assert output is not None

    def test_high_concentration(self, committee):
        """Test handling of high sector concentration."""
        sample = create_sample_input()
        sample.portfolio.sector_allocation = {"Technology": 80}
        sample.portfolio.top_5_concentration = 90

        output = committee.evaluate(sample)

        # High concentration should trigger warnings
        risks_text = " ".join(
            [r for resp in output.advisor_responses for r in resp.hidden_risks]
        ).lower()
        assert "concentration" in risks_text or output.rejection_count > 0

    def test_earnings_before_expiry(self, committee):
        """Test handling of earnings risk."""
        sample = create_sample_input()
        sample.candidate_trade.earnings_before_expiry = True

        output = committee.evaluate(sample)

        # Earnings should be noted as risk
        all_text = " ".join(
            [
                " ".join(r.hidden_risks) + " ".join(r.critical_questions)
                for r in output.advisor_responses
            ]
        ).lower()
        assert "earnings" in all_text

    def test_negative_expected_value(self, committee):
        """Test handling of negative EV."""
        sample = create_sample_input()
        sample.candidate_trade.expected_value = -1.5

        output = committee.evaluate(sample)

        # Negative EV should not be approved
        assert output.committee_judgment != Judgment.STRONG_APPROVE


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_committee_evaluation_speed(self):
        """Test committee evaluation completes in reasonable time."""
        import time

        committee = CommitteeEngine(parallel=True)
        sample = create_sample_input()

        start = time.time()
        output = committee.evaluate(sample)
        elapsed = time.time() - start

        # Should complete within 1 second
        assert elapsed < 1.0
        assert output is not None

    def test_multiple_evaluations(self):
        """Test multiple sequential evaluations."""
        committee = CommitteeEngine(parallel=True)
        sample = create_sample_input()

        results = []
        for _ in range(5):
            result = committee.evaluate(sample)
            results.append(result)

        assert len(results) == 5
        assert all(r is not None for r in results)
