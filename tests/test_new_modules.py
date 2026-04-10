"""Tests for wheel_runner, data_connector, and taleb advisor to boost coverage."""

import pandas as pd

from advisors.committee import CommitteeEngine
from advisors.schema import (
    ClosedTradeRecord,
    PortfolioReviewInput,
    PostMortemInput,
)
from advisors.taleb import TalebAdvisor


def _sample_input():
    """Create minimal advisor input for testing."""
    from advisors.schema import create_sample_input

    return create_sample_input()


class TestTalebAdvisor:
    """Test TalebAdvisor evaluation."""

    def test_taleb_evaluates(self):
        advisor = TalebAdvisor()
        result = advisor.evaluate(_sample_input())
        assert result.advisor_name == "Taleb"
        assert result.judgment is not None
        assert len(result.key_reasons) >= 2
        assert len(result.critical_questions) >= 1
        assert len(result.hidden_risks) >= 1

    def test_taleb_never_strong_approves(self):
        advisor = TalebAdvisor()
        result = advisor.evaluate(_sample_input())
        from advisors.schema import Judgment

        assert result.judgment != Judgment.STRONG_APPROVE

    def test_taleb_philosophy(self):
        advisor = TalebAdvisor()
        assert "tail" in advisor.philosophy.lower() or "fragil" in advisor.philosophy.lower()

    def test_taleb_has_system_prompt(self):
        advisor = TalebAdvisor()
        assert len(advisor.system_prompt) > 50


class TestPortfolioReview:
    """Test portfolio review mode."""

    def test_review_returns_output(self):
        committee = CommitteeEngine(parallel=False)
        sample = _sample_input()
        review_input = PortfolioReviewInput(
            portfolio=sample.portfolio,
            market=sample.market,
            strategy_description="Wheel strategy on S&P 500 names",
        )
        output = committee.review_portfolio(review_input)
        assert output is not None
        assert len(output.advisor_reviews) >= 3
        assert output.consensus_assessment in ("strong", "healthy", "concerning", "fragile")

    def test_review_has_strengths_and_weaknesses(self):
        committee = CommitteeEngine(parallel=False)
        sample = _sample_input()
        review_input = PortfolioReviewInput(
            portfolio=sample.portfolio,
            market=sample.market,
        )
        output = committee.review_portfolio(review_input)
        # At least some feedback
        total_items = (
            len(output.consensus_strengths)
            + len(output.consensus_weaknesses)
            + len(output.critical_blind_spots)
        )
        assert total_items > 0


class TestPostMortem:
    """Test post-mortem mode."""

    def test_post_mortem_returns_output(self):
        committee = CommitteeEngine(parallel=False)
        sample = _sample_input()
        trades = [
            ClosedTradeRecord(
                ticker="AAPL",
                trade_type="short_put",
                strike=170,
                entry_date="2025-01-15",
                exit_date="2025-02-15",
                entry_premium=3.50,
                exit_premium=0.10,
                pnl=340,
                pnl_pct=0.97,
                outcome="win",
                exit_reason="expired",
                hold_days=31,
            ),
            ClosedTradeRecord(
                ticker="MSFT",
                trade_type="short_put",
                strike=400,
                entry_date="2025-01-20",
                exit_date="2025-02-10",
                entry_premium=5.00,
                exit_premium=8.50,
                pnl=-350,
                pnl_pct=-0.70,
                outcome="loss",
                exit_reason="stop_loss",
                hold_days=21,
                max_drawdown_during=-15.0,
            ),
            ClosedTradeRecord(
                ticker="JPM",
                trade_type="short_put",
                strike=180,
                entry_date="2025-02-01",
                exit_date="2025-03-01",
                entry_premium=2.80,
                exit_premium=0.05,
                pnl=275,
                pnl_pct=0.98,
                outcome="win",
                exit_reason="expired",
                hold_days=28,
            ),
        ]
        pm_input = PostMortemInput(
            closed_trades=trades,
            portfolio=sample.portfolio,
            market=sample.market,
            period="2025-Q1",
        )
        output = committee.post_mortem(pm_input)
        assert output is not None
        assert output.total_trades == 3
        assert output.win_rate > 0.5
        assert output.total_pnl > 0
        assert len(output.advisor_reviews) >= 3

    def test_post_mortem_has_grade(self):
        committee = CommitteeEngine(parallel=False)
        sample = _sample_input()
        trades = [
            ClosedTradeRecord(
                ticker="TEST",
                trade_type="short_put",
                strike=100,
                entry_date="2025-01-01",
                exit_date="2025-02-01",
                entry_premium=2.0,
                exit_premium=0.0,
                pnl=200,
                pnl_pct=1.0,
                outcome="win",
                exit_reason="expired",
                hold_days=31,
            ),
        ]
        output = committee.post_mortem(
            PostMortemInput(
                closed_trades=trades,
                portfolio=sample.portfolio,
                market=sample.market,
            )
        )
        assert output.consensus_grade in ("A", "B", "C", "D", "F", "N/A")


class TestWheelRunnerImport:
    """Test wheel_runner imports and basic construction."""

    def test_import(self):
        from engine.wheel_runner import TickerAnalysis, WheelRunner

        assert WheelRunner is not None
        assert TickerAnalysis is not None

    def test_construct(self):
        from engine.wheel_runner import WheelRunner

        runner = WheelRunner(data_dir="data/bloomberg")
        assert runner.data_dir.name == "bloomberg"

    def test_ticker_analysis_summary(self):
        from engine.wheel_runner import TickerAnalysis

        ta = TickerAnalysis(
            ticker="AAPL",
            spot_price=175.0,
            market_cap=3e12,
            pe_ratio=28.0,
            beta=1.1,
            sector="Technology",
            iv_30d=25.0,
            rv_30d=22.0,
            iv_rank=65.0,
            wheel_score=72.0,
            wheel_recommendation="moderate",
        )
        s = ta.summary()
        assert "AAPL" in s
        assert "Technology" in s


class TestDataConnectorImport:
    """Test data_connector basic functionality."""

    def test_import(self):
        from engine.data_connector import MarketDataConnector, normalize_ticker

        assert MarketDataConnector is not None
        assert normalize_ticker is not None

    def test_normalize_ticker(self):
        from engine.data_connector import normalize_ticker

        assert normalize_ticker("AAPL UW Equity") == "AAPL"
        assert normalize_ticker("A UN Equity") == "A"
        assert normalize_ticker("MSFT UW") == "MSFT"
        assert normalize_ticker("JPM UN") == "JPM"

    def test_construct_with_missing_dir(self):
        from engine.data_connector import MarketDataConnector

        conn = MarketDataConnector("/tmp/nonexistent_bloomberg_dir")
        # Should not crash, just return empty
        ohlcv = conn.get_ohlcv("AAPL")
        assert isinstance(ohlcv, pd.DataFrame)


class TestDataIntegrationImport:
    """Test data_integration functions with missing data."""

    def test_import(self):
        from engine.data_integration import (
            load_earnings_from_bloomberg,
        )

        assert load_earnings_from_bloomberg is not None

    def test_missing_data_fallbacks(self):
        from engine.data_integration import get_current_risk_free_rate

        rate = get_current_risk_free_rate(data_dir="/tmp/nonexistent")
        assert rate == 0.05
