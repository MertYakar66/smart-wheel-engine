"""
Coverage backfill for engine/wheel_runner.py.

Targets the cleanly-unit-testable surface: the `TickerAnalysis`
dataclass + `.summary()`, `WheelRunner`'s lazy `connector` /
`strangle_engine` properties and provider selection, the pure
`_compute_wheel_score` composite (every weight branch), and
`screen_candidates`' empty-universe path.

The deep EV-path methods (`analyze_ticker`, `rank_candidates_by_ev`)
need the full Bloomberg dataset and are already exercised by the
real-data smoke tests (`test_audit_viii_real_data_smoke.py`,
`test_authority_hardening.py`); they are intentionally out of scope
here. No production code is touched.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.wheel_runner import TickerAnalysis, WheelRunner


# =====================================================================
# 1. TickerAnalysis.summary
# =====================================================================
class TestTickerAnalysisSummary:
    def test_summary_includes_header_and_core_fields(self):
        ta = TickerAnalysis(ticker="AAPL", spot_price=187.5, sector="Tech")
        s = ta.summary()
        assert "=== AAPL Wheel Analysis ===" in s
        assert "$187.50" in s
        assert "Tech" in s
        assert isinstance(s, str)

    def test_summary_with_events_shows_dates(self):
        from datetime import date

        ta = TickerAnalysis(
            ticker="AAPL",
            days_to_earnings=12,
            next_earnings_date=date(2024, 2, 1),
            next_div_date=date(2024, 2, 9),
            next_div_amount=0.24,
        )
        s = ta.summary()
        assert "2024-02-01" in s
        assert "(12d)" in s
        assert "2024-02-09" in s
        assert "N/A" not in s.split("Strangle Timing")[0]

    def test_summary_without_events_shows_na(self):
        # days_to_earnings=None and next_div_date=None -> both "N/A".
        ta = TickerAnalysis(ticker="AAPL")
        s = ta.summary()
        assert "Next Earnings: N/A" in s
        assert "Next Ex-Div: N/A" in s

    def test_summary_zero_days_to_earnings_is_na(self):
        # 0 is falsy -> the summary takes the N/A branch.
        ta = TickerAnalysis(ticker="AAPL", days_to_earnings=0)
        assert "Next Earnings: N/A" in ta.summary()


# =====================================================================
# 2. WheelRunner construction + lazy connector / provider selection
# =====================================================================
class TestWheelRunnerConnector:
    def test_default_provider_is_market_data_connector(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SWE_DATA_PROVIDER", raising=False)
        runner = WheelRunner(data_dir=tmp_path)
        assert type(runner.connector).__name__ == "MarketDataConnector"

    def test_explicit_bloomberg_provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SWE_DATA_PROVIDER", "bloomberg")
        assert type(WheelRunner(data_dir=tmp_path).connector).__name__ == "MarketDataConnector"

    def test_theta_provider_selected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SWE_DATA_PROVIDER", "theta")
        assert type(WheelRunner(data_dir=tmp_path).connector).__name__ == "ThetaConnector"

    def test_unknown_provider_falls_back_to_bloomberg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SWE_DATA_PROVIDER", "not-a-provider")
        assert type(WheelRunner(data_dir=tmp_path).connector).__name__ == "MarketDataConnector"

    def test_connector_is_cached(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SWE_DATA_PROVIDER", raising=False)
        runner = WheelRunner(data_dir=tmp_path)
        assert runner.connector is runner.connector

    def test_strangle_engine_lazy_loads_and_caches(self, tmp_path):
        runner = WheelRunner(data_dir=tmp_path)
        eng = runner.strangle_engine
        assert eng is not None
        assert runner.strangle_engine is eng


# =====================================================================
# 3. _compute_wheel_score — every weight branch
# =====================================================================
# Weights: iv 0.30, fundamentals 0.20, event 0.15, timing 0.20,
# liquidity 0.15. Expected totals below are hand-computed.
class TestComputeWheelScore:
    @pytest.fixture
    def runner(self, tmp_path):
        return WheelRunner(data_dir=tmp_path)

    def test_all_defaults(self, runner):
        # iv 30, fund 50, event 80, timing 0, liquidity 50 -> 38.5.
        score = runner._compute_wheel_score(TickerAnalysis(ticker="X"))
        assert score == pytest.approx(38.5)

    def test_strong_candidate(self, runner):
        # iv 95, fund 95, event 80, timing 70, liquidity 90 -> 87.0.
        ta = TickerAnalysis(
            ticker="X",
            iv_rank=0.8,
            vol_risk_premium=8.0,
            pe_ratio=20.0,
            beta=1.0,
            dividend_yield=2.0,
            credit_rating="A+",
            days_to_earnings=45,
            strangle_score=70.0,
            market_cap=150e9,
        )
        assert runner._compute_wheel_score(ta) == pytest.approx(87.0)

    def test_weak_candidate(self, runner):
        # iv 10, fund 15, event 0, timing 0, liquidity 50 -> 13.5.
        ta = TickerAnalysis(
            ticker="X",
            iv_rank=0.0,
            vol_risk_premium=-10.0,
            pe_ratio=60.0,
            beta=2.5,
            credit_rating="CCC",
            days_to_earnings=3,
            days_to_ex_div=1,
            market_cap=0.0,
        )
        assert runner._compute_wheel_score(ta) == pytest.approx(13.5)

    def test_earnings_under_14_days_lowers_event_score(self, runner):
        # days_to_earnings 10 -> event 40 -> 9 + 10 + 6 + 0 + 7.5 = 32.5.
        ta = TickerAnalysis(ticker="X", days_to_earnings=10)
        assert runner._compute_wheel_score(ta) == pytest.approx(32.5)

    def test_earnings_under_30_days(self, runner):
        # days_to_earnings 20 -> event 70 -> 9 + 10 + 10.5 + 0 + 7.5 = 37.0.
        ta = TickerAnalysis(ticker="X", days_to_earnings=20)
        assert runner._compute_wheel_score(ta) == pytest.approx(37.0)

    def test_market_cap_tiers(self, runner):
        # >20e9 -> liquidity 75 -> 9 + 10 + 12 + 0 + 11.25 = 42.25.
        assert runner._compute_wheel_score(
            TickerAnalysis(ticker="X", market_cap=50e9)
        ) == pytest.approx(42.25)
        # >5e9 -> liquidity 60 -> 9 + 10 + 12 + 0 + 9.0 = 40.0.
        assert runner._compute_wheel_score(
            TickerAnalysis(ticker="X", market_cap=8e9)
        ) == pytest.approx(40.0)
        # >0 -> liquidity 35 -> 9 + 10 + 12 + 0 + 5.25 = 36.25.
        assert runner._compute_wheel_score(
            TickerAnalysis(ticker="X", market_cap=1e9)
        ) == pytest.approx(36.25)

    def test_negative_pe_penalised(self, runner):
        # pe < 0 -> fund 35 -> 9 + 7 + 12 + 0 + 7.5 = 35.5.
        ta = TickerAnalysis(ticker="X", pe_ratio=-5.0)
        assert runner._compute_wheel_score(ta) == pytest.approx(35.5)

    def test_score_is_bounded_0_100(self, runner):
        strong = runner._compute_wheel_score(
            TickerAnalysis(
                ticker="X",
                iv_rank=1.0,
                vol_risk_premium=20.0,
                pe_ratio=15.0,
                beta=1.0,
                dividend_yield=3.0,
                credit_rating="AAA",
                strangle_score=100.0,
                market_cap=500e9,
                days_to_earnings=90,
            )
        )
        weak = runner._compute_wheel_score(
            TickerAnalysis(
                ticker="X",
                iv_rank=0.0,
                vol_risk_premium=-30.0,
                pe_ratio=200.0,
                beta=4.0,
                days_to_earnings=1,
                days_to_ex_div=0,
            )
        )
        assert 0.0 <= weak <= strong <= 100.0


# =====================================================================
# 4. screen_candidates — empty-universe short-circuit
# =====================================================================
class TestScreenCandidates:
    def test_empty_universe_returns_empty_frame(self, monkeypatch, tmp_path):
        # No fundamentals CSV -> screen_universe returns empty ->
        # screen_candidates short-circuits to an empty DataFrame.
        monkeypatch.delenv("SWE_DATA_PROVIDER", raising=False)
        runner = WheelRunner(data_dir=tmp_path)
        out = runner.screen_candidates()
        assert isinstance(out, pd.DataFrame)
        assert out.empty
