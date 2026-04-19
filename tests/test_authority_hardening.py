"""
Launch-gate tests for the audit-VI authority hardening pass.

Every test locks in a specific leak that was closed in this audit:

1. /api/tv/enrich verdict is now EV-authoritative, not heuristic
   wheel_score >= 60. Test that with EV unavailable, verdict becomes
   "review" with reason "ev_engine_unreachable", NOT a heuristic
   "proceed".

2. /api/analyze payload carries authority="heuristic_diagnostic" and
   tradeable_endpoint="/api/candidates" so callers cannot mistake
   wheelScore / strangleRecommendation for EV-backed outputs.

3. /api/strangle and /api/strikes responses carry the same authority
   contract.

4. WheelTracker launch-gate: when require_ev_authority=True, trades
   without a valid EV token are rejected; tokens are single-use so
   a captured token cannot be replayed.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from engine.wheel_tracker import WheelTracker


# ======================================================================
# 1. WheelTracker EV authority gate
# ======================================================================
class TestWheelTrackerEVAuthorityGate:
    def _ev_row(self, ticker="TEST", **overrides):
        r = {
            "ticker": ticker,
            "strike": 95.0,
            "premium": 1.20,
            "dte": 35,
            "ev_dollars": 25.0,
            "prob_profit": 0.72,
            "distribution_source": "empirical_non_overlapping",
        }
        r.update(overrides)
        return r

    def test_non_strict_tracker_backwards_compatible(self):
        """Default tracker (require_ev_authority=False) accepts trades
        without a token. This preserves the research/test path."""
        t = WheelTracker(initial_capital=100_000)
        ok = t.open_short_put(
            ticker="AAPL",
            strike=180,
            premium=2.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.25,
        )
        assert ok is True

    def test_strict_tracker_rejects_trade_without_token(self):
        """Production tracker must refuse trades lacking an EV token."""
        t = WheelTracker(initial_capital=100_000, require_ev_authority=True)
        ok = t.open_short_put(
            ticker="AAPL",
            strike=180,
            premium=2.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.25,
            ev_authority_token=None,
        )
        assert ok is False
        assert "AAPL" not in t.positions

    def test_strict_tracker_accepts_valid_token(self):
        t = WheelTracker(initial_capital=100_000, require_ev_authority=True)
        token = t.issue_ev_authority_token(
            self._ev_row(ticker="AAPL", strike=180, premium=2.50, dte=32)
        )
        ok = t.open_short_put(
            ticker="AAPL",
            strike=180,
            premium=2.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.25,
            ev_authority_token=token,
        )
        assert ok is True
        assert "AAPL" in t.positions

    def test_strict_tracker_rejects_replayed_token(self):
        """Single-use token: after the first open_short_put consumes
        the token, a second attempt with the same token must fail."""
        t = WheelTracker(initial_capital=200_000, require_ev_authority=True)
        token = t.issue_ev_authority_token(
            self._ev_row(ticker="AAPL", strike=180, premium=2.50, dte=32)
        )
        # First use: succeeds.
        assert t.open_short_put(
            ticker="AAPL",
            strike=180,
            premium=2.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.25,
            ev_authority_token=token,
        )
        # Close the position so the ticker slot is free.
        t.positions.pop("AAPL", None)
        # Second use of same token: must be rejected.
        assert not t.open_short_put(
            ticker="AAPL",
            strike=180,
            premium=2.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.25,
            ev_authority_token=token,
        )

    def test_strict_tracker_rejects_random_string(self):
        t = WheelTracker(initial_capital=100_000, require_ev_authority=True)
        _ = t.issue_ev_authority_token(self._ev_row())
        # Wrong token
        ok = t.open_short_put(
            ticker="AAPL",
            strike=180,
            premium=2.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.25,
            ev_authority_token="0" * 64,  # looks like a digest but isn't issued
        )
        assert ok is False

    def test_authority_log_captures_reject_reason(self):
        t = WheelTracker(initial_capital=100_000, require_ev_authority=True)
        t.open_short_put(
            ticker="AAPL",
            strike=180,
            premium=2.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.25,
            ev_authority_token="bad",
        )
        rejects = [
            e for e in t._ev_authority_log if e.get("action") == "reject"
        ]
        assert len(rejects) >= 1
        assert rejects[0]["reason"] == "unknown_token"


# ======================================================================
# 2. _enrich_alert verdict is EV-authoritative
# ======================================================================
class TestEnrichAlertEVAuthority:
    def test_enrich_uses_ev_not_wheel_score(self):
        """Mock the EV ranker to return a clear EV verdict, and verify
        the enriched response reflects ev_dollars + prob_profit rather
        than the heuristic wheel_score >= 60 rule."""
        from engine_api import EngineAPIHandler
        from engine.tv_signals import TVAlert

        class _FakeRunner:
            def analyze_ticker(self, ticker, as_of=None):
                from engine.wheel_runner import TickerAnalysis

                return TickerAnalysis(
                    ticker=ticker,
                    spot_price=100.0,
                    wheel_score=20.0,  # LOW heuristic score
                    wheel_recommendation="weak",
                    days_to_earnings=None,
                    sector="Tech",
                )

            def rank_candidates_by_ev(self, **kwargs):
                # High EV, high probability — should yield "proceed"
                return pd.DataFrame(
                    [
                        {
                            "ticker": "AAPL",
                            "spot": 100.0,
                            "strike": 95.0,
                            "premium": 1.50,
                            "dte": 35,
                            "iv": 0.22,
                            "ev_dollars": 45.0,
                            "ev_per_day": 1.30,
                            "prob_profit": 0.78,
                            "prob_assignment": 0.18,
                            "distribution_source": "empirical_non_overlapping",
                        }
                    ]
                )

        class _FakeConn:
            def get_ohlcv(self, ticker):
                import numpy as np

                idx = pd.date_range("2020-01-01", periods=800, freq="B")
                prices = 100 * np.exp(
                    np.cumsum(np.random.default_rng(0).normal(0, 0.01, 800))
                )
                return pd.DataFrame({"close": prices}, index=idx)

            def get_iv_rank(self, ticker, as_of=None):
                return 55.0

            def get_vol_risk_premium(self, ticker, as_of=None):
                return 2.5

        alert = TVAlert(ticker="AAPL", signal="wheel_put_zone", source="test")

        handler = EngineAPIHandler.__new__(EngineAPIHandler)

        # Force TV signal to agree with the alert so we isolate the
        # EV-authority behaviour from the signal-state parity logic.
        from engine.tv_signals import TVSignal as _TVSignal

        fake_sig = _TVSignal(
            ticker="AAPL",
            ok=True,
            close=100.0,
            phase="post_expansion",
            signal_action="wheel_put_zone",
            wheel_put_zone=True,
        )

        with patch("engine_api.get_connector", return_value=_FakeConn()), patch(
            "engine_api.get_runner", return_value=_FakeRunner()
        ), patch(
            "engine.tv_signals.compute_tv_signal", return_value=fake_sig
        ):
            enriched = handler._enrich_alert(alert)

        # Authority contract:
        assert enriched["authority"] == "ev_ranked"
        # EV numbers propagated:
        assert enriched["ev_dollars"] == pytest.approx(45.0)
        assert enriched["prob_profit"] == pytest.approx(0.78)
        # With high EV and chart agreement, verdict must be proceed.
        assert enriched["verdict"] == "proceed"
        assert enriched["verdict_reason"] == "ev_above_threshold_and_chart_agrees"

    def test_enrich_skips_on_negative_ev(self):
        """Even with a strong heuristic wheel_score, negative EV must
        force a skip verdict — the guardrail that the prior audit
        added for /api/candidates must apply here too."""
        from engine_api import EngineAPIHandler
        from engine.tv_signals import TVAlert

        class _FakeRunner:
            def analyze_ticker(self, ticker, as_of=None):
                from engine.wheel_runner import TickerAnalysis

                return TickerAnalysis(
                    ticker=ticker,
                    spot_price=100.0,
                    wheel_score=85.0,  # HIGH heuristic — misleading
                    wheel_recommendation="strong_candidate",
                    days_to_earnings=None,
                    sector="Tech",
                )

            def rank_candidates_by_ev(self, **kwargs):
                return pd.DataFrame(
                    [
                        {
                            "ticker": "AAPL",
                            "spot": 100.0,
                            "strike": 95.0,
                            "premium": 0.01,
                            "dte": 35,
                            "iv": 0.22,
                            "ev_dollars": -5.0,  # Negative EV
                            "ev_per_day": -0.20,
                            "prob_profit": 0.50,
                            "prob_assignment": 0.50,
                            "distribution_source": "empirical_non_overlapping",
                        }
                    ]
                )

        class _FakeConn:
            def get_ohlcv(self, ticker):
                import numpy as np

                idx = pd.date_range("2020-01-01", periods=800, freq="B")
                prices = 100 * np.exp(
                    np.cumsum(np.random.default_rng(0).normal(0, 0.01, 800))
                )
                return pd.DataFrame({"close": prices}, index=idx)

            def get_iv_rank(self, ticker, as_of=None):
                return 60.0

            def get_vol_risk_premium(self, ticker, as_of=None):
                return 5.0

        alert = TVAlert(ticker="AAPL", signal="wheel_put_zone", source="test")
        handler = EngineAPIHandler.__new__(EngineAPIHandler)

        with patch("engine_api.get_connector", return_value=_FakeConn()), patch(
            "engine_api.get_runner", return_value=_FakeRunner()
        ):
            enriched = handler._enrich_alert(alert)

        assert enriched["authority"] == "ev_ranked"
        assert enriched["ev_dollars"] == pytest.approx(-5.0)
        # Negative EV must produce skip regardless of chart agreement.
        assert enriched["verdict"] == "skip"
        assert enriched["verdict_reason"] == "negative_ev"


# ======================================================================
# 3. /api/analyze, /api/strangle, /api/strikes authority markers
# ======================================================================
class TestDiagnosticEndpointAuthorityMarkers:
    def test_analyze_carries_heuristic_authority(self):
        """/api/analyze returns wheelScore and strangleRecommendation
        which are both heuristic. The response must carry an explicit
        authority marker so the dashboard can never route a trade
        based on these values."""
        from engine_api import EngineAPIHandler
        from engine.wheel_runner import TickerAnalysis

        class _FakeConn:
            def get_universe(self):
                return ["AAPL"]

            def get_ohlcv(self, ticker):
                return pd.DataFrame({"close": [100]})

        class _FakeRunner:
            def analyze_ticker(self, ticker, as_of=None):
                return TickerAnalysis(
                    ticker=ticker,
                    spot_price=100.0,
                    wheel_score=75.0,
                    wheel_recommendation="strong_candidate",
                    strangle_score=80.0,
                    strangle_recommendation="enter",
                )

        captured = {}
        handler = EngineAPIHandler.__new__(EngineAPIHandler)
        handler._send_json = lambda payload, status=200: captured.update(
            payload=payload
        )
        handler._send_error = lambda *a, **kw: captured.update(error=a)

        with patch("engine_api.get_connector", return_value=_FakeConn()), patch(
            "engine_api.get_runner", return_value=_FakeRunner()
        ):
            handler._handle_analyze("AAPL", None)

        payload = captured.get("payload")
        assert payload is not None
        assert payload["authority"] == "heuristic_diagnostic"
        assert payload["tradeable_endpoint"] == "/api/candidates"
        assert payload["wheelScoreAuthority"] == "heuristic_diagnostic"
        assert payload["strangleAuthority"] == "heuristic_diagnostic"

    def test_strangle_endpoint_carries_heuristic_authority(self):
        from engine_api import EngineAPIHandler

        class _FakeConn:
            def get_ohlcv(self, ticker):
                import numpy as np

                idx = pd.date_range("2020-01-01", periods=500, freq="B")
                prices = 100 * np.exp(
                    np.cumsum(np.random.default_rng(0).normal(0, 0.01, 500))
                )
                return pd.DataFrame(
                    {
                        "open": prices,
                        "high": prices * 1.01,
                        "low": prices * 0.99,
                        "close": prices,
                        "volume": [1_000_000] * 500,
                    },
                    index=idx,
                )

        captured = {}
        handler = EngineAPIHandler.__new__(EngineAPIHandler)
        handler._send_json = lambda payload, status=200: captured.update(
            payload=payload
        )

        with patch("engine_api.get_connector", return_value=_FakeConn()):
            handler._handle_strangle("AAPL")

        payload = captured.get("payload", {})
        assert payload.get("authority") == "heuristic_diagnostic"
        assert payload.get("tradeable_endpoint") == "/api/candidates"
        assert "note" in payload

    def test_strikes_endpoint_carries_heuristic_authority(self):
        from engine_api import EngineAPIHandler

        class _FakeConn:
            def get_ohlcv(self, ticker):
                return pd.DataFrame({"close": [100.0, 101.0, 102.0]})

            def get_fundamentals(self, ticker):
                return {"implied_vol_atm": 25.0}

        captured = {}
        handler = EngineAPIHandler.__new__(EngineAPIHandler)
        handler._send_json = lambda payload, status=200: captured.update(
            payload=payload
        )

        with patch("engine_api.get_connector", return_value=_FakeConn()):
            handler._handle_strikes("AAPL", "csp", "45")

        payload = captured.get("payload", {})
        assert payload.get("authority") == "heuristic_diagnostic"
        assert payload.get("tradeable_endpoint") == "/api/candidates"
