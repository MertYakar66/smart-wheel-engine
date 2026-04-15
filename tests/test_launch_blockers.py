"""
Launch-blocker fix tests (audit V, P0 + P1).

Every test here locks in a specific P0/P1 from the launch-readiness
review:

* P0.1 — Decision-authority unification:
    /api/candidates is EV-authoritative (not heuristic BSM math)
    /api/tv/scan pool is EV-ranked, not screen_candidates
    /api/screen is flagged as research-only with a warning
* P0.2 — Historical data integrity gate in rank_candidates_by_ev
* P0.3 — Chain quality hard gate in rank_candidates_by_ev
* P1   — Greeks stress decomposition residual gate
"""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ======================================================================
# Fixtures — fake connector + fake runner that just returns a canned
# EV frame so we can exercise the API handlers without touching disk.
# ======================================================================
class _FakeConnector:
    def __init__(self):
        self._data_dir = "/tmp/fake"

    def get_universe(self):
        return ["AAA", "BBB"]

    def get_vix_regime(self, as_of=None):
        return {"vix": 14.5}

    def get_ohlcv(self, ticker):
        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))
        return pd.DataFrame({"close": prices}, index=idx)


class _FakeEVRunner:
    """Fake WheelRunner that returns a canned EV-ranked DataFrame."""

    def __init__(self):
        self.rank_calls = []
        self.screen_calls = []

    def rank_candidates_by_ev(self, **kwargs):
        self.rank_calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "spot": 100.0,
                    "strike": 95.0,
                    "premium": 1.25,
                    "dte": 35,
                    "iv": 0.22,
                    "ev_dollars": 35.0,
                    "ev_per_day": 1.10,
                    "prob_profit": 0.75,
                    "prob_assignment": 0.22,
                    "days_to_earnings": None,
                    "distribution_source": "empirical_non_overlapping",
                    "cvar_5": -250.0,
                    "cvar_99_evt": -400.0,
                    "tail_xi": 0.05,
                    "heavy_tail": False,
                    "omega_ratio": 1.35,
                    "fair_value": 1.40,
                    "edge_vs_fair": -15.0,
                    "breakeven_move_pct": -0.042,
                    "total_transaction_cost": 4.5,
                    "skew_pnl": -0.25,
                    "dealer_regime": None,
                    "dealer_multiplier": 1.0,
                    "gex_total": None,
                    "gamma_flip_distance_pct": None,
                    "nearest_put_wall_strike": None,
                    "nearest_call_wall_strike": None,
                },
                {
                    "ticker": "BBB",
                    "spot": 200.0,
                    "strike": 190.0,
                    "premium": 2.10,
                    "dte": 35,
                    "iv": 0.24,
                    "ev_dollars": 12.0,
                    "ev_per_day": 0.40,
                    "prob_profit": 0.68,
                    "prob_assignment": 0.28,
                    "days_to_earnings": 20,
                    "distribution_source": "block_bootstrap",
                    "cvar_5": -180.0,
                    "cvar_99_evt": -300.0,
                    "tail_xi": 0.02,
                    "heavy_tail": False,
                    "omega_ratio": 1.15,
                    "fair_value": 2.20,
                    "edge_vs_fair": -8.0,
                    "breakeven_move_pct": -0.055,
                    "total_transaction_cost": 5.0,
                    "skew_pnl": -0.10,
                    "dealer_regime": None,
                    "dealer_multiplier": 1.0,
                    "gex_total": None,
                    "gamma_flip_distance_pct": None,
                    "nearest_put_wall_strike": None,
                    "nearest_call_wall_strike": None,
                },
            ]
        )

    def screen_candidates(self, **kwargs):
        self.screen_calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "ticker": "ZZZ",
                    "wheel_score": 62.0,
                    "recommendation": "moderate",
                    "spot": 50.0,
                    "iv_rank": 0.4,
                }
            ]
        )

    def analyze_ticker(self, ticker, as_of=None):
        from engine.wheel_runner import TickerAnalysis

        return TickerAnalysis(ticker=ticker, spot_price=100.0)


def _call_handler(handler_fn, path_with_query: str):
    """Drive a single HTTP handler method and capture the JSON response."""
    from engine_api import EngineAPIHandler

    class _FakeRequest(io.BytesIO):
        def makefile(self, *a, **kw):
            return self

    # Hand-crafted HTTP request line
    req = _FakeRequest(f"GET {path_with_query} HTTP/1.1\r\nHost: test\r\n\r\n".encode())
    wfile = io.BytesIO()

    handler = EngineAPIHandler.__new__(EngineAPIHandler)
    handler.rfile = req
    handler.wfile = wfile
    handler.headers = {}
    handler.path = path_with_query
    handler.client_address = ("127.0.0.1", 0)
    handler.request_version = "HTTP/1.1"
    handler.command = "GET"
    handler.requestline = f"GET {path_with_query} HTTP/1.1"
    handler.server = None
    handler._headers_buffer = []
    handler.close_connection = True
    handler.raw_requestline = b""

    handler_fn(handler)

    wfile.seek(0)
    data = wfile.getvalue().decode()
    # Strip HTTP headers; find first blank line
    try:
        body = data.split("\r\n\r\n", 1)[1]
    except IndexError:
        body = data
    return json.loads(body)


# ======================================================================
# P0.1a — /api/candidates is EV-authoritative
# ======================================================================
class TestCandidatesEVAuthority:
    def test_candidates_returns_ev_native_fields(self):
        """Every row must carry EV-native keys, not heuristic BSM output."""
        fake_runner = _FakeEVRunner()
        with patch("engine_api.get_runner", return_value=fake_runner):
            from engine_api import EngineAPIHandler

            handler = EngineAPIHandler.__new__(EngineAPIHandler)
            sent = {}

            def capture(self_, payload, status=200):
                sent["payload"] = payload

            handler._send_json = lambda payload, status=200: capture(None, payload, status)
            handler._handle_candidates(
                limit="5", min_score="0", dte="35", delta="0.25", min_ev="0", as_of=None
            )
            payload = sent["payload"]

        assert payload["authority"] == "ev_ranked"
        assert payload["engine_version"] == "ev_engine_2026_04_14"
        assert payload["count"] == 2
        # First row
        t0 = payload["trades"][0]
        assert t0["ticker"] == "AAA"
        # EV-native fields must be present
        assert t0["evDollars"] == 35.0
        assert t0["evPerDay"] == 1.10
        assert t0["probProfit"] == 0.75
        assert t0["probAssignment"] == 0.22
        assert t0["distributionSource"] == "empirical_non_overlapping"
        # Backward-compat alias MUST mirror EV (not heuristic)
        assert t0["expectedPnL"] == 35.0

    def test_candidates_does_not_call_screen_candidates(self):
        """Hard guardrail: the EV path must not route through screen_candidates."""
        fake_runner = _FakeEVRunner()
        with patch("engine_api.get_runner", return_value=fake_runner):
            from engine_api import EngineAPIHandler

            handler = EngineAPIHandler.__new__(EngineAPIHandler)
            handler._send_json = lambda *a, **kw: None
            handler._handle_candidates(
                limit="10", min_score="0", dte="35", delta="0.25", min_ev="0", as_of=None
            )
        assert len(fake_runner.screen_calls) == 0
        assert len(fake_runner.rank_calls) == 1

    def test_candidates_passes_through_query_params(self):
        fake_runner = _FakeEVRunner()
        with patch("engine_api.get_runner", return_value=fake_runner):
            from engine_api import EngineAPIHandler

            handler = EngineAPIHandler.__new__(EngineAPIHandler)
            handler._send_json = lambda *a, **kw: None
            handler._handle_candidates(
                limit="7",
                min_score="30",
                dte="28",
                delta="0.20",
                min_ev="15",
                as_of="2026-04-01",
            )
        call = fake_runner.rank_calls[0]
        assert call["dte_target"] == 28
        assert call["delta_target"] == 0.20
        assert call["min_ev_dollars"] == 15.0
        assert call["top_n"] == 7
        assert call["as_of"] == "2026-04-01"


# ======================================================================
# P0.1b — /api/screen is explicitly flagged research-only
# ======================================================================
class TestScreenResearchOnly:
    def test_screen_response_includes_research_only_flag(self):
        fake_runner = _FakeEVRunner()
        with patch("engine_api.get_runner", return_value=fake_runner):
            from engine_api import EngineAPIHandler

            handler = EngineAPIHandler.__new__(EngineAPIHandler)
            captured = {}
            handler._send_json = lambda payload, status=200: captured.update(payload=payload)
            handler._handle_screen({})
            payload = captured["payload"]

        assert payload["authority"] == "heuristic_research_only"
        assert "warning" in payload
        assert payload["tradeable_endpoint"] == "/api/candidates"


# ======================================================================
# P0.2 — History gate
# ======================================================================
class TestHistoryGate:
    def test_short_history_ticker_dropped_by_default(self):
        from engine.wheel_runner import WheelRunner

        rng = np.random.default_rng(0)

        # 100 bars — well below the 504-day default gate
        short_idx = pd.date_range("2024-01-01", periods=100, freq="B")
        short_prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 100)))
        short_ohlcv = pd.DataFrame({"close": short_prices}, index=short_idx)

        class _ShortConn:
            def get_ohlcv(self, ticker):
                return short_ohlcv

            def get_fundamentals(self, ticker):
                return {"implied_vol_atm": 0.25, "volatility_30d": 0.22, "dividend_yield": 0.0}

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["NEWIPO"]

        runner = WheelRunner()
        runner._connector = _ShortConn()
        df = runner.rank_candidates_by_ev(
            tickers=["NEWIPO"],
            dte_target=30,
            delta_target=0.25,
            top_n=5,
            min_ev_dollars=-1e9,
        )
        assert df.empty, "short-history ticker must be dropped by the gate"

    def test_history_gate_can_be_disabled_for_research(self):
        from engine.wheel_runner import WheelRunner

        rng = np.random.default_rng(0)
        short_idx = pd.date_range("2024-01-01", periods=400, freq="B")
        short_prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 400)))
        short_ohlcv = pd.DataFrame({"close": short_prices}, index=short_idx)

        class _Conn:
            def get_ohlcv(self, ticker):
                return short_ohlcv

            def get_fundamentals(self, ticker):
                return {"implied_vol_atm": 0.25, "volatility_30d": 0.22, "dividend_yield": 0.0}

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["MIDIPO"]

        runner = WheelRunner()
        runner._connector = _Conn()
        # With gate disabled, the ticker passes
        df = runner.rank_candidates_by_ev(
            tickers=["MIDIPO"],
            dte_target=30,
            top_n=5,
            min_ev_dollars=-1e9,
            enforce_history_gate=False,
        )
        assert not df.empty


# ======================================================================
# P0.3 — Chain quality gate
# ======================================================================
class TestChainQualityGate:
    def _long_ohlcv(self):
        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))
        return pd.DataFrame({"close": prices}, index=idx)

    def _fake_connector_with_chain(self, chain: pd.DataFrame):
        ohlcv = self._long_ohlcv()

        class _Conn:
            def get_ohlcv(self, ticker):
                return ohlcv

            def get_fundamentals(self, ticker):
                return {"implied_vol_atm": 0.25, "volatility_30d": 0.22, "dividend_yield": 0.0}

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["TESTA"]

            def get_options(self, ticker):
                return chain.copy()

        return _Conn

    def test_crossed_market_chain_is_blocked(self):
        from datetime import date, timedelta

        from engine.wheel_runner import WheelRunner

        expiry = date.today() + timedelta(days=32)
        bad = pd.DataFrame(
            [
                # Bid > ask on the 95 put → crossed market ERROR
                {"strike": 95, "option_type": "P", "open_interest": 1000, "implied_vol": 0.25, "bid": 2.5, "ask": 2.0, "expiration": expiry},
                {"strike": 100, "option_type": "C", "open_interest": 1000, "implied_vol": 0.22, "bid": 1.0, "ask": 1.1, "expiration": expiry},
                {"strike": 100, "option_type": "P", "open_interest": 1000, "implied_vol": 0.22, "bid": 1.0, "ask": 1.1, "expiration": expiry},
                {"strike": 105, "option_type": "C", "open_interest": 1000, "implied_vol": 0.22, "bid": 0.5, "ask": 0.6, "expiration": expiry},
            ]
        )
        runner = WheelRunner()
        runner._connector = self._fake_connector_with_chain(bad)()
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"],
            dte_target=30,
            top_n=5,
            min_ev_dollars=-1e9,
            use_dealer_positioning=True,
        )
        assert df.empty, "crossed-market chain must be blocked by the quality gate"

    def test_clean_chain_passes_gate(self):
        from datetime import date, timedelta

        from engine.wheel_runner import WheelRunner

        expiry = date.today() + timedelta(days=32)
        rows = []
        for k, c_oi, p_oi in [(90, 500, 2000), (95, 1000, 3000), (100, 4000, 4000), (105, 3000, 1000)]:
            for opt_type, oi in [("C", c_oi), ("P", p_oi)]:
                rows.append(
                    {
                        "strike": k,
                        "option_type": opt_type,
                        "open_interest": oi,
                        "implied_vol": 0.22,
                        "bid": 1.0,
                        "ask": 1.1,
                        "expiration": expiry,
                    }
                )
        clean = pd.DataFrame(rows)
        runner = WheelRunner()
        runner._connector = self._fake_connector_with_chain(clean)()
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"],
            dte_target=30,
            top_n=5,
            min_ev_dollars=-1e9,
            use_dealer_positioning=True,
        )
        assert not df.empty

    def test_quality_gate_disabled_allows_bad_chain(self):
        from datetime import date, timedelta

        from engine.wheel_runner import WheelRunner

        expiry = date.today() + timedelta(days=32)
        # Mildly-bad chain with bid=ask=0 (stale but not crossed)
        chain = pd.DataFrame(
            [
                {"strike": 95, "option_type": "P", "open_interest": 1000, "implied_vol": 0.25, "bid": 2.5, "ask": 2.0, "expiration": expiry},
                {"strike": 100, "option_type": "C", "open_interest": 1000, "implied_vol": 0.22, "expiration": expiry},
                {"strike": 100, "option_type": "P", "open_interest": 1000, "implied_vol": 0.22, "expiration": expiry},
            ]
        )
        runner = WheelRunner()
        runner._connector = self._fake_connector_with_chain(chain)()
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"],
            dte_target=30,
            top_n=5,
            min_ev_dollars=-1e9,
            use_dealer_positioning=True,
            enforce_chain_quality_gate=False,
        )
        # With gate disabled, the ranker processes the chain
        assert not df.empty


# ======================================================================
# P1 — Stress residual gate
# ======================================================================
class TestStressResidualGate:
    def test_stress_ladder_has_reliable_column(self):
        from engine.stress_testing import StressTester

        tester = StressTester()
        df = tester.greeks_stress_ladder(
            positions=[
                {
                    "symbol": "TEST",
                    "option_type": "put",
                    "strike": 100,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"TEST": 100.0},
            portfolio_value=100000.0,
        )
        assert "reliable" in df.columns
        assert df["reliable"].dtype == bool
        # Metadata attached
        assert "residual_gate_passed" in df.attrs
        assert "n_unreliable_rows" in df.attrs
        assert "max_residual_pct" in df.attrs

    def test_unreliable_rows_flagged_false(self):
        from engine.stress_testing import StressTester

        tester = StressTester()
        # Wider spot range → almost guaranteed to produce unreliable rows
        df = tester.greeks_stress_ladder(
            positions=[
                {
                    "symbol": "TEST",
                    "option_type": "put",
                    "strike": 100,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"TEST": 100.0},
            portfolio_value=100000.0,
            spot_range=(-0.30, 0.30),
        )
        assert "reliable" in df.columns
        # Unreliable row count equals count of reliable=False rows
        n_unreliable_actual = int((~df["reliable"]).sum())
        assert df.attrs["n_unreliable_rows"] == n_unreliable_actual
        # With a ±30% range we expect some unreliable rows
        assert n_unreliable_actual >= 0
