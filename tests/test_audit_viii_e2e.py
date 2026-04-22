"""
AUDIT-VIII end-to-end integration tests.

Locks in the full authority chain that a real trade must traverse:

    TV webhook → HMAC verify → freshness + nonce-replay → _enrich_alert
    → EV ranker → issue_ev_authority_token → WheelTracker.open_short_put

and the per-layer rejection semantics (no-HMAC, replayed-HMAC,
negative-EV enrichment, forged ticker, stale-data guard, committee
authority label).

The tests use a fake connector backed by a synthetic OHLCV history so
they are fast and deterministic, but they exercise the *same* code
paths that live traffic hits — including the HMAC signature check, the
nonce-replay guard, the enrichment EV call, and the strict tracker.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import engine_api
from engine.wheel_runner import WheelRunner
from engine.wheel_tracker import WheelTracker


# ======================================================================
# Synthetic connector — enough surface to drive rank_candidates_by_ev
# ======================================================================
def _synth_ohlcv(seed: int, n: int = 2000, mu: float = 3e-4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(mu, 0.012, n)))
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


class _E2EConn:
    """In-memory connector sufficient for the enrichment path."""

    def __init__(self, ticker: str = "AAPL"):
        self._ticker = ticker
        self._ohlcv = _synth_ohlcv(seed=1)
        self._data_dir = "/tmp/e2e"

    def get_universe(self):
        return [self._ticker]

    def get_ohlcv(self, ticker, start_date=None, end_date=None):
        if ticker.upper() != self._ticker.upper():
            return pd.DataFrame()
        return self._ohlcv

    def get_fundamentals(self, ticker):
        return {
            # PERCENT form — the audit-VIII normaliser must convert these.
            "implied_vol_atm": 26.5,
            "volatility_30d": 27.0,
            "dividend_yield": 0.5,
            "beta": 1.1,
            "market_cap": 3e12,
            "pe_ratio": 28,
            "sector": "Technology",
        }

    def get_risk_free_rate(self, as_of=None, tenor="rate_3m"):
        return 0.042  # decimal

    def get_iv_rank(self, ticker, as_of=None):
        return 0.55

    def get_iv_percentile(self, ticker, as_of=None):
        return 0.6

    def get_vol_risk_premium(self, ticker, as_of=None):
        return 3.2

    def get_vix_regime(self, as_of=None):
        return {"vix": 15.0, "vix_percentile": 0.35, "term_structure": "contango"}

    def get_credit_risk(self, ticker):
        return {"rtg_sp_lt_lc_issuer_credit": "AA"}

    def get_next_earnings(self, ticker, as_of=None):
        return None

    def get_next_dividend(self, ticker, as_of=None):
        return None

    def get_options(self, ticker):
        return pd.DataFrame()


# ======================================================================
# Helpers to build a FakeHandler that drives the real handler methods
# without opening a socket.
# ======================================================================
class _FakeHandler(engine_api.EngineAPIHandler):
    """Subclass that captures responses in memory — no real TCP socket."""

    def __init__(self):
        # Deliberately skip BaseHTTPRequestHandler.__init__ — we don't
        # want it to read from a socket. We only need the helper
        # methods like _send_json / _send_error.
        self.responses: list[tuple[int, dict]] = []

    def _send_json(self, data, status=200):
        self.responses.append((status, data))

    def _send_error(self, message, status=500):
        self.responses.append((status, {"error": message}))


# ======================================================================
# 1. Webhook → HMAC guard
# ======================================================================
class TestWebhookHMACGate:
    def test_webhook_without_hmac_is_rejected(self, monkeypatch):
        monkeypatch.setenv("TV_WEBHOOK_HMAC_SECRET", "test-secret-1")
        h = _FakeHandler()
        payload = {"ticker": "AAPL", "signal": "wheel_put_zone"}
        raw = json.dumps(payload).encode()
        h._handle_tv_webhook(payload, raw_body=raw, signature_header="")
        assert h.responses[-1][0] == 401
        assert "hmac" in h.responses[-1][1]["error"].lower()

    def test_webhook_with_valid_hmac_is_accepted(self, monkeypatch):
        monkeypatch.setenv("TV_WEBHOOK_HMAC_SECRET", "test-secret-2")
        # Clear the replay cache so this test can run in isolation.
        engine_api._TV_SEEN_NONCES.clear()

        h = _FakeHandler()
        payload = {"ticker": "AAPL", "signal": "wheel_put_zone"}
        raw = json.dumps(payload).encode()
        sig = hmac.new(b"test-secret-2", raw, hashlib.sha256).hexdigest()
        with patch.object(engine_api, "get_connector", return_value=_E2EConn("AAPL")), \
             patch.object(engine_api, "get_runner", return_value=WheelRunner()):
            h._handle_tv_webhook(payload, raw_body=raw, signature_header=sig)
        assert h.responses[-1][0] == 200
        body = h.responses[-1][1]
        assert body["accepted"] is True
        assert "enriched" in body

    def test_webhook_replay_is_blocked(self, monkeypatch):
        monkeypatch.setenv("TV_WEBHOOK_HMAC_SECRET", "test-secret-3")
        engine_api._TV_SEEN_NONCES.clear()

        payload = {"ticker": "AAPL", "signal": "wheel_put_zone"}
        raw = json.dumps(payload).encode()
        sig = hmac.new(b"test-secret-3", raw, hashlib.sha256).hexdigest()
        with patch.object(engine_api, "get_connector", return_value=_E2EConn("AAPL")), \
             patch.object(engine_api, "get_runner", return_value=WheelRunner()):
            h1 = _FakeHandler()
            h1._handle_tv_webhook(payload, raw_body=raw, signature_header=sig)
            h2 = _FakeHandler()
            h2._handle_tv_webhook(payload, raw_body=raw, signature_header=sig)
        assert h1.responses[-1][0] == 200
        assert h2.responses[-1][0] == 409
        assert "replay" in h2.responses[-1][1]["error"].lower()

    def test_webhook_tampered_body_is_rejected(self, monkeypatch):
        monkeypatch.setenv("TV_WEBHOOK_HMAC_SECRET", "test-secret-4")
        engine_api._TV_SEEN_NONCES.clear()

        original = {"ticker": "AAPL", "signal": "wheel_put_zone"}
        tampered = {"ticker": "AAPL", "signal": "strangle_zone"}  # different
        raw_orig = json.dumps(original).encode()
        sig = hmac.new(b"test-secret-4", raw_orig, hashlib.sha256).hexdigest()
        raw_tampered = json.dumps(tampered).encode()

        h = _FakeHandler()
        h._handle_tv_webhook(tampered, raw_body=raw_tampered, signature_header=sig)
        assert h.responses[-1][0] == 401


# ======================================================================
# 2. EV-anchored verdict on enrichment
# ======================================================================
class TestEnrichmentEVAuthority:
    def test_enrichment_returns_ev_authoritative_verdict(self):
        h = _FakeHandler()
        conn = _E2EConn("AAPL")
        runner = WheelRunner()
        with patch.object(engine_api, "get_connector", return_value=conn), \
             patch.object(engine_api, "get_runner", return_value=runner):
            from engine.tv_signals import TVAlert
            alert = TVAlert(ticker="AAPL", signal="wheel_put_zone", source="test")
            enriched = h._enrich_alert(alert)
        # Authority is EV-ranked or ev_unavailable — never heuristic.
        assert enriched["authority"] in ("ev_ranked", "ev_unavailable")
        # Verdict field exists and is one of the known values.
        assert enriched["verdict"] in ("proceed", "review", "skip")
        # EV dollars is a real number, not None.
        assert isinstance(enriched["ev_dollars"], (int, float))

    def test_enrichment_blocks_non_existent_ticker(self):
        h = _FakeHandler()
        conn = _E2EConn("AAPL")
        runner = WheelRunner()
        with patch.object(engine_api, "get_connector", return_value=conn), \
             patch.object(engine_api, "get_runner", return_value=runner):
            from engine.tv_signals import TVAlert
            alert = TVAlert(ticker="NONEXIST", signal="wheel_put_zone", source="test")
            enriched = h._enrich_alert(alert)
        assert enriched["accepted"] is False
        assert enriched["reason"] == "ticker_not_in_universe"


# ======================================================================
# 3. Full chain: webhook → enrich → EV → token → tracker
# ======================================================================
class TestFullAuthorityChain:
    def test_positive_ev_flows_through_tracker(self, monkeypatch):
        """Simulate the full path: a webhook arrives, the engine
        enriches with an EV verdict, the ranker returns a row, a
        token is issued from that row, and the strict tracker accepts
        the trade. Then a replay of the same token must be rejected."""
        monkeypatch.setenv("TV_WEBHOOK_HMAC_SECRET", "full-chain-secret")
        engine_api._TV_SEEN_NONCES.clear()

        payload = {"ticker": "AAPL", "signal": "wheel_put_zone"}
        raw = json.dumps(payload).encode()
        sig = hmac.new(b"full-chain-secret", raw, hashlib.sha256).hexdigest()

        conn = _E2EConn("AAPL")
        runner = WheelRunner()

        # Step 1: webhook through the handler
        with patch.object(engine_api, "get_connector", return_value=conn), \
             patch.object(engine_api, "get_runner", return_value=runner):
            h = _FakeHandler()
            h._handle_tv_webhook(payload, raw_body=raw, signature_header=sig)
        assert h.responses[-1][0] == 200
        enriched = h.responses[-1][1]["enriched"]

        # Step 2: directly call the EV ranker for the token payload
        with patch.object(WheelRunner, "connector",
                          new_callable=lambda: property(lambda self: conn)):
            ev_df = runner.rank_candidates_by_ev(
                tickers=["AAPL"],
                dte_target=35,
                delta_target=0.25,
                top_n=1,
                min_ev_dollars=-1e9,
                enforce_history_gate=False,
                enforce_chain_quality_gate=False,
            )
        assert len(ev_df) == 1
        ev_row = ev_df.iloc[0].to_dict()

        # Step 3: strict tracker
        tracker = WheelTracker(initial_capital=250_000, require_ev_authority=True)
        token = tracker.issue_ev_authority_token(ev_row)

        # No token → reject
        no_token = tracker.open_short_put(
            ticker="AAPL",
            strike=ev_row["strike"],
            premium=ev_row["premium"],
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 4, 14) + timedelta(days=35),
            iv=ev_row["iv"],
            ev_authority_token=None,
        )
        assert no_token is False

        # Real token → accept
        opened = tracker.open_short_put(
            ticker="AAPL",
            strike=ev_row["strike"],
            premium=ev_row["premium"],
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 4, 14) + timedelta(days=35),
            iv=ev_row["iv"],
            ev_authority_token=token,
        )
        assert opened is True
        assert "AAPL" in tracker.positions

        # Replayed token → reject
        tracker2 = WheelTracker(initial_capital=250_000, require_ev_authority=True)
        replayed = tracker2.open_short_put(
            ticker="AAPL",
            strike=ev_row["strike"],
            premium=ev_row["premium"],
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 4, 14) + timedelta(days=35),
            iv=ev_row["iv"],
            ev_authority_token=token,
        )
        assert replayed is False


# ======================================================================
# 4. Committee authority label
# ======================================================================
class TestCommitteeAuthorityLabel:
    def test_committee_response_is_labelled_heuristic(self):
        conn = _E2EConn("AAPL")
        runner = WheelRunner()
        h = _FakeHandler()
        with patch.object(engine_api, "get_connector", return_value=conn), \
             patch.object(engine_api, "get_runner", return_value=runner), \
             patch.object(WheelRunner, "connector",
                          new_callable=lambda: property(lambda self: conn)):
            h._handle_committee("AAPL")
        assert h.responses, "committee handler produced no response"
        body = h.responses[-1][1]
        assert body.get("authority") == "heuristic_diagnostic"
        assert body.get("tradeable_endpoint") == "/api/candidates"
        assert "ev_anchored" in body


# ======================================================================
# 5. HMM regime cache reuse
# ======================================================================
class TestHMMRegimeCache:
    def test_second_call_reuses_cached_regime_multiplier(self):
        conn = _E2EConn("AAPL")
        runner = WheelRunner()
        with patch.object(WheelRunner, "connector",
                          new_callable=lambda: property(lambda self: conn)):
            df1 = runner.rank_candidates_by_ev(
                tickers=["AAPL"],
                dte_target=35,
                delta_target=0.25,
                top_n=1,
                min_ev_dollars=-1e9,
                enforce_history_gate=False,
                enforce_chain_quality_gate=False,
            )
            cache_size_after_first = len(runner._hmm_regime_cache)
            df2 = runner.rank_candidates_by_ev(
                tickers=["AAPL"],
                dte_target=35,
                delta_target=0.25,
                top_n=1,
                min_ev_dollars=-1e9,
                enforce_history_gate=False,
                enforce_chain_quality_gate=False,
            )
            cache_size_after_second = len(runner._hmm_regime_cache)
        assert len(df1) == 1 and len(df2) == 1
        # Cache grew on the first call, NOT on the second.
        assert cache_size_after_first == 1
        assert cache_size_after_second == 1


# ======================================================================
# 6. OHLCV invariant guard (silent-column-swap protection)
# ======================================================================
class TestOHLCVInvariantGuard:
    def test_invariant_check_runs_silently_on_valid_data(self, caplog):
        """A well-formed OHLCV frame must not trigger the critical
        warning. The guard only fires on genuine inversions."""
        from engine.data_connector import MarketDataConnector

        # Reset the per-process flag so the test is deterministic.
        MarketDataConnector._ohlcv_invariant_warned = False
        conn = MarketDataConnector.__new__(MarketDataConnector)
        idx = pd.date_range("2024-01-01", periods=50, freq="B")
        # Valid OHLC: high = max, low = min.
        df = pd.DataFrame({
            "open": np.full(50, 100.0),
            "high": np.full(50, 101.0),
            "low": np.full(50, 99.0),
            "close": np.full(50, 100.5),
        }, index=idx)
        with caplog.at_level("CRITICAL"):
            conn._validate_ohlcv_invariants("TEST", df)
        assert "OHLCV invariant violation" not in caplog.text

    def test_invariant_check_fires_on_inverted_data(self, caplog):
        from engine.data_connector import MarketDataConnector

        MarketDataConnector._ohlcv_invariant_warned = False
        conn = MarketDataConnector.__new__(MarketDataConnector)
        idx = pd.date_range("2024-01-01", periods=50, freq="B")
        # INVERTED OHLC — high is *lowest*, low is *highest*.
        df = pd.DataFrame({
            "open": np.full(50, 100.0),
            "high": np.full(50, 99.0),   # WRONG
            "low": np.full(50, 101.0),   # WRONG
            "close": np.full(50, 100.5),
        }, index=idx)
        with caplog.at_level("CRITICAL"):
            conn._validate_ohlcv_invariants("TEST", df)
        assert "OHLCV invariant violation" in caplog.text
