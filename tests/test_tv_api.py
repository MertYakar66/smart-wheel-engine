"""Tests for the TradingView bridge HTTP endpoints in engine_api.py.

Strategy
--------
Rather than booting a real HTTP server (slow, flaky, port conflicts in CI),
we drive ``EngineAPIHandler`` directly via a tiny in-memory socket harness.
This mirrors the pattern used by the rest of the repo's API tests and keeps
the surface area the tests exercise identical to the one production traffic
hits: ``do_GET`` / ``do_POST`` routing, param parsing, handler dispatch,
JSON response encoding.

The heavyweight ``WheelRunner`` is monkey-patched so tests do not touch the
503-ticker Bloomberg cache — they only exercise routing and serialization.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import engine_api
from engine.tv_signals import TVSignal


# ---------------------------------------------------------------------------
# Minimal in-memory request driver
# ---------------------------------------------------------------------------


class _FakeSocket(io.BytesIO):
    """Minimal BytesIO shim that pretends to be a socket for BaseHTTPRequestHandler."""

    def makefile(self, *args, **kwargs):  # pragma: no cover - not called
        return self


def _drive(method: str, path: str, body: bytes | None = None) -> tuple[int, dict]:
    """Invoke EngineAPIHandler with a synthesized request.

    We bypass ``BaseHTTPRequestHandler.handle_one_request`` entirely — the
    handler's ``do_GET`` / ``do_POST`` methods only read ``self.path``,
    ``self.headers``, and the POST body from ``self.rfile``. So we wire
    each of those manually and let the handler do its thing.

    Returns
    -------
    (status_code, parsed_json_body)
    """
    body = body or b""
    # rfile must contain *only* the body so Content-Length reads match
    rfile = io.BytesIO(body)
    wfile = io.BytesIO()

    handler = engine_api.EngineAPIHandler.__new__(engine_api.EngineAPIHandler)
    handler.rfile = rfile
    handler.wfile = wfile
    handler.client_address = ("127.0.0.1", 0)
    handler.command = method
    handler.path = path
    handler.request_version = "HTTP/1.1"
    handler.headers = {"Content-Length": str(len(body))}
    handler.server = MagicMock()
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.close_connection = True

    # The real do_GET/do_POST will call self._send_json which writes to wfile
    # through the normal send_response/send_header/end_headers path. We swap
    # _send_json for a capturing version so we avoid the socket shutdown
    # plumbing that BaseHTTPRequestHandler otherwise requires.
    captured = {"status": 200, "body": None}

    def _capture_json(data, status=200):
        captured["status"] = status
        captured["body"] = data

    def _capture_error(message, status=500):
        captured["status"] = status
        captured["body"] = {"error": message}

    handler._send_json = _capture_json  # type: ignore[assignment]
    handler._send_error = _capture_error  # type: ignore[assignment]

    if method == "GET":
        handler.do_GET()
    elif method == "POST":
        handler.do_POST()
    else:
        raise ValueError(f"unsupported method {method}")

    return captured["status"], captured["body"]


# ---------------------------------------------------------------------------
# Connector + runner stubs
# ---------------------------------------------------------------------------


def _fake_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close})


class _FakeConnector:
    def get_ohlcv(self, ticker):  # noqa: ARG002
        return _fake_ohlcv()

    def get_iv_rank(self, *_a, **_k):
        return 55.0

    def get_vol_risk_premium(self, *_a, **_k):
        return 2.5

    def get_universe(self):
        return ["AAPL", "MSFT", "MU"]

    def get_vix_regime(self, *_a, **_k):
        return {"vix": 18.0}


class _FakeRunner:
    def __init__(self):
        self.analysis = MagicMock(
            wheel_score=72.4,
            wheel_recommendation="moderate",
            days_to_earnings=21,
            sector="Technology",
        )

    def screen_candidates(self, **_kwargs):
        return pd.DataFrame(
            [
                {"ticker": "MU", "wheel_score": 72.0, "strangle_score": 65.0, "iv_rank": 55.0},
                {"ticker": "AAPL", "wheel_score": 68.0, "strangle_score": 61.0, "iv_rank": 48.0},
            ]
        )

    def analyze_ticker(self, ticker, as_of=None):  # noqa: ARG002
        self.analysis.ticker = ticker
        return self.analysis


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch):
    """Replace the engine globals for every test in this module."""
    fake_conn = _FakeConnector()
    fake_runner = _FakeRunner()
    monkeypatch.setattr(engine_api, "_connector", fake_conn, raising=False)
    monkeypatch.setattr(engine_api, "_runner", fake_runner, raising=False)
    # Ensure get_connector / get_runner return our fakes even if they re-init
    monkeypatch.setattr(engine_api, "get_connector", lambda: fake_conn)
    monkeypatch.setattr(engine_api, "get_runner", lambda: fake_runner)
    # Clear the alert log between tests
    monkeypatch.setattr(engine_api, "_TV_ALERT_LOG", [])
    yield


# ---------------------------------------------------------------------------
# GET /api/tv/signal
# ---------------------------------------------------------------------------


def test_tv_signal_missing_ticker_returns_400():
    status, body = _drive("GET", "/api/tv/signal")
    assert status == 400
    assert "ticker" in body["error"]


def test_tv_signal_happy_path():
    status, body = _drive("GET", "/api/tv/signal?ticker=MU")
    assert status == 200
    assert body["ticker"] == "MU"
    assert body["ok"] is True
    assert body["phase"] in {"compression", "expansion", "post_expansion", "trend", "normal"}


# ---------------------------------------------------------------------------
# GET /api/tv/scan
# ---------------------------------------------------------------------------


def test_tv_scan_returns_signals():
    status, body = _drive("GET", "/api/tv/scan?limit=5")
    assert status == 200
    assert "signals" in body
    assert "count" in body
    assert isinstance(body["signals"], list)
    assert body["count"] == len(body["signals"])


def test_tv_scan_phase_filter_rejects_mismatches():
    status, body = _drive("GET", "/api/tv/scan?limit=5&phase=compression")
    assert status == 200
    for row in body["signals"]:
        assert row["phase"] == "compression"


# ---------------------------------------------------------------------------
# POST /api/tv/webhook
# ---------------------------------------------------------------------------


def test_tv_webhook_accepts_valid_alert():
    payload = json.dumps(
        {
            "ticker": "MU",
            "signal": "wheel_put_zone",
            "price": 82.45,
            "phase": "post_expansion",
            "source": "smart_wheel_signals_v1",
        }
    ).encode()
    status, body = _drive("POST", "/api/tv/webhook", body=payload)
    assert status == 200
    assert body["accepted"] is True
    assert "enriched" in body
    enriched = body["enriched"]
    assert enriched["ticker"] == "MU"
    assert enriched["signal"] == "wheel_put_zone"
    assert enriched["verdict"] in {"proceed", "review", "skip"}
    assert enriched["preferred_dte"] in {31, 45}
    assert len(enriched["preferred_delta_range"]) == 2
    # Ring buffer updated
    assert len(engine_api._TV_ALERT_LOG) == 1


def test_tv_webhook_rejects_missing_fields():
    payload = json.dumps({"price": 1.0}).encode()
    status, body = _drive("POST", "/api/tv/webhook", body=payload)
    assert status == 400
    assert "ticker" in body["error"].lower() or "signal" in body["error"].lower()


def test_tv_webhook_rejects_invalid_json():
    status, body = _drive("POST", "/api/tv/webhook", body=b"not json{")
    assert status == 400
    assert "json" in body["error"].lower()


def test_tv_webhook_secret_validation(monkeypatch):
    monkeypatch.setenv("TV_WEBHOOK_SECRET", "s3cret")
    payload_wrong = json.dumps(
        {"ticker": "MU", "signal": "wheel_put_zone", "secret": "wrong"}
    ).encode()
    status, body = _drive("POST", "/api/tv/webhook", body=payload_wrong)
    assert status == 401

    payload_ok = json.dumps(
        {"ticker": "MU", "signal": "wheel_put_zone", "secret": "s3cret"}
    ).encode()
    status, body = _drive("POST", "/api/tv/webhook", body=payload_ok)
    assert status == 200


# ---------------------------------------------------------------------------
# GET /api/tv/alerts
# ---------------------------------------------------------------------------


def test_tv_alerts_ring_buffer_roundtrip():
    # Post two alerts
    for tkr in ("MU", "AAPL"):
        _drive(
            "POST",
            "/api/tv/webhook",
            body=json.dumps({"ticker": tkr, "signal": "wheel_put_zone"}).encode(),
        )
    status, body = _drive("GET", "/api/tv/alerts?limit=10")
    assert status == 200
    assert body["count"] == 2
    assert {r["ticker"] for r in body["alerts"]} == {"MU", "AAPL"}


# ---------------------------------------------------------------------------
# GET /api/tv/enrich
# ---------------------------------------------------------------------------


def test_tv_enrich_ticker_required():
    status, body = _drive("GET", "/api/tv/enrich")
    assert status == 400


def test_tv_enrich_returns_decision():
    status, body = _drive("GET", "/api/tv/enrich?ticker=MU&signal=wheel_put_zone")
    assert status == 200
    assert body["ticker"] == "MU"
    assert body["verdict"] in {"proceed", "review", "skip"}
    assert body["preferred_dte"] in {31, 45}
