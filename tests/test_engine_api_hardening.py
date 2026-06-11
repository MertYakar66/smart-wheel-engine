"""Error-path + hardening tests for the engine_api.py network surface (PR C).

These pin the robustness/security hardening landed alongside the audit items
R3 (loopback bind + CORS), R18 (malformed-param → 400), R19 (no exception
leak + correlation id), R20 (unknown-ticker 404 on payoff/expected_move/
strikes), R21 (4xx instead of 200 on no-data error bodies), and the R27
§2-adjacent verdict-LABEL alignment.

We reuse the in-memory ``_drive`` harness pattern from ``test_tv_api.py``:
the handler's ``do_GET`` / ``do_POST`` only touch ``self.path``,
``self.headers`` and the POST body, so we wire those manually and capture the
response by monkeypatching ``_send_json`` / ``_send_error``. The module-level
helpers (``_resolve_host``, ``_resolve_cors_origin``, ``_parse_param``) are
tested directly since they don't depend on the request object.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import engine_api

# ---------------------------------------------------------------------------
# Request driver (mirrors tests/test_tv_api.py::_drive, with header overrides)
# ---------------------------------------------------------------------------


def _drive(
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict | None = None,
) -> tuple[int, dict]:
    """Invoke EngineAPIHandler with a synthesized request and capture the reply.

    ``headers`` lets a test override the request headers (e.g. a bogus
    ``Content-Length`` for the 413 path). When omitted we default to a
    matching ``Content-Length`` like the production path expects.
    """
    body = body or b""
    rfile = io.BytesIO(body)
    wfile = io.BytesIO()

    handler = engine_api.EngineAPIHandler.__new__(engine_api.EngineAPIHandler)
    handler.rfile = rfile
    handler.wfile = wfile
    handler.client_address = ("127.0.0.1", 0)
    handler.command = method
    handler.path = path
    handler.request_version = "HTTP/1.1"
    handler.headers = headers if headers is not None else {"Content-Length": str(len(body))}
    handler.server = MagicMock()
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.close_connection = True

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
# Connector / runner stubs. Tickers in _KNOWN return OHLCV; everything else
# returns an empty frame so the R20 unknown-ticker guard fires.
# ---------------------------------------------------------------------------

_KNOWN = {"AAPL", "MSFT", "MU"}


def _fake_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = np.full(n, 1_000_000.0)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


class _FakeConnector:
    def get_ohlcv(self, ticker):
        if (ticker or "").upper() in _KNOWN:
            return _fake_ohlcv()
        return pd.DataFrame()  # unknown ticker → empty (R20 trigger)

    def get_fundamentals(self, ticker):
        if (ticker or "").upper() in _KNOWN:
            return {"implied_vol_atm": 30.0, "sector": "Technology"}
        return None

    def get_iv_history(self, ticker):
        return pd.DataFrame()  # always empty → R21 no-IV path

    def get_universe(self):
        return sorted(_KNOWN)

    def get_vix_regime(self, *_a, **_k):
        return {"vix": 18.0}


class _FakeRunner:
    def __init__(self):
        self.connector = _FakeConnector()


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch):
    fake_conn = _FakeConnector()
    fake_runner = _FakeRunner()
    monkeypatch.setattr(engine_api, "_connector", fake_conn, raising=False)
    monkeypatch.setattr(engine_api, "_runner", fake_runner, raising=False)
    monkeypatch.setattr(engine_api, "get_connector", lambda: fake_conn)
    monkeypatch.setattr(engine_api, "get_runner", lambda: fake_runner)
    yield


# ---------------------------------------------------------------------------
# R3 — loopback bind default + CORS allow-list (module helpers)
# ---------------------------------------------------------------------------


def test_resolve_host_defaults_to_loopback():
    assert engine_api._resolve_host({}) == "127.0.0.1"


def test_resolve_host_honours_env_override():
    assert engine_api._resolve_host({"SWE_API_HOST": "0.0.0.0"}) == "0.0.0.0"
    assert engine_api._resolve_host({"SWE_API_HOST": "  10.0.0.5 "}) == "10.0.0.5"


def test_cors_allows_localhost_origins():
    for origin in (
        "http://localhost:3000",
        "https://localhost",
        "http://127.0.0.1:8787",
        "http://[::1]:3000",
    ):
        assert engine_api._resolve_cors_origin(origin, env={}) == origin


def test_cors_denies_foreign_origin_by_default():
    assert engine_api._resolve_cors_origin("https://evil.example.com", env={}) is None


def test_cors_echoes_configured_origin():
    env = {"SWE_API_CORS_ORIGIN": "https://dash.internal"}
    assert (
        engine_api._resolve_cors_origin("https://dash.internal", env=env) == "https://dash.internal"
    )
    assert engine_api._resolve_cors_origin("https://other.internal", env=env) is None


def test_cors_no_origin_header_returns_none():
    # Server-side proxy / curl send no Origin — header is simply omitted.
    assert engine_api._resolve_cors_origin(None, env={}) is None
    assert engine_api._resolve_cors_origin("", env={}) is None


# ---------------------------------------------------------------------------
# R18 — malformed query params → clean 400 (helper + dispatch)
# ---------------------------------------------------------------------------


def test_parse_param_coerces_and_defaults():
    assert engine_api._parse_param("dte", "35", int, 99) == 35
    assert engine_api._parse_param("delta", "0.25", float, 1.0) == 0.25
    # Empty / None fall back to default
    assert engine_api._parse_param("dte", None, int, 45) == 45
    assert engine_api._parse_param("dte", "", int, 45) == 45


def test_parse_param_raises_badparam_on_garbage():
    with pytest.raises(engine_api.BadParam) as ei:
        engine_api._parse_param("dte", "abc", int)
    assert "dte" in str(ei.value)


def test_malformed_int_param_returns_400():
    status, body = _drive("GET", "/api/iv_history?ticker=AAPL&days=notanumber")
    assert status == 400
    assert "days" in body["error"]


def test_malformed_float_param_returns_400():
    status, body = _drive("GET", "/api/tv/ranked?delta=xyz")
    assert status == 400
    assert "delta" in body["error"]


def test_malformed_dte_on_payoff_returns_400():
    status, body = _drive("GET", "/api/payoff?ticker=AAPL&dte=soon")
    assert status == 400
    assert "dte" in body["error"]


# ---------------------------------------------------------------------------
# R19 — catch-all / handler 500s must not leak exception text
# ---------------------------------------------------------------------------


def test_handler_exception_returns_generic_500_with_error_id(monkeypatch):
    secret = "leak-me-filesystem-path-/etc/passwd"

    def _boom(self):
        raise RuntimeError(secret)

    monkeypatch.setattr(engine_api.EngineAPIHandler, "_handle_status", _boom, raising=True)
    status, body = _drive("GET", "/api/status")
    assert status == 500
    assert body["error"] == "internal server error"
    # Correlation id present and looks like an 8-hex token
    assert "error_id" in body
    assert len(body["error_id"]) == 8
    int(body["error_id"], 16)  # raises if not hex
    # Raw exception text must NOT be in the client body
    assert secret not in json.dumps(body)


def test_post_handler_exception_returns_generic_500(monkeypatch):
    def _boom(self, payload):
        raise RuntimeError("pandas internals leak /var/data/x.parquet")

    monkeypatch.setattr(engine_api.EngineAPIHandler, "_handle_news_ingest", _boom, raising=True)
    payload = json.dumps({"stories": []}).encode()
    status, body = _drive("POST", "/api/news/ingest", body=payload)
    assert status == 500
    assert body["error"] == "internal server error"
    assert "error_id" in body
    assert "parquet" not in json.dumps(body)


# ---------------------------------------------------------------------------
# R20 — unknown ticker on payoff / expected_move / strikes → 404
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/payoff?ticker=ZZZZ&strategy=csp",
        "/api/expected_move?ticker=ZZZZ&dte=45",
        "/api/strikes?ticker=ZZZZ&strategy=csp&dte=45",
    ],
)
def test_unknown_ticker_returns_404(path):
    status, body = _drive("GET", path)
    assert status == 404
    assert "ZZZZ" in body["error"]


@pytest.mark.parametrize(
    "path",
    [
        "/api/payoff?ticker=AAPL&strategy=csp",
        "/api/expected_move?ticker=AAPL&dte=45",
        "/api/strikes?ticker=AAPL&strategy=csp&dte=45",
    ],
)
def test_known_ticker_still_200(path):
    status, _body = _drive("GET", path)
    assert status == 200


# ---------------------------------------------------------------------------
# R21 — error bodies that used to be 200 now carry a 4xx status
# ---------------------------------------------------------------------------


def test_chart_no_data_returns_404():
    status, body = _drive("GET", "/api/chart/bollinger?ticker=ZZZZ")
    assert status == 404
    assert "error" in body
    assert body["data"] == []  # shape preserved for chart clients


def test_iv_history_no_data_returns_404():
    status, body = _drive("GET", "/api/iv_history?ticker=AAPL")
    assert status == 404
    assert "error" in body
    assert body["data"] == []


def test_fundamentals_not_found_returns_404():
    status, body = _drive("GET", "/api/fundamentals?ticker=ZZZZ")
    assert status == 404
    assert "error" in body


def test_strangle_insufficient_data_returns_422(monkeypatch):
    # Force a too-short OHLCV history so the < 100-bar guard fires.
    fake = _FakeConnector()
    monkeypatch.setattr(fake, "get_ohlcv", lambda t: _fake_ohlcv(n=50))
    monkeypatch.setattr(engine_api, "get_connector", lambda: fake)
    status, body = _drive("GET", "/api/strangle?ticker=AAPL")
    assert status == 422
    assert body["error"] == "Insufficient data"


# ---------------------------------------------------------------------------
# R32 / routing — unknown path, oversized body, invalid POST
# ---------------------------------------------------------------------------


def test_unknown_get_path_returns_404():
    status, body = _drive("GET", "/api/does-not-exist")
    assert status == 404
    assert "Unknown endpoint" in body["error"]


def test_unknown_post_path_returns_404():
    status, body = _drive("POST", "/api/nope", body=b"{}")
    assert status == 404
    assert "Unknown endpoint" in body["error"]


def test_oversized_post_body_returns_413():
    # Content-Length over the 16 KB cap → 413 before the body is read.
    big = b"x" * (16 * 1024 + 1)
    status, body = _drive("POST", "/api/tv/webhook", body=big)
    assert status == 413
    assert "too large" in body["error"].lower()


def test_invalid_json_post_returns_400():
    status, body = _drive("POST", "/api/tv/webhook", body=b"{not valid json")
    assert status == 400
    assert "json" in body["error"].lower()


def test_non_dict_json_post_returns_400():
    status, body = _drive("POST", "/api/tv/webhook", body=b"[1, 2, 3]")
    assert status == 400
    assert "object" in body["error"].lower()


def test_bogus_content_length_returns_400():
    status, body = _drive(
        "POST", "/api/tv/webhook", body=b"{}", headers={"Content-Length": "not-a-number"}
    )
    assert status == 400
    assert "content-length" in body["error"].lower()


# ---------------------------------------------------------------------------
# R27 — §2-adjacent verdict-LABEL alignment (negative / non-finite EV → blocked)
# ---------------------------------------------------------------------------


def test_enrich_alert_negative_ev_label_is_blocked(monkeypatch):
    """A negative-EV candidate must surface verdict='blocked' (was 'skip'),
    matching the dossier reviewer R1. Reason string unchanged."""
    from engine.tv_signals import TVAlert

    handler = engine_api.EngineAPIHandler.__new__(engine_api.EngineAPIHandler)

    # Stub the runner so the EV path returns a clean negative-EV row.
    neg_df = pd.DataFrame(
        [
            {
                "ev_dollars": -42.0,
                "ev_per_day": -1.0,
                "prob_profit": 0.4,
                "prob_assignment": 0.6,
                "strike": 90.0,
                "iv": 0.3,
                "dte": 35,
                "contracts": 1,
            }
        ]
    )
    runner = MagicMock()
    runner.rank_candidates_by_ev.return_value = neg_df
    runner.connector = _FakeConnector()
    runner.analyze_ticker.return_value = MagicMock(days_to_earnings=21, sector="Technology")
    monkeypatch.setattr(engine_api, "get_runner", lambda: runner)
    monkeypatch.setattr(engine_api, "get_connector", lambda: _FakeConnector())

    alert = TVAlert(ticker="AAPL", signal="wheel_put_zone", source="api")
    enriched = handler._enrich_alert(alert)

    assert enriched["verdict"] == "blocked"
    assert enriched["verdict_reason"] == "negative_ev"


def test_enrich_alert_non_finite_ev_label_is_blocked(monkeypatch):
    """Non-finite EV must surface verdict='blocked' with reason 'ev_non_finite',
    matching the dossier reviewer R1a."""
    from engine.tv_signals import TVAlert

    handler = engine_api.EngineAPIHandler.__new__(engine_api.EngineAPIHandler)

    inf_df = pd.DataFrame(
        [
            {
                "ev_dollars": float("inf"),
                "ev_per_day": 0.0,
                "prob_profit": 0.99,
                "prob_assignment": 0.1,
                "strike": 90.0,
                "iv": 0.3,
                "dte": 35,
                "contracts": 1,
            }
        ]
    )
    runner = MagicMock()
    runner.rank_candidates_by_ev.return_value = inf_df
    runner.connector = _FakeConnector()
    runner.analyze_ticker.return_value = MagicMock(days_to_earnings=21, sector="Technology")
    monkeypatch.setattr(engine_api, "get_runner", lambda: runner)
    monkeypatch.setattr(engine_api, "get_connector", lambda: _FakeConnector())

    alert = TVAlert(ticker="AAPL", signal="wheel_put_zone", source="api")
    enriched = handler._enrich_alert(alert)

    assert enriched["verdict"] == "blocked"
    assert enriched["verdict_reason"] == "ev_non_finite"
