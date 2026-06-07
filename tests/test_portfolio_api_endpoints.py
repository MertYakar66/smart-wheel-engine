"""HTTP-level tests for the D26 read-only performance-viewer endpoints.

Spins ``engine_api``'s stdlib handler in-process on an ephemeral port and
asserts each ``GET /api/portfolio/{summary,positions,returns,income,risk,
history}`` returns the dashboard shape, an unknown sub-path 404s, and — the
observational guard — that no response carries a tradeable verdict / EV
authority token / ev_dollars field.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "ibkr"


@pytest.fixture(scope="module")
def server():
    os.environ["SWE_IBKR_DATA_DIR"] = str(FIXTURES)
    os.environ.setdefault("SWE_DATA_PROVIDER", "bloomberg")
    from engine_api import EngineAPIHandler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), EngineAPIHandler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        srv.shutdown()
        thread.join(timeout=5)
        os.environ.pop("SWE_IBKR_DATA_DIR", None)


def _get(port: int, path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


SUBS = ["summary", "positions", "returns", "income", "risk", "history"]


@pytest.mark.parametrize("sub", SUBS)
def test_endpoint_returns_200(server, sub):
    status, _ = _get(server, f"/api/portfolio/{sub}")
    assert status == 200


def test_summary_shape(server):
    status, data = _get(server, "/api/portfolio/summary")
    assert status == 200
    for key in ("netLiq", "cash", "unrealizedPnl", "realizedYtd", "premium30d", "winRate"):
        assert key in data
    assert data["netLiq"] == 144507
    assert data["winRate"] == pytest.approx(0.68)


def test_positions_shape(server):
    _, data = _get(server, "/api/portfolio/positions")
    assert "holdings" in data and len(data["holdings"]) == 9
    cls = next(h for h in data["holdings"] if h["sym"] == "CLS")
    assert cls["state"] == "csp" and cls["breach"] is True


def test_returns_shape(server):
    _, data = _get(server, "/api/portfolio/returns")
    assert set(data["returns"]) == {"1D", "1W", "1M", "3M", "YTD", "1Y", "All"}
    assert data["returns"]["YTD"]["usd"] == -8593


def test_risk_shape(server):
    _, data = _get(server, "/api/portfolio/risk")
    assert data["singleName"][0]["sym"] == "CLS"
    assert data["margin"]["stressed"] is True
    assert "gates" in data


def test_history_shape(server):
    _, data = _get(server, "/api/portfolio/history")
    assert len(data["equity"]) == 13
    assert data["equity"][0]["m"] == "Jun '25"


def test_unknown_sub_404(server):
    status, _ = _get(server, "/api/portfolio/bogus")
    assert status == 404


@pytest.mark.parametrize("sub", SUBS)
def test_observational_no_tradeable_fields(server, sub):
    """The viewer is observational — no endpoint may leak a verdict / EV
    authority / ev_dollars field (CLAUDE.md §2/§3, finding I1)."""
    _, data = _get(server, f"/api/portfolio/{sub}")
    blob = json.dumps(data).lower()
    for forbidden in ("verdict", "ev_authority", "tradeable", "ev_dollars"):
        assert forbidden not in blob


@pytest.mark.parametrize("sub", SUBS)
def test_source_is_demo_for_fixtures(server, sub):
    """Provenance honesty (browser-QA §D): served from the committed demo
    fixtures (``source: 'fixture'``), every slice must report ``source ==
    'demo'`` — never 'live' — so the dashboard cannot label fixture data as a
    live IBKR pull. A real IBKR drop (no marker) reports 'live' (unit test
    below)."""
    _, data = _get(server, f"/api/portfolio/{sub}")
    assert data.get("source") == "demo"


def test_provenance_helper_live_vs_demo():
    """``provenance()``: fixtures (or explicit demo/mock markers) → 'demo';
    a real IBKR drop (no top-level ``source``) → 'live'."""
    from engine import ibkr_portfolio_adapter as adapter

    assert adapter.provenance({}) == "live"
    assert adapter.provenance({"source": "fixture"}) == "demo"
    assert adapter.provenance({"source": "demo"}) == "demo"
    assert adapter.provenance({"source": "mock"}) == "demo"
    assert adapter.provenance({"source": "live"}) == "live"
