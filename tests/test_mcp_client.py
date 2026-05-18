"""
Tests for engine/mcp_client.py — the Stage 2 tv-CLI MCPChartClient.

Every test MOCKS subprocess; none touches a real tradingview-mcp server
or TradingView Desktop. The mock dispatches on the `tv` subcommand and
returns scripted CompletedProcess objects (or raises scripted
exceptions), so the four-call capture sequence, the canonical failure
mapping, the no-retry contract, and the defensive JSON parsing are all
exercised deterministically.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from engine.mcp_client import (
    MCPCLIClient,
    _blob,
    _classify,
    _extract_price,
    _extract_str,
)
from engine.tradingview_bridge import MCPCaptureResult, MCPChartProvider, MCPClientError


# ---------------------------------------------------------------------
# subprocess mock
# ---------------------------------------------------------------------
def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["tv"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# The tv subcommands MCPCLIClient invokes. The mock locates the
# subcommand anywhere in argv so dispatch is robust to a multi-token
# cli_command prefix (e.g. ["node", "/path/cli.js"]).
_TV_SUBCOMMANDS = ("symbol", "timeframe", "state", "screenshot")


class _FakeRun:
    """Stands in for subprocess.run.

    Dispatches on the tv subcommand — located anywhere in argv, so it is
    robust to a multi-token cli_command prefix — to a scripted response:
    a CompletedProcess to return, or an exception instance to raise.
    Records every call so tests can assert call count and ordering
    (the no-retry contract).
    """

    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = dict(responses)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):  # loose signature: stands in for subprocess.run
        self.calls.append(list(cmd))
        subcommand = next((tok for tok in cmd if tok in _TV_SUBCOMMANDS), "")
        resp = self._responses.get(subcommand)
        if resp is None:
            resp = _completed(0, "{}")
        if isinstance(resp, BaseException):
            raise resp
        return resp


def _happy(screenshot_path: str = "/tmp/shot.png", price: float = 201.5) -> dict[str, object]:
    """A scripted response set where all four steps succeed."""
    return {
        "symbol": _completed(0, '{"ok": true}'),
        "timeframe": _completed(0, '{"ok": true}'),
        "state": _completed(0, json.dumps({"symbol": "AAPL", "timeframe": "D", "price": price})),
        "screenshot": _completed(0, json.dumps({"path": str(screenshot_path)})),
    }


def _patch(monkeypatch, fake: _FakeRun) -> _FakeRun:
    monkeypatch.setattr("engine.mcp_client.subprocess.run", fake)
    return fake


# =====================================================================
# 1. Happy path — the four-call capture sequence
# =====================================================================
class TestMCPCLIClientHappyPath:
    def test_four_calls_in_order(self, monkeypatch):
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        result = MCPCLIClient().capture("aapl", "1D")

        assert fake.calls == [
            ["tv", "symbol", "AAPL"],
            ["tv", "timeframe", "D"],
            ["tv", "state"],
            ["tv", "screenshot", "-r", "chart"],
        ]
        assert isinstance(result, MCPCaptureResult)
        assert result.screenshot_path == Path("/tmp/shot.png")
        assert result.visible_price == 201.5
        assert result.visible_symbol == "AAPL"
        assert result.visible_timeframe == "D"

    def test_timeframe_token_mapping(self, monkeypatch):
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        MCPCLIClient().capture("AAPL", "1h")
        assert ["tv", "timeframe", "60"] in fake.calls

    def test_custom_cli_command_prefix(self, monkeypatch):
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        MCPCLIClient(cli_command=["node", "/x/cli.js"]).capture("AAPL", "1D")
        assert fake.calls[0] == ["node", "/x/cli.js", "symbol", "AAPL"]


# =====================================================================
# 2. Failure mapping — every path → a canonical MCP_ERROR_MODES value
# =====================================================================
class TestMCPCLIClientFailureMapping:
    def test_missing_tv_binary_is_mcp_unavailable(self, monkeypatch):
        _patch(monkeypatch, _FakeRun({"symbol": FileNotFoundError("tv")}))
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "mcp_unavailable"

    def test_screenshot_timeout_is_screenshot_timeout(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"price": 1.0}'),
                    "screenshot": subprocess.TimeoutExpired(["tv", "screenshot"], 15.0),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "screenshot_timeout"

    def test_state_timeout_is_mcp_unavailable(self, monkeypatch):
        # A hang on a non-screenshot step means the MCP/CDP is unresponsive.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": subprocess.TimeoutExpired(["tv", "state"], 15.0),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "mcp_unavailable"

    def test_symbol_not_found(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun({"symbol": _completed(1, "", "Error: no such symbol ZZZZ")}),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("ZZZZ", "1D")
        assert ei.value.error == "symbol_not_found"

    def test_browser_disconnected(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(1, "", "WebSocket closed unexpectedly"),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "browser_disconnected"

    def test_connection_refused_is_mcp_unavailable(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun({"symbol": _completed(1, "", "connect ECONNREFUSED 127.0.0.1:9222")}),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "mcp_unavailable"

    def test_generic_nonzero_exit_is_unexpected_error(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(2, "", "kaboom"),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_non_json_output_is_unexpected_error(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, "this is not json"),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_empty_output_is_unexpected_error(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, "   "),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_error_envelope_is_classified(self, monkeypatch):
        # Exit 0 but a {"ok": false} envelope still counts as failure.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"ok": false, "error": "ECONNREFUSED :9222"}'),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "mcp_unavailable"

    def test_screenshot_without_path_is_unexpected_error(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"price": 1.0}'),
                    "screenshot": _completed(0, '{"ok": true}'),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_os_error_on_spawn_is_unexpected_error(self, monkeypatch):
        _patch(monkeypatch, _FakeRun({"symbol": OSError("Exec format error")}))
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"


# =====================================================================
# 3. No-retry contract
# =====================================================================
class TestMCPCLIClientNoRetries:
    def test_aborts_on_first_failure(self, monkeypatch):
        fake = _patch(monkeypatch, _FakeRun({"symbol": _completed(1, "", "kaboom")}))
        with pytest.raises(MCPClientError):
            MCPCLIClient().capture("AAPL", "1D")
        # symbol failed → timeframe/state/screenshot never attempted.
        assert fake.calls == [["tv", "symbol", "AAPL"]]

    def test_one_call_per_step_through_screenshot(self, monkeypatch):
        fake = _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"price": 1.0}'),
                    "screenshot": _completed(1, "", "kaboom"),
                }
            ),
        )
        with pytest.raises(MCPClientError):
            MCPCLIClient().capture("AAPL", "1D")
        # Exactly four calls, no retry of the failed screenshot step.
        assert len(fake.calls) == 4


# =====================================================================
# 4. Defensive JSON handling
# =====================================================================
class TestMCPCLIClientJSONHandling:
    def test_envelope_unwrapping(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(
                        0, '{"ok": true, "data": {"symbol": "AAPL", "price": 150.0}}'
                    ),
                    "screenshot": _completed(0, '{"path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_price == 150.0

    def test_nested_price_block(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL", "quote": {"last": 175.25}}'),
                    "screenshot": _completed(0, '{"path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_price == 175.25

    def test_screenshot_alternate_path_key(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"price": 1.0}'),
                    "screenshot": _completed(0, '{"filePath": "/tmp/alt.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.screenshot_path == Path("/tmp/alt.png")

    def test_state_without_price_yields_none(self, monkeypatch):
        # Price is supplemental; a state payload with no price is not a
        # failure as long as the screenshot still succeeds.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL", "timeframe": "D"}'),
                    "screenshot": _completed(0, '{"path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_price is None
        assert result.screenshot_path == Path("/tmp/x.png")


# =====================================================================
# 5. Module-level helpers
# =====================================================================
class TestModuleHelpers:
    def test_extract_price_non_dict(self):
        assert _extract_price(None) is None
        assert _extract_price("nope") is None
        assert _extract_price([1, 2]) is None

    def test_extract_price_skips_non_numeric(self):
        assert _extract_price({"price": "N/A", "last": 12.0}) == 12.0

    def test_extract_price_first_key_wins(self):
        assert _extract_price({"price": 1.0, "last": 2.0}) == 1.0

    def test_extract_price_none_when_absent(self):
        assert _extract_price({"symbol": "AAPL"}) is None

    def test_extract_str(self):
        assert _extract_str({"symbol": "AAPL"}, ("symbol", "ticker")) == "AAPL"
        assert _extract_str({"ticker": "MSFT"}, ("symbol", "ticker")) == "MSFT"
        assert _extract_str({"symbol": "   "}, ("symbol",)) is None
        assert _extract_str("not a dict", ("symbol",)) is None

    def test_classify_symbol_not_found_only_for_symbol_step(self):
        assert _classify("chart_set_symbol", "no such symbol").error == "symbol_not_found"
        # The same wording on a different step is NOT a symbol error.
        assert _classify("chart_get_state", "no such symbol").error == "unexpected_error"

    def test_classify_fallthrough_is_unexpected_error(self):
        assert _classify("chart_get_state", "something weird").error == "unexpected_error"

    def test_blob(self):
        assert _blob("a", "b") == "a b"
        assert _blob("", "x") == "x"
        assert _blob(None, None) == ""


# =====================================================================
# 6. Composition with MCPChartProvider (Stage 1 + Stage 2)
# =====================================================================
class TestMCPCLIClientWithProvider:
    def test_provider_with_cli_client_success(self, monkeypatch, tmp_path):
        png = tmp_path / "AAPL_1D.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        _patch(monkeypatch, _FakeRun(_happy(screenshot_path=str(png), price=190.0)))

        provider = MCPChartProvider(client=MCPCLIClient())
        ctx = provider.fetch("AAPL", "1D")
        assert ctx.is_ok() is True
        assert ctx.error == ""
        assert ctx.source == "mcp"
        assert ctx.visible_price == 190.0
        assert ctx.screenshot_path == png

    def test_provider_with_cli_client_failure_routes_to_errored_context(self, monkeypatch):
        # A client failure must surface as an errored ChartContext, not
        # an exception — the provider never raises.
        _patch(
            monkeypatch,
            _FakeRun({"symbol": _completed(1, "", "connect ECONNREFUSED :9222")}),
        )
        provider = MCPChartProvider(client=MCPCLIClient())
        ctx = provider.fetch("AAPL", "1D")
        assert ctx.is_ok() is False
        assert ctx.error == "mcp_unavailable"

    def test_provider_pit_guard_skips_cli_client(self, monkeypatch):
        # PIT short-circuit sits in front of the real client too: with
        # as_of set, subprocess.run must never be invoked.
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        provider = MCPChartProvider(client=MCPCLIClient())
        ctx = provider.fetch("AAPL", "1D", as_of=datetime(2024, 6, 1))
        assert ctx.error == "pit_violation"
        assert fake.calls == []
